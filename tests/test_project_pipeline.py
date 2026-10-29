"""Local integration tests. Inference replies are controlled fixtures, not real LLM output."""
import asyncio
from contextlib import contextmanager
import functools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, AsyncMock
import urllib.request

from fastapi import HTTPException
from core.orchestrator import run_orchestration, plan_task
from core.project import ProjectContract, validate_tree, validate_artifacts, write_packaging
from server.app import OrchestrateRequest, orchestrate_endpoint, get_models
from server import db


STATIC = dict(application_type='static', runtime='browser', runtime_version='modern',
              entry_point='pages/index.html', files=['pages/index.html', 'css/style.css'], port=8080)
HTML = '<!doctype html><html><head><link rel="stylesheet" href="../css/style.css"></head><body><h1>Protocol fixture</h1><button onclick="document.querySelector(\'output\').textContent=2+3">Add</button><output>0</output></body></html>'
CSS = 'body { color: rgb(0, 100, 0); }'


@contextmanager
def workdir():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='orchestrator-pipeline-') as td:
        os.chdir(td)
        with patch.object(db, 'DB_PATH', str(Path(td) / 'data' / 'test.db')):
            try:
                yield Path(td)
            finally:
                os.chdir(previous)


class FixtureServer:
    """Real local HTTP transport for the Ollama protocol, with deterministic test replies."""
    def __init__(self):
        self.requests = []
        self.project = dict(STATIC)
        self.sources = {'pages/index.html': HTML, 'css/style.css': CSS}
        self.review_files = {}
        self.review_project = None
        self.bad_plan = False
        self.empty_worker = False
        self.offline_models = False
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def reply(self, data, status=200):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_GET(self):
                owner.requests.append(('GET', self.path))
                if owner.offline_models:
                    return self.reply({}, 503)
                self.reply({'models':[{'name':'fixture-coder'}]})
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append(('POST', self.path, payload, dict(self.headers)))
                system, user = [m['content'] for m in payload['messages']]
                if 'master engineering orchestrator' in system:
                    content = {'plan_summary':'Two nested fixture files', 'project':owner.project,
                               **{f'agent_{i+1}':{'role':'source', 'filename':name, 'instructions':'Use shared files.'}
                                  for i, name in enumerate(owner.project['files'])}}
                    if owner.bad_plan:
                        content['agent_2']['filename'] = content['agent_1']['filename']
                elif 'final packaging reviewer' in system:
                    content = {'project':owner.review_project or owner.project, 'findings':['Fixture review: no model inference performed'], 'files':owner.review_files}
                else:
                    target = user.split('Target Filename: ',1)[1].splitlines()[0]
                    content = owner.sources.get(target, '')
                    if owner.empty_worker:
                        content = ''
                if not isinstance(content, str):
                    content = json.dumps(content)
                self.reply({'choices':[{'message':{'content':content}}],
                            'usage':{'prompt_tokens':10,'completion_tokens':20}})
        self.server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    def __enter__(self):
        self.thread.start()
        self.env = patch.dict(os.environ, {'OLLAMA_BASE_URL':f'http://127.0.0.1:{self.server.server_port}', 'OLLAMA_TIMEOUT_S':'5'})
        self.env.start()
        return self
    def __exit__(self, *args):
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def request(**kwargs):
    return OrchestrateRequest(task='Protocol fixture only', num_agents=2, provider='ollama',
        helm_model='ollama/fixture-coder', worker_models={'agent_1':'ollama/fixture-coder','agent_2':'ollama/fixture-coder'}, **kwargs)


def unused_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


class TestContract(unittest.TestCase):
    def test_paths_collisions_and_reserved_names(self):
        for files in [['../escape.py'], ['/tmp/escape'], ['a\\b'], ['a','a/b'], ['a','A'], ['setup.sh'], ['x//a'], ['.env']]:
            with self.subTest(files=files), self.assertRaises(ValueError):
                validate_tree(files)

    def test_reject_shell_and_nonexistent_entry(self):
        for update in [dict(start_command=['bash','-c','echo bad']), dict(entry_point='missing.html'), dict(environment_variables=['TOKEN=secret'])]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                ProjectContract.model_validate(STATIC | update)

    def test_node_dependency_and_script_validation(self):
        project = ProjectContract.model_validate(dict(application_type='node', runtime='node',runtime_version='22',
            entry_point='server.js', files=['server.js','package.json'], dependency_manifests=['package.json'],
            dependencies=[dict(ecosystem='npm',name='express',version='4.21.2')],
            start_command=['npm','run','start']))
        with self.assertRaises(ValueError):
            validate_artifacts(project, {'server.js':'// test', 'package.json':'{"scripts":{"start":"node server.js"}}'})


class TestOllamaPipeline(unittest.TestCase):
    def test_keyless_http_to_packaging_db_and_runnable_static(self):
        with workdir() as td, FixtureServer() as fixture, patch.dict(os.environ, {'OPENROUTER_API_KEY':''}):
            fixture.project['port'] = unused_port()
            report = asyncio.run(orchestrate_endpoint(request()))
            self.assertEqual(report['packaging']['status'],'ready')
            self.assertEqual(report['packaging']['startup'],'not_run')
            self.assertEqual(report['total_tokens_in'],40)
            self.assertEqual(report['total_tokens_out'],80)
            workspace = Path(report['output_dir'])
            for filename in ('setup.sh','start.sh','SETUP.md','project.json','packaging.json','first_attempt.json','run.json'):
                self.assertTrue((workspace/filename).is_file(),filename)
            self.assertEqual(db.get_run_by_id(report['run_id'])['project']['entry_point'],'pages/index.html')
            self.assertEqual(sum(m['tokens_in'] for m in db.get_usage_billing_report()['model_breakdown']),40)
            generated = json.loads((workspace/'run.json').read_text())
            self.assertEqual(generated['packaging']['status'],'ready')
            posts = [r for r in fixture.requests if r[0]=='POST']
            self.assertEqual(len(posts),4)
            self.assertTrue(all(r[1]=='/v1/chat/completions' for r in posts))
            self.assertTrue(all('Authorization' not in r[3] for r in posts))
            self.assertTrue(all(r[2]['model']=='fixture-coder' for r in posts))
            self.assertIn('response_format',posts[-1][2])
            subprocess.run(['bash','setup.sh'],cwd=workspace,check=True,capture_output=True)
            with (workspace/'startup.log').open('w') as log:
                proc = subprocess.Popen(['bash','start.sh'],cwd=workspace,stdout=log,stderr=log)
                try:
                    url=f"http://127.0.0.1:{fixture.project['port']}/pages/index.html"
                    for _ in range(50):
                        try:
                            with urllib.request.urlopen(url,timeout=1) as response:
                                self.assertIn(b'Protocol fixture',response.read())
                            break
                        except OSError:
                            if proc.poll() is not None: self.fail('Generated start script exited')
                            time.sleep(.05)
                    else: self.fail('Generated start script did not become ready')
                finally:
                    proc.terminate(); proc.wait(timeout=5)

    def test_invalid_plan_prevents_all_workers(self):
        with workdir(), FixtureServer() as fixture:
            fixture.bad_plan=True
            with self.assertRaises(ValueError):
                asyncio.run(run_orchestration('test',2,'ollama/fixture-coder'))
            self.assertEqual(len([r for r in fixture.requests if r[0]=='POST']),2)
            self.assertFalse(Path('workspaces').exists())

    def test_worker_failure_cannot_produce_ready_packaging(self):
        with workdir(), FixtureServer() as fixture:
            fixture.empty_worker=True
            report=asyncio.run(run_orchestration('test',2,'ollama/fixture-coder'))
            self.assertEqual(report.packaging['status'],'failed')
            self.assertFalse((Path(report.output_dir)/'start.sh').exists())
            self.assertTrue((Path(report.output_dir)/'run.json').exists())

    def test_packaging_cannot_overwrite_source(self):
        with workdir(), FixtureServer() as fixture:
            fixture.review_files={'pages/index.html':'overwrite'}
            report=asyncio.run(run_orchestration('test',2,'ollama/fixture-coder'))
            self.assertEqual(report.packaging['status'],'failed')
            self.assertEqual((Path(report.output_dir)/'pages/index.html').read_text(),HTML)

    def test_missing_model_and_mixed_provider_rejected_before_inference(self):
        with workdir(), FixtureServer() as fixture:
            req=request()
            req.worker_models['agent_1']='ollama/not-installed'
            with self.assertRaises(HTTPException) as error:
                asyncio.run(orchestrate_endpoint(req))
            self.assertEqual(error.exception.status_code,400)
            req.worker_models['agent_1']='openrouter/free'
            with self.assertRaises(HTTPException): asyncio.run(orchestrate_endpoint(req))
            self.assertFalse(any(r[0]=='POST' for r in fixture.requests))

    def test_unreachable_endpoint_and_login_node_do_not_start_anything(self):
        with FixtureServer() as fixture:
            fixture.offline_models=True
            with self.assertRaises(HTTPException) as error: asyncio.run(get_models('ollama'))
            self.assertEqual(error.exception.status_code,503)
            with patch('server.app.on_login_node',return_value=True):
                with self.assertRaises(HTTPException) as error: asyncio.run(orchestrate_endpoint(request()))
                self.assertEqual(error.exception.status_code,403)

    def test_missing_dependency_manifest_can_be_added(self):
        with workdir(), FixtureServer() as fixture:
            # Initial Python project is standard-library-only. Review adds the required manifest.
            fixture.project=dict(application_type='python',runtime='python',runtime_version='3.12',
                entry_point='app.py',files=['app.py','helper.py'],start_command=['.venv/bin/python','app.py'])
            fixture.sources={'app.py':'print("fixture")','helper.py':'# helper'}
            fixture.review_project=fixture.project | dict(files=['app.py','helper.py','requirements.txt'],
                dependency_manifests=['requirements.txt'],dependencies=[dict(ecosystem='pip',name='example',version='1.0')],
                install_commands=[['.venv/bin/python','-m','pip','install','-r','requirements.txt']])
            fixture.review_files={'requirements.txt':'example==1.0\n'}
            report=asyncio.run(run_orchestration('test',2,'ollama/fixture-coder'))
            self.assertEqual(report.packaging['status'],'ready')
            self.assertEqual((Path(report.output_dir)/'requirements.txt').read_text(),'example==1.0\n')
            # Installation was not executed by generation.
            self.assertFalse((Path(report.output_dir)/'.venv').exists())
