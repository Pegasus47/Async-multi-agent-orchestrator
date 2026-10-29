import asyncio
import json
from pathlib import Path
import subprocess
import time
import unittest
import urllib.request

from core.orchestrator import run_orchestration
from test_project_pipeline import FixtureServer, workdir, unused_port


class TestBackendLaunch(unittest.TestCase):
    def test_generated_python_and_node_launchers(self):
        for kind in ('python','node'):
            with self.subTest(kind=kind), workdir(), FixtureServer() as fixture:
                port=unused_port()
                if kind=='python':
                    fixture.project=dict(application_type='python',runtime='python',runtime_version='3.12',
                        entry_point='app.py',files=['app.py','helper.py'],start_command=['.venv/bin/python','app.py'],port=port)
                    fixture.sources={'app.py':f'from http.server import HTTPServer, SimpleHTTPRequestHandler\nHTTPServer(("127.0.0.1", {port}), SimpleHTTPRequestHandler).serve_forever()\n', 'helper.py':'# fixture'}
                else:
                    fixture.project=dict(application_type='node',runtime='node',runtime_version='18+',
                        entry_point='server.js',files=['server.js','package.json'],dependency_manifests=['package.json'],
                        start_command=['npm','run','start'],port=port)
                    fixture.sources={'server.js':f'require("http").createServer((q,r)=>r.end("Node fixture ready")).listen({port}, "127.0.0.1");',
                                     'package.json':json.dumps({'private':True,'scripts':{'start':'node server.js'}})}
                report=asyncio.run(run_orchestration('Backend fixture',2,'ollama/fixture-coder'))
                self.assertEqual(report.packaging['status'],'ready',report.packaging)
                workspace=Path(report.output_dir)
                subprocess.run(['bash','setup.sh'],cwd=workspace,check=True,capture_output=True,timeout=30)
                with (workspace/'startup.log').open('w') as log:
                    proc=subprocess.Popen(['bash','start.sh'],cwd=workspace,stdout=log,stderr=log,start_new_session=True)
                    try:
                        for _ in range(100):
                            try:
                                with urllib.request.urlopen(f'http://127.0.0.1:{port}/',timeout=1) as response:
                                    self.assertEqual(response.status,200)
                                break
                            except OSError:
                                if proc.poll() is not None: self.fail((workspace/'startup.log').read_text())
                                time.sleep(.05)
                        else: self.fail('Startup timeout')
                    finally:
                        import os,signal
                        os.killpg(proc.pid,signal.SIGTERM)
                        proc.wait(timeout=5)
