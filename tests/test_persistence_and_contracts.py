import asyncio
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from core.project import ProjectContract, Dependency, validate_artifacts
from core.orchestrator import run_orchestration
from server.app import orchestrate_endpoint
from server import db
from test_project_pipeline import STATIC,FixtureServer,workdir,request


class TestContractFailures(unittest.TestCase):
    def test_dependency_versions_must_be_pinned(self):
        for version in ('latest','^4.2.0','>=1.0','1.x','1.0.*','https://example.com/package'):
            with self.subTest(version=version), self.assertRaises(ValueError):
                Dependency(ecosystem='npm',name='example',version=version)

    def test_missing_html_asset_fails_packaging(self):
        with workdir(),FixtureServer() as fixture:
            fixture.sources['pages/index.html']='<h1>Broken fixture</h1><script src="missing.js"></script>'
            report=asyncio.run(run_orchestration('missing asset',2,'ollama/fixture-coder'))
            self.assertEqual(report.packaging['status'],'failed')
            self.assertIn('missing local asset',report.packaging['error'])
            self.assertFalse((Path(report.output_dir)/'start.sh').exists())

    def test_openrouter_is_optional_but_requires_its_own_key(self):
        req=request();req.provider='openrouter';req.helm_model='example/planner'
        req.worker_models={'agent_1':'example/worker','agent_2':'example/worker'}
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':''}):
            with self.assertRaises(HTTPException) as error:asyncio.run(orchestrate_endpoint(req))
            self.assertEqual(error.exception.status_code,400)

    def test_legacy_database_migrates_and_new_metadata_survives(self):
        with workdir():
            db.init_db()
            with sqlite3.connect(db.DB_PATH) as conn:
                conn.execute('ALTER TABLE runs DROP COLUMN report_json')
            record=dict(run_id='legacy',task='legacy task',helm_model='ollama/old',num_agents=2,
                wall_clock_s=1,sum_of_agent_times_s=1,concurrency_speedup=1,total_tokens_in=1,
                total_tokens_out=2,total_cost_est=0,output_dir='/old/path',files_written=[],agents=[])
            db.save_run(record)
            self.assertEqual(db.get_run_by_id('legacy')['task'],'legacy task')
            self.assertEqual(len(db.get_all_runs()),1)
            record.update(run_id='new',project=STATIC,packaging={'status':'ready'},
                          phase_usage={'planner':{'tokens_in':3,'tokens_out':4}})
            db.save_run(record)
            stored=db.get_run_by_id('new')
            self.assertEqual(stored['project'],STATIC)
            self.assertEqual(stored['packaging']['status'],'ready')
            self.assertEqual(stored['phase_usage']['planner']['tokens_in'],3)
