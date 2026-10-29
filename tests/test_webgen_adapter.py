import copy
import json
from pathlib import Path
import tempfile
import unittest
from tools.webgen_pilot import load_task,prepare,map_run

class TestWebGenAdapter(unittest.TestCase):
    def test_pinned_prompt_cases_mapping_and_scope(self):
        task=load_task()
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)
            result=prepare(output,task)
            self.assertEqual((output/'prompt.txt').read_text(),task['task']['instruction'])
            self.assertEqual(result['functional_cases'],7)
            manifest=output/'run.json'
            manifest.write_text(json.dumps({'task':task['task']['instruction'],'run_id':'local-test','packaging':{'status':'ready'}}))
            mapped=map_run(manifest,'http://127.0.0.1:8080/index.html',output,task)
            cases=[json.loads(line) for line in (output/'webvoyager_tasks.jsonl').read_text().splitlines()]
            self.assertEqual(len(cases),7)
            self.assertEqual(cases[0]['task'],task['task']['ui_instruct'][0]['task'])
            self.assertEqual(cases[0]['expected_result'],task['task']['ui_instruct'][0]['expected_result'])
            single=map_run(manifest,'http://localhost:8080',output/'single',task,case_indices=[0])
            self.assertEqual(single['included_case_ids'],['000080_0'])
            self.assertFalse(mapped['full_benchmark_result'])
            self.assertIsNone(mapped['functional_score'])
            with self.assertRaises(ValueError): map_run(manifest,'http://example.com',output,task)
            manifest.write_text(json.dumps({'task':'different task','packaging':{'status':'ready'}}))
            with self.assertRaises(ValueError): map_run(manifest,'http://localhost:8080',output,task)
