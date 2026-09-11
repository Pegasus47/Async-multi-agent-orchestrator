import asyncio
import os
import unittest
from unittest.mock import patch

from core.models import SubTaskSpec, HelmPlan, AgentResult, OrchestrationReport
from core.orchestrator import run_orchestration, extract_json_from_text
from server.db import init_db, save_run, get_all_runs, get_usage_billing_report, get_analytics_leaderboard

class TestOrchestrator(unittest.TestCase):
    def test_json_extraction(self):
        markdown_json = "```json\n{\"plan_summary\": \"test plan\", \"agent_1\": {\"role\": \"r\", \"filename\": \"f\", \"instructions\": \"i\"}}\n```"
        extracted = extract_json_from_text(markdown_json)
        self.assertEqual(extracted["plan_summary"], "test plan")

    @patch("core.orchestrator.call_openrouter")
    def test_live_orchestration_flow(self, mock_call):
        async def fake_call(model, system_prompt, user_prompt, **kwargs):
            if "Helm" in system_prompt or "master engineering orchestrator" in system_prompt:
                # Return Helm plan
                return '''{
                    "plan_summary": "Modular build test",
                    "agent_1": {"role": "html_architect", "filename": "index.html", "instructions": "Write HTML"},
                    "agent_2": {"role": "css_stylist", "filename": "style.css", "instructions": "Write CSS"},
                    "agent_3": {"role": "js_engineer", "filename": "script.js", "instructions": "Write JS"}
                }''', {"prompt_tokens": 120, "completion_tokens": 80}
            else:
                # Return code for worker
                return "console.log('Worker executed live');", {"prompt_tokens": 80, "completion_tokens": 50}

        mock_call.side_effect = fake_call

        report = asyncio.run(run_orchestration(
            task="build a test site",
            num_agents=3,
            api_key="test-key-mocked"
        ))

        self.assertTrue(report.wall_clock_s >= 0)
        self.assertEqual(len(report.files_written), 3)
        self.assertEqual(report.num_agents, 3)
        self.assertTrue(os.path.exists(report.output_dir))

    def test_database_persistence_and_billing(self):
        init_db()
        report_dict = {
            "run_id": "test_run_123",
            "task": "build test database project",
            "helm_model": "google/gemma-4-31b-it:free",
            "num_agents": 3,
            "plan_summary": "Test decomposition",
            "wall_clock_s": 2.1,
            "sum_of_agent_times_s": 4.5,
            "concurrency_speedup": 2.14,
            "total_tokens_in": 350,
            "total_tokens_out": 850,
            "total_cost_est": 0.0,
            "output_dir": "/tmp/test_workspace",
            "files_written": ["index.html", "style.css"],
            "agents": [
                {"agent_id": "agent_1", "model": "cohere/north-mini-code:free", "role": "html", "duration_s": 1.2, "tokens_in": 100, "tokens_out": 250, "cost_est": 0.0, "filename": "index.html"},
                {"agent_id": "agent_2", "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "role": "css", "duration_s": 1.5, "tokens_in": 150, "tokens_out": 300, "cost_est": 0.0, "filename": "style.css"}
            ]
        }
        save_run(report_dict)
        runs = get_all_runs(limit=5)
        self.assertTrue(len(runs) >= 1)

        billing = get_usage_billing_report()
        self.assertIn("summary", billing)
        self.assertIn("model_breakdown", billing)
        self.assertTrue(billing["summary"]["total_runs"] >= 1)

if __name__ == "__main__":
    unittest.main()
