from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from aeroguard.contracts import OrderUpdate, Policy, Settings, TrainRequest
from aeroguard.repository import Conflict, Repository


def packet(cycle=1, **values):
    return {"asset": "FD001/test/001", "cycle": cycle, "bundle_id": "fixture-model",
            "rul": 4., "lower": 1., "upper": 7., "anomaly": .8,
            "quality_valid": True, "data_quality_valid": True, "history_length": 30} | values


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "ag.sqlite"
        self.repo = Repository(self.path)

    def test_job_idempotency_conflicting_key_and_one_active_job(self):
        first, created = self.repo.create_job("train", {"seed": 42}, "request-1")
        self.assertTrue(created)
        repeated, created = self.repo.create_job("train", {"seed": 42}, "request-1")
        self.assertEqual(first, repeated)
        self.assertFalse(created)
        with self.assertRaisesRegex(Conflict, "different inputs"):
            self.repo.create_job("train", {"seed": 13}, "request-1")
        with self.assertRaisesRegex(Conflict, "already running"):
            self.repo.create_job("train", {"seed": 42}, "request-2")
        self.repo.update_job(first, status="complete", result={"run_id": "fixture"})
        self.assertEqual(self.repo.job(first)["result"], {"run_id": "fixture"})
        next_id, created = self.repo.create_job("pipeline", {"cycles": 5}, "request-2")
        self.assertTrue(created)
        self.assertNotEqual(first, next_id)

    def test_failed_stage_and_restart_recovery_are_persistent(self):
        job_id, _ = self.repo.create_job("train", {"subsets": ["FD001"]}, "recovery")
        self.repo.update_job(job_id, status="running")
        failed = self.repo.stage_start(job_id, "validation", "FD001")
        self.repo.stage_finish(failed, .23, records=0, error="malformed dataset")
        unfinished = self.repo.stage_start(job_id, "features", "FD001")
        restarted = Repository(self.path)
        job = restarted.job(job_id)
        self.assertEqual(job["status"], "interrupted")
        self.assertEqual(job["stages"][0]["status"], "failed")
        self.assertEqual(job["stages"][0]["error"], "malformed dataset")
        self.assertEqual(job["stages"][1]["status"], "interrupted")
        self.assertIsNotNone(job["finished_at"])

    def test_settings_are_validated_saved_and_audited(self):
        settings = self.repo.settings().model_dump()
        settings["workspace_name"] = "Engine laboratory"
        settings["policy"]["persistence"] = 5
        self.repo.save_settings(settings)
        self.assertEqual(Repository(self.path).settings().policy.persistence, 5)
        self.assertEqual(self.repo.history("settings", "workspace")[0]["action"], "updated")
        for value in [{"critical_lower": 40, "attention_lower": 30}, {"anomaly_attention": .99, "anomaly_extreme": .9}]:
            with self.assertRaises(ValidationError):
                Policy(**value)
        with self.assertRaises(ValidationError):
            Settings(promotion_rmse_tolerance_percent=-1)
        with self.assertRaises(ValidationError):
            TrainRequest(subsets=["FD001", "FD001"])

    def test_order_owner_progress_resolution_and_audit_are_mode_isolated(self):
        self.repo.store.process(packet(), "benchmark:fixture")
        self.repo.store.process(packet(asset="FD001/demo/001"), "demo")
        order = self.repo.orders("benchmark")[0]
        self.assertTrue(order["simulated"])
        self.assertEqual(order["predicted_rul"], 4.)
        self.repo.update_order(order["id"], "benchmark", OrderUpdate(action="assign", owner="Technician A", notes="Review scheduled"))
        self.repo.update_order(order["id"], "benchmark", OrderUpdate(action="acknowledge"))
        started = self.repo.update_order(order["id"], "benchmark", OrderUpdate(action="start"))
        self.assertEqual(started["status"], "in_progress")
        resolved = self.repo.update_order(order["id"], "benchmark", OrderUpdate(action="resolve", resolution="preventive_service", notes="Simulated service completed"))
        self.assertEqual(resolved["owner"], "Technician A")
        self.assertEqual(resolved["notes"], "Simulated service completed")
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(len(self.repo.history("order", order["id"])), 4)
        self.assertEqual(self.repo.orders("demo")[0]["status"], "open")
        with self.assertRaises(Conflict):
            self.repo.update_order(order["id"], "benchmark", OrderUpdate(action="start"))
        with self.assertRaises(ValueError):
            self.repo.update_order(order["id"], "demo", OrderUpdate(action="acknowledge"))

    def test_required_maintenance_outcome_and_owner_inputs(self):
        for values in [{"action": "assign", "owner": "  "}, {"action": "resolve", "resolution": "sensor_repair"}, {"action": "resolve", "notes": "done"}]:
            with self.assertRaises(ValidationError):
                OrderUpdate(**values)

    def test_decision_explanations_include_real_inputs_and_triggered_rules(self):
        decision = self.repo.store.process(packet(), "benchmark:fixture")
        self.assertEqual(decision["severity"], "critical")
        self.assertEqual(decision["inputs"]["rul"], 4.)
        self.assertIn("critical_point", decision["triggered_rules"])
        self.assertTrue(decision["recommended_action"])
        self.assertTrue(decision["timestamp"])
        self.assertGreater(len(decision["rules_evaluated"]), 3)
