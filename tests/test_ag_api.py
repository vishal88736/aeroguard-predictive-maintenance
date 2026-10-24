"""Isolated FastAPI integration checks, including a genuine small CPU fit.

All data and artifacts live in TemporaryDirectory. Fixtures are not benchmarks;
the test never changes the application's real corpus, models, or work orders.
"""

import io
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
import zipfile

from fastapi.testclient import TestClient

from aeroguard.api import create_app
from helpers import archive_payload
from test_ag_data import csv_fixture


class AeroGuardAPITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.app = create_app(self.root / "data", self.root / "artifacts", self.root / "runtime")
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.headers = {"X-AeroGuard-Client": "dashboard"}

    def mutate(self, method, route, payload=None, key=None, **kwargs):
        headers = dict(self.headers)
        if key:
            headers["Idempotency-Key"] = key
        return self.client.request(method, route, json=payload, headers=headers, **kwargs)

    def wait_job(self, job_id, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.client.get("/api/jobs/" + job_id)
            self.assertEqual(response.status_code, 200, response.text)
            job = response.json()
            if job["status"] in {"complete", "failed", "interrupted"}:
                return job
            time.sleep(.02)
        self.fail("Isolated pipeline did not finish within its bounded timeout")

    def import_fixture(self):
        response = self.client.post("/api/data/import", headers=self.headers,
                                    files={"file": ("synthetic-cmapss.zip", archive_payload(engines=20, cycles=35), "application/zip")})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_health_and_empty_benchmark_have_no_fabricated_results(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["application"], "AeroGuard")
        self.assertEqual(health.json()["model_status"], "training_required")
        self.assertEqual(health.json()["data_ready"], 0)
        dashboard = self.client.get("/api/dashboard").json()
        self.assertEqual(dashboard["mode"], "benchmark")
        self.assertEqual(dashboard["summary"]["total"], 0)
        self.assertEqual(dashboard["assets"], [])
        self.assertEqual(dashboard["orders"], [])
        self.assertEqual(dashboard["runs"], [])
        self.assertIsNone(dashboard["last_successful_run"])
        self.assertEqual(self.client.get("/api/models").json()["active"], {})

    def test_demo_filters_sensor_labels_search_and_namespaces(self):
        all_assets = self.client.get("/api/engines?mode=demo").json()
        self.assertEqual(all_assets["total"], 32)
        filtered = self.client.get("/api/engines?mode=demo&subset=FD004&engine=005&severity=critical").json()
        self.assertEqual(filtered["total"], 1)
        self.assertEqual(filtered["items"][0]["asset"], "FD004/demo/005")
        self.assertEqual(self.client.get("/api/engines?mode=demo&cycle_min=9999").json()["total"], 0)
        detail = self.client.get("/api/engines/FD001%2Fdemo%2F003?mode=demo&limit=15")
        self.assertEqual(detail.status_code, 200, detail.text)
        data = detail.json()
        self.assertTrue(data["demo"])
        self.assertIn("Synthetic", data["source_label"])
        self.assertEqual(len(data["history"]), 15)
        self.assertTrue(all("true_rul" not in point for point in data["history"]))
        self.assertTrue(any(sensor["symbol"] == "T30" for sensor in data["sensors"]))
        self.assertTrue(all(sensor["unit"] == "normalized synthetic signal" for sensor in data["sensors"]))
        self.assertEqual(len(data["operating_settings"]), 3)
        self.assertEqual(self.client.get("/api/engines/FD001%2Fdemo%2F003?mode=benchmark").status_code, 400)
        search = self.client.get("/api/search?mode=demo&q=FD001-005").json()
        self.assertEqual(len(search["engines"]), 1)
        self.assertGreater(len(self.client.get("/api/notifications?mode=demo").json()["items"]), 0)

    def test_host_origin_mutation_and_path_boundaries(self):
        self.assertEqual(self.client.get("/api/health", headers={"Host": "attacker.example"}).status_code, 400)
        settings = self.client.get("/api/settings").json()
        self.assertEqual(self.client.put("/api/settings", json=settings).status_code, 403)
        headers = self.headers | {"Origin": "https://attacker.example"}
        self.assertEqual(self.client.put("/api/settings", json=settings, headers=headers).status_code, 403)
        # Same-origin is scheme + hostname + port, not just matching netloc.
        headers = self.headers | {"Origin": "https://testserver"}
        self.assertEqual(self.client.put("/api/settings", json=settings, headers=headers).status_code, 403)
        self.assertEqual(self.client.get("/%2e%2e/.env").status_code, 404)
        self.assertEqual(self.client.get("/assets/%2e%2e/%2e%2e/requirements-app.txt").status_code, 404)
        self.assertEqual(self.client.get("/api/not-a-real-endpoint").status_code, 404)
        self.assertEqual(self.client.get("/assets/does-not-exist.png").status_code, 404)
        oversized = self.client.post("/api/pipeline/run", json={}, headers=self.headers | {"Content-Length": str(33 * 1024 * 1024)})
        self.assertEqual(oversized.status_code, 413)

    def test_typed_contract_errors_are_structured_and_do_not_save(self):
        invalid = self.mutate("POST", "/api/models/train", {"iterations": 2, "subsets": ["FD999"]})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["error"]["code"], "validation_error")
        self.assertTrue(invalid.json()["error"]["details"])
        self.assertEqual(self.client.get("/api/dashboard?mode=production").status_code, 422)
        self.assertEqual(self.client.get("/api/data/FD005/preview").status_code, 422)
        settings = self.client.get("/api/settings").json()
        invalid_settings = settings | {"promotion_rmse_tolerance_percent": -5}
        self.assertEqual(self.mutate("PUT", "/api/settings", invalid_settings).status_code, 422)
        self.assertEqual(self.client.get("/api/settings").json(), settings)
        self.assertEqual(self.client.get("/api/jobs").json()["items"], [])

    def test_csv_import_preview_validation_and_preservation(self):
        response = self.client.post("/api/data/import", headers=self.headers,
                                    data={"subset": "FD001", "partition": "train"},
                                    files={"file": ("input.csv", csv_fixture(missing=True), "text/csv")})
        self.assertEqual(response.status_code, 200, response.text)
        preview = self.client.get("/api/data/FD001/preview?partition=train&limit=3")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.json()["statistics"]["missing_values"], 1)
        self.assertIsNone(preview.json()["rows"][1][7])
        wrong = self.client.post("/api/data/import", headers=self.headers,
                                files={"file": ("input.csv", csv_fixture(sensor_delta=3), "text/csv")})
        self.assertEqual(wrong.status_code, 400)
        self.assertIn("preserved", wrong.json()["error"]["message"])
        bad = self.client.post("/api/data/import", headers=self.headers,
                              files={"file": ("bad.csv", b"unit,cycle\n1,2\n", "text/csv")})
        self.assertEqual(bad.status_code, 400)

    def test_demo_job_deduplication_and_explicit_skipped_ml_stages(self):
        request = {"mode": "demo", "subsets": ["FD001"], "cycles": 5}
        first = self.mutate("POST", "/api/pipeline/run", request, key="same-demo-click")
        self.assertEqual(first.status_code, 202, first.text)
        job = self.wait_job(first.json()["job_id"])
        self.assertEqual(job["status"], "complete", job)
        self.assertTrue(any(stage["status"] == "skipped" and "not live model outputs" in stage["detail"] for stage in job["stages"]))
        self.assertTrue(all(stage["duration_seconds"] is not None for stage in job["stages"]))
        second = self.mutate("POST", "/api/pipeline/run", request, key="same-demo-click")
        self.assertEqual(second.status_code, 202)
        self.assertEqual(first.json()["job_id"], second.json()["job_id"])
        conflict = self.mutate("POST", "/api/pipeline/run", request | {"cycles": 6}, key="same-demo-click")
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(len(self.client.get("/api/jobs").json()["items"]), 1)
        self.assertEqual(self.client.get("/api/orders?mode=benchmark").json()["items"], [])

    def test_dashboard_and_pipeline_history_keep_job_source_modes_separate(self):
        repository = self.app.state.service.repo
        first_demo, _ = repository.create_job("replay", {"mode": "demo", "subsets": ["FD001"]}, "fixture-demo-first")
        repository.update_job(first_demo, status="complete", result={"fixture": True})
        benchmark, _ = repository.create_job("train", {"subsets": ["FD001"]}, "fixture-benchmark")
        repository.update_job(benchmark, status="complete", result={"fixture": True})
        latest_demo, _ = repository.create_job("replay", {"mode": "demo", "subsets": ["FD001"]}, "fixture-demo-latest")
        repository.update_job(latest_demo, status="complete", result={"fixture": True})
        actual = self.client.get("/api/dashboard?mode=benchmark").json()
        demo = self.client.get("/api/dashboard?mode=demo").json()
        self.assertEqual(actual["last_successful_run"]["id"], benchmark)
        self.assertEqual({job["id"] for job in actual["jobs"]}, {benchmark})
        self.assertEqual(demo["last_successful_run"]["id"], latest_demo)
        self.assertEqual({job["id"] for job in demo["jobs"]}, {first_demo, latest_demo})
        actual_history = self.client.get("/api/jobs?mode=benchmark").json()["items"]
        demo_history = self.client.get("/api/jobs?mode=demo").json()["items"]
        self.assertEqual({job["id"] for job in actual_history}, {benchmark})
        self.assertEqual({job["id"] for job in demo_history}, {first_demo, latest_demo})

    def test_failed_pipeline_can_be_retried_without_fabricated_success(self):
        failed = self.mutate("POST", "/api/pipeline/run", {"mode": "benchmark", "subsets": ["FD001"]}, key="missing-model")
        self.assertEqual(failed.status_code, 202)
        original = self.wait_job(failed.json()["job_id"])
        self.assertEqual(original["status"], "failed")
        self.assertTrue(any(stage["status"] == "failed" for stage in original["stages"]))
        self.import_fixture()
        retry = self.mutate("POST", "/api/jobs/" + original["id"] + "/retry", {}, key="retry-missing-model")
        self.assertEqual(retry.status_code, 202)
        retried = self.wait_job(retry.json()["job_id"])
        self.assertEqual(retried["status"], "failed")
        self.assertIn("no active model", retried["error"])
        self.assertEqual(self.client.get("/api/models").json()["runs"], [])

    def test_demo_order_actions_audit_confirmation_inputs_and_mode_isolation(self):
        order = self.client.get("/api/orders?mode=demo").json()["items"][0]
        route = "/api/orders/" + order["id"] + "?mode=demo"
        invalid = self.mutate("PATCH", route, {"action": "resolve", "resolution": "preventive_service"})
        self.assertEqual(invalid.status_code, 422)
        for payload in [{"action": "acknowledge"}, {"action": "assign", "owner": "Test engineer", "notes": "Isolated test fixture"}, {"action": "start"}, {"action": "resolve", "resolution": "no_fault_found", "notes": "Synthetic maintenance outcome recorded"}]:
            result = self.mutate("PATCH", route, payload)
            self.assertEqual(result.status_code, 200, result.text)
        saved = self.client.get(route).json()
        self.assertEqual(saved["status"], "resolved")
        self.assertEqual(saved["owner"], "Test engineer")
        self.assertEqual(len(saved["audit"]), 4)
        self.assertEqual(self.mutate("PATCH", route, {"action": "start"}).status_code, 409)
        self.assertEqual(self.client.get("/api/orders/" + order["id"] + "?mode=benchmark").status_code, 400)
        self.assertEqual(self.client.get("/api/orders?mode=benchmark").json()["items"], [])

    def test_settings_persist_and_documented_paths_and_downloads_serve(self):
        original = self.client.get("/api/settings").json()
        saved = original | {"workspace_name": "Temporary API lab", "researcher_name": "Fixture engineer"}
        saved["policy"]["persistence"] = 4
        response = self.mutate("PUT", "/api/settings", saved)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get("/api/settings").json(), saved)
        for route in ["/", "/dashboard", "/engines", "/engines/FD001%2Fdemo%2F001", "/analytics", "/models", "/work-orders", "/pipeline", "/data", "/settings"]:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, route)
            self.assertIn("AeroGuard", response.text)
        notebook = self.client.get("/downloads/notebook")
        self.assertEqual(notebook.status_code, 200)
        self.assertEqual(json.loads(notebook.content)["nbformat"], 4)
        self.assertEqual(self.client.get("/openapi.json").status_code, 200)

    def test_real_fixture_training_evaluation_promotion_and_model_backed_replay(self):
        self.import_fixture()
        request = {"subsets": ["FD001"], "iterations": 10, "track_mlflow": False, "replay_cycles": 3, "promote": True}
        submitted = self.mutate("POST", "/api/models/train", request, key="fixture-training")
        self.assertEqual(submitted.status_code, 202, submitted.text)
        job = self.wait_job(submitted.json()["job_id"], timeout=40)
        self.assertEqual(job["status"], "complete", job)
        self.assertTrue(all(stage["status"] in {"complete", "skipped"} for stage in job["stages"]))
        models = self.client.get("/api/models").json()
        self.assertEqual(len(models["runs"]), 1)
        run = models["runs"][0]
        self.assertEqual(run["training_status"], "complete")
        self.assertEqual(models["active"]["FD001"], run["run_id"])
        self.assertTrue(math.isfinite(run["evaluation"]["metrics"]["rmse"]))
        self.assertEqual(run["evaluation"]["metrics"]["n_engines"], 20)
        repeated = self.mutate("POST", "/api/models/train", request, key="fixture-training")
        self.assertEqual(repeated.status_code, 202, repeated.text)
        self.assertEqual(repeated.json()["job_id"], job["id"])
        dashboard = self.client.get("/api/dashboard?subset=FD001").json()
        self.assertEqual(dashboard["summary"]["total"], 20)
        self.assertTrue(all(p["bundle_id"] == run["run_id"] for p in dashboard["assets"]))
        detail = self.client.get("/api/engines/FD001%2Ftest%2F001?limit=5")
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertFalse(detail.json()["demo"])
        self.assertEqual(len(detail.json()["history"]), 5)
        self.assertTrue(all("true_rul" in p for p in detail.json()["history"]))
        self.assertTrue(detail.json()["decisions"])
        self.assertTrue(all("true_rul" not in d["inputs"] for d in detail.json()["decisions"]))
        download = self.client.get("/api/models/" + run["run_id"] + "/download")
        self.assertEqual(download.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn(run["run_id"] + "/bundle_hashes.json", archive.namelist())
