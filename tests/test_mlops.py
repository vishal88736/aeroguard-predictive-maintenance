"""Actual isolated CPU/MLflow lifecycle checks; fixtures are not NASA results."""

import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from aeroguard.api import create_app
from aeroguard.baseline import BaselineBundle, model_folder, train_model
from aeroguard.contracts import LifecycleSettings
from aeroguard.mlops_data import DatasetVersions, protocol_split
from aeroguard.mlops_monitoring import drift_report, quality_report
from aeroguard.services import Service
from fleetpm.features import load_engines
from test_ag_baseline import write_fixture


def wait(service, submitted):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        job = service.repo.job(submitted["job_id"])
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(.02)
    raise AssertionError("Isolated lifecycle exceeded its bounded timeout")


class VersionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_fixture(self.root / "data", engines=50)
        self.store = DatasetVersions(self.root / "artifacts")
        self.content = (self.root / "data/train_FD001.txt").read_bytes()

    def add(self, content=None, **kwargs):
        return self.store.add("FD001", "train_FD001.txt", content or self.content,
                              kwargs.get("label_status", "complete_run_to_failure"), "Declared synthetic fixture", kwargs.get("namespace", "fixture"))

    def test_versions_are_immutable_idempotent_and_keep_source_files(self):
        first = self.add()
        self.assertTrue(self.add()["reused"])
        self.assertEqual((self.root / "data/train_FD001.txt").read_bytes(), self.content)
        snapshot = self.store.path(first["id"])
        snapshot.write_bytes(self.content + b"\n")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.store.get(first["id"])

    def test_label_declaration_changes_identity_and_cannot_be_tampered(self):
        complete, prefix = self.add(), self.add(label_status="observed_prefix")
        self.assertNotEqual(complete["id"], prefix["id"])
        path = self.store.root / prefix["id"] / "manifest.json"
        value = json.loads(path.read_text()); value["label_status"] = "complete_run_to_failure"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.store.get(prefix["id"])

    def test_official_test_targets_and_extra_label_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Official test"):
            self.store.add("FD001", "test_FD001.txt", self.content, "complete_run_to_failure", "Invalid training source")
        with self.assertRaises(ValueError):
            self.store.add("FD001", "batch.csv", b"unit,cycle,rul\n1,1,10\n", "observed_prefix", "Bad schema")
        with self.assertRaises(ValueError):
            self.store.get("../../private")

    def test_frozen_cohorts_accept_new_fit_engines_and_reject_changed_holdouts(self):
        version = self.add()
        cfg = {"split_seed": 42, "window": 15, "cap": 125, "minimum_history": 15}
        protocol = self.store.freeze_protocol(version["id"], cfg)
        write_fixture(self.root / "new", engines=55)
        raw = load_engines(self.root / "new/train_FD001.txt")
        split = protocol_split(raw, protocol, cfg)
        self.assertTrue(set(range(51, 56)) <= set(split["fit"]))
        self.assertFalse(set(split["fit"]) & set(split["selection"]))
        self.assertEqual(split["selection"], protocol["split"]["selection"])
        unit = split["selection"][0]
        raw[unit]["raw"][0, 4] += 1
        with self.assertRaisesRegex(ValueError, "Frozen validation"):
            protocol_split(raw, protocol, cfg)
        self.assertEqual(self.store.protocol("FD001")["id"], protocol["id"])

    def test_quality_does_not_invent_labels_and_rejects_missing_cycles(self):
        prefix = self.add(label_status="observed_prefix")
        self.assertFalse(quality_report(prefix, LifecycleSettings())["labels_available"])
        damaged = dict(prefix, missing_sensor_fraction=.5)
        self.assertFalse(quality_report(damaged, LifecycleSettings())["passed"])
        damaged = dict(prefix, label_status="complete_run_to_failure", missing_cycles=1)
        self.assertFalse(quality_report(damaged, LifecycleSettings())["passed"])


class LifecycleIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data, cls.artifacts = cls.root / "data", cls.root / "artifacts"
        write_fixture(cls.data, engines=50)
        cls.configuration = {"subsets": ["FD001"], "iterations": 10, "window": 15, "track_mlflow": True}
        cls.original = train_model(cls.configuration, cls.data, cls.artifacts)
        cls.service = Service(cls.data, cls.artifacts, cls.root / "runtime")
        cls.service.promote(cls.original["run_id"])
        initialized = wait(cls.service, cls.service.start("mlops_bootstrap", {"subsets": ["FD001"]}, "bootstrap"))
        if initialized["status"] != "complete":
            raise AssertionError(initialized)
        cls.protocol = cls.service.mlops.versions.protocol("FD001")

    @classmethod
    def tearDownClass(cls):
        cls.service.close()
        cls.temp.cleanup()

    def setUp(self):
        self.service = self.__class__.service
        if self.service.active().get("FD001") != self.original["run_id"]:
            self.service.promote(self.original["run_id"], rollback=True)
        self.service.mlops.save_settings(LifecycleSettings().model_dump())

    def version(self, engines=50, label="complete_run_to_failure"):
        directory = self.root / self._testMethodName
        write_fixture(directory, engines=engines)
        return self.service.mlops.add_version("FD001", "incoming.txt", (directory / "train_FD001.txt").read_bytes(),
                                               label, "Isolated synthetic complete/prefix fixtures", self._testMethodName[:48])

    def tearDown(self):
        for job in self.service.repo.jobs():
            if job["status"] in {"queued", "running"}:
                wait(self.service, {"job_id": job["id"]})

    def test_bootstrap_has_real_versions_aliases_and_idempotency(self):
        registry = self.service.mlops.registry
        record = registry.get(self.original["run_id"])
        self.assertTrue(record["packaged_inference_verified"])
        self.assertEqual(str(registry.aliases(self.original["run_id"])["champion"]), record["version"])
        self.assertTrue(registry.register(self.original["run_id"], self.data / "train_FD001.txt")["reused"])
        import mlflow
        client = registry.client()
        self.assertEqual(len(client.search_model_versions(f"name='{record['name']}'")), len(registry.list()))

    def test_registered_same_data_retrain_retains_frozen_lineage(self):
        metadata = train_model(self.configuration, self.data, self.artifacts)
        self.service.mlops.register_model(metadata["run_id"])
        lineage = self.service.mlops.lineage(metadata)
        self.assertEqual(lineage["protocol_id"], self.protocol["id"])
        eligible, reason = self.service.promotion_gate(metadata)
        self.assertTrue(eligible, reason)
        eligible, reason = self.service.promotion_gate(metadata, lifecycle_settings=LifecycleSettings(minimum_rmse_improvement_percent=1))
        self.assertFalse(eligible)

    def test_relocated_bundle_registers_without_fabricating_a_training_run(self):
        from aeroguard.mlops_registry import ModelRegistry
        root = self.root / "relocated" / "artifacts"
        source = model_folder(self.original["run_id"], self.artifacts)
        shutil.copytree(source, root / "aeroguard" / "runs" / self.original["run_id"])
        registry = ModelRegistry(root)
        result = registry.register(self.original["run_id"], self.data / "train_FD001.txt")
        self.assertTrue(result["packaged_inference_verified"])
        run = registry.client().get_run(result["mlflow_run_id"])
        self.assertEqual(run.data.tags["registration_only"], "true")
        self.assertEqual(run.data.tags["original_training_run_id"], self.original["mlflow_run_id"])

    def test_new_data_is_trained_on_frozen_cohorts_and_worse_candidate_rejected(self):
        version = self.version(55)
        with patch("aeroguard.baseline.parse_truth", side_effect=AssertionError("Lifecycle read official test truth")):
            result = wait(self.service, self.service.start("lifecycle", {"version_id": version["id"]}, self._testMethodName))
        self.assertEqual(result["status"], "complete", result)
        outcome = result["result"]
        self.assertEqual(outcome["status"], "rejected")
        self.assertEqual(self.service.active()["FD001"], self.original["run_id"])
        candidate = model_folder(outcome["run_id"], self.artifacts)
        split = json.loads((candidate / "split.json").read_text())
        self.assertEqual(split["selection"], self.protocol["split"]["selection"])
        self.assertTrue(set(range(51, 56)) <= set(split["fit"]))
        self.assertFalse(outcome["official_test_used_for_promotion"])
        self.assertTrue(self.service.mlops.registry.get(outcome["run_id"]))
        repeated = self.service.start("lifecycle", {"version_id": version["id"]}, self._testMethodName)
        self.assertTrue(repeated["reused"])

    def test_force_retrain_promotes_equal_candidate_and_rollback_updates_alias(self):
        version = self.version()
        job = wait(self.service, self.service.start("lifecycle", {"version_id": version["id"], "force_retrain": True}, self._testMethodName))
        self.assertEqual(job["status"], "complete", job)
        outcome = job["result"]
        self.assertEqual(outcome["status"], "promoted", outcome)
        candidate = outcome["run_id"]
        registry = self.service.mlops.registry
        self.assertEqual(str(registry.aliases(candidate)["champion"]), registry.get(candidate)["version"])
        reinitialized = wait(self.service, self.service.start("mlops_bootstrap", {"subsets": ["FD001"]}, self._testMethodName + "-initialize"))
        self.assertEqual(reinitialized["status"], "complete", reinitialized)
        self.assertEqual(self.service.active()["FD001"], candidate)
        self.service.promote(self.original["run_id"], rollback=True)
        self.assertEqual(str(registry.aliases(candidate)["champion"]), registry.get(self.original["run_id"])["version"])
        self.assertTrue((model_folder(candidate, self.artifacts) / "model.joblib").exists())
        switch = registry.champion
        def fail_after_alias_update(*args, **kwargs):
            switch(*args, **kwargs)
            raise RuntimeError("Injected failure after alias update")
        with patch.object(registry, "champion", side_effect=fail_after_alias_update):
            with self.assertRaisesRegex(RuntimeError, "Injected"):
                self.service.promote(candidate)
        self.assertEqual(self.service.active()["FD001"], self.original["run_id"])
        self.assertEqual(str(registry.aliases(candidate)["champion"]), registry.get(self.original["run_id"])["version"])
        self.assertFalse((self.service.runtime_dir / "promotion.pending.json").exists())

    def test_unlabelled_batch_never_trains_and_check_is_persisted(self):
        version = self.version(label="observed_prefix")
        with patch("aeroguard.mlops.train_model", side_effect=AssertionError("Unlabelled data trained")):
            job = wait(self.service, self.service.start("lifecycle", {"version_id": version["id"]}, self._testMethodName))
        self.assertEqual(job["status"], "complete", job)
        self.assertEqual(job["result"]["status"], "waiting_labels")
        overview = self.service.mlops.overview()
        self.assertTrue(any(c["version_id"] == version["id"] for c in overview["checks"]))
        incoming = [a for a in self.service.assets() if a.get("source_version_id") == version["id"]]
        self.assertEqual(len(incoming), 50)
        detail = self.service.asset_detail(incoming[0]["asset"])
        self.assertTrue(detail["history"])
        self.assertTrue(all("true_rul" not in p for p in detail["history"]))
        with self.service.repo.connect() as db:
            before = db.execute("SELECT COUNT(*) FROM events WHERE namespace LIKE '%:incoming:%'").fetchone()[0]
        repeat = wait(self.service, self.service.start("lifecycle", {"version_id": version["id"]}, self._testMethodName + "-repeat"))
        self.assertEqual(repeat["result"]["inference"]["processed_cycles"], 0)
        with self.service.repo.connect() as db:
            after = db.execute("SELECT COUNT(*) FROM events WHERE namespace LIKE '%:incoming:%'").fetchone()[0]
        self.assertEqual(before, after)

    def test_changed_committed_batch_history_is_quarantined(self):
        version = self.version(label="observed_prefix")
        first = wait(self.service, self.service.start("lifecycle", {"version_id": version["id"]}, self._testMethodName))
        self.assertEqual(first["result"]["status"], "waiting_labels")
        path = self.service.mlops.versions.path(version["id"])
        records = np.loadtxt(path);records[0, 5] += .01
        content = "\n".join(" ".join(map(str, row)) for row in records).encode()
        changed = self.service.mlops.add_version("FD001", "changed.txt", content, "observed_prefix", "Conflicting prefix fixture", self._testMethodName[:48])
        second = wait(self.service, self.service.start("lifecycle", {"version_id": changed["id"]}, self._testMethodName + "-conflict"))
        self.assertEqual(second["result"]["status"], "review_required")
        self.assertIn("rewrites", second["result"]["reason"])

    def test_downstream_failure_reuses_completed_candidate_on_retry(self):
        version = self.version(55)
        with patch.object(self.service.mlops.registry, "register", side_effect=RuntimeError("Registry unavailable")):
            failed = wait(self.service, self.service.start("lifecycle", {"version_id": version["id"]}, self._testMethodName))
        self.assertEqual(failed["status"], "failed")
        with patch("aeroguard.mlops.train_model", side_effect=AssertionError("Completed candidate was retrained")):
            retry = wait(self.service, self.service.retry(failed["id"], self._testMethodName + "-retry"))
        self.assertEqual(retry["status"], "complete", retry)
        self.assertTrue(any("reused" in stage["detail"] for stage in retry["stages"] if stage["name"] == "Train challenger and calibrate"))

    def test_unchanged_data_does_not_retrain(self):
        version = self.version()
        with patch("aeroguard.mlops.train_model", side_effect=AssertionError("Identical data retrained")):
            job = wait(self.service, self.service.start("lifecycle", {"version_id": version["id"]}, self._testMethodName))
        self.assertEqual(job["result"]["status"], "unchanged")

    def test_engine_level_regime_drift_and_small_sample_status(self):
        directory = self.root / "drift"
        write_fixture(directory, engines=50)
        path = directory / "train_FD001.txt"
        records = np.loadtxt(path)
        records[:, 0] += 100
        records[:, 5:9] += 50
        np.savetxt(path, records)
        reference = self.service.mlops.versions.path(self.protocol["reference_version_id"])
        bundle = self.service.bundle(self.original["run_id"])
        report = drift_report(reference, path, bundle, LifecycleSettings(), "observed_prefix")
        self.assertEqual(report["status"], "drift")
        self.assertGreaterEqual(len(report["drifted_sensors"]), 4)
        self.assertTrue(all(c["reference_engines"] <= 30 for c in report["comparisons"]))
        self.assertTrue(all(c["adjusted_p_value"] >= c["p_value"] for c in report["comparisons"]))
        np.savetxt(path, records[records[:, 0] <= 103])
        small = drift_report(reference, path, bundle, LifecycleSettings(), "observed_prefix")
        self.assertEqual(small["status"], "insufficient_data")
        np.savetxt(path, records[records[:, 1] <= 10])
        short = drift_report(reference, path, bundle, LifecycleSettings(), "observed_prefix")
        self.assertEqual(short["status"], "insufficient_data")

    def test_bad_quality_goes_to_review_and_resolution_does_not_authorize_training(self):
        directory = self.root / "missing"
        write_fixture(directory, engines=50)
        records = np.loadtxt(directory / "train_FD001.txt")
        records[:, 5:10] = np.nan
        content = "\n".join(" ".join(map(str, row)) for row in records).encode()
        version = self.service.mlops.add_version("FD001", "bad.txt", content, "observed_prefix", "Synthetic missing-channel test")
        job = wait(self.service, self.service.start("lifecycle", {"version_id": version["id"]}, self._testMethodName))
        self.assertEqual(job["result"]["status"], "review_required")
        self.service.mlops.resolve_review(job["result"]["review_id"], "Investigated the missing-channel input; labels remain absent")
        self.assertEqual(self.service.active()["FD001"], self.original["run_id"])

    def test_scheduler_disabled_and_job_lease_prevents_duplicate_submissions(self):
        self.assertEqual(self.service.mlops.tick()["status"], "disabled")
        settings = LifecycleSettings(enabled=True, auto_retrain=False)
        self.service.mlops.save_settings(settings.model_dump())
        self.version(label="observed_prefix")
        first = self.service.mlops.tick()
        self.assertEqual(first["status"], "submitted")
        second = self.service.mlops.tick()
        self.assertIn(second["status"], {"busy", "idle", "submitted"})
        self.assertNotEqual(second.get("job_id"), first["job_id"])
        wait(self.service, first)
        if second.get("job_id"):
            wait(self.service, second)


class LifecycleApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = create_app(self.root / "data", self.root / "artifacts", self.root / "runtime")
        self.client = TestClient(self.app)
        self.addCleanup(self.app.state.service.close)
        self.headers = {"X-AeroGuard-Client": "dashboard"}

    def test_empty_status_settings_validation_and_missing_input_errors(self):
        status = self.client.get("/api/mlops")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["registry"], [])
        self.assertFalse(status.json()["settings"]["enabled"])
        self.assertEqual(self.client.put("/api/mlops/settings", json={"interval_seconds": 0}, headers=self.headers).status_code, 422)
        self.assertEqual(self.client.post("/api/mlops/run", json={"version_id": "../../private"}, headers=self.headers).status_code, 422)
        self.assertEqual(self.client.post("/api/mlops/tick", headers=self.headers).json()["status"], "disabled")
        self.assertEqual(self.client.post("/api/mlops/tick").status_code, 403)

    def test_upload_validation_and_no_implicit_training(self):
        write_fixture(self.root / "fixture", engines=20)
        content = (self.root / "fixture/train_FD001.txt").read_bytes()
        response = self.client.post("/api/mlops/versions", files={"file": ("batch.txt", content)},
                                    data={"subset": "FD001", "label_status": "observed_prefix", "provenance": "Synthetic API fixture"}, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get("/api/jobs").json()["items"], [])
        identifier = response.json()["id"]
        self.assertEqual(self.client.get("/api/mlops/versions/" + identifier).json()["label_status"], "observed_prefix")
        self.assertFalse((self.root / "data/train_FD001.txt").exists())


if __name__ == "__main__":
    unittest.main()
