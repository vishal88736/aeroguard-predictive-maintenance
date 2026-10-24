"""Isolated export/API contracts, never benchmark results or fitted neural weights."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from aeroguard.api import create_app
from aeroguard.research import (FEATURE_COLUMNS, RAW_COLUMNS, REQUIRED_FILES, ResearchCatalog,
                                observed_window, sensor_frame)
from fleetpm.config import file_hash, write_json
from fleetpm.metrics import regression_metrics


def export_fixture(root, with_explanation=True):
    """Schema fixture with declared dummy model bytes; absolutely no neural quality claim."""
    root = Path(root)
    data, export = root / "data", root / "source-export"
    data.mkdir(parents=True, exist_ok=True); export.mkdir(parents=True, exist_ok=True)
    for partition, count, length in (("train", 4, 40), ("test", 2, 20)):
        rows = [[engine, cycle, *[engine + cycle * (index + 1) / 100 for index in range(24)]]
                for engine in range(1, count + 1) for cycle in range(1, length + 1)]
        np.savetxt(data / f"{partition}_FD001.txt", rows)
    np.savetxt(data / "RUL_FD001.txt", [7, 12])
    cfg = {"subset": "FD001", "variant": "paper", "seed": 42, "split_seed": 2025,
           "window": 30, "smoothing_window": 5, "normalization": "minmax", "target_scale": 100,
           "rul_cap": None, "horizons": 10, "encoder_layers": 6, "decoder_layers": 6,
           "d_model": 32, "n_heads": 8, "grid_size": 200, "effective_batch_size": 256}
    version = "xu2025-transformer-kan-bilstm-v2"
    identity = {"implementation": version, "config": cfg, "training_file_sha256": file_hash(data / "train_FD001.txt")}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    run_id = "KAN-FD001-paper-42-" + signature[:12]
    record = identity | {"signature": signature, "status": "completed", "best_validation_rmse": 1.0,
                         "best_sha256": "a" * 64, "parameter_count": 0,
                         "fixture_note": "Isolated unit contract; dummy weights, no training"}
    split = {"fit_engine_ids": [1, 2], "validation_engine_ids": [3, 4], "split_seed": 2025}
    preprocessor = {"columns": list(FEATURE_COLUMNS), "center": [0.] * 24, "scale": [1.] * 24,
                    "fit_engine_ids": [1, 2], "normalization": "minmax", "smoothing_window": 5}
    selection = {"subset": "FD001", "selected_variant": "paper", "mean_validation_rmse": {"paper": 1.0, "conditioned_kan": 1.2},
                 "seeds": [42], "criterion": "strictly_lower_mean_validation_rmse; ties retain paper",
                 "matched_paper_comparison": True, "official_test_used_for_selection": False,
                 "frozen_at": "2026-10-24T00:00:00+00:00"}
    predictions = pd.DataFrame({"engine_id": [1, 2], "cycle": [20, 20], "predicted_rul": [6., 13.], "true_rul": [7., 12.]})
    validation = pd.DataFrame({"engine_id": [3, 4], "cycle": [20, 20], "predicted_rul": [19., 21.], "true_rul": [20., 20.]})
    metrics = {"subset": "FD001", "variant": "paper", "seed": 42,
               "protocol": "official_test_last_window_per_engine", "raw_target_metrics": regression_metrics([7, 12], [6, 13]),
               "test_file_sha256": file_hash(data / "test_FD001.txt"), "rul_file_sha256": file_hash(data / "RUL_FD001.txt")}
    for name, value in {"run": record, "config": cfg, "preprocessor": preprocessor, "split": split,
                        "metrics": metrics, "selection_record": selection, "data_manifest": {"sha256": identity["training_file_sha256"]},
                        "environment": {"source": "isolated test, no installed neural stack"}}.items():
        write_json(export / f"{name}.json", value)
    predictions.to_csv(export / "test_predictions.csv", index=False)
    predictions.drop(columns="true_rul").to_csv(export / "test_predictions_sensor_only.csv", index=False)
    validation.to_csv(export / "validation_predictions.csv", index=False)
    validation.drop(columns="predicted_rul").to_csv(export / "validation_endpoints.csv", index=False)
    pd.DataFrame({"epoch": [1], "validation_rmse": [1.], "train_mse_scaled": [.01]}).to_csv(export / "history.csv", index=False)
    (export / "inference.ts").write_bytes(b"UNIT FIXTURE ONLY - NOT A FITTED NEURAL MODEL")
    np.savez_compressed(export / "inference_probes.npz", windows=np.zeros((2, 30, 24), dtype=np.float32),
                        predictions_cycles=np.zeros((2, 10), dtype=np.float32))
    groups = [f"sensor_{i}" for i in range(1, 22)]
    explanation = {"schema_version": 1, "run_id": run_id, "subset": "FD001", "variant": "paper",
                   "engine_id": 1, "cycle": 20, "prediction": 6.0, "model_sha256": record["best_sha256"],
                   "target": "current_cycle_predicted_rul", "units": "cycles", "feature_groups": groups,
                   "background_source": "fit_engines_only", "backgrounds": [{"partition": "fit", "engine_id": 1, "cycle": 15}],
                   "shap": {"method": "shap.KernelExplainer", "package_version": "fixture", "samples": 256,
                            "baseline": 5., "values": [1.] + [0.] * 20, "additivity_residual_cycles": 0.},
                   "lime": {"method": "lime.lime_base.LimeBase", "package_version": "fixture", "samples": 512,
                            "heldout_samples": 128, "intercept": 5., "coefficients": [1.] + [0.] * 20,
                            "local_prediction": 6., "fitting_r2": .9,
                            "heldout_fidelity": {"weighted_r2": .85, "weighted_mae_cycles": .2}}}
    if with_explanation:
        write_json(export / "explanations/engine-001-cycle-20.json", explanation)
    manifest = {"schema_version": 1, "run_id": run_id, "subset": "FD001", "variant": "paper", "seed": 42,
                "implementation": version, "architecture": "Transformer-KAN-BiLSTM", "status": "completed",
                "exported_at": "2026-10-24T00:00:00+00:00", "source": "Isolated contract fixture; no benchmark training",
                "selected_on_validation": True, "selection": selection,
                "explanation_status": "available" if with_explanation else "not_generated",
                "explanation_files": ["explanations/engine-001-cycle-20.json"] if with_explanation else [], "files": {}}
    write_json(export / "research_export.json", manifest)
    reseal(export)
    return data, export, run_id, explanation


def reseal(folder):
    manifest = json.loads((folder / "research_export.json").read_text())
    names = REQUIRED_FILES | set(manifest["explanation_files"])
    manifest["files"] = {name: {"sha256": file_hash(folder / name), "bytes": (folder / name).stat().st_size} for name in sorted(names)}
    write_json(folder / "research_export.json", manifest)


def archive(folder, extra=None):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(folder).as_posix())
        if extra:
            z.writestr(extra, "unsafe")
    return stream.getvalue()


class ResearchExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.export, self.run_id, self.explanation = export_fixture(self.root)
        self.catalog = ResearchCatalog(self.root / "artifacts", self.data)

    def test_empty_state_has_no_models_or_attributions(self):
        value = self.catalog.summary()
        self.assertEqual(value["status"], "awaiting_user_training")
        self.assertEqual(value["runs"], [])
        self.assertEqual(value["explanation_order"][-2:], ["SHAP", "LIME"])

    def test_import_detail_idempotency_and_engine_evidence(self):
        first = self.catalog.import_archive(archive(self.export))
        self.assertFalse(first["reused"])
        self.assertTrue(self.catalog.import_archive(archive(self.export))["reused"])
        summary = self.catalog.summary()
        self.assertEqual(len(summary["runs"]), 1)
        self.assertEqual(summary["invalid_exports"], [])
        detail = self.catalog.detail(self.run_id)
        self.assertEqual(detail["metrics"]["rmse"], 1.)
        self.assertEqual(len(detail["predictions"]), 2)
        engine = self.catalog.engine_detail(self.run_id, 1)
        self.assertEqual(engine["explanation"]["shap"]["method"], "shap.KernelExplainer")
        self.assertEqual(engine["explanation"]["lime"]["method"], "lime.lime_base.LimeBase")
        self.assertEqual(len(engine["sensors"]), 21)
        self.assertEqual(engine["sensors"][2]["symbol"], "T30")
        self.assertIsNone(self.catalog.engine_detail(self.run_id, 2)["explanation"])
        with zipfile.ZipFile(BytesIO(self.catalog.archive(self.run_id))) as z:
            self.assertIsNone(z.testzip())
            self.assertIn("inference.ts", z.namelist())

    def test_checksum_tampering_rejected_even_after_previous_read(self):
        self.catalog.import_archive(archive(self.export))
        self.catalog.detail(self.run_id)
        target = self.catalog.root / "imported" / self.run_id / "test_predictions.csv"
        target.write_text(target.read_text() + "\nchanged")
        self.assertEqual(self.catalog.summary()["runs"], [])
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.catalog.detail(self.run_id)

    def test_reported_metrics_cannot_disagree_with_saved_predictions(self):
        report = json.loads((self.export / "metrics.json").read_text())
        report["raw_target_metrics"]["rmse"] = .001
        write_json(self.export / "metrics.json", report); reseal(self.export)
        with self.assertRaisesRegex(ValueError, "rmse"):
            self.catalog.import_archive(archive(self.export))

    def test_frozen_sensor_only_predictions_cannot_change_after_labels(self):
        frozen = pd.read_csv(self.export / "test_predictions_sensor_only.csv")
        frozen.loc[0, "predicted_rul"] = 100
        frozen.to_csv(self.export / "test_predictions_sensor_only.csv", index=False); reseal(self.export)
        with self.assertRaisesRegex(ValueError, "sensor-only"):
            self.catalog.import_archive(archive(self.export))

    def test_last_observed_endpoint_is_required(self):
        for filename in ("test_predictions.csv", "test_predictions_sensor_only.csv"):
            table = pd.read_csv(self.export / filename); table.loc[0, "cycle"] = 19
            table.to_csv(self.export / filename, index=False)
        reseal(self.export)
        with self.assertRaisesRegex(ValueError, "last observed"):
            self.catalog.import_archive(archive(self.export))

    def test_official_target_and_validation_boundaries_verified(self):
        predictions = pd.read_csv(self.export / "validation_predictions.csv")
        predictions.loc[0, "engine_id"] = 1
        predictions.to_csv(self.export / "validation_predictions.csv", index=False); reseal(self.export)
        with self.assertRaisesRegex(ValueError, "held-out"):
            self.catalog.import_archive(archive(self.export))

    def test_bad_explanation_method_background_or_vector_rejected(self):
        path = self.export / "explanations/engine-001-cycle-20.json"
        for mutation in (lambda e: e["shap"].update(method="occlusion"),
                         lambda e: e["backgrounds"][0].update(engine_id=3),
                         lambda e: e["lime"].update(coefficients=[1]),
                         lambda e: e.update(prediction=42)):
            with self.subTest(mutation=mutation):
                report = json.loads(json.dumps(self.explanation)); mutation(report)
                write_json(path, report); reseal(self.export)
                with self.assertRaises(ValueError):
                    self.catalog.import_archive(archive(self.export))

    def test_poor_lime_fidelity_is_retained_and_visible(self):
        report = json.loads(json.dumps(self.explanation))
        report["lime"]["heldout_fidelity"]["weighted_r2"] = -2.0
        write_json(self.export / "explanations/engine-001-cycle-20.json", report); reseal(self.export)
        self.catalog.import_archive(archive(self.export))
        value = self.catalog.engine_detail(self.run_id, 1)
        self.assertEqual(value["explanation"]["lime"]["heldout_fidelity"]["weighted_r2"], -2.)

    def test_worse_extension_cannot_be_selected(self):
        manifest = json.loads((self.export / "research_export.json").read_text())
        manifest["selection"]["selected_variant"] = "conditioned_kan"
        manifest["selected_on_validation"] = False
        write_json(self.export / "selection_record.json", manifest["selection"])
        write_json(self.export / "research_export.json", manifest); reseal(self.export)
        with self.assertRaisesRegex(ValueError, "worse"):
            self.catalog.import_archive(archive(self.export))

    def test_zip_paths_and_missing_files_rejected(self):
        for extra in ("../.env", "/tmp/escape", "unexpected.py"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.catalog.import_archive(archive(self.export, extra))
        (self.export / "inference.ts").unlink()
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.catalog.import_archive(archive(self.export))

    def test_causal_web_preprocessing_matches_known_trailing_window(self):
        frame = sensor_frame(self.data / "test_FD001.txt")
        engine = frame[frame.engine_id == 1].reset_index(drop=True)
        cfg = json.loads((self.export / "config.json").read_text())
        preprocessor = json.loads((self.export / "preprocessor.json").read_text())
        before = observed_window(engine, 10, preprocessor, cfg)
        changed = engine.copy(); changed.loc[10:, FEATURE_COLUMNS] += 1e6
        np.testing.assert_array_equal(before, observed_window(changed, 10, preprocessor, cfg))
        # setting_1 = 1 + cycle/100; trailing cycles 6..10 have mean 1.08.
        self.assertAlmostEqual(float(before[0, -1, 0]), 1.08, places=6)
        self.assertAlmostEqual(float(before[0, 0, 0]), 1.01, places=6)

    def test_explanation_addition_does_not_replace_model(self):
        manifest = json.loads((self.export / "research_export.json").read_text())
        (self.export / "explanations/engine-001-cycle-20.json").unlink()
        manifest["explanation_files"] = []; manifest["explanation_status"] = "not_generated"
        write_json(self.export / "research_export.json", manifest); reseal(self.export)
        self.catalog.import_archive(archive(self.export))
        write_json(self.export / "explanations/engine-001-cycle-20.json", self.explanation)
        manifest["explanation_files"] = ["explanations/engine-001-cycle-20.json"]; manifest["explanation_status"] = "available"
        write_json(self.export / "research_export.json", manifest); reseal(self.export)
        result = self.catalog.import_archive(archive(self.export))
        self.assertTrue(result["reused"])
        self.assertEqual(result["run"]["explanations_count"], 1)


class ResearchAPITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.data, self.export, self.run_id, _ = export_fixture(root)
        self.app = create_app(self.data, root / "artifacts", root / "runtime")
        self.client = TestClient(self.app); self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_api_import_views_and_missing_inference_dependency(self):
        self.assertEqual(self.client.get("/api/research").json()["runs"], [])
        response = self.client.post("/api/research/import", headers={"X-AeroGuard-Client": "dashboard"},
                                    files={"file": ("unit-fixture.zip", archive(self.export), "application/zip")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get("/api/research/runs/" + self.run_id).status_code, 200)
        self.assertEqual(self.client.get(f"/api/research/runs/{self.run_id}/engines/1").status_code, 200)
        self.assertEqual(self.client.get(f"/api/research/runs/{self.run_id}/engines/0").status_code, 422)
        with patch("aeroguard.research.importlib.util.find_spec", return_value=None):
            unavailable = self.client.get(f"/api/research/runs/{self.run_id}/predict?engine_id=1&cycle=10")
        self.assertEqual(unavailable.status_code, 503, unavailable.text)
        self.assertEqual(unavailable.json()["error"]["code"], "research_dependency_unavailable")
        self.assertEqual(self.client.get("/downloads/kan-notebook").status_code, 200)
        self.assertEqual(self.client.get("/downloads/kan-requirements").status_code, 200)
        self.assertEqual(self.client.get(f"/api/research/runs/{self.run_id}/download").status_code, 200)
        self.assertEqual(self.app.state.service.active(), {})

    def test_invalid_upload_and_existing_cpu_models_remain_separate(self):
        response = self.client.post("/api/research/import", headers={"X-AeroGuard-Client": "dashboard"},
                                    files={"file": ("not-a-model.zip", b"broken", "application/zip")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/models").json()["runs"], [])
        self.assertEqual(self.client.get("/api/dashboard").json()["summary"]["total"], 0)
        self.assertEqual(self.client.get("/api/jobs").json()["items"], [])


if __name__ == "__main__":
    unittest.main()
