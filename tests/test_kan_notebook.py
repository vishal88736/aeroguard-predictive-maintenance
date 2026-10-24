"""Validate the standalone paper notebook without installing torch or training models."""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Transformer_KAN_BiLSTM_GPU.ipynb"
BOOK = json.loads(NOTEBOOK.read_text())
CODE = {cell["id"]: "".join(cell["source"]) for cell in BOOK["cells"] if cell["cell_type"] == "code"}
MARKDOWN = "\n".join("".join(cell["source"]) for cell in BOOK["cells"] if cell["cell_type"] == "markdown")


def data_namespace():
    module = types.ModuleType("kan_notebook_data_checks")
    sys.modules[module.__name__] = module
    with redirect_stdout(io.StringIO()):
        for name in ("configuration", "data-functions"):
            exec(compile(CODE[name], f"notebook:{name}", "exec"), module.__dict__)
    return module


class KANNotebookStructureTests(unittest.TestCase):
    def test_unexecuted_and_syntactically_valid(self):
        self.assertEqual((BOOK["nbformat"], BOOK["nbformat_minor"]), (4, 5))
        self.assertGreaterEqual(len(CODE), 15)
        ids = [cell["id"] for cell in BOOK["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in BOOK["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]), filename=cell["id"])

    def test_rebuild_is_reproducible(self):
        spec = importlib.util.spec_from_file_location("kan_builder", ROOT / "scripts/build_kan_notebook.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        self.assertEqual(BOOK, builder.build_notebook())

    def test_standalone_paper_architecture_and_training_boundary(self):
        all_code = "\n".join(CODE.values())
        for forbidden in ("from fleetpm", "from aeroguard", "pip install", "subprocess", "torch.hub", "requests.get"):
            self.assertNotIn(forbidden, all_code)
        for forbidden in ("nn.GRU(", "nn.Conv1d(", "HistGradientBoosting", "IsolationForest"):
            self.assertNotIn(forbidden, all_code)
        self.assertIn("nn.Transformer(", CODE["transformer-bilstm"])
        self.assertIn("bidirectional=True", CODE["transformer-bilstm"])
        self.assertIn("class SplineKANLayer", CODE["kan-model"])
        self.assertIn("torch.linalg.lstsq", CODE["kan-model"])
        self.assertIn("amplitudes = self.edges", CODE["kan-model"])
        tree = ast.parse(CODE["training-functions"])
        training = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_subset")
        training_source = ast.get_source_segment(CODE["training-functions"], training)
        self.assertNotIn('f"test_', training_source)
        self.assertNotIn('f"RUL_', training_source)
        self.assertIn('f"train_', training_source)
        self.assertIn('weights_only=True', all_code)
        self.assertNotIn('weights_only=False', all_code)

    def test_gpu_and_honest_protocol_documented(self):
        self.assertIn("if not torch.cuda.is_available()", CODE["cuda-runtime"])
        self.assertIn('DEVICE = torch.device("cuda:0")', CODE["cuda-runtime"])
        self.assertIn("Author", MARKDOWN)
        self.assertIn("not an\nexact numerical reproduction", MARKDOWN)
        self.assertIn("repartition", MARKDOWN)
        self.assertIn("official test stays untouched", MARKDOWN)
        self.assertIn("latent", MARKDOWN)
        self.assertIn("rul_cap: float | None = None", CODE["configuration"])

    def test_predictions_saved_before_labels_are_read(self):
        source = CODE["evaluation-functions"]
        self.assertLess(source.index('predictions.to_csv(run_dir / "test_predictions_sensor_only.csv"'),
                        source.index('np.loadtxt(truth_path'))
        self.assertIn('forecast[:, 0]', source)
        self.assertIn('"raw_target_metrics"', source)
        self.assertIn('"artifact_hashes"', source)

    def test_explanations_are_last_and_use_actual_libraries(self):
        ids = [cell["id"] for cell in BOOK["cells"]]
        self.assertLess(ids.index("start-training"), ids.index("variant-selection"))
        self.assertLess(ids.index("variant-selection"), ids.index("run-evaluation"))
        self.assertLess(ids.index("run-evaluation"), ids.index("post-training-explanation-config"))
        self.assertIn("import shap", CODE["post-training-explanation-config"])
        self.assertIn("from lime.lime_base import LimeBase", CODE["post-training-explanation-config"])
        source = CODE["post-training-explanation-functions"]
        self.assertIn("shap.KernelExplainer(predict_masks", source)
        self.assertIn("lime_base.explain_instance_with_data", source)
        self.assertIn("heldout_weighted_fidelity", source)
        self.assertNotIn("optimizer", ast.get_source_segment(source, next(n for n in ast.parse(source).body
                                                                          if isinstance(n, ast.FunctionDef) and n.name == "heldout_weighted_fidelity")))
        self.assertIn('VARIANTS = ["paper", "conditioned_kan"]', CODE["configuration"])


class KANNotebookDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = data_namespace()

    def test_documented_data_invariants_execute(self):
        with redirect_stdout(io.StringIO()):
            exec(compile(CODE["data-sanity-checks"], "notebook:data-sanity-checks", "exec"), self.module.__dict__)

    @unittest.skipUnless(all((ROOT / "data/cmapss" / f"train_FD00{i}.txt").is_file() for i in range(1, 5)),
                         "Optional NASA corpus absent; prepare FD001-FD004 to run real-data prefix checks")
    def test_actual_four_subsets_fit_only_and_prefix_invariance(self):
        m = self.module
        for subset, (expected_train, _) in m.EXPECTED_ENGINES.items():
            with self.subTest(subset=subset):
                cfg = m.config_for(subset)
                frame = m.load_sensor_file(ROOT / "data/cmapss" / f"train_{subset}.txt")
                self.assertEqual(frame.engine_id.nunique(), expected_train)
                split = m.split_training_engines(frame, cfg)
                self.assertFalse(set(split["fit_engine_ids"]) & set(split["validation_engine_ids"]))
                self.assertEqual(sorted(split["fit_engine_ids"] + split["validation_engine_ids"]), list(range(1, expected_train + 1)))
                self.assertEqual(split, m.split_training_engines(frame, m.replace(cfg, seed=99)))
                p = m.TrainOnlyPreprocessor(cfg).fit(frame, split["fit_engine_ids"])
                fit_frame = frame[frame.engine_id.isin(split["fit_engine_ids"])].reset_index(drop=True)
                p_fit_only = m.TrainOnlyPreprocessor(cfg).fit(fit_frame, split["fit_engine_ids"])
                self.assertEqual(p.to_dict(), p_fit_only.to_dict())
                restored = m.TrainOnlyPreprocessor.from_dict(cfg, p.to_dict())
                for engine_id in (split["fit_engine_ids"][0], split["validation_engine_ids"][0]):
                    engine = frame[frame.engine_id == engine_id].reset_index(drop=True)
                    whole = p.transform(engine)
                    np.testing.assert_array_equal(whole, restored.transform(engine))
                    self.assertTrue(np.isfinite(whole).all())
                    for endpoint in (1, 15, 30, len(engine) // 2):
                        prefix = engine.iloc[:endpoint].reset_index(drop=True)
                        transformed = p.transform(prefix)
                        np.testing.assert_array_equal(whole[:endpoint], transformed)
                        np.testing.assert_array_equal(m.historical_window(whole, endpoint - 1, cfg.window),
                                                      m.historical_window(transformed, endpoint - 1, cfg.window))

    @unittest.skipUnless(all((ROOT / "data/cmapss" / filename).is_file() for filename in ("train_FD001.txt", "test_FD001.txt")),
                         "Optional NASA corpus absent; prepare FD001 to run the held-out-data scaler check")
    def test_validation_and_test_features_cannot_refit_scaler(self):
        m = self.module
        cfg = m.config_for("FD001")
        frame = m.load_sensor_file(ROOT / "data/cmapss/train_FD001.txt")
        split = m.split_training_engines(frame, cfg)
        p = m.TrainOnlyPreprocessor(cfg).fit(frame, split["fit_engine_ids"])
        before = m.canonical_json(p.to_dict())
        test_frame = m.load_sensor_file(ROOT / "data/cmapss/test_FD001.txt")
        test_frame.loc[:, m.FEATURE_COLUMNS] += 1e6
        p.transform(test_frame)
        self.assertEqual(before, m.canonical_json(p.to_dict()))

    def test_schema_rejects_bad_records(self):
        m = self.module
        base = np.ones((4, 26), dtype=float)
        base[:, 1] = np.arange(1, 5)
        variants = {}
        variants["wrong_columns"] = base[:, :25]
        variants["duplicate"] = np.concatenate([base, base[:1]])
        bad = base.copy(); bad[0, 4] = np.nan; variants["missing"] = bad
        bad = base.copy(); bad[0, 1] = 1.5; variants["noninteger_cycle"] = bad
        bad = base.copy(); bad[1, 1] = 8; variants["cycle_gap"] = bad
        bad = base.copy(); bad[:, 0] = 0; variants["invalid_id"] = bad
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            for name, values in variants.items():
                with self.subTest(name=name):
                    np.savetxt(path, values)
                    with self.assertRaises(ValueError):
                        m.load_sensor_file(path)

    def test_targets_uncapped_and_post_failure_mask(self):
        m = self.module
        cfg = m.config_for("FD001")
        target, mask = m.horizon_targets(200, cfg)
        self.assertAlmostEqual(float(target[0] * cfg.target_scale), 200)
        np.testing.assert_array_equal(mask, np.ones(10))
        target, mask = m.horizon_targets(0, cfg)
        np.testing.assert_array_equal(target, np.zeros(10))
        np.testing.assert_array_equal(mask, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        target, mask = m.horizon_targets(200, m.replace(cfg, rul_cap=125))
        np.testing.assert_allclose(target * cfg.target_scale, np.full(10, 125))

    def test_metric_values_and_overflow_are_explicit(self):
        m = self.module
        metrics = m.regression_metrics([10, 20, 0], [9, 22, 1])
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2))
        self.assertAlmostEqual(metrics["mae"], 4 / 3)
        self.assertAlmostEqual(metrics["mape_percent_nonzero"], 10)
        self.assertEqual(metrics["mape_nonzero_count"], 2)
        self.assertAlmostEqual(metrics["nasa_score"], np.expm1(1 / 13) + np.expm1(2 / 10) + np.expm1(1 / 10))
        self.assertIsNone(m.regression_metrics([1], [1])["r2"])
        overflow = m.regression_metrics([1], [10001])
        self.assertTrue(overflow["nasa_overflow"])
        self.assertIsNone(overflow["nasa_score"])
        with self.assertRaises(ValueError):
            m.regression_metrics([1, 2], [1])

    def test_configuration_validation(self):
        m = self.module
        for subset in m.SUBSETS:
            cfg = m.config_for(subset)
            m.validate_config(cfg)
            self.assertEqual(cfg.encoder_layers, 6)
            self.assertEqual(cfg.decoder_layers, 6)
            self.assertEqual(cfg.effective_batch_size, 256)
        for changes in ({"micro_batch_size": 30}, {"validation_fraction": 0.8}, {"rul_cap": -1},
                        {"n_heads": 7}, {"grid_uniform_mix": 0}, {"num_workers": 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.validate_config(m.replace(m.config_for("FD001"), **changes))

    def test_grouped_explanation_masks_preserve_settings_and_complete_histories(self):
        m = self.module
        window = np.arange(24, dtype=np.float32).reshape(6, 4)
        backgrounds = np.stack([window + 100, window + 200])
        mixed = m.mix_sensor_windows(window, backgrounds, [[1, 1], [0, 0], [1, 0]], [2, 3])
        self.assertEqual(mixed.shape, (3, 2, 6, 4))
        np.testing.assert_array_equal(mixed[0, 0], window)
        np.testing.assert_array_equal(mixed[0, 1], window)
        np.testing.assert_array_equal(mixed[1, :, :, 2:], backgrounds[:, :, 2:])
        for mask_index in range(3):
            for background_index in range(2):
                np.testing.assert_array_equal(mixed[mask_index, background_index, :, :2], window[:, :2])
        np.testing.assert_array_equal(mixed[2, 1, :, 2], window[:, 2])
        np.testing.assert_array_equal(mixed[2, 1, :, 3], backgrounds[1, :, 3])
        with self.assertRaises(ValueError):
            m.mix_sensor_windows(window, backgrounds, [[2, 0]], [2, 3])

    def test_novelty_selection_uses_matched_validation_and_retains_reference_on_ties(self):
        m = data_namespace()
        node = next(n for n in ast.parse(CODE["variant-selection"]).body if isinstance(n, ast.FunctionDef))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "selector", "exec"), m.__dict__)
        with tempfile.TemporaryDirectory() as directory:
            m.OUTPUT_DIR = Path(directory); m.SELECTED_SUBSETS = ["FD001"]; m.SEEDS = [42]
            runs = {}
            for variant, value in (("paper", 10), ("conditioned_kan", 12)):
                folder = Path(directory) / variant; folder.mkdir()
                record = {"status": "completed", "best_validation_rmse": value, "training_file_sha256": "same-data",
                          "config": m.asdict(m.config_for("FD001", variant=variant))}
                m.save_json(folder / "run.json", record)
                m.save_json(folder / "split.json", {"fit_engine_ids": [1,2], "validation_engine_ids": [3]})
                (folder / "validation_endpoints.csv").write_text("engine_id,cycle,true_rul\n3,20,40\n")
                runs[("FD001", variant, 42)] = folder
            result = m.select_variants_on_validation(runs)
            self.assertEqual(result["FD001"]["selected_variant"], "paper")
            again = m.select_variants_on_validation(runs)
            self.assertEqual(again["FD001"]["frozen_at"], result["FD001"]["frozen_at"])
            candidate_path = runs[("FD001", "conditioned_kan", 42)] / "run.json"
            candidate = json.loads(candidate_path.read_text())
            candidate["best_validation_rmse"] = 10; m.save_json(candidate_path, candidate)
            self.assertEqual(m.select_variants_on_validation(runs)["FD001"]["selected_variant"], "paper")
            candidate["best_validation_rmse"] = 8; m.save_json(candidate_path, candidate)
            result = m.select_variants_on_validation(runs)
            self.assertEqual(result["FD001"]["selected_variant"], "conditioned_kan")
            self.assertFalse(result["FD001"]["official_test_used_for_selection"])
            candidate["config"]["window"] = 50; m.save_json(candidate_path, candidate)
            with self.assertRaisesRegex(ValueError, "configuration"):
                m.select_variants_on_validation(runs)

    def test_evaluation_resume_preserves_sealed_evidence_and_detects_changed_sources(self):
        m = data_namespace()
        node = next(n for n in ast.parse(CODE["evaluation-functions"]).body
                    if isinstance(n, ast.FunctionDef) and n.name == "cached_official_evaluation")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "cached-evaluation", "exec"), m.__dict__)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / "data"; run = root / "run"
            data.mkdir(); run.mkdir()
            for path in (data / "test_FD001.txt", data / "RUL_FD001.txt", run / "best.pt",
                         run / "test_predictions.csv", run / "test_predictions_sensor_only.csv"):
                path.write_text("fixture-only")
            record = {"status": "completed", "implementation": m.IMPLEMENTATION_VERSION, "config": {"subset": "FD001"}}
            metrics = {"best_checkpoint_sha256": m.sha256_file(run / "best.pt"),
                       "test_file_sha256": m.sha256_file(data / "test_FD001.txt"),
                       "rul_file_sha256": m.sha256_file(data / "RUL_FD001.txt"), "evaluated_at": "frozen-original-time"}
            m.save_json(run / "run.json", record); m.save_json(run / "metrics.json", metrics)
            m.save_json(run / "research_export.json", {"files": {name: {"sha256": m.sha256_file(run / name)}
                         for name in ("run.json", "metrics.json", "test_predictions.csv", "test_predictions_sensor_only.csv")}})
            self.assertEqual(m.cached_official_evaluation(run, data), metrics)
            (data / "test_FD001.txt").write_text("changed source")
            self.assertIsNone(m.cached_official_evaluation(run, data))


if __name__ == "__main__":
    unittest.main()
