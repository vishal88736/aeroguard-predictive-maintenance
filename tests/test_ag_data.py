import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from aeroguard.datasets import COLUMNS, COLUMN_NAMES, DatasetManager
from fleetpm.config import file_hash
from helpers import archive_payload


def csv_fixture(engines=20, cycles=40, sensor_delta=0, missing=False):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(COLUMN_NAMES)
    for unit in range(1, engines + 1):
        for cycle in range(1, cycles + 1):
            sensors = [500 + unit * .2 + cycle * .03 + sensor_delta + j for j in range(21)]
            if missing and unit == 1 and cycle == 2:
                sensors[2] = "nan"
            writer.writerow([unit, cycle, 0, 0, 100, *sensors])
    return stream.getvalue().encode()


class DatasetManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manager = DatasetManager(self.root / "data", self.root / "artifacts")

    def test_all_subsets_are_reported_with_explicit_provenance(self):
        summary = self.manager.summary()
        self.assertEqual([r["subset"] for r in summary["datasets"]], ["FD001", "FD002", "FD003", "FD004"])
        self.assertFalse(any(r["ready"] for r in summary["datasets"]))
        self.assertEqual(len(summary["columns"]), 26)
        self.assertIn("not been verified", summary["provenance"]["acquisition"])
        self.assertEqual(COLUMNS[7]["symbol"], "T30")
        self.assertFalse(any("vibration" in c["label"].lower() for c in COLUMNS))

    def test_csv_import_preview_and_missing_sensor_analysis(self):
        result = self.manager.import_file("training.csv", csv_fixture(missing=True))
        self.assertEqual(result["rows"], 800)
        preview = self.manager.preview("FD001", "train", 3)
        self.assertEqual(preview["statistics"]["engine_count"], 20)
        self.assertEqual(preview["statistics"]["missing_values"], 1)
        self.assertEqual(preview["statistics"]["missing_by_column"]["sensor_3"], 1)
        self.assertIsNone(preview["rows"][1][7])
        self.assertIsNone(preview["split"])
        json.dumps(preview, allow_nan=False)

    def test_csv_reimport_is_idempotent_and_different_file_preserved(self):
        payload = csv_fixture()
        first = self.manager.import_file("train.csv", payload)
        self.assertTrue(self.manager.import_file("train.csv", payload)["unchanged"])
        with self.assertRaisesRegex(ValueError, "different contents"):
            self.manager.import_file("train.csv", csv_fixture(sensor_delta=1))
        self.assertEqual(file_hash(self.root / "data/train_FD001.txt"), first["sha256"])

    def test_schema_duplicate_invalid_setting_and_label_column_rejected(self):
        payload = csv_fixture(engines=1, cycles=2).decode()
        lines = payload.splitlines()
        bad_inputs = [
            "\n".join([lines[0] + ",RUL", lines[1] + ",15"]),
            "\n".join([lines[0], lines[1], lines[1]]),
            "\n".join([lines[0], lines[1].replace("1,1,0,0,100", "1,1,nan,0,100")]),
            "unit,cycle\n1,1\n",
        ]
        for bad in bad_inputs:
            with self.subTest(bad=bad[:30]), self.assertRaises(ValueError):
                self.manager.import_file("input.csv", bad.encode())
        self.assertFalse((self.root / "data/train_FD001.txt").exists())

    def test_test_truth_mapping_validated_before_commit(self):
        self.manager.import_file("test.csv", csv_fixture(engines=2, cycles=5), partition="test")
        with self.assertRaisesRegex(ValueError, "expected 2"):
            self.manager.import_file("truth.csv", b"RUL\n12\n", partition="RUL")
        self.assertFalse((self.root / "data/RUL_FD001.txt").exists())
        self.manager.import_file("truth.csv", b"RUL\n12\n15\n", partition="RUL")
        preview = self.manager.preview("FD001", "RUL")
        self.assertEqual(preview["rows"], [[12], [15]])
        self.assertIn("evaluation only", preview["statistics"]["label_role"])

    def test_zip_supports_original_twelve_files(self):
        result = self.manager.import_file("cmapss.zip", archive_payload())
        self.assertEqual(result["imported_files"], 12)
        self.assertTrue(all(r["ready"] for r in result["datasets"]))
        self.assertEqual(len(list((self.root / "data").glob("*.txt"))), 12)

    def test_prepare_never_reads_test_truth_and_keeps_engine_groups_disjoint(self):
        self.manager.import_file("train.csv", csv_fixture())
        # Poisoned held-out files must not be parsed or normalized during prepare.
        (self.root / "data/test_FD001.txt").write_text("not test data")
        (self.root / "data/RUL_FD001.txt").write_text("not labels")
        result = self.manager.prepare(["FD001"])
        audit = result["subsets"][0]
        split = audit["split"]
        groups = [set(ids) for ids in split.values()]
        self.assertEqual(set.union(*groups), set(range(1, 21)))
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                self.assertFalse(left & right)
        self.assertEqual(audit["preprocessing_fit_ids"], split["fit"])
        self.assertTrue(audit["features_finite"])
        self.assertEqual(audit["input_shape"], [30, 66])
        self.assertTrue((self.root / "artifacts/aeroguard/prepared/FD001/preprocessing.joblib").is_file())

    def test_heldout_engine_values_do_not_change_fitted_preprocessing(self):
        import joblib
        import numpy as np
        self.manager.import_file("train.csv", csv_fixture())
        first = self.manager.prepare("FD001")["subsets"][0]
        artifact = self.root / "artifacts/aeroguard/prepared/FD001/preprocessing.joblib"
        before = joblib.load(artifact).sensor_mean.copy()
        excluded = set(first["split"]["selection"] + first["split"]["calibration"])
        source = self.root / "data/train_FD001.txt"
        rewritten = []
        for line in source.read_text().splitlines():
            fields = line.split()
            if int(float(fields[0])) in excluded:
                fields[5:] = ["1000000"] * 21
            rewritten.append(" ".join(fields))
        source.write_text("\n".join(rewritten) + "\n")
        second = self.manager.prepare("FD001")["subsets"][0]
        self.assertEqual(first["split"], second["split"])
        np.testing.assert_array_equal(before, joblib.load(artifact).sensor_mean)

    def test_prepare_summary_is_invalidated_when_training_source_changes(self):
        self.manager.import_file("train.csv", csv_fixture())
        self.manager.prepare("FD001")
        self.assertIsNotNone(self.manager.preview("FD001")["split"])
        path = self.root / "data/train_FD001.txt"
        path.write_text(path.read_text() + "\n")
        self.assertIsNone(self.manager.preview("FD001")["split"])

    def test_missing_and_invalid_routes_are_actionable(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            self.manager.preview("FD001")
        for subset, partition in [("FD005", "train"), ("FD001", "validation")]:
            with self.assertRaises(ValueError):
                self.manager.preview(subset, partition)
        with self.assertRaises(ValueError):
            self.manager.import_file("file.exe", b"1 2")
        for filename, content, partition in [("empty.csv", b"\xef\xbb\xbf", "train"), ("empty.txt", b"\n", "RUL")]:
            with self.assertRaises(ValueError):
                self.manager.import_file(filename, content, partition=partition)
