import io
import math
from pathlib import Path
import tempfile
import unittest
import zipfile

from fleetpm.config import file_hash
from fleetpm.data import import_zip,inventory,parse_rows,parse_truth
from fleetpm.metrics import conformal_radius,prediction_interval,regression_metrics
from helpers import archive_payload,row


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write(self,name,text):
        path=self.root/name;path.write_text(text);return path

    def test_rows_sort_and_allow_explicit_missing_sensors(self):
        result=parse_rows(self.write("train.txt",row(2,1)+row(1,2,"nan")+row(1,1)))
        self.assertEqual([(r[0],r[1]) for r in result],[(1,1),(1,2),(2,1)])
        self.assertTrue(math.isnan(result[1][5]))

    def test_invalid_sensor_schema_identity_and_settings_rejected(self):
        for text in ["1 1 0\n",row()+row(),row(1.5),row(1,0),row(sensor="inf"),row().replace("0 0 100","nan 0 100")]:
            with self.subTest(text=text[:30]),self.assertRaises(ValueError):parse_rows(self.write("bad.txt",text))

    def test_truth_count_and_values(self):
        self.assertEqual(parse_truth(self.write("RUL.txt","8\n20\n"),2),[8,20])
        for text in ["8\n","8 20\n","-1\n20\n","nan\n20\n"]:
            with self.assertRaises(ValueError):parse_truth(self.write("RUL.txt",text),2)

    def test_zip_validates_all_subsets_and_is_idempotent(self):
        data=self.root/"data";result=import_zip(archive_payload(),data)
        self.assertTrue(all(r["ready"] for r in result));self.assertEqual(len(list(data.glob('*.txt'))),12)
        self.assertTrue(all(r["official_counts_match"] is False for r in result))
        original=file_hash(data/"train_FD001.txt")
        import_zip(archive_payload(),data)
        self.assertEqual(file_hash(data/"train_FD001.txt"),original)
        with self.assertRaisesRegex(ValueError,"different contents"):import_zip(archive_payload(sensor=900),data)
        self.assertEqual(file_hash(data/"train_FD001.txt"),original)
        self.assertFalse(list(data.glob('.import-*')))

    def test_zip_uses_only_canonical_basenames(self):
        import_zip(archive_payload(prefix="../../escape/"),self.root/"data")
        self.assertFalse((self.root/"escape").exists())
        self.assertEqual(len(list((self.root/"data").glob('*.txt'))),12)

    def test_missing_zip_members_leave_dataset_untouched(self):
        content=io.BytesIO()
        with zipfile.ZipFile(content,'w') as z:z.writestr('train_FD001.txt',row())
        with self.assertRaisesRegex(ValueError,"all 12"):import_zip(content.getvalue(),self.root/"data")
        self.assertFalse((self.root/"data").exists())

    def test_duplicate_canonical_filenames_rejected(self):
        content=io.BytesIO(archive_payload())
        with zipfile.ZipFile(content,'a') as z:z.writestr('other/train_fd001.txt',row())
        with self.assertRaisesRegex(ValueError,"Duplicate"):import_zip(content.getvalue(),self.root/"data")

    def test_inventory_empty_and_partial(self):
        self.assertFalse(any(r['ready'] for r in inventory(self.root,True)))
        self.write('train_FD001.txt',row());self.assertEqual(len(inventory(self.root)[0]['files']),1)


class MetricsTests(unittest.TestCase):
    def test_nasa_penalizes_late_prediction_more(self):
        late=regression_metrics([30],[40]);early=regression_metrics([30],[20])
        self.assertGreater(late['nasa_score'],early['nasa_score'])
        self.assertAlmostEqual(late['nasa_score'],math.expm1(1))
        self.assertEqual(late['rmse'],10)

    def test_raw_truth_is_not_silently_capped(self):
        result=regression_metrics([150],[125]);self.assertEqual(result['rmse'],25);self.assertEqual(result['capped_rmse'],0)

    def test_invalid_metrics_rejected(self):
        for truth,pred in [([],[]),([1],[1,2]),([math.nan],[1])]:
            with self.assertRaises(ValueError):regression_metrics(truth,pred)

    def test_finite_sample_quantile_and_insufficient_calibration(self):
        result=conformal_radius(range(20));self.assertEqual(result['order_statistic'],19);self.assertEqual(result['radius'],18)
        self.assertEqual(prediction_interval(120,result),(102,125))
        small=conformal_radius([1,2]);self.assertIsNone(small['radius']);self.assertEqual(prediction_interval(20,small),(None,None))
