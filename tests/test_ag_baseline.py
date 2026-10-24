"""Numerical CPU integration tests; fixtures are synthetic, never benchmark data."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from aeroguard.baseline import (BaselineBundle, FEATURE_NAMES, SENSOR_SCHEMA, causal_features,
                                evaluate_model, list_models, load_engines, model_folder, train_model)
from fleetpm.features import Preprocessor, split_engines


def write_fixture(directory, engines=20):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    train=[];test=[];truth=[]
    for unit in range(1,engines+1):
        lifetime=55+unit
        observed=lifetime-12-unit%5
        for cycle in range(1,lifetime+1):
            sensors=[450+j*3+cycle*(.03+j*.001)+np.sin(cycle/9+j)*.1+unit*.02 for j in range(21)]
            record=' '.join(map(str,[unit,cycle,0,0,100,*sensors]))+'\n'
            train.append(record)
            if cycle<=observed:test.append(record)
        truth.append(str(lifetime-observed)+'\n')
    (directory/'train_FD001.txt').write_text(''.join(train))
    (directory/'test_FD001.txt').write_text(''.join(test))
    (directory/'RUL_FD001.txt').write_text(''.join(truth))


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.data=cls.root/'data';cls.artifacts=cls.root/'artifacts'
        write_fixture(cls.data)
        cls.config={'subsets':['FD001'],'iterations':12,'window':15,'stride':2,'track_mlflow':False}
        cls.metadata=train_model(cls.config,cls.data,cls.artifacts)
        cls.folder=model_folder(cls.metadata['run_id'],cls.artifacts)
        cls.bundle=BaselineBundle(cls.folder)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_engine_cohorts_and_saved_mapping_never_overlap(self):
        split=json.loads((self.folder/'split.json').read_text())
        self.assertFalse(set(split['fit'])&set(split['selection']))
        self.assertFalse(set(split['fit'])&set(split['calibration']))
        self.assertFalse(set(split['selection'])&set(split['calibration']))
        self.assertEqual(self.bundle.processor.fit_ids,split['fit'])
        mapping=json.loads((self.folder/'training_mapping.json').read_text())
        self.assertTrue({row['engine_id'] for row in mapping}<=set(split['fit']))
        self.assertTrue(all(row['cycle']>=15 for row in mapping))
        self.assertEqual(self.metadata['features'],len(FEATURE_NAMES))

    def test_causal_features_and_predictions_ignore_future_suffix(self):
        raw=load_engines(self.data/'test_FD001.txt')[1]
        before=self.bundle.processor.transform(raw)
        features,_=causal_features(before,15,125)
        prediction=self.bundle.engine_predictions(raw,[24])[0]
        changed=dict(raw);changed['raw']=raw['raw'].copy();changed['raw'][25:,3:]+=50000
        after_features,_=causal_features(self.bundle.processor.transform(changed),15,125)
        np.testing.assert_array_equal(features[:25],after_features[:25])
        after=self.bundle.engine_predictions(changed,[24])[0]
        self.assertEqual(prediction['rul'],after['rul'])
        self.assertEqual(prediction['anomaly'],after['anomaly'])
        # Truncating the supplied trajectory at t cannot change its prediction.
        prefix={**raw,'raw':raw['raw'][:25],'cycles':raw['cycles'][:25]}
        truncated=self.bundle.engine_predictions(prefix,[24])[0]
        self.assertEqual(prediction['rul'],truncated['rul'])

    def test_masked_statistics_are_finite_and_short_prefixes_supported(self):
        raw=load_engines(self.data/'test_FD001.txt')[1]
        raw['raw'][8:12,5]=np.nan
        transformed=self.bundle.processor.transform(raw)
        features,quality=causal_features(transformed,15,125)
        self.assertTrue(np.isfinite(features).all());self.assertLess(float(quality['quality'][11]),1)
        packets=self.bundle.engine_predictions(raw,[0,11,30])
        self.assertEqual(packets[0]['history_length'],1)
        self.assertEqual(packets[0]['required_history'],15)
        self.assertTrue(all(np.isfinite(packet['rul']) for packet in packets))
        self.assertFalse(packets[1]['quality_valid'])
        self.assertNotIn('true_rul',packets[1])

    def test_independent_anomaly_reference_excludes_age_and_reserved_engines(self):
        split=json.loads((self.folder/'split.json').read_text())
        self.assertEqual(self.bundle.anomaly['feature_columns'],list(range(84)))
        self.assertEqual(self.bundle.anomaly['reference_engine_ids'],split['fit'])
        reference=self.bundle.anomaly['reference'];self.assertGreater(len(reference),20)
        self.assertTrue(np.all(reference[:-1]<=reference[1:]))
        packets=self.bundle.engine_predictions(load_engines(self.data/'test_FD001.txt')[2],[30,35])
        self.assertTrue(all(0<=packet['anomaly']<=1 for packet in packets))

    def test_official_evaluation_has_one_endpoint_per_engine_and_separate_truth(self):
        report=evaluate_model(self.metadata['run_id'],self.data,self.artifacts)
        predictions=json.loads((self.folder/'predictions.json').read_text())
        truth=json.loads((self.folder/'evaluation_targets.json').read_text())
        self.assertEqual(report['metrics']['n_engines'],20)
        self.assertEqual(len(predictions),len(truth))
        self.assertEqual([p['engine_id'] for p in predictions],list(range(1,21)))
        self.assertTrue(all('true_rul' not in p for p in predictions))
        self.assertIsInstance(report['metrics']['rmse'],float)
        self.assertIsInstance(report['metrics']['nasa_score'],float)
        self.assertIn('not',report['anomaly_evaluation'].lower())

    def test_changed_truth_cannot_rewrite_existing_evaluation(self):
        evaluate_model(self.metadata['run_id'],self.data,self.artifacts)
        truth_path=self.data/'RUL_FD001.txt';old=truth_path.read_text()
        try:
            truth_path.write_text('99\n'*20)
            with self.assertRaisesRegex(ValueError,'different official'):evaluate_model(self.metadata['run_id'],self.data,self.artifacts)
        finally:truth_path.write_text(old)

    def test_training_never_opens_official_sensor_or_truth_files(self):
        with tempfile.TemporaryDirectory() as folder:
            write_fixture(folder)
            Path(folder,'test_FD001.txt').unlink();Path(folder,'RUL_FD001.txt').unlink()
            with patch('aeroguard.baseline.parse_truth',side_effect=AssertionError('Training read official truth')):
                trained=train_model(self.config,folder,Path(folder)/'artifacts')
            self.assertEqual(trained['training_status'],'complete')

    def test_bundle_integrity_and_registry_path_validation(self):
        self.assertTrue(any(r['run_id']==self.metadata['run_id'] for r in list_models(self.artifacts)))
        with self.assertRaises(ValueError):model_folder('../../private',self.artifacts)
        original=(self.folder/'feature_names.json').read_bytes()
        try:
            (self.folder/'feature_names.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'integrity'):BaselineBundle(self.folder)
        finally:(self.folder/'feature_names.json').write_bytes(original)

    def test_sensor_schema_uses_actual_cmapss_names(self):
        self.assertEqual(len(SENSOR_SCHEMA),21)
        self.assertEqual(SENSOR_SCHEMA[2]['symbol'],'T30')
        self.assertEqual(SENSOR_SCHEMA[6]['symbol'],'P30')
        self.assertEqual(SENSOR_SCHEMA[18]['symbol'],'PCNfR_dmd')
        self.assertFalse(any('vibration' in sensor['label'].lower() for sensor in SENSOR_SCHEMA))

    def test_mlflow_sqlite_records_real_fit_and_evaluation(self):
        with tempfile.TemporaryDirectory() as base:
            root=Path(base);write_fixture(root/'data')
            metadata=train_model(self.config|{'iterations':10,'track_mlflow':True},root/'data',root/'artifacts')
            self.assertEqual(metadata['tracking_status'],'complete',metadata.get('tracking_error'))
            self.assertTrue((root/'artifacts/aeroguard/mlflow.db').is_file())
            report=evaluate_model(metadata['run_id'],root/'data',root/'artifacts')
            self.assertNotIn('tracking_error',report)
            import mlflow
            client=mlflow.MlflowClient(tracking_uri='sqlite:///'+str((root/'artifacts/aeroguard/mlflow.db').resolve()))
            run=client.get_run(metadata['mlflow_run_id'])
            self.assertEqual(run.info.status,'FINISHED')
            self.assertIn('validation_rmse',run.data.metrics)
            self.assertIn('official_test_rmse',run.data.metrics)


if __name__=='__main__':unittest.main()
