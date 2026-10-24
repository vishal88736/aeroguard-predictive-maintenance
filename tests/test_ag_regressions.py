"""Regression checks for evidence integrity and policy/source boundaries."""

import json
from pathlib import Path
import tempfile
import unittest

from aeroguard.baseline import evaluate_model,model_folder,train_model
from aeroguard.repository import Repository
from aeroguard.services import Service
from fleetpm.data import parse_rows
from fleetpm.store import Store
from test_ag_baseline import write_fixture


def packet(cycle=1,**values):
    return {'asset':'FD001/test/001','cycle':cycle,'bundle_id':'test-regression-model',
            'rul':4,'lower':1,'upper':7,'anomaly':.8,'quality_valid':True,
            'data_quality_valid':True,'history_length':30}|values


class EvidenceRegressions(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.repo=Repository(self.root/'orders.sqlite')

    def test_creation_rul_survives_later_evidence_revisions(self):
        self.repo.store.process(packet(),'benchmark:test')
        self.repo.store.process(packet(2,rul=2,lower=0,upper=5),'benchmark:test')
        order=self.repo.orders('benchmark')[0]
        self.assertEqual(order['predicted_rul'],4)
        self.assertEqual(order['current_rul'],2)
        self.assertEqual(len(self.repo.orders('benchmark')),1)

    def test_unavailable_prediction_is_quality_exception_not_numeric_comparison(self):
        decision=self.repo.store.process(packet(rul=None,lower=None,upper=None,quality_valid=False,exception_type='CALIBRATION'))
        self.assertEqual(decision['state'],'DATA_EXCEPTION')
        critical=next(rule for rule in decision['rules_evaluated'] if rule['rule']=='critical_point')
        self.assertFalse(critical['evaluated']);self.assertFalse(critical['matched'])
        self.assertFalse(self.repo.orders('benchmark'))

    def test_configured_context_and_persistence_are_used_in_explanation(self):
        store=Store(self.root/'policy.sqlite',policy={'persistence':4})
        for cycle in range(1,5):
            decision=store.process(packet(cycle,rul=24,lower=15,upper=35,history_length=15,required_history=15))
            self.assertNotEqual(decision['state'],'WARMING_UP')
        self.assertEqual(decision['state'],'HIGH');self.assertIn('4 cycles',decision['reason'])

    def test_first_promotion_requires_finite_nasa_evidence(self):
        service=Service(self.root/'data',self.root/'artifacts',self.root/'runtime')
        self.addCleanup(service.close)
        candidate={'run_id':'AG-FD001-'+'a'*12,'subset':'FD001','training_status':'complete',
                   'validation_metrics':{'rmse':1.0,'nasa_score':None}}
        eligible,reason=service.promotion_gate(candidate)
        self.assertFalse(eligible);self.assertIn('NASA',reason)

    def test_oversized_cycle_is_rejected_before_dense_allocation(self):
        path=self.root/'bad.txt';path.write_text(' '.join(map(str,[1,10001,0,0,100]+[10]*21))+'\n')
        with self.assertRaisesRegex(ValueError,'allocation limit'):parse_rows(path)


class ModelEvidenceRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.data=cls.root/'data';cls.artifacts=cls.root/'artifacts'
        write_fixture(cls.data,engines=50)
        cls.metadata=train_model({'subsets':['FD001'],'iterations':10,'window':15,'stride':3,'track_mlflow':False},cls.data,cls.artifacts)
        cls.model_id=cls.metadata['run_id'];cls.folder=model_folder(cls.model_id,cls.artifacts)
        evaluate_model(cls.model_id,cls.data,cls.artifacts)
        cls.service=Service(cls.data,cls.artifacts,cls.root/'runtime')
        cls.service.promote(cls.model_id)

    @classmethod
    def tearDownClass(cls):
        cls.service.close();cls.temp.cleanup()

    def test_cached_bundle_still_rejects_changed_artifact(self):
        self.service.bundle(self.model_id)
        path=self.folder/'model.joblib';original=path.read_bytes()
        try:
            path.write_bytes(b'INVALID ARTIFACT')
            with self.assertRaisesRegex(ValueError,'integrity'):self.service.bundle(self.model_id)
        finally:path.write_bytes(original)

    def test_engine_truth_uses_frozen_evaluation_not_new_current_label_file(self):
        path=self.data/'RUL_FD001.txt';original=path.read_bytes()
        frozen=json.loads((self.folder/'evaluation_targets.json').read_text())[0]['true_rul']
        try:
            path.write_text('99\n'*50)
            detail=self.service.asset_detail('FD001/test/001','benchmark',limit=10)
            self.assertEqual(detail['history'][-1]['true_rul'],frozen)
            self.assertNotEqual(frozen,99)
        finally:path.write_bytes(original)

    def test_dashboard_quality_uses_archived_same_cycle_policy_packet(self):
        original=next(a for a in self.service.assets('benchmark') if a['quality_valid'])
        changed={k:v for k,v in original.items() if k not in {'state','health','decision'}}
        changed.update(quality_valid=False,quality_reason='Archived narrow uncertainty gate',exception_type='UNCERTAINTY')
        self.service.repo.store.process(changed,'benchmark:'+self.model_id)
        displayed=next(a for a in self.service.assets('benchmark') if a['asset']==original['asset'])
        self.assertFalse(displayed['quality_valid']);self.assertEqual(displayed['quality_reason'],changed['quality_reason'])
        self.assertEqual(displayed['health'],'attention')

    def test_promotion_and_explicit_rollback_retain_previous_artifacts(self):
        second=train_model({'subsets':['FD001'],'iterations':10,'window':15,'stride':3,'track_mlflow':False},self.data,self.artifacts)
        self.service.promote(second['run_id'])
        self.assertEqual(self.service.active()['FD001'],second['run_id'])
        self.service.promote(self.model_id,rollback=True)
        self.assertEqual(self.service.active()['FD001'],self.model_id)
        self.assertTrue((model_folder(second['run_id'],self.artifacts)/'model.joblib').is_file())
        self.assertTrue((self.folder/'model.joblib').is_file())
