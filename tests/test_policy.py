import json
from pathlib import Path
import tempfile
import unittest

from fleetpm.store import Store


def packet(cycle=1,**values):
    return {"asset":"FD001/test/001","cycle":cycle,"bundle_id":"fixture-bundle","rul":22,
            "lower":15,"upper":29,"anomaly":.6,"quality_valid":True,"data_quality_valid":True,
            "history_length":30}|values


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'orders.sqlite';self.store=Store(self.path)

    def test_high_persists_three_cycles_and_duplicate_does_not_advance(self):
        self.assertEqual(self.store.process(packet())['state'],'WATCH')
        self.assertTrue(self.store.process(packet())['replayed'])
        self.assertEqual(self.store.process(packet(2))['state'],'WATCH')
        result=self.store.process(packet(3));self.assertEqual(result['state'],'HIGH')
        self.assertEqual(len(self.store.orders()),1)

    def test_critical_escalates_existing_order_and_survives_restart(self):
        for cycle in range(1,4):self.store.process(packet(cycle))
        original=self.store.orders()[0]
        self.store.process(packet(4,rul=4,lower=1,upper=7))
        order=Store(self.path).orders()[0]
        self.assertEqual(order['id'],original['id']);self.assertEqual(order['priority'],'CRITICAL')
        self.assertLessEqual(order['target_cycle'],original['target_cycle'])
        result=Store(self.path).process(packet(5,quality_valid=False,data_quality_valid=False,exception_type='DATA'))
        self.assertEqual(result['state'],'DATA_EXCEPTION');self.assertEqual(result['latched_priority'],'CRITICAL')

    def test_anomaly_only_creates_inspection_order(self):
        for cycle in range(1,4):result=self.store.process(packet(cycle,rul=95,lower=80,upper=110,anomaly=.998))
        self.assertEqual(result['state'],'MEDIUM');self.assertEqual(self.store.orders()[0]['priority'],'MEDIUM')

    def test_quality_gate_precedes_equipment_risk(self):
        for cycle in range(1,3):self.store.process(packet(cycle,rul=2,lower=1,upper=3,quality_valid=False,data_quality_valid=False,exception_type='DATA'))
        orders=self.store.orders();self.assertEqual(len(orders),1);self.assertEqual(orders[0]['kind'],'SENSOR')

    def test_model_uncertainty_does_not_become_sensor_repair(self):
        for kind in ['DOMAIN','CALIBRATION','UNCERTAINTY']:
            for cycle in range(1,4):self.store.process(packet(cycle,asset=kind,quality_valid=False,exception_type=kind))
        self.assertFalse(self.store.orders())
        self.store.process(packet(4,asset='UNCERTAINTY',quality_valid=False,exception_type='DATA'))
        self.assertFalse(self.store.orders(),"Uncertainty must not advance the persistent sensor-fault counter")

    def test_urgent_loss_and_short_context(self):
        result=self.store.process(packet(quality_valid=False,data_quality_valid=False,complete_loss=True))
        self.assertEqual(result['priority'],'HIGH');self.assertEqual(self.store.orders()[0]['kind'],'SENSOR')
        result=self.store.process(packet(asset='short',rul=2,lower=1,upper=3,history_length=15))
        self.assertEqual(result['state'],'WARMING_UP');self.assertEqual(len(self.store.orders()),1)

    def test_gap_and_invalid_interval_reset_persistence(self):
        self.store.process(packet());self.store.process(packet(2))
        self.store.process(packet(3,lower=None,upper=None));self.assertEqual(self.store.process(packet(4))['state'],'WATCH')
        self.assertEqual(self.store.process(packet(9))['state'],'WATCH');self.assertFalse(self.store.orders())

    def test_late_and_conflicting_events_are_rejected(self):
        self.store.process(packet(3))
        with self.assertRaisesRegex(ValueError,'Conflicting'):self.store.process(packet(3,rul=21))
        with self.assertRaisesRegex(ValueError,'Late'):self.store.process(packet(2))

    def test_namespaces_and_lifecycles_are_isolated(self):
        for namespace in ['demo','dataset']:
            self.store.process(packet(rul=4,lower=1,upper=7),namespace)
        self.store.process(packet(rul=4,lower=1,upper=7,lifecycle='2'),'dataset')
        self.assertEqual(len(self.store.orders('demo')),1);self.assertEqual(len(self.store.orders('dataset')),2)
        with self.assertRaises(ValueError):self.store.update_order(self.store.orders('demo')[0]['id'],'acknowledged',namespace='dataset')

    def test_outcomes_idempotent_acknowledgement_and_rearm(self):
        self.store.process(packet(rul=4,lower=1,upper=7));oid=self.store.orders()[0]['id']
        first=self.store.update_order(oid,'acknowledged');second=self.store.update_order(oid,'acknowledged')
        self.assertEqual(first['revision'],second['revision'])
        with self.assertRaises(ValueError):self.store.update_order(oid,'resolved')
        self.store.update_order(oid,'resolved','preventive_service')
        self.store.process(packet(2,rul=4,lower=1,upper=7));self.assertEqual(len(self.store.orders()),1)
        for cycle in range(3,8):self.store.process(packet(cycle,rul=90,lower=75,upper=105,anomaly=.4))
        self.store.process(packet(8,rul=4,lower=1,upper=7));self.assertEqual(len(self.store.orders()),2)
        with self.store.connect() as db:
            count,unique=db.execute('SELECT count(*),count(DISTINCT idempotency_key) FROM outbox').fetchone()
        self.assertEqual(count,unique);self.assertGreater(self.store.dispatch_local(),0);self.assertEqual(self.store.dispatch_local(),0)
