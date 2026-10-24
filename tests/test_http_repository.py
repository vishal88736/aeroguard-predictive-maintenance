import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch,MagicMock

from fleetpm.config import file_hash,write_json
from fleetpm.repository import activate_run,active_runs,run_folder
from fleetpm.webapp import Application,make_handler
from http.server import ThreadingHTTPServer
from helpers import archive_payload


class QuietHandler:
    def log_message(self,*args):pass


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();root=Path(cls.tmp.name)
        cls.app=Application(root/'data',root/'artifacts',root/'runtime')
        handler=type('Quiet',(QuietHandler,make_handler(cls.app)),{})
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.port=cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join();cls.tmp.cleanup()

    def request(self,path,payload=None,headers=None,binary=False):
        connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=10)
        hdr={'X-Fleet-Client':'dashboard'}| (headers or {})
        body=payload if binary else json.dumps(payload) if payload is not None else None
        connection.request('POST' if payload is not None else 'GET',path,body,hdr)
        response=connection.getresponse();raw=response.read();status=response.status;ct=response.getheader('Content-Type');connection.close()
        return status,json.loads(raw) if ct=='application/json' else raw

    def test_modes_are_explicit_and_isolated(self):
        status,real=self.request('/api/overview');self.assertEqual(status,200);self.assertFalse(real['demo']);self.assertEqual(real['assets'],[]);self.assertEqual(real['orders'],[])
        _,demo=self.request('/api/overview?mode=demo');self.assertEqual(len(demo['assets']),32);self.assertTrue(all(a['demo'] for a in demo['assets']))
        _,subset=self.request('/api/overview?mode=demo&subset=FD004');self.assertEqual(len(subset['assets']),8)

    def test_static_files_downloads_and_notebook_are_real(self):
        for path in ['/','/app.js','/styles.css','/assets/favicon.svg','/documents/model.svg','/documents/system.svg','/downloads/requirements','/downloads/environment','/downloads/architecture']:
            status,body=self.request(path);self.assertEqual(status,200,path);self.assertGreater(len(body),20)
        status,body=self.request('/downloads/notebook');self.assertEqual(status,200);self.assertEqual(json.loads(body)['nbformat'],4)

    def test_loopback_host_origin_and_path_boundaries(self):
        self.assertEqual(self.request('/api/health',headers={'Host':'attacker.example'})[0],403)
        self.assertEqual(self.request('/api/train',{},headers={'Origin':'https://attacker.example'})[0],403)
        self.assertEqual(self.request('/api/train',{},headers={'X-Fleet-Client':''})[0],403)
        self.assertEqual(self.request('/../requirements.txt')[0],404)
        self.assertEqual(self.request('/%2e%2e/.env')[0],404)
        self.assertEqual(self.request('/api/overview?mode=invalid')[0],400)

    def test_demo_asset_detail_is_labelled_synthetic(self):
        status,detail=self.request('/api/asset?mode=demo&id=FD001/demo/004');self.assertEqual(status,200)
        self.assertTrue(detail['demo']);self.assertEqual(len(detail['history']),60)
        self.assertIn('Synthetic',detail['source'])
        self.assertEqual(self.request('/api/asset?mode=dataset&id=FD001/demo/004')[0],404)

    def test_demo_order_outcome_persists_without_touching_dataset(self):
        _,demo=self.request('/api/overview?mode=demo');order=next(o for o in demo['orders'] if o['status']=='open')
        self.assertEqual(self.request('/api/orders/update?mode=dataset',{'id':order['id'],'status':'acknowledged'})[0],400)
        status,_=self.request('/api/orders/update?mode=demo',{'id':order['id'],'status':'acknowledged'});self.assertEqual(status,200)
        status,_=self.request('/api/orders/update?mode=demo',{'id':order['id'],'status':'resolved','resolution':'sensor_repair'});self.assertEqual(status,200)
        persisted=next(o for o in self.app.store.orders('demo') if o['id']==order['id']);self.assertEqual(persisted['resolution'],'sensor_repair')
        self.assertFalse(self.app.store.orders('dataset'))

    def test_import_validates_without_starting_training(self):
        status,_=self.request('/api/import?mode=demo',archive_payload(),binary=True);self.assertEqual(status,400)
        status,_=self.request('/api/import?mode=dataset',b'not a zip',binary=True);self.assertEqual(status,400)
        status,result=self.request('/api/import?mode=dataset',archive_payload(),binary=True);self.assertEqual(status,200)
        self.assertTrue(all(d['ready'] for d in result['datasets']));self.assertFalse(self.app.jobs)

    def test_training_validation_and_missing_dependency_error(self):
        self.assertEqual(self.request('/api/train?mode=demo',{})[0],400)
        self.assertEqual(self.request('/api/train',{'subsets':['FD005']})[0],400)
        with patch('fleetpm.webapp.dependency_status',return_value={'torch':False}):
            status,result=self.request('/api/train',{'subsets':['FD001']});self.assertEqual(status,409);self.assertIn('Missing: torch',result['error'])

    def test_evaluation_and_replay_require_real_mode_and_dependencies(self):
        for path in ['/api/evaluate','/api/replay']:
            self.assertEqual(self.request(path+'?mode=demo',{})[0],400)
            with patch('fleetpm.webapp.dependency_status',return_value={'torch':False}):
                self.assertEqual(self.request(path,{})[0],409)

    def test_job_process_arguments_database_and_monitor_without_training(self):
        fake=MagicMock();fake.pid=12345;fake.poll.return_value=None;fake.wait.return_value=0
        with patch('fleetpm.webapp.dependency_status',return_value={'torch':True}),patch('fleetpm.webapp.subprocess.Popen',return_value=fake) as launch,patch('fleetpm.webapp.threading.Thread'):
            self.app.inventory_stamp=None
            # This fixture is isolated in a temporary directory; no model is executed.
            if not (self.app.data_dir/'train_FD001.txt').exists():
                from fleetpm.data import import_zip
                import_zip(archive_payload(),self.app.data_dir)
            result=self.app.start_training({'subsets':['FD001','FD004'],'device':'cpu','profile':'smoke'})
            args,kwargs=launch.call_args
            self.assertIn('fleetpm.train',args[0]);self.assertNotIn('shell',kwargs)
            self.assertEqual(kwargs['env']['FLEET_DATA_DIR'],str(self.app.data_dir))
            self.assertEqual(self.app.job_list()[0]['status'],'running')
            self.app._monitor_job(result['job_id'],fake)
            self.assertEqual(self.app.job_list()[0]['status'],'complete')
            fake.poll.return_value=0


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.run='FD001-'+'a'*16;self.folder=self.root/'runs'/self.run;self.folder.mkdir(parents=True)
        names=('model.pt','preprocessing.joblib','anomaly.joblib','residual.json','calibration.json','config.json','split.json','folds.json')
        for name in names:(self.folder/name).write_text('INTEGRITY TEST FIXTURE; NOT A TRAINED MODEL')
        write_json(self.folder/'bundle_hashes.json',{n:file_hash(self.folder/n) for n in names})
        write_json(self.folder/'metadata.json',{'run_id':self.run,'subset':'FD001','training_status':'complete','smoke_only':False})

    def test_integrity_activation_and_path_validation(self):
        activate_run(self.run,self.root);self.assertEqual(active_runs(self.root),{'FD001':self.run})
        (self.folder/'model.pt').write_text('CHANGED')
        with self.assertRaisesRegex(ValueError,'integrity'):activate_run(self.run,self.root)
        with self.assertRaises(ValueError):run_folder('../../secret',self.root)

    def test_incomplete_and_smoke_bundles_rejected(self):
        write_json(self.folder/'bundle_hashes.json',{})
        with self.assertRaisesRegex(ValueError,'incomplete'):activate_run(self.run,self.root)
        write_json(self.folder/'metadata.json',{'training_status':'complete','smoke_only':True})
        with self.assertRaisesRegex(ValueError,'non-smoke'):activate_run(self.run,self.root)
