"""SQLite application repository, separate from model computation and HTTP."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import uuid

from fleetpm.store import Store, now
from .contracts import Settings


class Conflict(ValueError):
    pass


class Repository:
    def __init__(self, path):
        self.store = Store(path)
        with self.connect() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS ag_settings(id INTEGER PRIMARY KEY CHECK(id=1), value TEXT, updated_at TEXT);
              CREATE TABLE IF NOT EXISTS ag_jobs(id TEXT PRIMARY KEY, kind TEXT, status TEXT, configuration TEXT,
                idempotency_key TEXT UNIQUE, fingerprint TEXT, created_at TEXT, started_at TEXT, finished_at TEXT,
                error TEXT, result TEXT);
              CREATE UNIQUE INDEX IF NOT EXISTS ag_one_running ON ag_jobs((1)) WHERE status IN ('queued','running');
              CREATE TABLE IF NOT EXISTS ag_stages(id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, name TEXT,
                subset TEXT, status TEXT, started_at TEXT, finished_at TEXT, duration_seconds REAL,
                records INTEGER, detail TEXT, error TEXT);
              CREATE TABLE IF NOT EXISTS ag_audit(id INTEGER PRIMARY KEY AUTOINCREMENT, entity TEXT, entity_id TEXT,
                action TEXT, details TEXT, created_at TEXT);
              CREATE TABLE IF NOT EXISTS ag_order_metadata(order_id TEXT PRIMARY KEY, owner TEXT, notes TEXT);
              CREATE TABLE IF NOT EXISTS ag_promotions(id INTEGER PRIMARY KEY AUTOINCREMENT, subset TEXT,
                run_id TEXT, previous_run_id TEXT, details TEXT, created_at TEXT);
              CREATE TABLE IF NOT EXISTS ag_batches(id TEXT PRIMARY KEY, filename TEXT, subset TEXT, partition TEXT,
                rows INTEGER, sha256 TEXT, created_at TEXT);
              CREATE INDEX IF NOT EXISTS ag_latest_event ON events(namespace,asset,cycle);
            ''')
            db.execute("INSERT OR IGNORE INTO ag_settings VALUES(1,?,?)", (json.dumps(Settings().model_dump()),now()))
            db.execute("UPDATE ag_jobs SET status='interrupted',finished_at=?,error='Application restarted. Retry explicitly; completed artifacts are retained.' WHERE status IN ('queued','running')",(now(),))
            db.execute("UPDATE ag_stages SET status='interrupted',finished_at=?,error='Application restarted' WHERE status='running'",(now(),))

    @contextmanager
    def connect(self):
        connection=self.store.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def settings(self):
        with self.connect() as db:
            value=db.execute("SELECT value FROM ag_settings WHERE id=1").fetchone()[0]
        return Settings.model_validate_json(value)

    def save_settings(self, value):
        value=Settings.model_validate(value)
        with self.connect() as db:
            db.execute("UPDATE ag_settings SET value=?,updated_at=? WHERE id=1",(value.model_dump_json(),now()))
        self.audit('settings','workspace','updated',value.model_dump())
        return value.model_dump()

    def audit(self, entity, entity_id, action, details):
        with self.connect() as db:
            db.execute("INSERT INTO ag_audit(entity,entity_id,action,details,created_at) VALUES(?,?,?,?,?)",(entity,entity_id,action,json.dumps(details,allow_nan=False),now()))

    def history(self, entity, entity_id):
        with self.connect() as db:
            rows=db.execute("SELECT * FROM ag_audit WHERE entity=? AND entity_id=? ORDER BY id DESC LIMIT 100",(entity,entity_id)).fetchall()
        return [dict(r)|{'details':json.loads(r['details'])} for r in rows]

    def create_job(self, kind, configuration, key):
        payload=json.dumps({'kind':kind,'configuration':configuration},sort_keys=True)
        fingerprint=hashlib.sha256(payload.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous=db.execute("SELECT id,fingerprint FROM ag_jobs WHERE idempotency_key=?",(key,)).fetchone()
            if previous:
                if previous['fingerprint']!=fingerprint:raise Conflict('Idempotency key was already used for different inputs')
                return previous['id'],False
            if db.execute("SELECT id FROM ag_jobs WHERE status IN ('queued','running')").fetchone():
                raise Conflict('A pipeline job is already running. Wait for it to finish before starting another.')
            job_id='JOB-'+uuid.uuid4().hex[:12]
            db.execute("INSERT INTO ag_jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)",(job_id,kind,'queued',json.dumps(configuration),key,fingerprint,now(),None,None,None,None))
        return job_id,True

    def job_by_key(self,key):
        with self.connect() as db:row=db.execute('SELECT * FROM ag_jobs WHERE idempotency_key=?',(key,)).fetchone()
        return self._job(row) if row else None

    def update_job(self, job_id, **values):
        allowed={'status','started_at','finished_at','error','result'}
        if not values or not set(values)<=allowed:raise ValueError('Unsupported job fields')
        fields=','.join(f'{k}=?' for k in values)
        with self.connect() as db:db.execute(f"UPDATE ag_jobs SET {fields} WHERE id=?",(*[json.dumps(v) if k=='result' else v for k,v in values.items()],job_id))

    def jobs(self):
        with self.connect() as db:rows=db.execute("SELECT * FROM ag_jobs ORDER BY rowid DESC LIMIT 40").fetchall()
        return [self._job(row) for row in rows]

    def job(self, job_id):
        with self.connect() as db:
            row=db.execute("SELECT * FROM ag_jobs WHERE id=?",(job_id,)).fetchone()
            stages=db.execute("SELECT * FROM ag_stages WHERE job_id=? ORDER BY id",(job_id,)).fetchall()
        if row is None:raise ValueError('Pipeline run not found')
        return self._job(row)|{'stages':[dict(s) for s in stages]}

    @staticmethod
    def _job(row):
        value=dict(row)
        value['configuration']=json.loads(value['configuration'])
        value['result']=json.loads(value['result']) if value['result'] else None
        return value

    def stage_start(self, job_id, name, subset):
        with self.connect() as db:
            cursor=db.execute("INSERT INTO ag_stages(job_id,name,subset,status,started_at) VALUES(?,?,?,'running',?)",(job_id,name,subset,now()))
            return cursor.lastrowid

    def stage_finish(self, stage_id, duration, records=0, detail='', error=None):
        with self.connect() as db:db.execute("UPDATE ag_stages SET status=?,finished_at=?,duration_seconds=?,records=?,detail=?,error=? WHERE id=?",('failed' if error else 'complete',now(),round(duration,4),records,detail,str(error) if error else None,stage_id))

    def decisions(self, namespaces):
        if not namespaces:return {}
        placeholders=','.join('?' for _ in namespaces)
        with self.connect() as db:
            rows=db.execute(f"SELECT e.namespace,e.asset,e.cycle,e.decision,e.packet,e.created_at FROM events e JOIN (SELECT namespace,asset,MAX(cycle) AS cycle FROM events WHERE namespace IN ({placeholders}) GROUP BY namespace,asset) latest ON e.namespace=latest.namespace AND e.asset=latest.asset AND e.cycle=latest.cycle",tuple(namespaces)).fetchall()
        return {(r['namespace'],r['asset']):json.loads(r['decision'])|{'packet':json.loads(r['packet']),'timestamp':r['created_at']} for r in rows}

    def orders(self, mode):
        result=self.store.orders()
        result=[r for r in result if (r['namespace']=='demo' if mode=='demo' else r['namespace'].startswith('benchmark:'))]
        with self.connect() as db:metadata={r['order_id']:dict(r) for r in db.execute('SELECT * FROM ag_order_metadata')}
        with self.connect() as db:
            creation={r['id']:json.loads(r['packet']).get('rul') for r in db.execute("SELECT o.id,e.packet FROM orders o JOIN events e ON e.namespace=o.namespace AND e.asset=o.asset AND e.cycle=o.created_cycle AND json_extract(e.packet,'$.lifecycle')=o.lifecycle")}
        return [r|metadata.get(r['id'],{'owner':None,'notes':''})|{'subset':r['asset'].split('/')[0],
            'predicted_rul':creation.get(r['id']),'current_rul':r['evidence'].get('rul'),'simulated':True} for r in result]

    def update_order(self, order_id, mode, request):
        order=next((o for o in self.orders(mode) if o['id']==order_id),None)
        if not order:raise ValueError('Order not found in this data mode')
        if order['status']=='resolved':raise Conflict('Resolved orders are immutable')
        if order['status']=='in_progress' and request.action=='acknowledge':raise Conflict('An in-progress order cannot move backwards to acknowledged')
        if request.action in {'acknowledge','start','resolve'}:
            status={'acknowledge':'acknowledged','start':'in_progress','resolve':'resolved'}[request.action]
            self.store.update_order(order_id,status,request.resolution,order['namespace'])
        owner=request.owner if request.action=='assign' else order.get('owner')
        notes=request.notes.strip() or order.get('notes','')
        with self.connect() as db:db.execute("INSERT OR REPLACE INTO ag_order_metadata VALUES(?,?,?)",(order_id,owner,notes))
        self.audit('order',order_id,request.action,request.model_dump())
        return next(o for o in self.orders(mode) if o['id']==order_id)
