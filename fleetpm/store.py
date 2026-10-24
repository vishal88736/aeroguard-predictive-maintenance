"""Durable, deterministic triage and deduplicated local work orders."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import uuid

PRIORITY = {"MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
POLICY_VERSION = "triage-v2"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path, policy=None):
        self.policy=dict(policy or {})
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS events (
              namespace TEXT, event_id TEXT, asset TEXT, cycle INTEGER,
              digest TEXT, packet TEXT, decision TEXT, created_at TEXT,
              PRIMARY KEY(namespace,event_id));
            CREATE TABLE IF NOT EXISTS states (
              namespace TEXT,asset TEXT,lifecycle TEXT,state TEXT,
              PRIMARY KEY(namespace,asset,lifecycle));
            CREATE TABLE IF NOT EXISTS orders (
              id TEXT PRIMARY KEY,namespace TEXT,asset TEXT,lifecycle TEXT,kind TEXT,
              priority TEXT,status TEXT,revision INTEGER,created_cycle INTEGER,
              target_cycle INTEGER,evidence TEXT,reason TEXT,bundle_id TEXT,
              created_at TEXT,updated_at TEXT,resolution TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS one_open_order ON orders(namespace,asset,lifecycle,kind)
              WHERE status != 'resolved';
            CREATE TABLE IF NOT EXISTS outbox (
              id INTEGER PRIMARY KEY AUTOINCREMENT,order_id TEXT,revision INTEGER,
              idempotency_key TEXT UNIQUE,payload TEXT,status TEXT DEFAULT 'pending',created_at TEXT);
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,status TEXT,pid INTEGER,configuration TEXT,
              log_path TEXT,created_at TEXT,updated_at TEXT,error TEXT);
            ''')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=20000")
        return db

    def process(self, packet, namespace="real"):
        required = {"asset", "cycle", "bundle_id"}
        if not required <= set(packet):
            raise ValueError("Prediction packet is incomplete")
        if not isinstance(packet["cycle"], int) or packet["cycle"] < 1:
            raise ValueError("Cycle must be a positive integer")
        packet = dict(packet)
        packet.setdefault("lifecycle", "1")
        event_id = packet.get("event_id", f'{packet["asset"]}/{packet["lifecycle"]}/{packet["cycle"]}')
        serialized = json.dumps(packet, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT * FROM events WHERE namespace=? AND event_id=?", (namespace,event_id)).fetchone()
            if prior:
                if prior["digest"] != digest:
                    raise ValueError("Conflicting duplicate prediction event; archived evidence is immutable")
                return json.loads(prior["decision"]) | {"replayed": True}
            key = (namespace, packet["asset"], packet["lifecycle"])
            row = db.execute("SELECT state FROM states WHERE namespace=? AND asset=? AND lifecycle=?",key).fetchone()
            state = json.loads(row[0]) if row else {"last_cycle":0,"critical":0,"high":0,"medium":0,"bad":0}
            if packet["cycle"] <= state["last_cycle"]:
                raise ValueError("Late event cannot rewrite an already committed decision")
            if packet["cycle"] != state["last_cycle"]+1:
                state.update(critical=0,high=0,medium=0,bad=0)
                state.update(healthy_count=0,sensor_good=0)
            sensor_good=packet.get("data_quality_valid",packet.get("quality_valid",False))
            state["sensor_good"]=state.get("sensor_good",0)+1 if sensor_good else 0
            required_history=packet.get("required_history",30)
            healthy=(packet.get("quality_valid",False) and packet.get("history_length",30)>=required_history
                     and packet.get("lower") is not None and packet["lower"]>self.policy.get("watch_lower",50)+5
                     and (packet.get("anomaly") is None or packet["anomaly"]<self.policy.get("anomaly_attention",.99)))
            state["healthy_count"]=state.get("healthy_count",0)+1 if healthy else 0
            rearm=self.policy.get("rearm_cycles",5)
            if state["sensor_good"]>=rearm: state["sensor_armed"]=True
            if state["healthy_count"]>=rearm: state["equipment_armed"]=True
            decision = self._decide(packet,state)
            if decision.get("kind") and not state.get(decision["kind"].lower()+"_armed",True):
                decision["kind"]=None
                decision["state"]="DATA_EXCEPTION"
                decision["reason"]=f"Risk persists after resolution; review outcome or wait for {rearm} healthy cycles before a new episode"
            if decision["state"]=="WATCH": state["watch_latched"]=True
            elif decision["state"]=="NORMAL" and state.get("watch_latched"):
                if state["healthy_count"]<rearm: decision["state"]="WATCH"
                else: state["watch_latched"]=False
            if decision.get("kind"):
                order = self._order(db,packet,namespace,decision)
                decision["order_id"] = order["id"]
                decision["priority"] = order["priority"]
            active = db.execute("SELECT priority FROM orders WHERE namespace=? AND asset=? AND lifecycle=? AND kind='EQUIPMENT' AND status!='resolved'",key).fetchone()
            if active:
                decision["latched_priority"] = active[0]
            effective=decision.get("latched_priority",decision["state"])
            decision["severity"]="critical" if effective=="CRITICAL" else "healthy" if effective=="NORMAL" else "attention"
            decision["inputs"]={k:packet.get(k) for k in ("asset","cycle","rul","lower","upper","anomaly","trend","quality_valid","bundle_id")}
            eligible=(packet.get("quality_valid",False) and packet.get("history_length",30)>=required_history
                      and all(packet.get(k) is not None for k in ("rul","lower","upper")))
            decision["rules_evaluated"]=[
                {"rule":"quality_gate","evaluated":True,"matched":not packet.get("quality_valid",False)},
                {"rule":"warmup","evaluated":bool(packet.get("quality_valid",False)),"matched":bool(packet.get("quality_valid",False) and packet.get("history_length",30)<required_history)},
                {"rule":"critical_point","evaluated":eligible,"matched":bool(eligible and packet["rul"]<=self.policy.get("critical_rul",5))},
                {"rule":"critical_upper","evaluated":eligible,"matched":bool(eligible and packet["upper"]<=self.policy.get("critical_upper",10))},
                {"rule":"critical_persistence","evaluated":eligible,"matched":bool(eligible and state["critical"]>=self.policy.get("critical_persistence",2))},
                {"rule":"low_life_persistence","evaluated":eligible,"matched":bool(eligible and state["high"]>=self.policy.get("persistence",3))},
                {"rule":"anomaly_or_trend_persistence","evaluated":eligible,"matched":bool(eligible and state["medium"]>=self.policy.get("persistence",3))},
                {"rule":"active_order_latch","evaluated":True,"matched":bool(active)},
            ]
            decision["triggered_rules"]=[r["rule"] for r in decision["rules_evaluated"] if r["matched"]]
            decision["recommended_action"]={"critical":"Urgent engineering review; simulate removal from service", "healthy":"Continue historical monitoring", "attention":"Inspect equipment and sensor evidence; plan maintenance"}[decision["severity"]]
            if decision["state"]=="DATA_EXCEPTION":decision["recommended_action"]="Review data quality or uncertainty before equipment action"
            decision["timestamp"]=now()
            decision["policy_snapshot"]=self.policy
            state["last_cycle"] = packet["cycle"]
            db.execute("INSERT OR REPLACE INTO states VALUES (?,?,?,?)",key+(json.dumps(state),))
            decision["policy_version"] = POLICY_VERSION
            decision["replayed"] = False
            db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",(namespace,event_id,packet["asset"],packet["cycle"],digest,serialized,json.dumps(decision),now()))
            return decision

    def _decide(self,p,s):
        if not p.get("quality_valid",False):
            sensor_fault=p.get("exception_type","DATA")=="DATA"
            s.update(critical=0,high=0,medium=0,bad=s["bad"]+1 if sensor_fault else 0)
            urgent = p.get("complete_loss",False) or p.get("urgent_exception",False)
            return {"state":"DATA_EXCEPTION","kind":"SENSOR" if sensor_fault and (s["bad"]>=2 or urgent) else None,
                    "priority":"HIGH" if urgent else "MEDIUM","reason":p.get("quality_reason","Unsupported data or uncertainty")}
        s["bad"] = 0
        if p.get("history_length",30)<p.get("required_history",30):
            s.update(critical=0,high=0,medium=0)
            return {"state":"WARMING_UP","reason":f"Waiting for a {p.get('required_history',30)}-cycle context"}
        r,l,u = p.get("rul"),p.get("lower"),p.get("upper")
        if any(x is None or not math.isfinite(float(x)) for x in (r,l,u)) or not 0 <= l <= r <= u:
            s.update(critical=0,high=0,medium=0)
            return {"state":"DATA_EXCEPTION","reason":"Prediction or interval is unavailable"}
        anomaly = p.get("anomaly")
        anomaly = float(anomaly) if anomaly is not None else -1
        s["critical"] = s["critical"]+1 if l<=self.policy.get("critical_lower",10) else 0
        s["high"] = s["high"]+1 if l<=self.policy.get("attention_lower",30) else 0
        condition = ((l<=self.policy.get("watch_lower",50) and anomaly>=self.policy.get("anomaly_attention",.99))
                     or anomaly>=self.policy.get("anomaly_extreme",.995)
                     or (abs(p.get("trend",0))>=self.policy.get("trend_threshold",.04) and anomaly>=self.policy.get("anomaly_attention",.99)))
        s["medium"] = s["medium"]+1 if condition else 0
        if r<=self.policy.get("critical_rul",5) or u<=self.policy.get("critical_upper",10) or s["critical"]>=self.policy.get("critical_persistence",2):
            severity,reason="CRITICAL","Remaining-life critical threshold"
        elif s["high"]>=self.policy.get("persistence",3):
            severity,reason="HIGH",f"Low remaining life persisted for {self.policy.get('persistence',3)} cycles"
        elif s["medium"]>=self.policy.get("persistence",3):
            severity,reason="MEDIUM","Persistent unusual behavior requires inspection"
        else:
            return {"state":"WATCH" if l<=self.policy.get("watch_lower",50) or anomaly>=self.policy.get("anomaly_attention",.99) else "NORMAL","reason":"Continue monitoring"}
        return {"state":severity,"priority":severity,"kind":"EQUIPMENT","reason":reason}

    def _order(self,db,p,namespace,d):
        key=(namespace,p["asset"],p["lifecycle"],d["kind"])
        row=db.execute("SELECT * FROM orders WHERE namespace=? AND asset=? AND lifecycle=? AND kind=? AND status!='resolved'",key).fetchone()
        stamp=now(); priority=d["priority"]
        due=p["cycle"] if priority=="CRITICAL" else p["cycle"]+max(0,int((p.get("lower") or 10)-5))
        if row:
            priority=max((row["priority"],priority),key=PRIORITY.get)
            revision=row["revision"]+1
            due=min(due,row["target_cycle"])
            order_id=row["id"]
            db.execute("UPDATE orders SET priority=?,revision=?,target_cycle=?,evidence=?,reason=?,bundle_id=?,updated_at=? WHERE id=?",(priority,revision,due,json.dumps(p),d["reason"],p["bundle_id"],stamp,order_id))
        else:
            order_id="WO-"+uuid.uuid4().hex[:10].upper(); revision=1
            db.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(order_id,*key,priority,"open",revision,p["cycle"],due,json.dumps(p),d["reason"],p["bundle_id"],stamp,stamp,None))
        payload={"order_id":order_id,"priority":priority,"revision":revision,"target_cycle":due,"reason":d["reason"]}
        db.execute("INSERT INTO outbox(order_id,revision,idempotency_key,payload,created_at) VALUES (?,?,?,?,?)",(order_id,revision,f"{order_id}/{revision}",json.dumps(payload),stamp))
        return payload | {"id":order_id}

    def orders(self,namespace=None):
        with self.connect() as db:
            query="SELECT * FROM orders"; args=()
            if namespace is not None:
                query+=" WHERE namespace=?"; args=(namespace,)
            result=[]
            for row in db.execute(query+" ORDER BY created_at DESC",args):
                item=dict(row); item["evidence"]=json.loads(item["evidence"]); result.append(item)
            return result

    def update_order(self,order_id,status,resolution=None,namespace=None):
        if status not in {"acknowledged","in_progress","resolved"}:
            raise ValueError("Supported actions: acknowledge, start work or resolve")
        if status=="resolved" and resolution not in {"preventive_service","sensor_repair","confirmed_failure","no_fault_found","deferred"}:
            raise ValueError("Choose an explicit maintenance outcome")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT * FROM orders WHERE id=?",(order_id,)).fetchone()
            if not row or namespace is not None and row["namespace"]!=namespace:
                raise ValueError("Work order not found in this data mode")
            if row["status"]=="resolved":
                return dict(row)
            if row["status"]==status:
                return {"order_id":order_id,"revision":row["revision"],"status":status,"resolution":row["resolution"]}
            revision=row["revision"]+1
            db.execute("UPDATE orders SET status=?,revision=?,resolution=?,updated_at=? WHERE id=?",(status,revision,resolution,now(),order_id))
            if status=="resolved":
                key=(row["namespace"],row["asset"],row["lifecycle"])
                saved=db.execute("SELECT state FROM states WHERE namespace=? AND asset=? AND lifecycle=?",key).fetchone()
                if saved:
                    state=json.loads(saved[0]); state[row["kind"].lower()+"_armed"]=False
                    state.update(healthy_count=0,sensor_good=0,critical=0,high=0,medium=0,bad=0)
                    db.execute("UPDATE states SET state=? WHERE namespace=? AND asset=? AND lifecycle=?",(json.dumps(state),*key))
            payload={"order_id":order_id,"revision":revision,"status":status,"resolution":resolution}
            db.execute("INSERT INTO outbox(order_id,revision,idempotency_key,payload,created_at) VALUES (?,?,?,?,?)",(order_id,revision,f"{order_id}/{revision}",json.dumps(payload),now()))
            return payload

    def dispatch_local(self):
        """A local queue acknowledgement, not an external CMMS integration."""
        with self.connect() as db:
            count=db.execute("SELECT count(*) FROM outbox WHERE status='pending'").fetchone()[0]
            db.execute("UPDATE outbox SET status='delivered_local' WHERE status='pending'")
        return count
