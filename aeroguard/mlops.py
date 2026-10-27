"""Persistent MLOps lifecycle used by the app, local scheduler and Airflow adapter."""

import hashlib
import json
import logging
from pathlib import Path
import subprocess
import threading
import time
import uuid

from fleetpm.config import ROOT, file_hash, write_json
from fleetpm.features import load_engines
from fleetpm.store import Store, now
from .baseline import model_folder, train_model
from .contracts import LifecycleSettings, TrainRequest
from .mlops_data import DatasetVersions, digest_json, engine_digest, protocol_split
from .mlops_monitoring import drift_report, quality_report
from .repository import Conflict

LOGGER = logging.getLogger(__name__)


class Lifecycle:
    def __init__(self, service):
        self.service = service
        self.root = service.artifact_dir / "mlops"
        self.versions = DatasetVersions(service.artifact_dir)
        self.stop_event = threading.Event()
        self.thread = None
        self.last_scheduler_error = None
        with service.repo.connect() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS ag_mlops_settings(id INTEGER PRIMARY KEY CHECK(id=1), value TEXT);
              CREATE TABLE IF NOT EXISTS ag_mlops_versions(id TEXT PRIMARY KEY, status TEXT, attempts INTEGER DEFAULT 0,
                job_id TEXT, result TEXT, error TEXT, updated_at TEXT);
              CREATE TABLE IF NOT EXISTS ag_mlops_checks(id TEXT PRIMARY KEY, version_id TEXT, report TEXT, created_at TEXT);
              CREATE TABLE IF NOT EXISTS ag_mlops_reviews(id TEXT PRIMARY KEY, version_id TEXT, reason TEXT,
                status TEXT, notes TEXT, created_at TEXT, resolved_at TEXT);
              CREATE TABLE IF NOT EXISTS ag_mlops_served(namespace TEXT, asset TEXT, bundle_id TEXT,
                version_id TEXT, packet TEXT, PRIMARY KEY(namespace,asset));
            ''')
            db.execute("INSERT OR IGNORE INTO ag_mlops_settings VALUES(1,?)", (LifecycleSettings().model_dump_json(),))
            db.execute("UPDATE ag_mlops_versions SET status='failed',error='Worker interrupted; retained artifacts may be retried',updated_at=? WHERE status='processing' AND job_id IN (SELECT id FROM ag_jobs WHERE status='interrupted')", (now(),))

    @property
    def registry(self):
        if not hasattr(self, "_registry"):
            from .mlops_registry import ModelRegistry
            self._registry = ModelRegistry(self.service.artifact_dir)
        return self._registry

    def settings(self):
        with self.service.repo.connect() as db:
            value = db.execute("SELECT value FROM ag_mlops_settings WHERE id=1").fetchone()[0]
        return LifecycleSettings.model_validate_json(value)

    def save_settings(self, settings):
        settings = LifecycleSettings.model_validate(settings)
        with self.service.repo.connect() as db:
            db.execute("UPDATE ag_mlops_settings SET value=? WHERE id=1", (settings.model_dump_json(),))
        self.service.repo.audit("mlops", "scheduler", "settings_updated", settings.model_dump())
        return settings.model_dump()

    def add_version(self, subset, filename, content, label_status, provenance, namespace="incoming"):
        manifest = self.versions.add(subset, filename, content, label_status, provenance, namespace)
        with self.service.repo.connect() as db:
            db.execute("INSERT OR IGNORE INTO ag_mlops_versions(id,status,updated_at) VALUES(?,'new',?)", (manifest["id"], now()))
        self.service.repo.audit("dataset_version", manifest["id"], "reused" if manifest["reused"] else "created", {k: manifest[k] for k in ("subset", "sha256", "label_status", "rows", "source_description")})
        return manifest

    def lineage(self, run):
        if isinstance(run, str):
            run = json.loads((model_folder(run, self.service.artifact_dir) / "metadata.json").read_text())
        if run.get("lifecycle"):
            return run["lifecycle"]
        path = self.root / "lineage" / (run["run_id"] + ".json")
        return json.loads(path.read_text()) if path.exists() else None

    def _save_lineage(self, run_id, value):
        path = self.root / "lineage" / (run_id + ".json")
        if path.exists() and json.loads(path.read_text()) != value:
            raise Conflict("Existing immutable model lineage differs")
        write_json(path, value)

    def _code_identity(self):
        files = sorted([* (ROOT / "aeroguard").glob("*.py"), * (ROOT / "fleetpm").glob("*.py")])
        code = digest_json({p.relative_to(ROOT).as_posix(): file_hash(p) for p in files})
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            revision = "unavailable"
        return {"git_revision": revision, "working_source_sha256": code}

    def register_model(self, run_id):
        folder = model_folder(run_id, self.service.artifact_dir)
        metadata = json.loads((folder / "metadata.json").read_text())
        lineage = self.lineage(metadata)
        source = self.versions.path(lineage["data_version_id"]) if lineage else self.service.data_dir / f"train_{metadata['subset']}.txt"
        if not source.exists() or file_hash(source) != metadata["source_sha256"]:
            raise ValueError("The exact original training snapshot is required for model registration")
        protocol = self.versions.protocol(metadata["subset"])
        if lineage is None and protocol:
            reference = self.versions.get(protocol["reference_version_id"])
            if reference["sha256"] == metadata["source_sha256"]:
                try:
                    expected = protocol_split(load_engines(source), protocol, metadata["configuration"])
                    saved = json.loads((folder / "split.json").read_text())
                    if saved == expected:
                        lineage = {"data_version_id": reference["id"], "data_sha256": reference["sha256"],
                                   "protocol_id": protocol["id"], "validation_cohort_sha256": protocol["validation_cohort_sha256"],
                                   "training_code_sha256": metadata.get("code_sha256"),
                                   "lineage_origin": "verified protocol-compatible CPU run"}
                except ValueError:
                    pass  # Registered for inspection, but incompatible protocols cannot be promoted.
            if lineage is not None:
                self._save_lineage(run_id, lineage)
        with self.service.lock:
            return self.registry.register(run_id, source, lineage)

    def register_job(self, job_id, configuration):
        with self.service.stage(job_id, "Register and verify model", "ALL") as stage:
            result = self.register_model(configuration["run_id"])
            with self.service.lock:
                active = self.service.active().get(result["name"].split("-")[1])
                if active == configuration["run_id"]:
                    self.registry.champion(active)
            stage.records = 1
            stage.detail = "Full inference bundle loaded and numerically checked before MLflow registration"
        return result

    def track_research_job(self, job_id, configuration):
        with self.service.stage(job_id, "Track verified neural export", "ALL") as stage:
            result = self.registry.track_research(self.service.research, configuration["run_id"])
            stage.records = 1
            stage.detail = "Actual imported metrics, configuration, weights and SHAP/LIME artifacts tracked; no maintenance promotion implied"
        return result

    def bootstrap(self, job_id, configuration):
        output = []
        for subset in configuration["subsets"]:
            with self.service.stage(job_id, "Freeze dataset and evaluation protocol", subset) as stage:
                source = self.service.data_dir / f"train_{subset}.txt"
                active = self.service.active().get(subset)
                metadata = json.loads((model_folder(active, self.service.artifact_dir) / "metadata.json").read_text()) if active else None
                existing_lineage = self.lineage(metadata) if metadata else None
                if existing_lineage:
                    source = self.versions.path(existing_lineage["data_version_id"])
                if metadata and metadata["source_sha256"] != file_hash(source):
                    raise ValueError("Restore the active model's original training data before freezing its comparison protocol")
                version = self.versions.get(existing_lineage["data_version_id"]) if existing_lineage else self.add_version(
                    subset, source.name, source.read_bytes(), "complete_run_to_failure",
                    "Reference C-MAPSS run-to-failure training corpus; official test and targets excluded", "benchmark_reference")
                cfg = metadata["configuration"] if metadata else TrainRequest(subsets=[subset]).model_dump()
                split = json.loads((model_folder(active, self.service.artifact_dir) / "split.json").read_text()) if active else None
                protocol = self.versions.freeze_protocol(version["id"], cfg, split)
                if version["id"] == protocol["reference_version_id"]:
                    with self.service.repo.connect() as db:
                        db.execute("UPDATE ag_mlops_versions SET status='reference',updated_at=? WHERE id=?", (now(), version["id"]))
                stage.records = version["engines"]
                stage.detail = "Immutable training snapshot and unchanged development/calibration trajectories; no official test access"
            registered = None
            if active:
                lineage = existing_lineage or {"data_version_id": version["id"], "data_sha256": version["sha256"], "protocol_id": protocol["id"],
                           "validation_cohort_sha256": protocol["validation_cohort_sha256"], "training_code_sha256": metadata.get("code_sha256"),
                           "training_git_revision": "not recorded by the original run", "lineage_origin": "verified legacy model bootstrap"}
                self._save_lineage(active, lineage)
                with self.service.stage(job_id, "Register and verify champion bundle", subset) as stage:
                    with self.service.lock:
                        registered = self.registry.register(active, source, lineage)
                        if self.service.active().get(subset) == active:
                            self.registry.champion(active)
                    stage.records = 1
                    stage.detail = "Actual MLflow version, raw-history inference probe and champion alias verified; active model unchanged"
            output.append({"subset": subset, "version_id": version["id"], "protocol_id": protocol["id"], "registry": registered})
        return {"initialized": output}

    def overview(self):
        with self.service.repo.connect() as db:
            ledger = {r["id"]: dict(r) for r in db.execute("SELECT * FROM ag_mlops_versions")}
            checks = [json.loads(r[0]) for r in db.execute("SELECT report FROM ag_mlops_checks ORDER BY rowid DESC LIMIT 30")]
            reviews = [dict(r) for r in db.execute("SELECT * FROM ag_mlops_reviews ORDER BY rowid DESC LIMIT 50")]
        versions = []
        for version in self.versions.list():
            state = ledger.get(version["id"], {})
            versions.append(version | {"processing_status": state.get("status", "new"), "attempts": state.get("attempts", 0),
                                       "job_id": state.get("job_id"), "error": state.get("error"),
                                       "result": json.loads(state["result"]) if state.get("result") else None})
        models = self.registry.list() if (self.root / "registry").exists() else []
        research = self.service.research.summary()["runs"]
        tracked = {p.stem: json.loads(p.read_text()) for p in (self.root / "research-tracking").glob("*.json")} if (self.root / "research-tracking").exists() else {}
        return {"settings": self.settings().model_dump(), "versions": versions, "checks": checks, "reviews": reviews,
                "research_tracking": [r | {"tracking": tracked.get(r["run_id"])} for r in research],
                "protocols": [p for subset in ("FD001", "FD002", "FD003", "FD004") if (p := self.versions.protocol(subset))],
                "registry": models, "scheduler_running": bool(self.thread and self.thread.is_alive()),
                "orchestration_error": self.last_scheduler_error,
                "source": "Actual immutable datasets, engine-level drift checks and MLflow model versions",
                "neural_status": "Research exports retain their separate contract; no automatic promotion into calibrated CPU triage"}

    def _review(self, version_id, reason):
        key = "REV-" + digest_json({"version_id": version_id, "reason": reason})[:16]
        with self.service.repo.connect() as db:
            db.execute("INSERT OR IGNORE INTO ag_mlops_reviews VALUES(?,?,?,'open','',?,NULL)", (key, version_id, reason, now()))
        return key

    def resolve_review(self, review_id, notes):
        if not notes.strip() or len(notes) > 1500:
            raise ValueError("Provide review notes in 1-1500 characters")
        with self.service.repo.connect() as db:
            row = db.execute("SELECT * FROM ag_mlops_reviews WHERE id=?", (review_id,)).fetchone()
            if row is None:
                raise ValueError("Review item not found")
            if row["status"] != "open":
                raise Conflict("Review is already resolved")
            db.execute("UPDATE ag_mlops_reviews SET status='resolved',notes=?,resolved_at=? WHERE id=?", (notes.strip(), now(), review_id))
        self.service.repo.audit("mlops_review", review_id, "resolved", {"notes": notes.strip(), "training_authorized_by_resolution": False})
        return {"id": review_id, "status": "resolved", "note": "Resolution records review only; dataset eligibility and promotion gates remain enforced"}

    def _finish(self, version_id, result):
        with self.service.repo.connect() as db:
            db.execute("UPDATE ag_mlops_versions SET status=?,result=?,error=NULL,updated_at=? WHERE id=?",
                       (result["status"], json.dumps(result, allow_nan=False), now(), version_id))
        self.service.repo.audit("dataset_version", version_id, result["status"], result)
        return result

    def incoming_assets(self, active):
        if not active:
            return []
        ids = list(active.values())
        with self.service.repo.connect() as db:
            rows = db.execute("SELECT packet FROM ag_mlops_served WHERE bundle_id IN (" + ",".join("?" for _ in ids) + ")", ids).fetchall()
        return [json.loads(row[0]) for row in rows]

    def _serve_prefixes(self, job_id, manifest, source, champion_id, settings):
        """Replay new observed cycles only; prevent changes to committed histories."""
        subset, cohort = manifest["subset"], manifest["namespace"]
        namespace = f"benchmark:{champion_id}:incoming:{cohort}"
        engines = load_engines(source)
        with self.service.repo.connect() as db:
            previous = {row["asset"]: dict(row) for row in db.execute("SELECT * FROM ag_mlops_served WHERE namespace=?", (namespace,))}
        for (_, asset), archived in self.service.repo.decisions([namespace]).items():
            packet = archived["packet"]
            if asset not in previous and packet.get("source_version_id"):
                previous[asset] = {"version_id": packet["source_version_id"], "packet": json.dumps(packet)}
        previous_engines = {key: load_engines(self.versions.path(key)) for key in {r["version_id"] for r in previous.values()}}
        # Validate all histories before committing any decision from this batch.
        for unit, engine in engines.items():
            asset = f"{subset}/incoming/{cohort}/{unit:03d}"
            if asset in previous:
                old_version = previous[asset]["version_id"]
                old = previous_engines[old_version][unit]
                length = min(len(old["raw"]), len(engine["raw"]))
                a = dict(old, raw=old["raw"][:length])
                b = dict(engine, raw=engine["raw"][:length])
                if engine_digest(a) != engine_digest(b):
                    raise Conflict("Incoming batch rewrites previously committed sensor history; use a new cohort or resolve the data conflict")
        store = Store(self.service.repo.store.path, policy=settings["policy"])
        bundle = self.service.bundle(champion_id)
        before = len(store.orders(namespace))
        processed = 0
        with self.service.stage(job_id, "Serve observed prefixes and triage", subset) as stage:
            for unit, engine in engines.items():
                asset = f"{subset}/incoming/{cohort}/{unit:03d}"
                with self.service.repo.connect() as db:
                    row = db.execute("SELECT state FROM states WHERE namespace=? AND asset=? AND lifecycle='1'", (namespace, asset)).fetchone()
                committed = json.loads(row[0])["last_cycle"] if row else 0
                ends = [i for i in range(max(0, len(engine["raw"]) - settings["replay_cycles"]), len(engine["raw"])) if i + 1 > committed]
                if not ends:
                    if asset in previous:
                        old = previous[asset]
                        with self.service.repo.connect() as db:
                            db.execute("INSERT OR IGNORE INTO ag_mlops_served VALUES(?,?,?,?,?)", (namespace, asset, champion_id, old["version_id"], old["packet"]))
                    continue
                packets = bundle.engine_predictions(engine, ends, settings["uncertainty_gate"])
                for packet in packets:
                    packet.update(asset=asset, namespace=namespace, partition="incoming", source_version_id=manifest["id"])
                    decision = store.process(packet, namespace)
                    if decision.get("order_id") and not decision["replayed"]:
                        self.service.repo.audit("order", decision["order_id"], "incoming_batch_rule", {"version_id": manifest["id"], "decision": decision})
                    processed += 1
                with self.service.repo.connect() as db:
                    db.execute("INSERT OR REPLACE INTO ag_mlops_served VALUES(?,?,?,?,?)", (namespace, asset, champion_id, manifest["id"], json.dumps(packets[-1], allow_nan=False)))
            stage.records = processed
            stage.detail = "Actual champion RUL/anomaly inference and persistent triage; targets are not inputs; replayed cycles skipped"
        with self.service.stage(job_id, "Persist simulated incoming work orders", subset) as stage:
            created = len(store.orders(namespace)) - before
            delivered = store.dispatch_local()
            stage.records = created
            stage.detail = f"{created} new simulated orders; {delivered} local outbox revisions acknowledged; no external dispatch"
        return {"processed_cycles": processed, "engines": len(engines), "new_orders": created,
                "source": "Uploaded compatible observed sensor history; no accuracy labels supplied"}

    def run(self, job_id, configuration):
        version_id = configuration["version_id"]
        with self.service.repo.connect() as db:
            db.execute("UPDATE ag_mlops_versions SET status='processing',attempts=attempts+1,job_id=?,updated_at=? WHERE id=?", (job_id, now(), version_id))
        try:
            return self._run(job_id, configuration)
        except Exception as error:
            with self.service.repo.connect() as db:
                db.execute("UPDATE ag_mlops_versions SET status='failed',error=?,updated_at=? WHERE id=?", (str(error), now(), version_id))
            raise

    def _run(self, job_id, configuration):
        version_id = configuration["version_id"]
        policy = LifecycleSettings.model_validate(configuration["_mlops_settings"])
        with self.service.stage(job_id, "Verify immutable dataset", "ALL") as stage:
            manifest = self.versions.get(version_id)
            source = self.versions.path(version_id)
            subset = manifest["subset"]
            protocol = self.versions.protocol(subset)
            if not protocol:
                raise ValueError("Initialize MLOps for this subset to freeze a comparison protocol first")
            stage.records = manifest["rows"]
            stage.detail = "Dataset digest verified; original corpus and official test files remain unchanged"
        with self.service.stage(job_id, "Data quality and drift screening", subset) as stage:
            quality = quality_report(manifest, policy)
            champion_id = self.service.active().get(subset)
            drift = {"status": "reference_required" if quality["passed"] else "not_checked_quality_failed", "comparisons": [], "drifted_sensors": [], "out_of_domain_fraction": 0}
            if champion_id and quality["passed"]:
                drift = drift_report(self.versions.path(protocol["reference_version_id"]), source,
                                     self.service.bundle(champion_id), policy, manifest["label_status"])
            report = {"id": "MON-" + uuid.uuid4().hex[:16], "version_id": version_id, "subset": subset,
                      "reference_model": champion_id, "created_at": now(), "quality": quality, "drift": drift, "policy_snapshot": policy.model_dump()}
            with self.service.repo.connect() as db:
                db.execute("INSERT INTO ag_mlops_checks VALUES(?,?,?,?)", (report["id"], version_id, json.dumps(report, allow_nan=False), now()))
            write_json(self.root / "checks" / (report["id"] + ".json"), report)
            stage.records = manifest["engines"]
            stage.detail = f"Quality {'passed' if quality['passed'] else 'failed'}; drift {drift['status']}; independent engine/regime summaries and Holm correction"
        base_result = {"version_id": version_id, "check_id": report["id"], "subset": subset}
        if not quality["passed"] or drift["out_of_domain_fraction"] > .05:
            reason = "; ".join(quality["reasons"]) or "More than 5% of observed prefix settings are outside the champion's supported domain"
            return self._finish(version_id, base_result | {"status": "review_required", "reason": reason, "review_id": self._review(version_id, reason)})
        if not quality["labels_available"] and champion_id:
            try:
                base_result["inference"] = self._serve_prefixes(job_id, manifest, source, champion_id, configuration["_settings"])
            except Conflict as error:
                reason = str(error)
                return self._finish(version_id, base_result | {"status": "review_required", "reason": reason, "review_id": self._review(version_id, reason)})
        with self.service.stage(job_id, "Check labels and training eligibility", subset) as stage:
            stage.records = manifest["engines"]
            if not quality["labels_available"]:
                stage.detail = "Observed prefixes supply no supervised RUL labels; champion retained"
                return self._finish(version_id, base_result | {"status": "waiting_labels", "reason": stage.detail})
            if configuration["monitor_only"]:
                stage.detail = "Monitoring-only invocation; training was not requested"
                return self._finish(version_id, base_result | {"status": "monitored", "reason": stage.detail})
            current = self.lineage(champion_id) if champion_id else None
            if current and current["data_sha256"] == manifest["sha256"] and not configuration["force_retrain"]:
                stage.detail = "Champion already uses this dataset content; no duplicate retraining"
                return self._finish(version_id, base_result | {"status": "unchanged", "reason": stage.detail})
            original = self.service.bundle(champion_id).configuration if champion_id else TrainRequest(subsets=[subset]).model_dump()
            supplied = configuration.get("training") or {k: v for k, v in original.items() if k in TrainRequest.model_fields}
            supplied = supplied | {"subsets": [subset], "track_mlflow": True}
            cfg = TrainRequest.model_validate(supplied).model_dump()
            contract = cfg | {"minimum_history": min(15, cfg["window"])}
            protocol_split(load_engines(source), protocol, contract)
            stage.detail = "Complete trajectories declared; validation/calibration hashes unchanged; new engines reserved for fitting"
        lineage = {"data_version_id": version_id, "data_sha256": manifest["sha256"], "protocol_id": protocol["id"],
                   "validation_cohort_sha256": protocol["validation_cohort_sha256"], "monitoring_check_id": report["id"],
                   **self._code_identity()}
        with self.service.stage(job_id, "Train challenger and calibrate", subset) as stage:
            with self.service.repo.connect() as db:
                row = db.execute("SELECT result FROM ag_mlops_versions WHERE id=?", (version_id,)).fetchone()
            resume = json.loads(row[0]).get("resume_candidate") if row and row[0] else None
            candidate = None
            if resume and resume.get("configuration_sha256") == digest_json(cfg) and resume.get("working_source_sha256") == lineage["working_source_sha256"]:
                try:
                    saved = self.service.bundle(resume["run_id"]).metadata
                    if saved.get("training_status") == "complete" and saved.get("tracking_status") == "complete" and saved.get("lifecycle", {}).get("data_version_id") == version_id:
                        candidate = saved
                        lineage = candidate["lifecycle"]
                except (OSError, ValueError):
                    candidate = None
            if candidate is None:
                candidate = train_model(cfg, source.parent, self.service.artifact_dir, protocol=protocol, lineage=lineage)
            if candidate.get("tracking_status") != "complete":
                raise RuntimeError("Challenger saved, but MLflow tracking failed: " + candidate.get("tracking_error", "unknown tracking failure"))
            stage.records = candidate["training_samples"]
            stage.detail = "Verified completed candidate reused after interrupted downstream work" if resume and resume.get("run_id") == candidate["run_id"] else "Actual CPU fitting; frozen development cohort and separate calibration engines; official test targets not opened"
            checkpoint = {"resume_candidate": {"run_id": candidate["run_id"], "configuration_sha256": digest_json(cfg), "working_source_sha256": lineage["working_source_sha256"]}}
            with self.service.repo.connect() as db:
                db.execute("UPDATE ag_mlops_versions SET result=? WHERE id=?", (json.dumps(checkpoint), version_id))
        run_id = candidate["run_id"]
        self._save_lineage(run_id, lineage)
        with self.service.stage(job_id, "Verify and register inference bundle", subset) as stage:
            with self.service.lock:
                registered = self.registry.register(run_id, source, lineage)
            client = self.registry.client()
            client.log_dict(registered["mlflow_run_id"], report, f"monitoring/{report['id']}.json")
            stage.records = 1
            stage.detail = f"MLflow {registered['name']} version {registered['version']}; packaged raw-history inference verified"
        # Prepare sensor-only serving results. No official endpoint targets are
        # read or used for comparison, and no test accuracy is fabricated.
        test_path = self.service.data_dir / f"test_{subset}.txt"
        if test_path.exists():
            with self.service.stage(job_id, "Prepare sensor-only fleet inference", subset) as stage:
                bundle = self.service.bundle(run_id)
                predictions = [bundle.engine_predictions(engine, [len(engine["raw"]) - 1])[0] for engine in load_engines(test_path).values()]
                write_json(model_folder(run_id, self.service.artifact_dir) / "predictions.json", predictions)
                write_json(model_folder(run_id, self.service.artifact_dir) / "serving_inputs.json", {"file": test_path.name, "sha256": file_hash(test_path), "targets_read": False})
                stage.records = len(predictions)
                stage.detail = "Frozen candidate predicts observed engine endpoints; no test RUL targets opened and no accuracy score claimed"
        with self.service.stage(job_id, "Compare and promote or reject", subset) as stage:
            eligible, reason = self.service.promotion_gate(candidate, lifecycle_settings=policy)
            if champion_id != self.service.active().get(subset):
                eligible, reason = False, "Champion changed during training; repeat comparison before promotion"
            radius = candidate["calibration"].get("radius")
            if radius is None:
                eligible, reason = False, "Independent calibration cohort cannot supply a finite interval"
            status = "candidate_ready" if eligible else "rejected"
            if eligible and cfg["promote"]:
                self.service.promote(run_id,lifecycle_settings=policy)
                status = "promoted"
            client.set_model_version_tag(registered["name"], registered["version"], "promotion_decision", status)
            client.set_model_version_tag(registered["name"], registered["version"], "promotion_reason", reason)
            stage.records = 1
            stage.detail = reason + "; previous versions and immutable evidence retained"
        return self._finish(version_id, base_result | {"status": status, "run_id": run_id, "registry_version": registered["version"],
                                                       "reason": reason, "validation_metrics": candidate["validation_metrics"],
                                                       "official_test_used_for_promotion": False})

    def start_scheduler(self):
        if self.thread and self.thread.is_alive():
            return
        try:
            self.reconcile_deployment()
        except Exception as error:
            self.last_scheduler_error = str(error)
            LOGGER.exception("Pending model deployment needs reconciliation")
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._schedule, name="aeroguard-monitor", daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _schedule(self):
        last = 0.0
        while not self.stop_event.wait(1):
            try:
                settings = self.settings()
                if not settings.enabled or settings.orchestrator != "local":
                    continue
                if time.monotonic() - last < settings.interval_seconds:
                    continue
                last = time.monotonic()
                self.tick()
                self.last_scheduler_error = None
            except Exception as error:
                self.last_scheduler_error = str(error)
                LOGGER.exception("Scheduled MLOps tick failed; no success fabricated")

    def reconcile_deployment(self):
        journal = self.service.runtime_dir / "promotion.pending.json"
        if not journal.exists():
            return
        pending = json.loads(journal.read_text())
        deployed = self.service.active().get(pending["subset"])
        if deployed:
            self.registry.champion(deployed)
        else:
            self.registry.restore_aliases(pending["target_run_id"], pending.get("previous_aliases") or {})
        self.service.repo.audit("model", deployed or pending["target_run_id"], "deployment_reconciled", {"subset": pending["subset"], "active_run_id": deployed})
        journal.unlink()

    def tick(self):
        settings = self.settings()
        if not settings.enabled:
            return {"status": "disabled", "reason": "Enable the scheduler in MLOps controls"}
        if (self.service.runtime_dir / "promotion.pending.json").exists():
            return {"status": "review_required", "reason": "Reconcile the pending deployment before scheduled processing"}
        with self.service.repo.connect() as db:
            if db.execute("SELECT 1 FROM ag_jobs WHERE status IN ('queued','running')").fetchone():
                return {"status": "busy", "reason": "Another durable job is running"}
            rows = db.execute("SELECT * FROM ag_mlops_versions WHERE status IN ('new','failed') AND attempts<? ORDER BY updated_at,id", (settings.max_attempts,)).fetchall()
        for row in rows:
            manifest = self.versions.get(row["id"])
            if not self.versions.protocol(manifest["subset"]):
                continue
            key = "mlops-" + row["id"] + "-attempt-" + str(row["attempts"] + 1)
            try:
                job = self.service.start("lifecycle", {"version_id": row["id"], "monitor_only": not settings.auto_retrain}, key)
                return {**job, "status": "submitted", "job_status": job["status"]}
            except Conflict:
                return {"status": "busy", "reason": "Another invocation acquired the job lease"}
        return {"status": "idle", "reason": "No eligible unprocessed dataset versions"}
