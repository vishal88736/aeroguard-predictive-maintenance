"""HTTP orchestration and neural evidence tracking, not GPU/Airflow execution."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from fastapi.testclient import TestClient

from aeroguard.api import create_app
from aeroguard.contracts import LifecycleSettings
from aeroguard.orchestration import process_pending_version, request_json
from test_ag_baseline import write_fixture
from test_mlops import wait
from test_research_exports import archive, export_fixture


class OperationalAdapterTests(unittest.TestCase):
    def test_http_adapter_drives_actual_backend_job_and_label_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);write_fixture(root / "data", engines=50)
            app = create_app(root / "data", root / "artifacts", root / "runtime")
            service = app.state.service
            client = TestClient(app)

            class Handler(BaseHTTPRequestHandler):
                def forward(self):
                    response = client.request(self.command, self.path, headers=dict(self.headers))
                    self.send_response(response.status_code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers();self.wfile.write(response.content)
                do_POST = forward
                do_GET = forward
                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                self.assertEqual(process_pending_version(base)["status"], "disabled")
                initialized = wait(service, service.start("mlops_bootstrap", {"subsets": ["FD001"]}, "init"))
                self.assertEqual(initialized["status"], "complete")
                version = service.mlops.add_version("FD001", "batch.txt", (root / "data/test_FD001.txt").read_bytes(), "observed_prefix", "Synthetic HTTP adapter fixture")
                service.mlops.save_settings(LifecycleSettings(enabled=True).model_dump())
                result = process_pending_version(base, timeout_seconds=10, poll_seconds=.02)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["result"]["version_id"], version["id"])
                self.assertEqual(result["result"]["status"], "waiting_labels")
                self.assertEqual(process_pending_version(base)["status"], "idle")
                with self.assertRaises(ValueError):
                    request_json("file:///private", "/api/mlops")
            finally:
                server.shutdown();server.server_close();thread.join(timeout=3);service.close()

    def test_verified_neural_fixture_reports_track_without_execution_or_promotion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, export, run_id, _ = export_fixture(root)
            app = create_app(data, root / "artifacts", root / "runtime")
            service = app.state.service
            try:
                service.research.import_archive(archive(export))
                record = service.mlops.registry.track_research(service.research, run_id)
                self.assertEqual(record["status"], "tracked")
                self.assertIn("blocked", record["maintenance_promotion"])
                again = service.mlops.registry.track_research(service.research, run_id)
                self.assertTrue(again["reused"])
                self.assertEqual(record["mlflow_run_id"], again["mlflow_run_id"])
                client = service.mlops.registry.client()
                self.assertEqual(client.get_run(record["mlflow_run_id"]).info.status, "FINISHED")
                self.assertEqual(client.search_registered_models(), [])
                self.assertEqual(service.active(), {})
            finally:
                service.close()


if __name__ == "__main__":
    unittest.main()
