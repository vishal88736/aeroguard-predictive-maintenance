"""Run browser contracts against two disposable API workspaces, without real data/models."""

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def available_port():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def wait_ready(process, port):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Temporary API exited; inspect its browser-test server log")
        try:
            with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(.2)
    raise RuntimeError("Temporary browser-test API did not become ready")


def main():
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    processes, handles, owned_workspaces = [], [], []
    with tempfile.TemporaryDirectory(prefix="browser-contract-", dir=runtime) as temporary:
        try:
            live_port, fixture_port = available_port(), available_port()
            while fixture_port == live_port:
                fixture_port = available_port()
            for name, port, extra in (("portal", live_port, ["--empty"]), ("fixture", fixture_port, [])):
                log_path = Path(temporary) / f"{name}.log"
                handle = log_path.open("w")
                handles.append(handle)
                process = subprocess.Popen([sys.executable, "scripts/research_qa_server.py", "--port", str(port), *extra],
                                           cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                processes.append(process)
                wait_ready(process, port)
                for line in log_path.read_text().splitlines():
                    if not line.startswith('{"fixture_zip"'):
                        continue
                    state = json.loads(line)
                    owned_workspaces.append(Path(state["workspace"]))
                    if name == "fixture":
                        fixture = state["fixture_zip"]
            environment = os.environ.copy()
            environment.update(FLEET_QA_URL=f"http://127.0.0.1:{live_port}",
                               RESEARCH_QA_URL=f"http://127.0.0.1:{fixture_port}", RESEARCH_QA_FIXTURE=fixture)
            return subprocess.run(["node", "scripts/browser_research.cjs"], cwd=ROOT, env=environment).returncode
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            for handle in handles:
                handle.close()
            for folder in owned_workspaces:
                if folder.is_relative_to(runtime) and folder.name.startswith("research-ui-fixture-") and folder.exists():
                    shutil.rmtree(folder)


if __name__ == "__main__":
    raise SystemExit(main())
