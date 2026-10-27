"""Start a disposable CPU workspace, run browser workflows, clean owned state."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import time

from run_browser_contract import available_port, wait_ready

ROOT = Path(__file__).resolve().parents[1]


def main():
    (ROOT / "runtime").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlops-browser-", dir=ROOT / "runtime") as temporary:
        log = Path(temporary) / "server.log"
        with log.open("w") as stream:
            port = available_port()
            process = subprocess.Popen([sys.executable, "scripts/mlops_qa_server.py", "--port", str(port)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            try:
                wait_ready(process, port)
                state = next(json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"mlops_fixture"'))
                fixture = Path(temporary) / "fixture.json";fixture.write_text(json.dumps(state))
                environment = os.environ.copy();environment.update(MLOPS_QA_URL=f"http://127.0.0.1:{port}", MLOPS_QA_FIXTURE=str(fixture))
                return subprocess.run(["node", "scripts/browser_mlops.cjs"], cwd=ROOT, env=environment).returncode
            finally:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                for line in log.read_text().splitlines():
                    if line.startswith('{"mlops_fixture"'):
                        owned=Path(json.loads(line)["workspace"])
                        if owned.parent==ROOT/"runtime" and owned.name.startswith("mlops-ui-fixture-") and owned.exists():
                            shutil.rmtree(owned)


if __name__ == "__main__":
    raise SystemExit(main())
