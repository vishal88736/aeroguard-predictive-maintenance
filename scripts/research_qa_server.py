"""Isolated browser-test server with explicitly labelled contract fixtures, no trained neural model."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from aeroguard.api import create_app
from test_research_exports import archive, export_fixture
import uvicorn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--empty", action="store_true", help="Serve a clean source-only portal without a research fixture")
    args = parser.parse_args()
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="research-ui-fixture-", dir=runtime) as temporary:
        root = Path(temporary)
        if args.empty:
            data, bundle, run_id = root / "data", None, None
        else:
            data, export, run_id, _ = export_fixture(root)
            bundle = root / "isolated-ui-fixture.zip"
            bundle.write_bytes(archive(export))
        print(json.dumps({"fixture_zip": str(bundle) if bundle else None, "workspace": str(root),
                          "run_id": run_id, "url": f"http://127.0.0.1:{args.port}",
                          "source": "Isolated UI contract fixture, dummy model bytes, no benchmark training"}), flush=True)
        uvicorn.run(create_app(data, root / "artifacts", root / "runtime"), host="127.0.0.1", port=args.port)
