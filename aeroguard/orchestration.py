"""Dependency-free HTTP adapter shared by Airflow and local operational checks."""

import json
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def request_json(base_url, path, method="GET"):
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Configure an HTTP(S) API base URL without embedded credentials")
    headers = {"Accept": "application/json"}
    if method != "GET":
        headers["X-AeroGuard-Client"] = "dashboard"
    request = Request(base_url.rstrip("/") + path, method=method, headers=headers)
    try:
        with urlopen(request, timeout=15) as response:
            return json.load(response)
    except HTTPError as error:
        try:
            payload = json.loads(error.read())
            message = payload.get("error", {}).get("message", "API request failed")
        except (ValueError, AttributeError):
            message = "API request failed"
        raise RuntimeError(f"AeroGuard returned HTTP {error.code}: {message}") from error


def process_pending_version(base_url, timeout_seconds=1800, poll_seconds=3):
    """Run one durable backend invocation; Airflow retries cannot bypass its lease."""
    invocation = request_json(base_url, "/api/mlops/tick", "POST")
    if not invocation.get("job_id"):
        return invocation
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = request_json(base_url, "/api/jobs/" + invocation["job_id"])
        if job["status"] == "complete":
            return {"status": "complete", "job_id": job["id"], "result": job["result"]}
        if job["status"] in {"failed", "interrupted"}:
            raise RuntimeError(f"Lifecycle {job['id']} {job['status']}: {job.get('error')}")
        time.sleep(poll_seconds)
    raise TimeoutError("Lifecycle is still recorded by AeroGuard; inspect it before retrying")
