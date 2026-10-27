"""Airflow 3 adapter: the tested AeroGuard service owns data/model transactions."""

from datetime import datetime, timedelta, timezone
import os

from airflow.sdk import dag, task

from aeroguard.orchestration import process_pending_version


@dag(
    dag_id="aeroguard_closed_loop",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "aeroguard", "retries": 2, "retry_delay": timedelta(minutes=2)},
    tags=["cmapss", "mlops", "simulated-maintenance"],
)
def aeroguard_closed_loop():
    @task(execution_timeout=timedelta(minutes=35))
    def monitor_and_process_one_version():
        return process_pending_version(os.environ.get("AEROGUARD_API_URL", "http://127.0.0.1:8765"))

    monitor_and_process_one_version()


aeroguard_closed_loop()
