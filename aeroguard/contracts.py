"""Validated public API/configuration contracts; /openapi.json is authoritative."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Subset = Literal["FD001", "FD002", "FD003", "FD004"]
Mode = Literal["benchmark", "demo"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Policy(StrictModel):
    critical_rul: float = Field(5, ge=0, le=30)
    critical_lower: float = Field(10, ge=0, le=40)
    critical_upper: float = Field(10, ge=0, le=40)
    attention_lower: float = Field(30, ge=5, le=100)
    watch_lower: float = Field(50, ge=10, le=125)
    anomaly_attention: float = Field(.99, ge=.8, le=1)
    anomaly_extreme: float = Field(.995, ge=.8, le=1)
    trend_threshold: float = Field(.04, ge=.001, le=2)
    persistence: int = Field(3, ge=1, le=20)
    critical_persistence: int = Field(2, ge=1, le=10)
    rearm_cycles: int = Field(5, ge=2, le=30)

    @model_validator(mode="after")
    def ordered(self):
        if not self.critical_lower < self.attention_lower < self.watch_lower:
            raise ValueError("Require critical_lower < attention_lower < watch_lower")
        if self.anomaly_extreme < self.anomaly_attention:
            raise ValueError("Extreme anomaly threshold must be at least the attention threshold")
        return self


class Settings(StrictModel):
    workspace_name: str = Field("AeroGuard Research", min_length=1, max_length=60)
    researcher_name: str = Field("Local researcher", min_length=1, max_length=60)
    default_subset: Literal["ALL", "FD001", "FD002", "FD003", "FD004"] = "ALL"
    policy: Policy = Field(default_factory=Policy)
    promotion_rmse_tolerance_percent: float = Field(0, ge=0, le=10)
    promotion_nasa_tolerance_percent: float = Field(5, ge=0, le=20)
    replay_cycles: int = Field(30, ge=3, le=150)
    uncertainty_gate: float = Field(80, ge=10, le=250)


class TrainRequest(StrictModel):
    subsets: list[Subset] = Field(default_factory=lambda: ["FD001"], min_length=1, max_length=4)
    iterations: int = Field(100, ge=10, le=500)
    learning_rate: float = Field(.08, ge=.005, le=.5)
    max_leaf_nodes: int = Field(15, ge=5, le=63)
    seed: int = Field(42, ge=0, le=2147483647)
    split_seed: int = Field(42, ge=0, le=2147483647)
    window: int = Field(30, ge=5, le=60)
    stride: int = Field(3, ge=1, le=15)
    cap: float = Field(125, ge=30, le=200)
    replay_cycles: int = Field(30, ge=3, le=150)
    promote: bool = True
    track_mlflow: bool = True

    @model_validator(mode="after")
    def unique_subsets(self):
        if len(set(self.subsets)) != len(self.subsets):
            raise ValueError("Subsets must be unique")
        return self


class PipelineRequest(StrictModel):
    mode: Mode = "benchmark"
    subsets: list[Subset] = Field(default_factory=lambda: ["FD001"], min_length=1, max_length=4)
    cycles: int = Field(30, ge=3, le=150)


class MLOpsBootstrap(StrictModel):
    subsets: list[Subset] = Field(default_factory=lambda: ["FD001", "FD002", "FD003", "FD004"], min_length=1, max_length=4)

    @model_validator(mode="after")
    def unique_subsets(self):
        if len(set(self.subsets)) != len(self.subsets):
            raise ValueError("Subsets must be unique")
        return self


class LifecycleRequest(StrictModel):
    version_id: str = Field(pattern=r"^DV-FD00[1-4]-[a-f0-9]{20}$")
    training: TrainRequest | None = None
    force_retrain: bool = False
    monitor_only: bool = False


class LifecycleSettings(StrictModel):
    enabled: bool = False
    auto_retrain: bool = False
    orchestrator: Literal["local", "airflow"] = "local"
    interval_seconds: int = Field(60, ge=5, le=86400)
    max_attempts: int = Field(3, ge=1, le=10)
    min_engines_per_regime: int = Field(20, ge=5, le=1000)
    ks_alpha: float = Field(.01, gt=0, le=.1)
    ks_min_effect: float = Field(.2, gt=0, le=1)
    drift_sensor_count: int = Field(3, ge=1, le=21)
    maximum_missing_fraction: float = Field(.02, ge=0, le=.2)
    reference_cycles: int = Field(30, ge=5, le=100)
    minimum_rmse_improvement_percent: float = Field(0, ge=0, le=50)
    minimum_nasa_improvement_percent: float = Field(0, ge=0, le=50)


class RegisterRequest(StrictModel):
    run_id: str = Field(pattern=r"^AG-FD00[1-4]-[a-f0-9]{12}$")


class ReviewResolve(StrictModel):
    notes: str = Field(min_length=1, max_length=1500)


class ResearchTrackRequest(StrictModel):
    run_id: str = Field(pattern=r"^KAN-FD00[1-4]-(?:paper|conditioned_kan)-\d{1,10}-[a-f0-9]{12}$")


class PromoteRequest(StrictModel):
    run_id: str = Field(pattern=r"^AG-FD00[1-4]-[a-f0-9]{12}$")
    rollback: bool = False


class OrderUpdate(StrictModel):
    action: Literal["acknowledge", "assign", "start", "resolve"]
    owner: str | None = Field(None, max_length=80)
    notes: str = Field("", max_length=1500)
    resolution: Literal["preventive_service", "sensor_repair", "confirmed_failure", "no_fault_found", "deferred"] | None = None

    @model_validator(mode="after")
    def required_details(self):
        if self.action == "assign" and not (self.owner or "").strip():
            raise ValueError("Provide an owner to assign this order")
        if self.action == "resolve" and (self.resolution is None or not self.notes.strip()):
            raise ValueError("An explicit outcome and resolution notes are required")
        return self


class EnginePrediction(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    asset: str
    engine_id: int
    subset: Subset
    cycle: int
    rul: float
    lower: float | None = None
    upper: float | None = None
    anomaly: float | None = None
    state: str
    health: str
    bundle_id: str
    quality_valid: bool
    timestamp: str | None = None


class DashboardResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    mode: Mode
    source_label: str
    summary: dict
    assets: list[EnginePrediction]
    orders: list[dict]
    runs: list[dict]
    jobs: list[dict]
