/** Browser contracts mirror FastAPI /openapi.json and docs/API_CONTRACT.md. */
export type Subset = 'FD001' | 'FD002' | 'FD003' | 'FD004';
export type DataMode = 'benchmark' | 'demo';
export type Health = 'healthy' | 'attention' | 'critical' | 'unknown';
export interface EnginePrediction {asset:string;engine_id:number;subset:Subset;cycle:number;rul:number;lower:number|null;upper:number|null;anomaly:number|null;quality_valid:boolean;quality:number;bundle_id:string;timestamp?:string|null;state:string;health:Health;trend?:number;quality_reason?:string;decision?:Decision}
export interface HistoryPoint {cycle:number;rul:number;lower?:number|null;upper?:number|null;true_rul?:number|null}
export interface Sensor {key:string;index:number;symbol:string;label:string;unit:string;value:number|null;series:{cycle:number;value:number|null}[]}
export interface Decision {reason:string;state:string;severity:Health;inputs:Record<string,unknown>;rules_evaluated:{rule:string;matched:boolean}[];triggered_rules:string[];recommended_action:string;timestamp:string}
export interface Order {id:string;asset:string;subset:Subset;namespace:string;priority:'MEDIUM'|'HIGH'|'CRITICAL';status:'open'|'acknowledged'|'in_progress'|'resolved';predicted_rul:number;created_at:string;updated_at:string;target_cycle:number;revision:number;reason:string;owner:string|null;notes:string;resolution:string|null;simulated:boolean;evidence:EnginePrediction;audit?:Audit[]}
export interface Audit {action:string;created_at:string;details:Record<string,unknown>}
export interface Metrics {n_engines:number;rmse:number;mae:number;r2:number|null;nasa_score:number|null;capped_rmse:number}
export interface ModelRun {run_id:string;subset:Subset;family:string;training_status:string;created_at:string;finished_at?:string;configuration:TrainRequest;validation_metrics?:Metrics;evaluation?:{metrics:Metrics};calibration?:Record<string,unknown>;artifact_path:string;mlflow_run_id?:string;promotion_eligible?:boolean;promotion_reason?:string}
export interface Job {id:string;kind:string;status:'queued'|'running'|'complete'|'failed'|'interrupted';configuration:Record<string,unknown>;created_at:string;started_at:string|null;finished_at:string|null;error:string|null;result:unknown;stages?:Stage[]}
export interface Stage {id:number;name:string;subset:Subset|null;status:string;started_at:string;finished_at:string|null;duration_seconds:number|null;records:number|null;detail:string|null;error:string|null}
export interface TrainRequest {subsets:Subset[];iterations:number;learning_rate:number;max_leaf_nodes:number;seed:number;split_seed:number;window:number;stride:number;cap:number;replay_cycles:number;promote:boolean;track_mlflow:boolean}
export interface PipelineRequest {mode:DataMode;subsets:Subset[];cycles:number}
export interface Policy {critical_rul:number;critical_lower:number;critical_upper:number;attention_lower:number;watch_lower:number;anomaly_attention:number;anomaly_extreme:number;trend_threshold:number;persistence:number;critical_persistence:number;rearm_cycles:number}
export interface Settings {workspace_name:string;researcher_name:string;default_subset:Subset|'ALL';policy:Policy;promotion_rmse_tolerance_percent:number;promotion_nasa_tolerance_percent:number;replay_cycles:number;uncertainty_gate:number}
export interface Dashboard {mode:DataMode;source_label:string;summary:{total:number;healthy:number;attention:number;critical:number;unknown:number;open_orders:number;mean_rul:number|null;models_active:number};assets:EnginePrediction[];orders:Order[];runs:ModelRun[];jobs:Job[];active:Partial<Record<Subset,string>>;last_successful_run:Job|null}

export type ResearchVariant = 'paper' | 'conditioned_kan';
export interface ResearchSelection {subset:Subset;selected_variant:ResearchVariant;mean_validation_rmse:Partial<Record<ResearchVariant,number>>;seeds:number[];matched_paper_comparison:boolean;official_test_used_for_selection:false;frozen_at:string}
export interface ResearchRun {run_id:string;subset:Subset;variant:ResearchVariant;seed:number;architecture:'Transformer-KAN-BiLSTM';status:'completed';selected_on_validation:boolean;selection:ResearchSelection;validation_rmse:number;metrics:Metrics;configuration:Record<string,unknown>;explanation_status:'available'|'not_generated';explanations_count:number;local_data_status:'matched'|'missing';inference_dependency_available:boolean;source:string;artifact_count:number}
export interface ResearchCatalog {status:'available'|'awaiting_user_training';runs:ResearchRun[];invalid_exports:{folder:string;error:string}[];dependencies:{torch:boolean;shap:boolean;lime:boolean};notebook_url:string;requirements_url:string;explanation_order:string[]}
export interface LocalExplanation {run_id:string;subset:Subset;variant:ResearchVariant;engine_id:number;cycle:number;target:'current_cycle_predicted_rul';units:'cycles';prediction:number;feature_groups:string[];background_source:'fit_engines_only';backgrounds:{partition:'fit';engine_id:number;cycle:number}[];shap:{method:'shap.KernelExplainer';baseline:number;values:number[];additivity_residual_cycles:number;samples:number};lime:{method:'lime.lime_base.LimeBase';coefficients:number[];intercept:number;local_prediction:number;fitting_r2:number|null;heldout_fidelity:{weighted_r2:number|null;weighted_mae_cycles:number};samples:number;heldout_samples:number}}

export interface LifecycleSettings {
  enabled:boolean;auto_retrain:boolean;orchestrator:'local'|'airflow';interval_seconds:number;max_attempts:number;
  min_engines_per_regime:number;ks_alpha:number;ks_min_effect:number;drift_sensor_count:number;
  maximum_missing_fraction:number;reference_cycles:number;
  minimum_rmse_improvement_percent:number;minimum_nasa_improvement_percent:number;
}
export interface LifecycleRequest {version_id:string;training?:Partial<TrainRequest>|null;force_retrain?:boolean;monitor_only?:boolean}
export interface DatasetVersion {
  id:string;subset:Subset;sha256:string;label_status:'complete_run_to_failure'|'observed_prefix';namespace:string;
  filename:string;source_description:string;created_at:string;rows:number;engines:number;
  missing_sensor_fraction:number;missing_cycles:number;file:string;schema:'cmapss-26-v1';
  engine_hashes:Record<string,string>;integrity_sha256:string;processing_status?:string;
  attempts?:number;job_id?:string|null;error?:string|null;result?:Record<string,unknown>|null;
}
export interface MLflowRegisteredModel {
  run_id:string;name:string;version:string;model_uri:string;mlflow_run_id:string;bundle_sha256:string;
  created_at:string;status:'registered';packaged_inference_verified?:boolean;aliases:Record<string,string>;
}
export interface MonitoringCheck {
  id:string;version_id:string;subset:Subset;reference_model:string|null;created_at:string;
  policy_snapshot:LifecycleSettings;
  quality:{passed:boolean;reasons:string[];missing_sensor_fraction:number;missing_cycles:number;engines:number;labels_available:boolean;label_provenance:string};
  drift:{status:'stable'|'drift'|'insufficient_data'|'reference_required'|'not_checked_quality_failed';drifted_sensors:string[];out_of_domain_fraction:number;
    comparisons:{regime:number;sensor:string;reference_engines:number;incoming_engines:number;ks_statistic:number;p_value:number;adjusted_p_value:number;drifted:boolean}[]};
}
export interface MLOpsReview {id:string;version_id:string;reason:string;status:'open'|'resolved';notes:string;created_at:string;resolved_at:string|null}
export interface MLOpsOverview {
  settings:LifecycleSettings;versions:(DatasetVersion|{id:string;status:'invalid';error:string})[];
  checks:MonitoringCheck[];reviews:MLOpsReview[];protocols:Record<string,unknown>[];
  registry:(MLflowRegisteredModel|{run_id:string;status:'error';error:string})[];
  scheduler_running:boolean;orchestration_error:string|null;source:string;neural_status:string;
  research_tracking:(ResearchRun&{tracking:Record<string,unknown>|null})[];
}
export interface EnginePrediction {partition?:'incoming';namespace?:string;source_version_id?:string}
