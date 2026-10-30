// Types mirror backend/schemas.py

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  checkpoint_exists: boolean;
  device: string | null;
  model_load_seconds: number | null;
  message: string;
}

export interface Probabilities {
  real: number;
  fake: number;
}

export type PredictionLabel = "REAL" | "FAKE" | "Real" | "Fake";

export interface PredictionResponse {
  filename: string;
  file_size_bytes: number;
  prediction: PredictionLabel;
  real_probability?: number;
  fake_probability?: number;
  status?: string;
  probabilities: Probabilities;
  processing_time_seconds: number;
  frames_processed: number;
  model_loaded: boolean;
  timing: {
    upload_seconds: number;
    decode_seconds: number;
    preprocess_seconds: number;
    inference_seconds: number;
    model_seconds: number;
  };
  video: {
    total_frames: number | null;
    fps: number | null;
    width: number | null;
    height: number | null;
    duration_seconds: number | null;
    decode_method: string | null;
    sampled_indices: number[];
  };
  diagnostic?: Record<string, unknown>;
  warning: string;
}


export interface HistoryRow {
  epoch: number;
  train_loss: number;
  train_accuracy: number | null;
  val_loss: number | null;
  val_accuracy: number | null;
  val_f1: number | null;
  val_precision: number | null;
  val_recall: number | null;
  epoch_seconds: number;
}

export interface SplitSummary {
  videos: number;
  real: number;
  fake: number;
  groups: number;
}

export interface TrainingRecord {
  status: string;
  config: Record<string, unknown>;
  device: string;
  best_epoch: number | null;
  selection_criterion: string;
  history: HistoryRow[];
  class_distribution: { real: number; fake: number; total: number };
  grouping_method: string;
  num_source_groups: number;
  split_summary: Record<string, SplitSummary>;
  split_warnings: string[];
  failed_videos: { path: string; reason: string }[];
  timing: {
    total_training_seconds?: number;
    epochs_seconds?: number;
    pretrained_weights_downloaded?: boolean;
    feature_extraction?: { total_seconds: number; decode_seconds: number; cnn_seconds: number };
  };
  time_budget_minutes: number;
  time_budget_messages: string[];
  finished_at: string;
}

export interface Metrics {
  num_samples: number;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  roc_auc: number | null;
  roc_auc_note: string | null;
  confusion_matrix: number[][];
}

export interface EvaluationRecord {
  split: string;
  metrics: Metrics;
  meaningful: boolean;
  warnings: string[];
  failed_videos: { path: string; reason: string }[];
  num_frames: number;
  best_epoch: number | null;
  best_val_f1: number | null;
  evaluation_seconds: number;
  evaluated_at: string;
}

export interface ModelInfoResponse {
  checkpoint_available: boolean;
  model_loaded: boolean;
  device: string | null;
  architecture: Record<string, unknown>;
  num_frames: number | null;
  training_config: Record<string, unknown> | null;
  best_epoch: number | null;
  best_val_f1: number | null;
  dataset_split: Record<string, SplitSummary> | null;
  split_warnings: string[];
  training: TrainingRecord | null;
  evaluation: EvaluationRecord | null;
  message: string;
}

/** One finished analysis, kept for this browser session only. */
export interface AnalysisRecord {
  id: string;
  filename: string;
  sizeBytes: number;
  prediction: PredictionLabel;
  confidence?: number;
  real_probability?: number;
  fake_probability?: number;
  probabilities: Probabilities;
  processingSeconds: number;
  framesProcessed: number;
  analyzedAt: string; // ISO
}

