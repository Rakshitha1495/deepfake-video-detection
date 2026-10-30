import axios, { AxiosError } from "axios";
import type { HealthResponse, ModelInfoResponse, PredictionResponse } from "../types/api";

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") || "http://127.0.0.1:8000";

const http = axios.create({ baseURL: API_BASE_URL, timeout: 15000 });

export const ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"];
export const MAX_UPLOAD_MB = 100;

/** Turn any axios error into one sentence the user can act on. */
export function describeError(err: unknown): string {
  if (axios.isCancel(err)) return "Upload cancelled.";
  const e = err as AxiosError<{ detail?: unknown }>;
  if (e?.isAxiosError) {
    if (!e.response) {
      if (e.code === "ECONNABORTED") return "The backend took too long to answer. Try a shorter video.";
      return `Cannot reach the backend at ${API_BASE_URL}. Start it with: uvicorn backend.main:app --port 8000`;
    }
    const detail = e.response.data?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return "The request was rejected: a video file is required.";
    return `Backend error (HTTP ${e.response.status}).`;
  }
  return err instanceof Error ? err.message : "Unexpected error.";
}

export async function getHealth(): Promise<HealthResponse> {
  return (await http.get<HealthResponse>("/api/health", { timeout: 5000 })).data;
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  return (await http.get<ModelInfoResponse>("/api/model-info")).data;
}

export async function predictVideo(
  file: File,
  onUploadProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await http.post<PredictionResponse>("/api/predict", form, {
    timeout: 5 * 60 * 1000,
    signal,
    onUploadProgress: (ev) => {
      if (ev.total) onUploadProgress(ev.loaded / ev.total);
    },
  });
  return res.data;
}

export function validateFile(file: File): string | null {
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return `"${file.name}" is not a supported video. Use MP4, AVI, MOV or MKV.`;
  }
  if (file.size === 0) return "This file is empty.";
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return `This file is ${formatBytes(file.size)}. The limit is ${MAX_UPLOAD_MB} MB.`;
  }
  return null;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function pct(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined ? "n/a" : `${(x * 100).toFixed(digits)}%`;
}
