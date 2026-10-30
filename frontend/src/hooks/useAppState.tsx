import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { describeError, getHealth, getModelInfo, predictVideo } from "../services/api";
import type { AnalysisRecord, HealthResponse, ModelInfoResponse, PredictionResponse } from "../types/api";

export type Connection = "checking" | "online" | "offline";
export type AnalysisStatus = "idle" | "ready" | "uploading" | "analyzing" | "done" | "error";

export interface CurrentAnalysis {
  file: File | null;
  previewUrl: string | null;
  status: AnalysisStatus;
  uploadProgress: number;
  result: PredictionResponse | null;
  error: string | null;
  startedAt: number | null;
}

interface AppState {
  connection: Connection;
  health: HealthResponse | null;
  modelInfo: ModelInfoResponse | null;
  refreshBackend: () => Promise<void>;
  history: AnalysisRecord[];
  clearHistory: () => void;
  current: CurrentAnalysis;
  selectFile: (file: File) => void;
  analyze: () => Promise<void>;
  reset: () => void;
}

const HISTORY_KEY = "deepguard.history.v1";
const EMPTY: CurrentAnalysis = {
  file: null, previewUrl: null, status: "idle", uploadProgress: 0, result: null, error: null, startedAt: null,
};

const Ctx = createContext<AppState | null>(null);

function loadHistory(): AnalysisRecord[] {
  try {
    return JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]") as AnalysisRecord[];
  } catch {
    return [];
  }
}

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [connection, setConnection] = useState<Connection>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [history, setHistory] = useState<AnalysisRecord[]>(loadHistory);
  const [current, setCurrent] = useState<CurrentAnalysis>(EMPTY);
  const inFlight = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

  const refreshBackend = useCallback(async () => {
    try {
      const h = await getHealth();
      setHealth(h);
      setConnection("online");
      try {
        setModelInfo(await getModelInfo());
      } catch {
        /* model info is optional for the status badge */
      }
    } catch {
      setConnection("offline");
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    void refreshBackend();
    const id = window.setInterval(() => void refreshBackend(), 15000);
    return () => window.clearInterval(id);
  }, [refreshBackend]);

  useEffect(() => {
    try {
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch {
      /* storage may be unavailable (private mode); history then lives in memory only */
    }
  }, [history]);

  const selectFile = useCallback((file: File) => {
    if (inFlight.current) return;
    setCurrent((prev) => {
      if (prev.previewUrl) URL.revokeObjectURL(prev.previewUrl);
      return { ...EMPTY, file, previewUrl: URL.createObjectURL(file), status: "ready" };
    });
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    inFlight.current = false;
    setCurrent((prev) => {
      if (prev.previewUrl) URL.revokeObjectURL(prev.previewUrl);
      return EMPTY;
    });
  }, []);

  const analyze = useCallback(async () => {
    const file = current.file;
    if (!file || inFlight.current) return; // prevents duplicate submissions
    inFlight.current = true;
    const controller = new AbortController();
    abortRef.current = controller;
    setCurrent((p) => ({ ...p, status: "uploading", uploadProgress: 0, error: null, result: null, startedAt: Date.now() }));
    try {
      const result = await predictVideo(
        file,
        (f) => setCurrent((p) => ({ ...p, uploadProgress: f, status: f >= 1 ? "analyzing" : "uploading" })),
        controller.signal,
      );
      setCurrent((p) => ({ ...p, status: "done", result, uploadProgress: 1 }));
      setHistory((h) => [
        {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          filename: result.filename,
          sizeBytes: result.file_size_bytes,
          prediction: result.prediction,
          probabilities: result.probabilities,
          processingSeconds: result.processing_time_seconds,
          framesProcessed: result.frames_processed,
          analyzedAt: new Date().toISOString(),
        },
        ...h,
      ]);
    } catch (err) {
      if (!controller.signal.aborted) {
        setCurrent((p) => ({ ...p, status: "error", error: describeError(err) }));
        void refreshBackend();
      }
    } finally {
      inFlight.current = false;
    }
  }, [current.file, refreshBackend]);

  const value = useMemo<AppState>(
    () => ({
      connection, health, modelInfo, refreshBackend, history,
      clearHistory: () => setHistory([]),
      current, selectFile, analyze, reset,
    }),
    [connection, health, modelInfo, refreshBackend, history, current, selectFile, analyze, reset],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAppState(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAppState must be used inside AppStateProvider");
  return v;
}
