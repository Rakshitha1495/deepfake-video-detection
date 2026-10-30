import { useRef, useState } from "react";
import type { DragEvent } from "react";
import { useNavigate } from "react-router-dom";
import { FileVideo, ScanSearch, UploadCloud, X } from "lucide-react";
import { useAppState } from "../hooks/useAppState";
import { ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, formatBytes, validateFile } from "../services/api";
import Notice from "../components/Notice";

export default function Upload() {
  const { current, selectFile, analyze, reset, connection, health } = useAppState();
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const busy = current.status === "uploading" || current.status === "analyzing";

  const accept = (file: File | undefined) => {
    if (!file) return;
    const problem = validateFile(file);
    setError(problem);
    if (!problem) selectFile(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (!busy) accept(e.dataTransfer.files?.[0]);
  };

  const start = () => {
    void analyze();
    navigate("/analysis");
  };

  const modelReady = connection === "online" && !!health?.model_loaded;
  const file = current.file;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold sm:text-3xl">Analyze a video</h1>
        <p className="mt-2 text-haze">The video is sent only to your local backend and deleted right after analysis.</p>
      </header>

      {!file && (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label="Choose a video file"
          className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-16 text-center transition ${
            dragging ? "border-signal bg-signal/5" : "border-rule bg-hull/50 hover:border-haze/60"
          }`}
        >
          <UploadCloud size={36} className={dragging ? "text-signal" : "text-haze"} />
          <p className="mt-4 font-display text-lg">Drop a video here or click to browse</p>
          <p className="mt-2 text-sm text-haze">
            {ALLOWED_EXTENSIONS.map((e) => e.slice(1).toUpperCase()).join(", ")} up to {MAX_UPLOAD_MB} MB. Short clips (under a minute) analyse fastest.
          </p>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept={ALLOWED_EXTENSIONS.join(",") + ",video/*"}
        className="hidden"
        onChange={(e) => { accept(e.target.files?.[0]); e.target.value = ""; }}
      />

      {error && <Notice tone="error">{error}</Notice>}

      {file && (
        <section className="panel grid gap-6 p-5 sm:p-6 md:grid-cols-[1.3fr_1fr]">
          <div className="overflow-hidden rounded-xl border border-rule bg-black">
            {current.previewUrl && (
              <video src={current.previewUrl} controls muted className="aspect-video w-full bg-black object-contain">
                Preview not supported for this format.
              </video>
            )}
          </div>
          <div className="flex flex-col">
            <div className="flex items-start gap-3">
              <FileVideo size={20} className="mt-0.5 shrink-0 text-pulse" />
              <div className="min-w-0">
                <div className="break-all font-medium">{file.name}</div>
                <div className="mt-1 text-sm text-haze">
                  {formatBytes(file.size)}, {file.type || "video"}
                </div>
              </div>
            </div>
            <p className="mt-4 text-sm leading-relaxed text-haze">
              If the preview stays black, your browser cannot play this format. The backend can still analyse it.
            </p>
            <div className="mt-auto flex flex-wrap gap-3 pt-6">
              <button className="btn-primary" onClick={start} disabled={busy || !modelReady}>
                <ScanSearch size={18} /> {busy ? "Analyzing" : "Analyze video"}
              </button>
              <button className="btn-ghost" onClick={() => { reset(); setError(null); }} disabled={busy}>
                <X size={16} /> Choose another
              </button>
            </div>
            {!modelReady && (
              <p className="mt-3 text-sm text-amber">
                {connection === "offline" ? "Start the backend to enable analysis." : "Train the model first (python initialize.py)."}
              </p>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
