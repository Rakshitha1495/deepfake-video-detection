import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Cpu, RotateCcw, ScanSearch, Timer } from "lucide-react";
import { useAppState } from "../hooks/useAppState";
import { formatBytes } from "../services/api";
import ProbabilitySplit from "../components/ProbabilitySplit";
import FrameStrip from "../components/FrameStrip";
import Notice from "../components/Notice";
import EmptyState from "../components/EmptyState";

function useElapsed(running: boolean, startedAt: number | null) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setNow(Date.now()), 200);
    return () => window.clearInterval(id);
  }, [running]);
  return startedAt ? (now - startedAt) / 1000 : 0;
}

function separationText(top: number) {
  if (top >= 0.9) return "The model strongly prefers this class.";
  if (top >= 0.7) return "The model leans toward this class.";
  return "The two classes are close. The model is unsure about this video.";
}

export default function Analysis() {
  const { current, reset, analyze, modelInfo } = useAppState();
  const navigate = useNavigate();
  const running = current.status === "uploading" || current.status === "analyzing";
  const elapsed = useElapsed(running, current.startedAt);
  const r = current.result;

  if (!current.file) {
    return (
      <div className="panel">
        <EmptyState
          icon={ScanSearch}
          title="No video selected"
          action={<Link to="/upload" className="btn-primary">Choose a video</Link>}
        >
          Pick a video on the upload page to see its analysis here.
        </EmptyState>
      </div>
    );
  }

  const realProb = r?.real_probability ?? r?.probabilities?.real ?? 0;
  const fakeProb = r?.fake_probability ?? r?.probabilities?.fake ?? 0;
  const isReal = realProb > fakeProb;
  const isFake = !isReal;
  const top = r ? (r.confidence ?? (isReal ? realProb : fakeProb)) : 0;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold sm:text-3xl">Analysis</h1>
          <p className="mt-1 break-all text-haze">{current.file.name} ({formatBytes(current.file.size)})</p>
        </div>
        <button className="btn-ghost" onClick={() => { reset(); navigate("/upload"); }} disabled={running}>
          <RotateCcw size={16} /> Analyze another video
        </button>
      </header>

      <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
        <div className="panel overflow-hidden p-0">
          <div className="relative bg-black">
            {current.previewUrl && (
              <video src={current.previewUrl} controls={!running} muted className="aspect-video w-full object-contain" />
            )}
            {running && (
              <div className="pointer-events-none absolute inset-0 overflow-hidden bg-abyss/40">
                <div className="scanline h-[10%] w-full bg-gradient-to-b from-transparent via-signal/40 to-transparent" />
              </div>
            )}
          </div>
        </div>

        <section className="panel flex flex-col p-6" aria-live="polite">
          {running && (
            <>
              <p className="text-sm text-haze">{current.status === "uploading" ? "Uploading" : "Sampling frames and running CNN-LSTM"}</p>
              <p className="mt-2 font-display text-4xl font-semibold">{elapsed.toFixed(1)} s</p>
              <div className="mt-6 h-1.5 overflow-hidden rounded-full bg-deck">
                <div
                  className="h-full bg-pulse transition-[width]"
                  style={{ width: `${current.status === "uploading" ? current.uploadProgress * 50 : 75}%` }}
                />
              </div>
              <p className="mt-3 text-sm text-haze">
                {current.status === "uploading"
                  ? `${Math.round(current.uploadProgress * 100)}% uploaded`
                  : "Upload complete. Running ResNet18 feature extraction & LSTM analysis..."}
              </p>
            </>
          )}

          {current.status === "error" && (
            <div className="space-y-4">
              <Notice tone="error">{current.error}</Notice>
              <button className="btn-primary" onClick={() => void analyze()}>Try again</button>
            </div>
          )}

          {current.status === "ready" && (
            <div className="space-y-4">
              <p className="text-haze">This video has not been analysed yet.</p>
              <button className="btn-primary" onClick={() => void analyze()}><ScanSearch size={18} /> Analyze video</button>
            </div>
          )}

          {r && current.status === "done" && (
            <>
              <div className="flex items-center justify-between">
                <p className="text-sm text-haze">Classification Result</p>
              </div>
              <p
                className={`mt-1 font-display text-5xl font-semibold ${
                  isReal ? "text-signal" : "text-alarm"
                }`}
              >
                {isReal ? "REAL VIDEO" : "FAKE VIDEO"}
              </p>
              <p className="mt-1 text-sm font-medium text-haze">
                {isReal
                  ? "Likely Authentic / Genuine"
                  : "Likely Manipulated / Deepfake"}
              </p>
              <p className="mt-3 text-sm text-ink/90">
                {separationText(top)}
              </p>
              <div className="mt-6"><ProbabilitySplit probabilities={r.probabilities} /></div>
              <div className="mt-4 rounded-lg border border-rule/50 bg-abyss/40 p-3 text-xs text-haze">
                <div className="flex justify-between">
                  <span>Real probability:</span>
                  <span className="font-semibold text-signal">{(realProb * 100).toFixed(1)}%</span>
                </div>
                <div className="mt-1 flex justify-between">
                  <span>Fake probability:</span>
                  <span className="font-semibold text-alarm">{(fakeProb * 100).toFixed(1)}%</span>
                </div>
                <div className="mt-1 flex justify-between">
                  <span>Confidence:</span>
                  <span className="font-semibold text-ink">{(top * 100).toFixed(1)}%</span>
                </div>
                <div className="mt-1 flex justify-between">
                  <span>Frames evaluated:</span>
                  <span className="font-semibold text-ink">{r.frames_processed} frames</span>
                </div>
              </div>
              <div className="mt-4">
                <Notice tone="info">
                  These are statistical model probabilities, not certified certainty. Genuine evaluation on Celeb-DF v2 test splits.
                </Notice>
              </div>
            </>
          )}
        </section>
      </div>


      {r && current.status === "done" && (
        <>
          <section className="panel p-5 sm:p-6">
            <h2 className="text-lg font-semibold">Frames the model looked at</h2>
            <p className="mb-4 mt-1 text-sm text-haze">
              {r.frames_processed} frames sampled evenly across the clip. The model sees a centered square crop of each.
            </p>
            <FrameStrip
              previewUrl={current.previewUrl}
              sampledIndices={r.video.sampled_indices}
              fps={r.video.fps}
              totalFrames={r.video.total_frames}
            />
          </section>

          <div className="grid gap-6 md:grid-cols-2">
            <section className="panel p-5 sm:p-6">
              <h2 className="flex items-center gap-2 text-lg font-semibold"><Timer size={18} className="text-pulse" /> Processing time</h2>
              <p className="mt-3 font-display text-3xl font-semibold">{r.processing_time_seconds.toFixed(2)} s</p>
              <p className="text-sm text-haze">measured by the backend, upload included</p>
              <dl className="mt-5 space-y-2 text-sm">
                {[
                  ["Receiving upload", r.timing.upload_seconds],
                  ["Decoding frames", r.timing.decode_seconds],
                  ["Preprocessing", r.timing.preprocess_seconds],
                  ["CNN-LSTM inference", r.timing.inference_seconds],
                ].map(([k, v]) => (
                  <div key={k as string} className="flex justify-between border-b border-rule/60 pb-2">
                    <dt className="text-haze">{k as string}</dt>
                    <dd>{(v as number).toFixed(3)} s</dd>
                  </div>
                ))}
              </dl>
            </section>

            <section className="panel p-5 sm:p-6">
              <h2 className="flex items-center gap-2 text-lg font-semibold"><Cpu size={18} className="text-pulse" /> Video and model</h2>
              <dl className="mt-4 grid grid-cols-2 gap-y-2 text-sm">
                <dt className="text-haze">Resolution</dt>
                <dd>{r.video.width && r.video.height ? `${r.video.width} x ${r.video.height}` : "n/a"}</dd>
                <dt className="text-haze">Duration</dt>
                <dd>{r.video.duration_seconds != null ? `${r.video.duration_seconds.toFixed(1)} s` : "n/a"}</dd>
                <dt className="text-haze">Total frames</dt><dd>{r.video.total_frames ?? "n/a"}</dd>
                <dt className="text-haze">Architecture</dt><dd>ResNet18 + LSTM</dd>
                <dt className="text-haze">LSTM hidden size</dt>
                <dd>{String(modelInfo?.architecture?.lstm_hidden_size ?? "n/a")}</dd>
                <dt className="text-haze">Device</dt><dd>{modelInfo?.device?.toUpperCase() ?? "n/a"}</dd>
              </dl>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
