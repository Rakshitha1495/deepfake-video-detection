import { Link } from "react-router-dom";
import { BarChart3, Clock, FileVideo, ScanSearch, ShieldCheck } from "lucide-react";
import { useAppState } from "../hooks/useAppState";
import { pct } from "../services/api";
import ProbabilitySplit from "../components/ProbabilitySplit";
import EmptyState from "../components/EmptyState";
import Notice from "../components/Notice";

function timeAgo(iso: string) {
  const s = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export default function Dashboard() {
  const { connection, health, modelInfo, history } = useAppState();
  const realCount = history.filter((h) => {
    const isReal = h.probabilities ? h.probabilities.real > h.probabilities.fake : h.prediction.toUpperCase() === "REAL";
    return isReal;
  }).length;
  const fakeCount = history.length - realCount;
  const evaluation = modelInfo?.evaluation;
  const m = evaluation?.metrics;

  return (
    <div className="space-y-8">
      {/* Hero: the one action this product exists for */}
      <section className="panel relative overflow-hidden px-6 py-9 sm:px-10">
        <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-pulse/20 blur-3xl" />
        <div className="relative grid items-end gap-8 md:grid-cols-[1.4fr_1fr]">
          <div>
            <h1 className="text-3xl font-semibold leading-tight sm:text-[2.6rem]">
              Check whether a video was manipulated.
            </h1>
            <p className="mt-4 max-w-xl leading-relaxed text-haze">
              DeepGuard samples frames across the clip, reads each one with ResNet18 and lets an LSTM
              judge how they change over time. Every video receives a definitive REAL or FAKE prediction.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <Link to="/upload" className="btn-primary">
                <ScanSearch size={18} /> Analyze a video
              </Link>
              <Link to="/performance" className="btn-ghost">
                <BarChart3 size={16} /> How accurate is it?
              </Link>
            </div>
          </div>
          <div className="rounded-xl border border-rule bg-abyss/50 p-5">
            <div className="flex items-center gap-2 text-sm text-haze">
              <ShieldCheck size={16} className="text-signal" /> Model status
            </div>
            <p className="mt-2 font-display text-xl">
              {connection === "offline" ? "Backend not reachable" : health?.model_loaded ? "Ready to analyze" : connection === "checking" ? "Checking" : "Waiting for training"}
            </p>
            <dl className="mt-4 grid grid-cols-2 gap-y-2 text-sm">
              <dt className="text-haze">Device</dt><dd>{health?.device?.toUpperCase() ?? "n/a"}</dd>
              <dt className="text-haze">Frames per video</dt><dd>{modelInfo?.num_frames ?? "n/a"}</dd>
              <dt className="text-haze">Best epoch</dt><dd>{modelInfo?.best_epoch ?? "n/a"}</dd>
            </dl>
          </div>
        </div>
      </section>

      {connection === "online" && health && !health.model_loaded && (
        <Notice tone="warning">
          No trained model is loaded. Put videos in <code>dataset/raw/real</code> and <code>dataset/raw/fake</code>,
          run <code>python initialize.py</code>, then restart the backend.
        </Notice>
      )}

      {/* This session in numbers: counts only, from real analyses */}
      <section aria-labelledby="session-h">
        <h2 id="session-h" className="mb-3 text-lg font-semibold">This session</h2>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 rounded-2xl border border-rule bg-hull/60 p-2 sm:p-0 sm:divide-x sm:divide-rule">
          {[
            { label: "Videos analyzed", value: history.length, color: "text-ink" },
            { label: "Predicted REAL", value: realCount, color: "text-signal" },
            { label: "Predicted FAKE", value: fakeCount, color: "text-alarm" },
          ].map((s) => (
            <div key={s.label} className="px-4 py-4 sm:px-6">
              <div className={`font-display text-3xl font-semibold ${s.color}`}>{s.value}</div>
              <div className="mt-1 text-sm text-haze">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
        <section className="panel p-5 sm:p-6" aria-labelledby="recent-h">
          <div className="mb-4 flex items-center justify-between">
            <h2 id="recent-h" className="text-lg font-semibold">Recent analyses</h2>
            {history.length > 0 && <Link to="/history" className="text-sm text-signal hover:underline">See all</Link>}
          </div>
          {history.length === 0 ? (
            <EmptyState icon={FileVideo} title="Nothing analyzed yet">
              Results from this browser session will appear here.
            </EmptyState>
          ) : (
            <ul className="divide-y divide-rule">
              {history.slice(0, 5).map((h) => {
                const isReal = h.probabilities ? h.probabilities.real > h.probabilities.fake : h.prediction.toUpperCase() === "REAL";
                const label = isReal ? "REAL" : "FAKE";
                const color = isReal ? "text-signal" : "text-alarm";
                return (
                  <li key={h.id} className="flex items-center gap-4 py-3">
                    <span className={`w-20 shrink-0 text-xs font-bold uppercase tracking-wider ${color}`}>
                      {label}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm">{h.filename}</div>
                      <div className="mt-1.5"><ProbabilitySplit probabilities={h.probabilities} size="sm" /></div>
                    </div>
                    <div className="hidden shrink-0 text-right text-xs text-haze sm:block">
                      <div>{timeAgo(h.analyzedAt)}</div>
                      <div className="mt-0.5 flex items-center justify-end gap-1"><Clock size={11} />{h.processingSeconds.toFixed(1)} s</div>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="panel p-5 sm:p-6" aria-labelledby="perf-h">
          <h2 id="perf-h" className="mb-4 text-lg font-semibold">Test-set performance</h2>
          {m && m.accuracy !== null ? (
            <>
              <dl className="grid grid-cols-2 gap-4">
                {[["Accuracy", m.accuracy], ["F1-score", m.f1], ["Precision", m.precision], ["Recall", m.recall]].map(([k, v]) => (
                  <div key={k as string}>
                    <dt className="text-sm text-haze">{k as string}</dt>
                    <dd className="font-display text-2xl font-semibold">{pct(v as number | null)}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-4 text-sm text-haze">
                Measured on {m.num_samples} held-out test videos.
                {!evaluation?.meaningful && " The test split is too small or one-sided, so treat these as rough."}
              </p>
            </>
          ) : (
            <EmptyState icon={BarChart3} title="No evaluation yet">
              Metrics appear after <code>python initialize.py</code> trains and evaluates the model.
            </EmptyState>
          )}
        </section>
      </div>
    </div>
  );
}
