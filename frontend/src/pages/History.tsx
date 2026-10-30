import { Link } from "react-router-dom";
import { History as HistoryIcon, Trash2 } from "lucide-react";
import { useAppState } from "../hooks/useAppState";
import { formatBytes, pct } from "../services/api";
import EmptyState from "../components/EmptyState";
import Notice from "../components/Notice";

export default function History() {
  const { history, clearHistory } = useAppState();

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold sm:text-3xl">History</h1>
          <p className="mt-2 text-haze">Every video you analysed in this browser tab.</p>
        </div>
        {history.length > 0 && (
          <button className="btn-ghost" onClick={clearHistory}><Trash2 size={16} /> Clear history</button>
        )}
      </header>

      <Notice tone="info">
        Session-only: history is kept in this tab's session storage and disappears when you close the tab.
        Nothing is saved on the server, and uploaded videos are deleted after analysis.
      </Notice>

      <section className="panel overflow-hidden">
        {history.length === 0 ? (
          <EmptyState icon={HistoryIcon} title="No analyses in this session" action={<Link to="/upload" className="btn-primary">Analyze a video</Link>}>
            Analysed videos will be listed here with their result and timing.
          </EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead className="border-b border-rule text-haze">
                <tr>
                  <th className="px-5 py-3 font-normal">File</th>
                  <th className="px-3 py-3 font-normal">Prediction</th>
                  <th className="px-3 py-3 font-normal">Fake probability</th>
                  <th className="px-3 py-3 font-normal">Date and time</th>
                  <th className="px-5 py-3 text-right font-normal">Processing</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h) => (
                  <tr key={h.id} className="border-b border-rule/50 last:border-0">
                    <td className="max-w-[260px] px-5 py-3">
                      <div className="truncate">{h.filename}</div>
                      <div className="text-xs text-haze">{formatBytes(h.sizeBytes)}, {h.framesProcessed} frames</div>
                    </td>
                    <td className={`px-3 py-3 font-semibold uppercase ${
                      (h.probabilities ? h.probabilities.real > h.probabilities.fake : h.prediction.toUpperCase() === "REAL")
                        ? "text-signal"
                        : "text-alarm"
                    }`}>
                      {(h.probabilities ? h.probabilities.real > h.probabilities.fake : h.prediction.toUpperCase() === "REAL") ? "REAL" : "FAKE"}
                    </td>

                    <td className="px-3 py-3">{pct(h.probabilities.fake)}</td>
                    <td className="px-3 py-3 text-haze">{new Date(h.analyzedAt).toLocaleString()}</td>
                    <td className="px-5 py-3 text-right">{h.processingSeconds.toFixed(2)} s</td>

                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
