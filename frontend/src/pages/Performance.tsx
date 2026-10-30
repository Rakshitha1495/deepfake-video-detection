import { useEffect } from "react";
import { BarChart3 } from "lucide-react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useAppState } from "../hooks/useAppState";
import { pct } from "../services/api";
import ConfusionMatrix from "../components/ConfusionMatrix";
import EmptyState from "../components/EmptyState";
import Notice from "../components/Notice";

const AXIS = { stroke: "#8C99B6", fontSize: 12 };
const TOOLTIP = {
  contentStyle: { background: "#101935", border: "1px solid #24314F", borderRadius: 10, color: "#E6ECF8" },
  labelFormatter: (l: unknown) => `Epoch ${l}`,
};

function fmtMinutes(s?: number) {
  if (s === undefined) return "n/a";
  return s < 90 ? `${s.toFixed(1)} s` : `${(s / 60).toFixed(1)} min`;
}

export default function Performance() {
  const { modelInfo, refreshBackend, connection } = useAppState();
  useEffect(() => { void refreshBackend(); }, [refreshBackend]);

  const training = modelInfo?.training;
  const evaluation = modelInfo?.evaluation;
  const m = evaluation?.metrics;

  if (!training && !evaluation) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold sm:text-3xl">Model performance</h1>
        <div className="panel">
          <EmptyState icon={BarChart3} title={connection === "offline" ? "Backend offline" : "No training results yet"}>
            {connection === "offline"
              ? "Start the backend to load training results."
              : <>Run <code>python initialize.py</code>. Loss curves, test metrics and the confusion matrix appear here afterwards.</>}
          </EmptyState>
        </div>
      </div>
    );
  }

  const hist = training?.history ?? [];
  const warnings = [...new Set([...(training?.split_warnings ?? []), ...(evaluation?.warnings ?? [])])];
  const cfg = training?.config ?? modelInfo?.training_config ?? {};
  const shownCfg: [string, unknown][] = [
    ["Frames per video", cfg.num_frames], ["Epochs", cfg.epochs], ["Batch size", cfg.batch_size],
    ["Learning rate", cfg.learning_rate], ["LSTM hidden size", cfg.hidden_size],
    ["Backbone frozen", cfg.freeze_backbone === false ? "No" : "Yes"], ["Seed", cfg.seed],
  ];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold sm:text-3xl">Model performance</h1>
        <p className="mt-2 max-w-2xl text-haze">
          Everything on this page is read from files written by training and evaluation on your machine.
          Test metrics come from videos the model never saw during training or model selection.
        </p>
      </header>

      {warnings.length > 0 && (
        <Notice tone="warning">
          <ul className="list-disc space-y-1 pl-4">{warnings.map((w) => <li key={w}>{w}</li>)}</ul>
        </Notice>
      )}

      {m && m.accuracy !== null && (
        <section className="panel p-5 sm:p-6">
          <h2 className="text-lg font-semibold">Test set ({m.num_samples} videos)</h2>
          <dl className="mt-4 grid grid-cols-2 gap-5 sm:grid-cols-5">
            {[["Accuracy", m.accuracy], ["Precision", m.precision], ["Recall", m.recall], ["F1-score", m.f1], ["ROC-AUC", m.roc_auc]].map(([k, v]) => (
              <div key={k as string}>
                <dt className="text-sm text-haze">{k as string}</dt>
                <dd className="font-display text-3xl font-semibold">{v === null ? "n/a" : pct(v as number)}</dd>
              </div>
            ))}
          </dl>
          {m.roc_auc === null && m.roc_auc_note && <p className="mt-3 text-sm text-haze">ROC-AUC: {m.roc_auc_note}</p>}
          <p className="mt-3 text-sm text-haze">Precision and recall treat "Fake" as the positive class.</p>
        </section>
      )}

      {hist.length > 0 && (
        <div className="grid gap-6 xl:grid-cols-2">
          <section className="panel p-5 sm:p-6">
            <h2 className="text-lg font-semibold">Loss per epoch</h2>
            <div className="mt-4 h-64">
              <ResponsiveContainer>
                <LineChart data={hist} margin={{ left: -10, right: 10 }}>
                  <CartesianGrid stroke="#24314F" strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" tick={AXIS} stroke="#24314F" />
                  <YAxis tick={AXIS} stroke="#24314F" />
                  <Tooltip {...TOOLTIP} />
                  <Legend />
                  <Line type="monotone" dataKey="train_loss" name="Training loss" stroke="#9B7BF0" strokeWidth={2} dot />
                  <Line type="monotone" dataKey="val_loss" name="Validation loss" stroke="#3FD0E0" strokeWidth={2} dot connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
          <section className="panel p-5 sm:p-6">
            <h2 className="text-lg font-semibold">Validation accuracy and F1</h2>
            <div className="mt-4 h-64">
              <ResponsiveContainer>
                <LineChart data={hist} margin={{ left: -10, right: 10 }}>
                  <CartesianGrid stroke="#24314F" strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" tick={AXIS} stroke="#24314F" />
                  <YAxis domain={[0, 1]} tick={AXIS} stroke="#24314F" />
                  <Tooltip {...TOOLTIP} formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))} />
                  <Legend />
                  <Line type="monotone" dataKey="train_accuracy" name="Train accuracy" stroke="#9B7BF0" strokeWidth={2} strokeDasharray="5 4" />
                  <Line type="monotone" dataKey="val_accuracy" name="Val accuracy" stroke="#3FD0E0" strokeWidth={2} connectNulls />
                  <Line type="monotone" dataKey="val_f1" name="Val F1" stroke="#E9B14C" strokeWidth={2} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
            {training?.best_epoch && <p className="mt-2 text-sm text-haze">Saved model: epoch {training.best_epoch}, chosen by {training.selection_criterion}.</p>}
          </section>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {m && (
          <section className="panel overflow-x-auto p-5 sm:p-6">
            <h2 className="mb-4 text-lg font-semibold">Confusion matrix (test)</h2>
            <ConfusionMatrix matrix={m.confusion_matrix} />
          </section>
        )}

        <section className="panel p-5 sm:p-6">
          <h2 className="text-lg font-semibold">Training setup</h2>
          <dl className="mt-4 grid grid-cols-2 gap-y-2 text-sm">
            {shownCfg.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-haze">{k}</dt><dd>{v === undefined || v === null ? "n/a" : String(v)}</dd>
              </div>
            ))}
            <dt className="text-haze">Training time</dt><dd>{fmtMinutes(training?.timing.total_training_seconds)}</dd>
            <dt className="text-haze">Status</dt><dd>{training?.status ?? "n/a"}</dd>
          </dl>
        </section>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <section className="panel p-5 sm:p-6">
          <h2 className="mb-3 text-lg font-semibold">ROC Curve (Test Split)</h2>
          <img
            src="/api/artifacts/roc-curve"
            alt="ROC Curve"
            className="w-full rounded-lg border border-rule/50 bg-white"
            onError={(e) => { (e.currentTarget as HTMLElement).style.display = "none"; }}
          />
        </section>

        <section className="panel p-5 sm:p-6">
          <h2 className="mb-3 text-lg font-semibold">Precision-Recall Curve (Test Split)</h2>
          <img
            src="/api/artifacts/pr-curve"
            alt="Precision-Recall Curve"
            className="w-full rounded-lg border border-rule/50 bg-white"
            onError={(e) => { (e.currentTarget as HTMLElement).style.display = "none"; }}
          />
        </section>
      </div>


      {training?.split_summary && (
        <section className="panel overflow-x-auto p-5 sm:p-6">
          <h2 className="text-lg font-semibold">Dataset split</h2>
          <p className="mt-1 text-sm text-haze">
            {training.class_distribution.real} real and {training.class_distribution.fake} fake videos in {training.num_source_groups} independent
            source groups. Related videos always stay in the same split. Grouping: {training.grouping_method}.
          </p>
          <table className="mt-4 w-full min-w-[420px] text-left text-sm">
            <thead className="text-haze">
              <tr className="border-b border-rule"><th className="py-2 font-normal">Split</th><th className="font-normal">Videos</th><th className="font-normal">Real</th><th className="font-normal">Fake</th><th className="font-normal">Source groups</th></tr>
            </thead>
            <tbody>
              {(["train", "val", "test"] as const).map((k) => {
                const s = training.split_summary[k];
                return s ? (
                  <tr key={k} className="border-b border-rule/50">
                    <td className="py-2 capitalize">{k === "val" ? "Validation" : k}</td><td>{s.videos}</td><td>{s.real}</td><td>{s.fake}</td><td>{s.groups}</td>
                  </tr>
                ) : null;
              })}
            </tbody>
          </table>
          {training.failed_videos.length > 0 && (
            <p className="mt-3 text-sm text-amber">{training.failed_videos.length} video(s) could not be decoded and were skipped.</p>
          )}
        </section>
      )}
    </div>
  );
}
