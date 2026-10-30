import { RefreshCw } from "lucide-react";
import { useAppState } from "../hooks/useAppState";
import { API_BASE_URL } from "../services/api";

export function useBackendStatus() {
  const { connection, health } = useAppState();
  if (connection === "checking") return { tone: "haze", text: "Connecting to backend" } as const;
  if (connection === "offline") return { tone: "alarm", text: "Backend offline" } as const;
  if (!health?.model_loaded) return { tone: "amber", text: "Model not trained" } as const;
  return { tone: "signal", text: `Model ready on ${health.device?.toUpperCase() ?? "CPU"}` } as const;
}

const DOT: Record<string, string> = {
  haze: "bg-haze", alarm: "bg-alarm", amber: "bg-amber", signal: "bg-signal",
};

export default function ConnectionBadge({ detailed = false }: { detailed?: boolean }) {
  const { refreshBackend, connection, health } = useAppState();
  const s = useBackendStatus();
  return (
    <div className={detailed ? "rounded-xl border border-rule bg-hull/70 p-3" : ""}>
      <div className="flex items-center gap-2 text-sm" role="status" aria-live="polite">
        <span className="relative flex h-2.5 w-2.5">
          {s.tone === "signal" && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-signal/50 motion-reduce:hidden" />}
          <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${DOT[s.tone]}`} />
        </span>
        <span className="text-ink">{s.text}</span>
        {detailed && (
          <button onClick={() => void refreshBackend()} className="ml-auto text-haze hover:text-ink" aria-label="Check backend again">
            <RefreshCw size={14} />
          </button>
        )}
      </div>
      {detailed && (
        <p className="mt-2 break-all text-xs leading-relaxed text-haze">
          {connection === "offline"
            ? `Nothing answering at ${API_BASE_URL}.`
            : connection === "online" && !health?.model_loaded
              ? "Run python initialize.py, then restart the backend."
              : health?.model_load_seconds != null
                ? `Checkpoint loaded once in ${health.model_load_seconds.toFixed(2)} s.`
                : API_BASE_URL}
        </p>
      )}
    </div>
  );
}
