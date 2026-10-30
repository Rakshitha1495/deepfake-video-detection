import type { ReactNode } from "react";
import { AlertTriangle, Info, XCircle } from "lucide-react";

const STYLES = {
  info: { box: "border-pulse/30 bg-pulse/10", icon: Info, color: "text-pulse" },
  warning: { box: "border-amber/30 bg-amber/10", icon: AlertTriangle, color: "text-amber" },
  error: { box: "border-alarm/40 bg-alarm/10", icon: XCircle, color: "text-alarm" },
} as const;

export default function Notice({ tone = "info", children }: { tone?: keyof typeof STYLES; children: ReactNode }) {
  const s = STYLES[tone];
  const Icon = s.icon;
  return (
    <div className={`flex gap-3 rounded-xl border px-4 py-3 text-sm leading-relaxed ${s.box}`} role={tone === "error" ? "alert" : undefined}>
      <Icon size={18} className={`mt-0.5 shrink-0 ${s.color}`} />
      <div className="text-ink/90">{children}</div>
    </div>
  );
}
