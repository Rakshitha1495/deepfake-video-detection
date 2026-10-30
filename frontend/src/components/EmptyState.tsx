import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

export default function EmptyState({
  icon: Icon, title, children, action,
}: { icon: LucideIcon; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center px-6 py-12 text-center">
      <div className="mb-4 rounded-2xl border border-rule bg-deck p-3">
        <Icon size={22} className="text-haze" />
      </div>
      <h3 className="text-lg font-semibold">{title}</h3>
      {children && <div className="mt-2 max-w-md text-sm leading-relaxed text-haze">{children}</div>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}
