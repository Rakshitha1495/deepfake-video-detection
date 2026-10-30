export default function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <svg width="34" height="34" viewBox="0 0 32 32" aria-hidden="true">
        <rect width="32" height="32" rx="9" fill="#16213F" />
        <path d="M16 5l9 3.5v6.8c0 5.6-3.8 10-9 11.7-5.2-1.7-9-6.1-9-11.7V8.5z" fill="none" stroke="#3FD0E0" strokeWidth="1.8" />
        <path d="M11.5 16.5l3 3 6-6.5" fill="none" stroke="#9B7BF0" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div className="leading-tight">
        <div className="font-display text-[17px] font-semibold text-ink">DeepGuard AI</div>
        {!compact && <div className="text-xs text-haze">Detect. Analyze. Verify.</div>}
      </div>
    </div>
  );
}
