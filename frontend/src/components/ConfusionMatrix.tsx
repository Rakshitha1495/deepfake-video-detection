/** 2x2 confusion matrix: rows = actual, columns = predicted. Index 0 = Real, 1 = Fake. */
export default function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const max = Math.max(1, ...matrix.flat());
  const labels = ["Real", "Fake"];
  const names = [["True real", "Real called fake"], ["Fake missed", "True fake"]];
  return (
    <div className="inline-grid grid-cols-[auto_1fr_1fr] gap-2 text-sm">
      <div />
      {labels.map((l) => (
        <div key={l} className="pb-1 text-center text-haze">Predicted {l.toLowerCase()}</div>
      ))}
      {matrix.map((row, i) => (
        <div key={i} className="contents">
          <div className="flex items-center pr-2 text-haze">Actual {labels[i].toLowerCase()}</div>
          {row.map((v, j) => {
            const correct = i === j;
            const alpha = 0.12 + 0.55 * (v / max);
            return (
              <div
                key={j}
                className="flex min-h-[84px] min-w-[96px] flex-col items-center justify-center rounded-xl border border-rule"
                style={{ backgroundColor: correct ? `rgba(63,208,224,${alpha})` : `rgba(240,112,138,${alpha})` }}
              >
                <span className="font-display text-2xl font-semibold">{v}</span>
                <span className="text-xs text-ink/70">{names[i][j]}</span>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
