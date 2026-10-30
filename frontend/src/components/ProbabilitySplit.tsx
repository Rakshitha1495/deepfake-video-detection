import type { Probabilities } from "../types/api";

/** One bar split between Real (cyan) and Fake (rose), with the 50% decision line marked. */
export default function ProbabilitySplit({ probabilities, size = "lg" }: { probabilities: Probabilities; size?: "lg" | "sm" }) {
  const real = probabilities.real * 100;
  const fake = probabilities.fake * 100;
  const h = size === "lg" ? "h-4" : "h-1.5";
  return (
    <div>
      {size === "lg" && (
        <div className="mb-2 flex justify-between text-sm">
          <span className="text-signal">Real {real.toFixed(1)}%</span>
          <span className="text-alarm">Fake {fake.toFixed(1)}%</span>
        </div>
      )}
      <div
        className={`relative flex ${h} w-full overflow-hidden rounded-full bg-deck`}
        role="img"
        aria-label={`Real ${real.toFixed(1)} percent, fake ${fake.toFixed(1)} percent`}
      >
        <div className="bg-signal transition-[width] duration-700" style={{ width: `${real}%` }} />
        <div className="bg-alarm transition-[width] duration-700" style={{ width: `${fake}%` }} />
        {size === "lg" && <div className="absolute inset-y-0 left-1/2 w-px bg-abyss/80" aria-hidden="true" />}
      </div>
      {size === "lg" && (
        <div className="mt-1.5 text-center text-xs text-haze">50% decision threshold</div>
      )}
    </div>
  );
}
