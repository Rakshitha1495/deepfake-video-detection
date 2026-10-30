import { useEffect, useState } from "react";

interface Props {
  previewUrl: string | null;
  sampledIndices: number[];
  fps: number | null;
  totalFrames: number | null;
}

function fmtTime(s: number) {
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

/**
 * Shows exactly which moments of the video the model looked at.
 * Thumbnails are grabbed in the browser from the local preview (no extra
 * upload). If the browser cannot play the format (often AVI/MKV), only the
 * timeline positions are shown.
 */
export default function FrameStrip({ previewUrl, sampledIndices, fps, totalFrames }: Props) {
  const [thumbs, setThumbs] = useState<(string | null)[]>([]);
  const times = fps && fps > 0 ? sampledIndices.map((i) => i / fps) : [];

  useEffect(() => {
    if (!previewUrl || times.length === 0) return;
    let cancelled = false;
    const video = document.createElement("video");
    video.muted = true;
    video.preload = "auto";
    video.src = previewUrl;
    const canvas = document.createElement("canvas");
    canvas.width = 192;
    canvas.height = 108;
    const ctx = canvas.getContext("2d");

    const seek = (t: number) =>
      new Promise<void>((resolve, reject) => {
        const done = () => { video.removeEventListener("seeked", done); resolve(); };
        video.addEventListener("seeked", done);
        video.addEventListener("error", () => reject(new Error("decode")), { once: true });
        video.currentTime = Math.min(t, Math.max(0, (video.duration || t) - 0.05));
      });

    (async () => {
      try {
        await new Promise<void>((res, rej) => {
          video.addEventListener("loadeddata", () => res(), { once: true });
          video.addEventListener("error", () => rej(new Error("unsupported")), { once: true });
        });
        const out: (string | null)[] = [];
        for (const t of times) {
          if (cancelled) return;
          await seek(t);
          if (!ctx) { out.push(null); continue; }
          const vw = video.videoWidth, vh = video.videoHeight;
          const scale = Math.max(canvas.width / vw, canvas.height / vh);
          const w = vw * scale, h = vh * scale;
          ctx.drawImage(video, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
          out.push(canvas.toDataURL("image/jpeg", 0.7));
          if (!cancelled) setThumbs([...out]);
        }
      } catch {
        if (!cancelled) setThumbs([]);
      }
    })();
    return () => { cancelled = true; video.removeAttribute("src"); video.load(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewUrl, sampledIndices.join(","), fps]);

  if (sampledIndices.length === 0) return null;
  const total = totalFrames && totalFrames > 0 ? totalFrames : Math.max(...sampledIndices) + 1;

  return (
    <div>
      <div className="flex gap-2 overflow-x-auto pb-2">
        {sampledIndices.map((idx, k) => (
          <figure key={`${idx}-${k}`} className="w-[120px] shrink-0">
            <div className="aspect-video overflow-hidden rounded-lg border border-rule bg-deck">
              {thumbs[k] ? (
                <img src={thumbs[k] as string} alt={`Analysed frame ${k + 1}`} className="h-full w-full object-cover" />
              ) : (
                <div className="flex h-full items-center justify-center text-xs text-haze">Frame {idx}</div>
              )}
            </div>
            <figcaption className="mt-1 text-xs text-haze">
              {times[k] !== undefined ? fmtTime(times[k]) : `#${idx}`}
            </figcaption>
          </figure>
        ))}
      </div>
      {/* where those frames sit within the whole clip */}
      <div className="relative mt-2 h-2 rounded-full bg-deck" aria-hidden="true">
        {sampledIndices.map((idx, k) => (
          <span
            key={k}
            className="absolute top-1/2 h-3.5 w-0.5 -translate-y-1/2 rounded bg-pulse"
            style={{ left: `${(idx / Math.max(total - 1, 1)) * 100}%` }}
          />
        ))}
      </div>
    </div>
  );
}
