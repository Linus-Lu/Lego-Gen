import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { starGalleryBuild, parseBrickString } from '../api/legogen';
import type { GalleryBuild } from '../api/legogen';

interface GalleryCardProps {
  build: GalleryBuild;
  onStarUpdate?: (b: GalleryBuild) => void;
  index?: number;
}

function formatAge(iso: string): string {
  const diff = Date.now() - new Date(iso + 'Z').getTime();
  if (diff < 60_000) return 'just now';
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

/** Tiny SVG isometric-ish brick stack, derived from brick coords. */
function ThumbGraphic({ bricks }: { bricks: ReturnType<typeof parseBrickString> }) {
  if (!bricks.length) {
    return (
      <div className="w-full h-full bp-dot grid place-items-center">
        <span className="mono text-[10px] text-[var(--color-mute)] tracking-[0.2em]">NO GEOMETRY</span>
      </div>
    );
  }
  // Normalize and project bricks to a simple isometric preview.
  const xs = bricks.map(b => b.x);
  const ys = bricks.map(b => b.y);
  const zs = bricks.map(b => b.z);
  const minX = Math.min(...xs), maxX = Math.max(...xs.map((_, i) => xs[i] + bricks[i].h));
  const minY = Math.min(...ys), maxY = Math.max(...ys.map((_, i) => ys[i] + bricks[i].w));
  const maxZ = Math.max(...zs);

  const W = maxX - minX, D = maxY - minY, H = maxZ + 1;
  const SCALE = 10;
  const vw = (W + D) * SCALE + 40;
  const vh = (H + (W + D) / 2) * SCALE + 40;

  const sorted = [...bricks].sort((a, b) =>
    a.z - b.z ||
    (a.x + a.y) - (b.x + b.y)
  );

  return (
    <svg viewBox={`0 0 ${vw} ${vh}`} className="w-full h-full" preserveAspectRatio="xMidYMid meet">
      {sorted.map((b, i) => {
        const x = b.x - minX, y = b.y - minY, z = H - 1 - b.z;
        // Isometric projection
        const iso = (px: number, py: number, pz: number) => {
          const sx = (px - py) * SCALE + vw / 2;
          const sy = (px + py) * SCALE * 0.5 + pz * SCALE + 20;
          return [sx, sy];
        };
        const [trx, try_] = iso(x + b.h, y, z);
        const [brx, bry] = iso(x + b.h, y + b.w, z);
        const [blx, bly] = iso(x, y + b.w, z);
        const [tbx, tby] = iso(x, y, z + 1);
        const [tbrx, tbry] = iso(x + b.h, y, z + 1);
        const [bbrx, bbry] = iso(x + b.h, y + b.w, z + 1);
        const [bbx, bby] = iso(x, y + b.w, z + 1);

        return (
          <g key={i}>
            {/* top face (lighter) */}
            <polygon
              points={`${tbx},${tby} ${tbrx},${tbry} ${bbrx},${bbry} ${bbx},${bby}`}
              fill={b.color}
              stroke="rgba(0,0,0,0.25)"
              strokeWidth={0.5}
            />
            {/* right face (darker) */}
            <polygon
              points={`${tbrx},${tbry} ${trx},${try_} ${brx},${bry} ${bbrx},${bbry}`}
              fill={b.color}
              fillOpacity={0.55}
              stroke="rgba(0,0,0,0.3)"
              strokeWidth={0.5}
            />
            {/* front face */}
            <polygon
              points={`${bbx},${bby} ${bbrx},${bbry} ${brx},${bry} ${blx},${bly}`}
              fill={b.color}
              fillOpacity={0.72}
              stroke="rgba(0,0,0,0.3)"
              strokeWidth={0.5}
            />
          </g>
        );
      })}
    </svg>
  );
}

export default function GalleryCard({ build, onStarUpdate, index = 0 }: GalleryCardProps) {
  const navigate = useNavigate();
  const [hover, setHover] = useState(0);
  const stable = typeof build.stable === 'boolean' ? build.stable : !!build.stable;
  const bricks = useMemo(() => parseBrickString(build.bricks), [build.bricks]);

  const handleStar = async (stars: number) => {
    try {
      const updated = await starGalleryBuild(build.id, stars);
      onStarUpdate?.(updated);
    } catch { /* silent */ }
  };

  return (
    <article
      className="group bp-frame-4 border border-[var(--color-line)] bg-[var(--color-ink-2)] hover:border-[var(--color-line-2)] transition-colors animate-fade-up"
      style={{ animationDelay: `${Math.min(index * 40, 400)}ms` }}
    >
      <span className="bp-corner-tl" /><span className="bp-corner-br" />

      {/* Thumbnail */}
      <button
        onClick={() => navigate(`/guide/${build.id}`)}
        className="block w-full aspect-[4/3] relative overflow-hidden border-b border-[var(--color-line)] cursor-pointer"
      >
        <ThumbGraphic bricks={bricks} />
        <div className="absolute top-2 left-2 mono text-[9px] tracking-[0.2em] uppercase text-[var(--color-acid)]">
          BUILD / {build.id.slice(0, 6)}
        </div>
        <div className="absolute top-2 right-2 mono text-[9px] tracking-[0.2em] uppercase"
             style={{ color: stable ? 'var(--color-acid)' : 'var(--color-heat)' }}>
          {stable ? '■ STABLE' : '▲ UNSTABLE'}
        </div>
        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-[var(--color-ink-2)] to-transparent h-8" />
      </button>

      {/* Body */}
      <div className="p-4 space-y-3">
        <div className="flex items-start justify-between gap-3">
          <h3 className="serif text-[18px] leading-[1.15] text-[var(--color-fg-strong)] truncate flex-1">
            {build.title || 'Untitled'}
          </h3>
          <span className="mono text-[10px] tracking-[0.16em] uppercase text-[var(--color-mute)] whitespace-nowrap mt-1">
            {formatAge(build.created_at)}
          </span>
        </div>

        {build.caption && (
          <p className="text-[12px] text-[var(--color-dim)] leading-[1.5] line-clamp-2">
            {build.caption}
          </p>
        )}

        <div className="grid grid-cols-2 gap-0 border border-[var(--color-line)]">
          <div className="px-3 py-2 border-r border-[var(--color-line)]">
            <p className="tick">BRICKS</p>
            <p className="mono text-[14px] text-[var(--color-fg-strong)] tabular-nums">
              {String(build.brick_count).padStart(3, '0')}
            </p>
          </div>
          <div className="px-3 py-2">
            <p className="tick">LAYERS</p>
            <p className="mono text-[14px] text-[var(--color-fg-strong)] tabular-nums">
              {String(new Set(bricks.map(b => b.z)).size).padStart(2, '0')}
            </p>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-0.5" onMouseLeave={() => setHover(0)}>
            {[1, 2, 3, 4, 5].map(s => (
              <button
                key={s}
                onClick={() => handleStar(s)}
                onMouseEnter={() => setHover(s)}
                className={`w-5 h-5 text-[14px] leading-none transition-colors ${
                  s <= (hover || Math.round(build.stars))
                    ? 'text-[var(--color-acid)]'
                    : 'text-[var(--color-line-2)]'
                }`}
                aria-label={`Rate ${s} stars`}
              >
                ◆
              </button>
            ))}
            {build.star_count > 0 && (
              <span className="mono text-[10px] text-[var(--color-mute)] ml-1">({build.star_count})</span>
            )}
          </div>
          <button
            onClick={() => navigate(`/guide/${build.id}`)}
            className="mono text-[10px] tracking-[0.18em] uppercase text-[var(--color-mute)] hover:text-[var(--color-acid)] transition-colors"
          >
            open &rarr;
          </button>
        </div>
      </div>
    </article>
  );
}
