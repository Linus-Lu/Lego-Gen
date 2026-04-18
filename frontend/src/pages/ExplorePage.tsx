import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';
import GalleryCard from '../components/GalleryCard';
import { listGalleryBuilds } from '../api/legogen';
import type { GalleryBuild } from '../api/legogen';

type SortKey = 'newest' | 'stars' | 'bricks';

const SORTS: { id: SortKey; label: string }[] = [
  { id: 'newest', label: 'NEWEST' },
  { id: 'bricks', label: 'BRICK COUNT' },
  { id: 'stars',  label: 'RATED' },
];

export default function ExplorePage() {
  const [builds, setBuilds] = useState<GalleryBuild[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [sort, setSort] = useState<SortKey>('newest');
  const [q, setQ] = useState('');
  const requestIdRef = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    const requestId = ++requestIdRef.current;
    const t = setTimeout(() => {
      setLoading(true);
      setError('');
      listGalleryBuilds({ sort, q: q || undefined }, controller.signal)
        .then(res => {
          if (controller.signal.aborted || requestId !== requestIdRef.current) return;
          setBuilds(res);
        })
        .catch((e: any) => {
          if (controller.signal.aborted || requestId !== requestIdRef.current) return;
          setError(e?.message ?? 'Failed to load archive');
        })
        .finally(() => {
          if (controller.signal.aborted || requestId !== requestIdRef.current) return;
          setLoading(false);
        });
    }, 100);
    return () => {
      clearTimeout(t);
      controller.abort();
    };
  }, [sort, q]);

  const totals = useMemo(() => ({
    builds: builds.length,
    bricks: builds.reduce((s, b) => s + b.brick_count, 0),
    stable: builds.filter(b => (typeof b.stable === 'boolean' ? b.stable : !!b.stable)).length,
  }), [builds]);

  return (
    <div className="min-h-screen flex flex-col bp-grid text-[var(--color-fg)]">
      <Header />

      <main className="flex-1 max-w-[1440px] w-full mx-auto px-6 md:px-10 py-8">
        <div className="flex items-end justify-between border-b border-[var(--color-line)] pb-3 mb-6">
          <div>
            <p className="label mb-1">ARCHIVE // INDEX</p>
            <h1 className="display text-[40px] md:text-[52px] text-[var(--color-fg-strong)]">
              Saved builds.
            </h1>
          </div>
          <div className="hidden md:flex items-center gap-5 pb-1">
            <Stat label="BUILDS" value={totals.builds} />
            <Stat label="BRICKS" value={totals.bricks} />
            <Stat label="STABLE" value={totals.stable} tint="var(--color-acid)" />
          </div>
        </div>

        {/* Control strip */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-6">
          <div className="flex items-center gap-1 border border-[var(--color-line)]">
            {SORTS.map(s => (
              <button
                key={s.id}
                onClick={() => setSort(s.id)}
                className={`mono text-[10px] tracking-[0.18em] uppercase px-3 py-2 border-r border-[var(--color-line)] last:border-r-0 ${
                  sort === s.id
                    ? 'bg-[var(--color-acid)] text-[var(--color-ink-0)]'
                    : 'text-[var(--color-mute)] hover:text-[var(--color-fg)]'
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>

          <div className="relative md:w-80">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 mono text-[10px] text-[var(--color-acid)]">
              &gt;
            </span>
            <input
              value={q}
              onChange={e => setQ(e.target.value)}
              placeholder="filter / title or caption"
              className="input-blueprint w-full pl-8"
            />
          </div>
        </div>

        {/* Grid */}
        {loading ? (
          <LoadingGrid />
        ) : error ? (
          <EmptyState
            label="ERR // ARCHIVE"
            title="Couldn't load archive"
            msg={error}
          />
        ) : builds.length === 0 ? (
          <EmptyState
            label="ARCHIVE // EMPTY"
            title="No builds yet."
            msg="Save one from the build session and it will appear here."
            action={<Link to="/build" className="btn-primary">▸ run pipeline</Link>}
          />
        ) : (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
            {builds.map((b, i) => (
              <GalleryCard key={b.id} build={b} onStarUpdate={u => {
                setBuilds(prev => prev.map(p => p.id === u.id ? u : p));
              }} index={i} />
            ))}
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}

function Stat({ label, value, tint }: { label: string; value: number; tint?: string }) {
  return (
    <div>
      <p className="tick mb-0.5">{label}</p>
      <p className="mono text-[16px] tabular-nums" style={{ color: tint ?? 'var(--color-fg-strong)' }}>
        {String(value).padStart(3, '0')}
      </p>
    </div>
  );
}

function LoadingGrid() {
  return (
    <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="border border-[var(--color-line)] bg-[var(--color-ink-2)] scan-line">
          <div className="aspect-[4/3] bp-grid" />
          <div className="p-4 space-y-2">
            <div className="h-4 bg-[var(--color-line)]" />
            <div className="h-3 w-2/3 bg-[var(--color-line)]" />
          </div>
        </div>
      ))}
    </div>
  );
}

function EmptyState({
  label, title, msg, action,
}: {
  label: string; title: string; msg: string; action?: React.ReactNode;
}) {
  return (
    <div className="border border-[var(--color-line)] bg-[var(--color-ink-2)] p-16 text-center bp-frame-4 relative">
      <span className="bp-corner-tl" /><span className="bp-corner-br" />
      <p className="label-accent mb-3">{label}</p>
      <p className="display text-[30px] text-[var(--color-fg-strong)] mb-3">{title}</p>
      <p className="mono text-[12px] text-[var(--color-mute)] mb-6 max-w-md mx-auto leading-[1.6]">{msg}</p>
      {action}
    </div>
  );
}
