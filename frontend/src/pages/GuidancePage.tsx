import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';
import BrickCoordViewer from '../components/BrickCoordViewer';
import {
  downloadLDraw,
  getGalleryBuild,
  parseBrickString,
  bricksToLayers,
} from '../api/legogen';
import type { GalleryBuild } from '../api/legogen';

export default function GuidancePage() {
  const { buildId } = useParams<{ buildId: string }>();
  const navigate = useNavigate();

  const [build, setBuild] = useState<GalleryBuild | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [currentStep, setCurrentStep] = useState(1);
  const [showFuture, setShowFuture] = useState(true);
  const [autoplay, setAutoplay] = useState(false);

  useEffect(() => {
    if (!buildId) return;
    setLoading(true);
    getGalleryBuild(buildId)
      .then(b => { setBuild(b); setLoading(false); })
      .catch(e => { setError(e?.message ?? 'Failed to load'); setLoading(false); });
  }, [buildId]);

  const bricks = useMemo(() => (build ? parseBrickString(build.bricks) : []), [build]);
  const { steps: layers, zLevels } = useMemo(() => bricksToLayers(bricks), [bricks]);

  // Autoplay
  useEffect(() => {
    if (!autoplay || !layers.length) return;
    const id = setInterval(() => {
      setCurrentStep(s => {
        if (s >= layers.length) {
          setAutoplay(false);
          return s;
        }
        return s + 1;
      });
    }, 1400);
    return () => clearInterval(id);
  }, [autoplay, layers.length]);

  const goPrev = useCallback(() => setCurrentStep(s => Math.max(1, s - 1)), []);
  const goNext = useCallback(() => setCurrentStep(s => Math.min(layers.length, s + 1)), [layers.length]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === 'j' || e.key === 'n') goNext();
      if (e.key === 'ArrowLeft' || e.key === 'k' || e.key === 'p') goPrev();
      if (e.key === ' ') { e.preventDefault(); setAutoplay(a => !a); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [goNext, goPrev]);

  if (loading) {
    return (
      <PageShell>
        <div className="flex-1 grid place-items-center">
          <div className="text-center">
            <p className="label-accent mb-2">LOADING_BUILD</p>
            <p className="mono text-[13px] text-[var(--color-fg-strong)]">
              resolving {buildId?.slice(0, 8)}<span className="caret" />
            </p>
          </div>
        </div>
      </PageShell>
    );
  }

  if (error || !build) {
    return (
      <PageShell>
        <div className="flex-1 grid place-items-center">
          <div className="text-center max-w-md">
            <p className="label-accent mb-2" style={{ color: 'var(--color-danger)' }}>ERR // BUILD_NOT_FOUND</p>
            <p className="mono text-[13px] text-[var(--color-fg-strong)] mb-6">{error || 'No such build in archive.'}</p>
            <button onClick={() => navigate('/explore')} className="btn-ghost">← back to archive</button>
          </div>
        </div>
      </PageShell>
    );
  }

  const current = layers[currentStep - 1];

  return (
    <PageShell>
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-6 md:px-10 py-8">
        {/* Title */}
        <div className="flex items-end justify-between border-b border-[var(--color-line)] pb-3 mb-6">
          <div>
            <p className="label mb-1">GUIDE // {build.id.slice(0, 8).toUpperCase()}</p>
            <h1 className="display text-[34px] md:text-[44px] text-[var(--color-fg-strong)]">
              {build.title || 'Untitled'}
            </h1>
          </div>
          <div className="hidden md:flex items-center gap-5 text-right pb-1">
            <Stat label="BRICKS" value={String(build.brick_count).padStart(3, '0')} />
            <Stat label="LAYERS" value={String(layers.length).padStart(2, '0')} />
            <Stat
              label="STATE"
              value={build.stable ? 'STABLE' : 'UNSTABLE'}
              tint={build.stable ? 'var(--color-acid)' : 'var(--color-heat)'}
            />
          </div>
        </div>

        {build.caption && (
          <p className="mono text-[12px] text-[var(--color-dim)] mb-6 max-w-3xl leading-[1.6]">
            <span className="text-[var(--color-acid)]">&gt; </span>
            {build.caption}
          </p>
        )}

        <div className="grid xl:grid-cols-12 gap-6">
          {/* Viewport */}
          <section className="xl:col-span-8 flex flex-col gap-4">
            <div className="min-h-[540px] flex-1 border border-[var(--color-line)] bp-frame-4 relative">
              <span className="bp-corner-tl" /><span className="bp-corner-br" />
              <BrickCoordViewer
                bricks={bricks}
                zLevels={zLevels}
                currentStep={currentStep}
                showFuture={showFuture}
              />
            </div>

            {/* Transport bar */}
            <div className="border border-[var(--color-line)] bg-[var(--color-ink-2)]">
              <div className="px-4 py-3 border-b border-[var(--color-line)] flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <button onClick={goPrev} disabled={currentStep <= 1} className="btn-ghost !py-2 !px-3">← prev</button>
                  <button
                    onClick={() => setAutoplay(a => !a)}
                    className={autoplay ? 'btn-primary !py-2 !px-4' : 'btn-ghost !py-2 !px-4'}
                  >
                    {autoplay ? '■ pause' : '▶ autoplay'}
                  </button>
                  <button onClick={goNext} disabled={currentStep >= layers.length} className="btn-ghost !py-2 !px-3">next →</button>
                </div>
                <div className="flex items-center gap-4 mono text-[10px] tracking-[0.18em] uppercase text-[var(--color-mute)]">
                  <button
                    onClick={() => setShowFuture(v => !v)}
                    className={showFuture ? 'text-[var(--color-acid)]' : 'hover:text-[var(--color-fg)]'}
                  >
                    ghost future {showFuture ? 'on' : 'off'}
                  </button>
                  <span className="hidden md:inline">← → to step · space toggles auto</span>
                </div>
              </div>
              {/* Progress bar */}
              <div className="h-1 bg-[var(--color-line)] relative">
                <div
                  className="absolute top-0 left-0 h-full bg-[var(--color-acid)] transition-all duration-300"
                  style={{ width: `${(currentStep / Math.max(layers.length, 1)) * 100}%` }}
                />
              </div>
              {/* Layer pills */}
              <div className="p-3 flex gap-2 overflow-x-auto">
                {layers.map((l, i) => (
                  <button
                    key={l.z}
                    onClick={() => setCurrentStep(i + 1)}
                    className={`flex-shrink-0 px-3 py-2 border transition-colors mono text-[10px] tracking-[0.16em] uppercase ${
                      currentStep === i + 1
                        ? 'border-[var(--color-acid)] text-[var(--color-acid)] bg-[var(--color-acid)]/5'
                        : 'border-[var(--color-line)] text-[var(--color-mute)] hover:border-[var(--color-line-2)] hover:text-[var(--color-fg)]'
                    }`}
                  >
                    L{String(i + 1).padStart(2, '0')} <span className="opacity-60">·</span> {l.brick_count}
                  </button>
                ))}
              </div>
            </div>
          </section>

          {/* Right rail: current step parts */}
          <aside className="xl:col-span-4 space-y-4">
            <div className="border border-[var(--color-line)] bg-[var(--color-ink-2)] bp-frame-4 relative">
              <span className="bp-corner-tl" /><span className="bp-corner-br" />
              <div className="px-4 py-2 border-b border-[var(--color-line)] flex items-center justify-between">
                <span className="label-accent">CURRENT_STEP</span>
                <span className="tick">L{String(currentStep).padStart(2, '0')} / {String(layers.length).padStart(2, '0')}</span>
              </div>
              {current ? (
                <div className="p-4 space-y-4">
                  <div>
                    <p className="tick mb-1">Altitude</p>
                    <p className="mono text-[20px] text-[var(--color-fg-strong)] tabular-nums">
                      z = {String(current.z).padStart(3, '0')}
                      <span className="text-[var(--color-mute)] ml-2">({current.brick_count} bricks)</span>
                    </p>
                  </div>

                  <div>
                    <p className="tick mb-2">Parts required</p>
                    <ul className="border-t border-[var(--color-line)] divide-y divide-[var(--color-line)]">
                      {current.tally.map((t, i) => (
                        <li key={i} className="flex items-center gap-3 py-2">
                          <span
                            className="w-4 h-4 border border-[var(--color-line-2)] flex-shrink-0"
                            style={{ background: t.color }}
                            aria-hidden
                          />
                          <span className="mono text-[12px] tracking-[0.1em] text-[var(--color-fg-strong)] uppercase flex-1">
                            {t.dims}
                          </span>
                          <span className="mono text-[11px] text-[var(--color-mute)] uppercase">
                            {t.color}
                          </span>
                          <span className="mono text-[13px] text-[var(--color-acid)] tabular-nums w-8 text-right">
                            ×{t.count}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              ) : (
                <div className="p-6 text-center">
                  <p className="mono text-[11px] text-[var(--color-mute)]">no layer selected</p>
                </div>
              )}
            </div>

            <button
              onClick={() => downloadLDraw(build.title || 'legogen-build', build.bricks).catch(e => {
                setError(e instanceof Error ? e.message : 'Export failed');
              })}
              className="btn-ghost w-full"
            >
              ▸ export .ldr
            </button>

            <button onClick={() => navigate('/explore')} className="btn-ghost w-full">
              ← back to archive
            </button>
          </aside>
        </div>
      </main>
    </PageShell>
  );
}

function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col bp-grid text-[var(--color-fg)]">
      <Header />
      {children}
      <Footer />
    </div>
  );
}

function Stat({ label, value, tint }: { label: string; value: string; tint?: string }) {
  return (
    <div>
      <p className="tick mb-0.5">{label}</p>
      <p className="mono text-[14px] tabular-nums" style={{ color: tint ?? 'var(--color-fg-strong)' }}>
        {value}
      </p>
    </div>
  );
}
