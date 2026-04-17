import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import UploadPanel from '../components/UploadPanel';
import PromptInput from '../components/PromptInput';
import ProgressIndicator from '../components/ProgressIndicator';
import BrickCoordViewer from '../components/BrickCoordViewer';
import {
  generateBricksStream,
  createGalleryBuild,
  parseBrickString,
  bricksToLayers,
} from '../api/legogen';
import type { BrickResponse, StreamEvent } from '../api/legogen';

type Stage = 'idle' | 'stage1' | 'stage2' | 'done' | 'error';

export default function BuildSession() {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState('');
  const [stage, setStage] = useState<Stage>('idle');
  const [caption, setCaption] = useState<string>('');
  const [brickCount, setBrickCount] = useState(0);
  const [rollbacks, setRollbacks] = useState(0);
  const [result, setResult] = useState<BrickResponse | null>(null);
  const [error, setError] = useState<string>('');
  const [currentStep, setCurrentStep] = useState(1);
  const [showFuture, setShowFuture] = useState(true);
  const [title, setTitle] = useState('');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const bricks = useMemo(() => (result ? parseBrickString(result.bricks) : []), [result]);
  const { steps: layers, zLevels } = useMemo(() => bricksToLayers(bricks), [bricks]);

  const reset = useCallback(() => {
    setStage('idle');
    setCaption('');
    setBrickCount(0);
    setRollbacks(0);
    setResult(null);
    setError('');
    setCurrentStep(1);
    setSaved(false);
  }, []);

  const run = useCallback(async () => {
    if (stage === 'stage1' || stage === 'stage2') return;
    if (!file && !prompt.trim()) return;

    reset();
    setStage(file ? 'stage1' : 'stage2');

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await generateBricksStream({
        image: file ?? undefined,
        prompt: prompt.trim() || undefined,
        signal: controller.signal,
        onEvent: (evt: StreamEvent) => {
          if (evt.type === 'progress') {
            if (evt.stage === 'stage1') setStage('stage1');
            if (evt.stage === 'stage2') setStage('stage2');
            if (evt.caption) setCaption(evt.caption);
          } else if (evt.type === 'brick') {
            setBrickCount(evt.count);
          } else if (evt.type === 'rollback') {
            setRollbacks(evt.count);
          }
        },
      });
      if (res.caption) setCaption(res.caption);
      setResult(res);
      setBrickCount(res.brick_count);
      setStage('done');
      setCurrentStep(1);
      const firstWord = (res.caption ?? prompt ?? 'build').split(/\s+/).slice(0, 3).join(' ');
      setTitle(firstWord ? firstWord.charAt(0).toUpperCase() + firstWord.slice(1) : 'Untitled');
    } catch (e: any) {
      if (e?.name === 'AbortError' || controller.signal.aborted) return;
      setError(e?.message ?? 'Generation failed');
      setStage('error');
    }
  }, [file, prompt, stage, reset]);

  const save = useCallback(async () => {
    if (!result || !title.trim()) return;
    setSaving(true);
    try {
      await createGalleryBuild({
        title: title.trim(),
        caption: result.caption ?? '',
        bricks: result.bricks,
        brick_count: result.brick_count,
        stable: result.stable,
      });
      setSaved(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  }, [result, title]);

  const running = stage === 'stage1' || stage === 'stage2';
  const hasResult = stage === 'done' && result;

  return (
    <div className="min-h-screen flex flex-col bp-grid text-[var(--color-fg)]">
      <Header />

      <main className="flex-1 max-w-[1440px] w-full mx-auto px-6 md:px-10 py-8">
        {/* top breadcrumb + status line */}
        <div className="flex items-end justify-between border-b border-[var(--color-line)] pb-3 mb-6">
          <div>
            <p className="label mb-1">SESSION // BUILD</p>
            <h1 className="display text-[32px] md:text-[40px] text-[var(--color-fg-strong)]">
              {running ? 'Generating…' : hasResult ? 'Review output' : 'Configure input'}
            </h1>
          </div>
          <div className="hidden md:flex items-center gap-4 pb-1">
            <StatusDot running={running} errored={stage === 'error'} done={stage === 'done'} />
            <span className="mono text-[10px] tracking-[0.2em] uppercase text-[var(--color-mute)]">
              {running ? 'STREAMING' : stage === 'done' ? 'READY' : stage === 'error' ? 'ERRED' : 'IDLE'}
            </span>
          </div>
        </div>

        <div className="grid xl:grid-cols-12 gap-6">
          {/* Left column: inputs + status */}
          <section className="xl:col-span-5 space-y-6">
            <div className="grid sm:grid-cols-2 gap-4">
              <UploadPanel file={file} onFileSelected={setFile} />
              <PromptInput
                value={prompt}
                onChange={setPrompt}
                onSubmit={run}
                disabled={running}
              />
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={run}
                disabled={running || (!file && !prompt.trim())}
                className="btn-primary flex-1"
              >
                {running ? '▸ streaming…' : '▸ run pipeline'}
              </button>
              {(hasResult || stage === 'error') && (
                <button onClick={reset} className="btn-ghost">
                  ↻ clear
                </button>
              )}
            </div>

            <ProgressIndicator
              stage={stage}
              caption={caption}
              brickCount={brickCount}
              rollbacks={rollbacks}
              error={error}
            />

            {/* Save panel appears on done */}
            {hasResult && (
              <div className="animate-fade-up border border-[var(--color-line)] bg-[var(--color-ink-2)] bp-frame-4 relative">
                <span className="bp-corner-tl" /><span className="bp-corner-br" />
                <div className="px-4 py-2 border-b border-[var(--color-line)] flex items-center justify-between">
                  <span className="label-accent">ARCHIVE_ENTRY</span>
                  <span className="tick">{result!.brick_count} bricks · {result!.stable ? 'stable' : 'unstable'}</span>
                </div>
                <div className="p-4 space-y-3">
                  <div>
                    <p className="tick mb-1.5">Label</p>
                    <input
                      value={title}
                      onChange={e => setTitle(e.target.value)}
                      disabled={saved}
                      className="input-blueprint w-full"
                      placeholder="Name this build…"
                    />
                  </div>
                  <button
                    onClick={save}
                    disabled={saving || saved || !title.trim()}
                    className={saved ? 'btn-ghost w-full' : 'btn-primary w-full'}
                  >
                    {saving ? '▸ saving…' : saved ? '✓ saved to archive' : '▸ save to archive'}
                  </button>
                </div>
              </div>
            )}
          </section>

          {/* Right column: 3D viewport */}
          <section className="xl:col-span-7 flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-[var(--color-line)] pb-2">
              <p className="label">VIEWPORT // 3D</p>
              {hasResult && (
                <div className="flex items-center gap-4 mono text-[10px] tracking-[0.18em] uppercase text-[var(--color-mute)]">
                  <span>LAYER <span className="text-[var(--color-fg-strong)]">{String(currentStep).padStart(2, '0')}</span> / {String(layers.length).padStart(2, '0')}</span>
                  <span>Z <span className="text-[var(--color-fg-strong)]">{String(zLevels[currentStep - 1] ?? 0).padStart(3, '0')}</span></span>
                  <span>Δt <span className="text-[var(--color-fg-strong)]">{(result!.metadata.generation_time_ms / 1000).toFixed(2)}s</span></span>
                </div>
              )}
            </div>

            <div className="flex-1 min-h-[520px] border border-[var(--color-line)] bp-frame-4 relative">
              <span className="bp-corner-tl" /><span className="bp-corner-br" />
              <BrickCoordViewer
                bricks={bricks}
                zLevels={zLevels}
                currentStep={hasResult ? currentStep : undefined}
                showFuture={showFuture}
              />
            </div>

            {/* Step control rail */}
            {hasResult && layers.length > 0 && (
              <div className="border border-[var(--color-line)] bg-[var(--color-ink-2)] divide-y divide-[var(--color-line)]">
                <div className="p-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
                      disabled={currentStep <= 1}
                      className="btn-ghost !py-2 !px-3"
                    >
                      ← prev
                    </button>
                    <button
                      onClick={() => setCurrentStep(Math.min(layers.length, currentStep + 1))}
                      disabled={currentStep >= layers.length}
                      className="btn-ghost !py-2 !px-3"
                    >
                      next →
                    </button>
                    <label className="flex items-center gap-2 ml-4 cursor-pointer select-none">
                      <span
                        className={`w-9 h-5 border border-[var(--color-line-2)] flex items-center transition-colors ${
                          showFuture ? 'bg-[var(--color-acid)]/20 border-[var(--color-acid)]' : ''
                        }`}
                        onClick={() => setShowFuture(v => !v)}
                      >
                        <span
                          className={`w-3 h-3 transition-transform mx-1 ${
                            showFuture ? 'bg-[var(--color-acid)] translate-x-4' : 'bg-[var(--color-mute)]'
                          }`}
                        />
                      </span>
                      <span className="mono text-[10px] tracking-[0.18em] uppercase text-[var(--color-mute)]">
                        ghost future
                      </span>
                    </label>
                  </div>
                  <button
                    onClick={() => setCurrentStep(layers.length)}
                    disabled={currentStep >= layers.length}
                    className="mono text-[10px] tracking-[0.18em] uppercase text-[var(--color-mute)] hover:text-[var(--color-acid)] disabled:opacity-40 disabled:hover:text-[var(--color-mute)]"
                  >
                    jump to full ↗
                  </button>
                </div>

                {/* Layer tally */}
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
            )}
          </section>
        </div>
      </main>

      <Footer />
    </div>
  );
}

function StatusDot({ running, errored, done }: { running: boolean; errored: boolean; done: boolean }) {
  const color = errored ? 'var(--color-danger)' : done ? 'var(--color-acid)' : running ? 'var(--color-acid)' : 'var(--color-mute)';
  return (
    <span
      className={`inline-block w-2 h-2 ${running ? 'animate-pulse-acid' : ''}`}
      style={{ background: color }}
    />
  );
}
