interface ProgressIndicatorProps {
  /** Current generation stage. */
  stage: 'idle' | 'stage1' | 'stage2' | 'done' | 'error';
  /** Caption produced by Stage 1 (if any). */
  caption?: string;
  /** Bricks placed so far (SSE stream). */
  brickCount: number;
  /** Cumulative candidate rejections (SSE stream). */
  rejections: number;
  /** Rollback events (SSE stream). */
  rollbacks: number;
  /** Error message when stage === 'error'. */
  error?: string;
}

const STAGE_LABEL: Record<ProgressIndicatorProps['stage'], string> = {
  idle:   'STANDBY',
  stage1: 'STAGE 01 / IMAGE → CAPTION',
  stage2: 'STAGE 02 / CAPTION → BRICKS',
  done:   'COMPLETE',
  error:  'HALTED',
};

export default function ProgressIndicator({
  stage,
  caption,
  brickCount,
  rejections,
  rollbacks,
  error,
}: ProgressIndicatorProps) {
  const active = stage === 'stage1' || stage === 'stage2';
  const errored = stage === 'error';
  const statusColor =
    errored ? 'var(--color-danger)' :
    stage === 'done' ? 'var(--color-acid)' :
    active ? 'var(--color-acid)' : 'var(--color-mute)';

  return (
    <div className={`relative border border-[var(--color-line)] bg-[var(--color-ink-2)] ${active ? 'scan-line' : ''}`}>
      {/* header strip */}
      <div className="flex items-center justify-between border-b border-[var(--color-line)] px-4 py-2">
        <span className="mono text-[10px] tracking-[0.24em] uppercase flex items-center gap-2" style={{ color: statusColor }}>
          <span
            className={`inline-block w-2 h-2 ${active ? 'animate-pulse-acid' : ''}`}
            style={{ background: statusColor }}
          />
          {STAGE_LABEL[stage]}
        </span>
        <span className="mono text-[10px] tracking-[0.2em] text-[var(--color-dim)] uppercase">
          RT // Live stream
        </span>
      </div>

      {/* body */}
      <div className="p-4 space-y-3">
        {/* Stage 1 caption stream */}
        <div className={`transition-opacity ${caption ? 'opacity-100' : 'opacity-40'}`}>
          <p className="tick mb-1">Stage 1 · Caption</p>
          <p className="mono text-[13px] text-[var(--color-fg-strong)] leading-[1.6] min-h-[1.6em]">
            {caption ? (
              <>
                <span className="text-[var(--color-acid)]">&gt; </span>
                {caption}
                {stage === 'stage1' && <span className="caret" />}
              </>
            ) : (
              <span className="text-[var(--color-mute)]">— awaiting —</span>
            )}
          </p>
        </div>

        <hr className="border-[var(--color-line)]" />

        {/* Numeric readouts */}
        <div className="grid grid-cols-4 gap-0 border border-[var(--color-line)]">
          <Readout label="BRICKS" value={brickCount} emphasis={active} />
          <Readout label="REJECTS" value={rejections} emphasis={rejections > 0} tint={rejections > 0 ? 'var(--color-heat)' : undefined} />
          <Readout label="ROLLBACKS" value={rollbacks} emphasis={rollbacks > 0} tint={rollbacks > 0 ? 'var(--color-heat)' : undefined} />
          <Readout
            label="STATE"
            text={stage === 'done' ? 'OK' : errored ? 'ERR' : active ? 'RUN' : 'IDLE'}
            tint={errored ? 'var(--color-danger)' : stage === 'done' ? 'var(--color-acid)' : undefined}
          />
        </div>

        {errored && error && (
          <p className="mono text-[11px] text-[var(--color-danger)] leading-[1.5] pt-2">
            &gt; {error}
          </p>
        )}
      </div>
    </div>
  );
}

function Readout({
  label, value, text, tint, emphasis,
}: {
  label: string; value?: number; text?: string; tint?: string; emphasis?: boolean;
}) {
  return (
    <div className="px-4 py-3 border-r border-[var(--color-line)] last:border-r-0 flex flex-col gap-1">
      <span className="mono text-[9px] tracking-[0.2em] text-[var(--color-mute)] uppercase">{label}</span>
      <span
        className={`mono text-[22px] tabular-nums tracking-tight ${emphasis ? 'text-[var(--color-acid)]' : 'text-[var(--color-fg-strong)]'}`}
        style={tint ? { color: tint } : undefined}
      >
        {text ?? String(value ?? 0).padStart(3, '0')}
      </span>
    </div>
  );
}
