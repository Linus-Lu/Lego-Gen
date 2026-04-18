interface PromptInputProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit?: () => void;
  placeholder?: string;
  disabled?: boolean;
  error?: string | null;
  maxChars?: number;
}

export default function PromptInput({
  value,
  onChange,
  onSubmit,
  placeholder = 'a small red house with a dark red roof…',
  disabled,
  error,
  maxChars,
}: PromptInputProps) {
  return (
    <div className="h-full flex flex-col">
      <p className="label mb-2">INPUT.TXT</p>
      <div
        className={`relative flex-1 bp-frame-4 bg-[var(--color-ink-2)] border transition-colors ${
          error
            ? 'border-[var(--color-danger)]'
            : 'border-[var(--color-line)] focus-within:border-[var(--color-acid)]'
        }`}
      >
        <span className="bp-corner-tl" />
        <span className="bp-corner-br" />
        <span className="absolute top-3 left-3 mono text-[10px] text-[var(--color-acid)] opacity-60">&gt;</span>
        <textarea
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={e => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
              e.preventDefault();
              onSubmit?.();
            }
          }}
          placeholder={placeholder}
          disabled={disabled}
          rows={7}
          className="w-full h-full bg-transparent resize-none outline-none pl-8 pr-3 pt-3 pb-3 mono text-[13px] leading-[1.6] text-[var(--color-fg-strong)] placeholder:text-[var(--color-mute)]"
        />
        <p
          className={`absolute bottom-2 right-3 mono text-[9px] tracking-[0.18em] uppercase ${
            error ? 'text-[var(--color-danger)]' : 'text-[var(--color-mute)]'
          }`}
        >
          ⌘↵ to run · {maxChars ? `${value.length} / ${maxChars}` : value.length} ch
        </p>
      </div>
      <p
        className={`mt-2 min-h-[1rem] mono text-[10px] tracking-[0.14em] uppercase ${
          error ? 'text-[var(--color-danger)]' : 'text-transparent'
        }`}
      >
        {error ?? 'within prompt limit'}
      </p>
    </div>
  );
}
