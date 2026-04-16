import { useCallback, useRef, useState } from 'react';

interface UploadPanelProps {
  onFileSelected: (file: File | null) => void;
  file?: File | null;
}

export default function UploadPanel({ onFileSelected, file }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const take = useCallback((f: File) => {
    setPreview(URL.createObjectURL(f));
    onFileSelected(f);
  }, [onFileSelected]);

  return (
    <div className="h-full">
      <p className="label mb-2">INPUT.IMG <span className="text-[var(--color-acid)]">— optional</span></p>
      <div
        className={`relative h-[220px] bp-frame-4 border transition-colors cursor-pointer overflow-hidden ${
          dragging ? 'border-[var(--color-acid)] bg-[var(--color-ink-3)]' : 'border-[var(--color-line)] bg-[var(--color-ink-2)] hover:border-[var(--color-line-2)]'
        }`}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) take(f);
        }}
        onClick={() => inputRef.current?.click()}
      >
        <span className="bp-corner-tl" />
        <span className="bp-corner-br" />
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={e => e.target.files?.[0] && take(e.target.files[0])}
        />
        {preview && file ? (
          <>
            <img src={preview} alt="" className="w-full h-full object-cover opacity-70" />
            <div className="absolute inset-0 bg-gradient-to-t from-[var(--color-ink-1)] to-transparent" />
            <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between">
              <span className="mono text-[11px] text-[var(--color-fg-strong)] truncate max-w-[70%]">
                {file.name}
              </span>
              <button
                type="button"
                onClick={e => {
                  e.stopPropagation();
                  setPreview(null);
                  onFileSelected(null);
                }}
                className="mono text-[10px] tracking-[0.2em] uppercase text-[var(--color-danger)] hover:text-white px-2 py-1 border border-[var(--color-danger)]/40 hover:bg-[var(--color-danger)]"
              >
                × clear
              </button>
            </div>
          </>
        ) : (
          <div className="absolute inset-0 grid place-items-center text-center">
            <div>
              <p className="mono text-[11px] tracking-[0.2em] text-[var(--color-mute)] uppercase mb-3">
                Drop image here<span className="caret" />
              </p>
              <p className="mono text-[10px] tracking-[0.14em] text-[var(--color-dim)] uppercase">
                JPG / PNG / WEBP · ≤ 8MB
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
