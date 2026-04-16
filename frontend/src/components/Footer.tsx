export default function Footer() {
  return (
    <footer className="border-t border-[var(--color-line)] bg-[var(--color-ink-1)]">
      <div className="max-w-[1440px] mx-auto px-6 md:px-10 py-5 flex items-center justify-between">
        <p className="mono text-[10px] tracking-[0.18em] text-[var(--color-mute)] uppercase">
          LEGO//GEN · Two-stage Qwen3.5 pipeline · {new Date().getFullYear()}
        </p>
        <p className="mono text-[10px] tracking-[0.18em] text-[var(--color-mute)] uppercase">
          Stage 1: 9B · Stage 2: 4B · NF4
        </p>
      </div>
    </footer>
  );
}
