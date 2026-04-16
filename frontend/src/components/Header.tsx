import { Link, useLocation } from 'react-router-dom';

const NAV = [
  { to: '/',        label: 'INDEX',   id: '00' },
  { to: '/build',   label: 'BUILD',   id: '01' },
  { to: '/explore', label: 'ARCHIVE', id: '02' },
  { to: '/about',   label: 'SPEC',    id: '03' },
];

export default function Header() {
  const { pathname } = useLocation();
  return (
    <header className="relative z-10 border-b border-[var(--color-line)] bg-[var(--color-ink-1)]/80 backdrop-blur-sm">
      <div className="max-w-[1440px] mx-auto px-6 md:px-10 h-14 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 group">
          <div className="w-6 h-6 border border-[var(--color-acid)] grid place-items-center">
            <div className="w-2 h-2 bg-[var(--color-acid)] group-hover:animate-pulse-acid" />
          </div>
          <span className="mono text-[13px] tracking-[0.18em] text-[var(--color-fg-strong)]">
            LEGO<span className="text-[var(--color-acid)]">//</span>GEN
          </span>
          <span className="tick hidden md:inline ml-2">v0.2 · two-stage</span>
        </Link>

        <nav className="flex items-center">
          {NAV.map(n => {
            const active = n.to === '/' ? pathname === '/' : pathname.startsWith(n.to);
            return (
              <Link
                key={n.to}
                to={n.to}
                className={`mono text-[11px] tracking-[0.16em] uppercase px-3 py-2 border-b-2 transition-colors ${
                  active
                    ? 'border-[var(--color-acid)] text-[var(--color-fg-strong)]'
                    : 'border-transparent text-[var(--color-mute)] hover:text-[var(--color-fg)]'
                }`}
              >
                <span className="text-[var(--color-acid)] mr-1.5 opacity-60">{n.id}</span>
                {n.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
