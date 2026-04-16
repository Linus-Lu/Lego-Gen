import Header from '../components/Header';
import Footer from '../components/Footer';

const SPECS: { group: string; rows: [string, string][] }[] = [
  {
    group: 'STAGE 1 / IMAGE → CAPTION',
    rows: [
      ['BASE_MODEL',  'Qwen/Qwen3.5-9B'],
      ['QUANTIZATION', 'NF4 · 4-bit · double-quant'],
      ['ADAPTER',      'LoRA r=32 α=64 (all-linear)'],
      ['STRATEGY',     'DoRA + rsLoRA · vision frozen'],
      ['MAX_TOKENS',   '256'],
      ['TEMPERATURE',  '0.70 · top-p 0.90'],
    ],
  },
  {
    group: 'STAGE 2 / CAPTION → BRICKS',
    rows: [
      ['BASE_MODEL',  'Qwen/Qwen3.5-4B'],
      ['QUANTIZATION', 'NF4 · 4-bit'],
      ['ADAPTER',      'LoRA r=32 α=64 (q_proj, v_proj)'],
      ['STRATEGY',     'DoRA + rsLoRA + PiSSA'],
      ['DECODER',      'Outlines RegexLogitsProcessor'],
      ['GRAMMAR',      'HxW (x,y,z) #RRGGBB · 14 dims'],
      ['TEMPERATURE',  '0.60 (fixed, no ramp)'],
    ],
  },
  {
    group: 'PHYSICS / STABILITY LP',
    rows: [
      ['MODEL',        'Force-equilibrium per brick'],
      ['SOLVER',       'scipy.optimize.linprog · HiGHS'],
      ['VARIABLES',    'per-stud contact forces'],
      ['CONSTRAINTS',  'vertical balance · zero moment'],
      ['STUD_STRENGTH','1.0 (tension pull-off bound)'],
      ['ROLLBACK',     'binary search · ≤100 rollbacks'],
    ],
  },
];

const CHANGELOG: { tag: string; date: string; note: string }[] = [
  { tag: 'v0.2.0', date: '2026-04', note: 'Strip legacy JSON path. Pipeline is now strictly two-stage brick-coordinate.' },
  { tag: 'v0.1.2', date: '2026-03', note: 'Grammar-constrained brick decoding (outlines). Parse failures now impossible.' },
  { tag: 'v0.1.1', date: '2026-03', note: 'Force-equilibrium LP replaces connectivity-graph stability.' },
  { tag: 'v0.1.0', date: '2026-03', note: 'Two-stage pipeline online. Stage 1: 9B captioner · Stage 2: 4B bricker.' },
];

export default function About() {
  return (
    <div className="min-h-screen flex flex-col bp-grid text-[var(--color-fg)]">
      <Header />

      <main className="flex-1 max-w-[1240px] w-full mx-auto px-6 md:px-10 py-8">
        {/* Masthead */}
        <div className="flex items-end justify-between border-b border-[var(--color-line)] pb-3 mb-10">
          <div>
            <p className="label mb-1">DOC // SPEC</p>
            <h1 className="display text-[40px] md:text-[56px] text-[var(--color-fg-strong)]">
              System specification.
            </h1>
          </div>
          <p className="tick hidden md:block pb-1">rev 0.2 · april 2026</p>
        </div>

        {/* Intro */}
        <section className="grid md:grid-cols-12 gap-8 mb-16">
          <div className="md:col-span-8">
            <p className="mono text-[15px] leading-[1.7] text-[var(--color-dim)]">
              LEGOGEN is a two-stage generative pipeline for producing physically
              stable LEGO structures from images or text. It exists to explore
              how far grammar-constrained decoding plus continuous physics
              validation can push a language model toward{' '}
              <span className="text-[var(--color-acid)]">buildable output</span>,
              rather than plausible-looking output.
            </p>
            <p className="mono text-[15px] leading-[1.7] text-[var(--color-dim)] mt-4">
              Every brick the model emits is checked against a voxel occupancy
              grid, every completed structure is verified with a per-stud
              force-equilibrium linear program, and any unstable prefix is
              rolled back mid-generation. The result is that the model cannot,
              by construction, return geometry that would topple when built.
            </p>
          </div>
          <aside className="md:col-span-4 border border-[var(--color-line)] bg-[var(--color-ink-2)] bp-frame-4 relative">
            <span className="bp-corner-tl" /><span className="bp-corner-br" />
            <div className="px-4 py-2 border-b border-[var(--color-line)]">
              <span className="label-accent">AT A GLANCE</span>
            </div>
            <dl className="divide-y divide-[var(--color-line)] mono text-[12px]">
              {[
                ['STAGES',        '2'],
                ['PARAMS.TRAIN',  '~1% via LoRA'],
                ['OUTPUT',        'HxW (x,y,z) #RRGGBB'],
                ['ALLOWED_DIMS',  '14'],
                ['MAX_BRICKS',    '500'],
                ['MAX_ROLLBACKS', '100'],
              ].map(([k, v]) => (
                <div key={k} className="flex items-center justify-between px-4 py-2">
                  <dt className="text-[var(--color-mute)]">{k}</dt>
                  <dd className="text-[var(--color-fg-strong)]">{v}</dd>
                </div>
              ))}
            </dl>
          </aside>
        </section>

        {/* Spec tables */}
        <section className="space-y-10 mb-16">
          {SPECS.map(block => (
            <div key={block.group}>
              <div className="flex items-center justify-between border-b border-[var(--color-line)] pb-2 mb-3">
                <span className="label-accent">{block.group}</span>
                <span className="tick">{block.rows.length} parameters</span>
              </div>
              <table className="w-full mono text-[12px] border border-[var(--color-line)]">
                <tbody className="divide-y divide-[var(--color-line)]">
                  {block.rows.map(([k, v]) => (
                    <tr key={k} className="hover:bg-[var(--color-ink-2)]">
                      <td className="px-4 py-2.5 text-[var(--color-mute)] w-[40%] border-r border-[var(--color-line)]">
                        {k}
                      </td>
                      <td className="px-4 py-2.5 text-[var(--color-fg-strong)]">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </section>

        {/* Changelog */}
        <section className="mb-16">
          <div className="flex items-center justify-between border-b border-[var(--color-line)] pb-2 mb-4">
            <span className="label-accent">REVISION / LOG</span>
            <span className="tick">most recent first</span>
          </div>
          <ol className="border border-[var(--color-line)] divide-y divide-[var(--color-line)]">
            {CHANGELOG.map(c => (
              <li key={c.tag} className="grid grid-cols-[120px_120px_1fr] px-4 py-3 gap-4 hover:bg-[var(--color-ink-2)]">
                <span className="mono text-[11px] tracking-[0.14em] uppercase text-[var(--color-acid)]">{c.tag}</span>
                <span className="mono text-[11px] text-[var(--color-mute)]">{c.date}</span>
                <span className="mono text-[12px] text-[var(--color-fg-strong)]">{c.note}</span>
              </li>
            ))}
          </ol>
        </section>
      </main>

      <Footer />
    </div>
  );
}
