import { Link } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bp-grid text-[var(--color-fg)]">
      <Header />

      <main className="flex-1 relative">
        {/* Hero */}
        <section className="max-w-[1440px] mx-auto px-6 md:px-10 pt-16 md:pt-24 pb-20">
          {/* Top system bar */}
          <div className="flex items-center justify-between border-b border-[var(--color-line)] pb-3 mb-12 animate-fade-up">
            <div className="flex items-center gap-6">
              <span className="label-accent">SYSTEM_READY</span>
              <span className="tick">SESSION // 0x{Math.floor(Date.now() / 1000).toString(16).toUpperCase().slice(-6)}</span>
            </div>
            <span className="tick hidden md:inline">TWO-STAGE · QWEN3.5-9B → QWEN3.5-4B</span>
          </div>

          {/* Display */}
          <div className="grid md:grid-cols-12 gap-8 items-start">
            <div className="md:col-span-8">
              <p className="label mb-6 animate-fade-up">DIRECTIVE — 001</p>

              <h1 className="display text-[64px] md:text-[92px] lg:text-[112px] text-[var(--color-fg-strong)] mb-6 animate-fade-up delay-100">
                From pixel
                <span className="block text-[var(--color-acid)]">to brick.</span>
              </h1>

              <p className="text-[17px] md:text-[19px] text-[var(--color-dim)] max-w-[560px] leading-[1.55] mb-10 animate-fade-up delay-200">
                A two-stage generative pipeline. Stage 1 captions your image with a
                9B vision-language model. Stage 2 emits grammar-constrained brick
                coordinates with per-stud physics validation. No renderer tricks —
                every brick you see can actually be built.
              </p>

              <div className="flex items-center gap-3 animate-fade-up delay-300">
                <Link to="/build" className="btn-primary">
                  ▸ Start build session
                </Link>
                <Link to="/about" className="btn-ghost">
                  View pipeline spec
                </Link>
              </div>
            </div>

            {/* Side panel: live-looking system readout */}
            <div className="md:col-span-4 md:mt-4 animate-fade-up delay-400">
              <div className="border border-[var(--color-line)] bg-[var(--color-ink-2)] bp-frame-4 relative">
                <span className="bp-corner-tl" /><span className="bp-corner-br" />
                <div className="px-4 py-2 border-b border-[var(--color-line)] flex items-center justify-between">
                  <span className="mono text-[10px] tracking-[0.2em] uppercase text-[var(--color-acid)]">SPEC / PIPELINE</span>
                  <span className="mono text-[10px] tracking-[0.2em] uppercase text-[var(--color-mute)]">live</span>
                </div>
                <dl className="divide-y divide-[var(--color-line)] mono text-[12px]">
                  {[
                    ['STAGE_1.MODEL',    'Qwen3.5-9B'],
                    ['STAGE_1.QUANT',    'NF4 · 4-bit'],
                    ['STAGE_1.ADAPTER',  'LoRA r=32'],
                    ['STAGE_2.MODEL',    'Qwen3.5-4B'],
                    ['STAGE_2.DECODER',  'regex-grammar'],
                    ['STAGE_2.PHYSICS',  'stud-LP'],
                    ['STAGE_2.TEMP',     'T = 0.60'],
                  ].map(([k, v]) => (
                    <div key={k} className="flex items-center justify-between px-4 py-2">
                      <dt className="text-[var(--color-mute)]">{k}</dt>
                      <dd className="text-[var(--color-fg-strong)]">{v}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>
          </div>
        </section>

        {/* Feature triad */}
        <section className="max-w-[1440px] mx-auto px-6 md:px-10 pb-24">
          <div className="flex items-end justify-between border-b border-[var(--color-line)] pb-4 mb-8">
            <p className="label">PIPELINE / 3 STAGES</p>
            <p className="tick hidden md:inline">hover modules to inspect</p>
          </div>

          <div className="grid md:grid-cols-3 gap-0 border border-[var(--color-line)]">
            {[
              {
                id: '01',
                title: 'Perceive',
                model: 'Qwen3.5-9B + Stage 1 LoRA',
                desc: 'Multimodal encoder produces a short geometry/color/scale description. Vision encoder frozen. DoRA + rsLoRA on top of 4-bit NF4 base.',
              },
              {
                id: '02',
                title: 'Build',
                model: 'Qwen3.5-4B + Stage 2 LoRA',
                desc: 'Grammar-constrained regex decoder emits HxW (x,y,z) #hex lines. Fourteen allowed dims. Structure-aware loss weighting during training.',
              },
              {
                id: '03',
                title: 'Validate',
                model: 'VoxelGrid + SciPy HiGHS',
                desc: 'Every brick checked for collision and bounds. Force-equilibrium LP verifies per-stud tension + compression. Unstable placements roll back.',
              },
            ].map(card => (
              <article
                key={card.id}
                className="group relative p-7 border-r border-b md:border-b-0 border-[var(--color-line)] last:border-r-0 hover:bg-[var(--color-ink-2)] transition-colors"
              >
                <div className="flex items-baseline gap-4 mb-8">
                  <span className="mono text-[11px] tracking-[0.24em] text-[var(--color-acid)]">{card.id}</span>
                  <div className="flex-1 h-px bg-[var(--color-line)] group-hover:bg-[var(--color-acid)] transition-colors" />
                </div>
                <h3 className="display text-[38px] text-[var(--color-fg-strong)] mb-2">{card.title}</h3>
                <p className="mono text-[11px] tracking-[0.14em] uppercase text-[var(--color-mute)] mb-5">{card.model}</p>
                <p className="text-[14px] leading-[1.6] text-[var(--color-dim)]">{card.desc}</p>
              </article>
            ))}
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
