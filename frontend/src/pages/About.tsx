import Header from '../components/Header';
import Footer from '../components/Footer';

const About: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-gray-950 text-gray-100 font-sans">
      <Header />

      <main className="flex-grow relative overflow-hidden">
        <div className="absolute inset-0 bg-mesh pointer-events-none" />

        <div className="relative container mx-auto px-4 py-16 max-w-4xl">
          {/* Title */}
          <div className="animate-slide-up mb-16">
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4">
              About <span className="text-gradient-warm">LegoGen</span>
            </h1>
            <p className="text-lg text-gray-400 max-w-2xl leading-relaxed">
              An AI-powered system that generates LEGO building instructions from images,
              built as a Final Year Project exploring vision-language models.
            </p>
          </div>

          {/* Architecture cards */}
          <div className="space-y-6">
            {[
              {
                title: 'Vision Encoder',
                subtitle: 'BLIP-2 + QLoRA',
                desc: 'Fine-tuned Salesforce BLIP-2 vision-language model with 4-bit quantization and LoRA adapters on the OPT-2.7B language model. The frozen ViT encoder and Q-Former extract visual features from LEGO images.',
                color: 'from-blue-500',
              },
              {
                title: 'Structured Output',
                subtitle: 'JSON Description Schema',
                desc: 'The model generates structured JSON with part inventories, spatial relationships, color data (hex codes from Rebrickable), subassembly groupings, and build complexity estimates.',
                color: 'from-purple-500',
              },
              {
                title: 'Constraint Engine',
                subtitle: 'Validation + Repair',
                desc: 'Post-processing pipeline validates JSON schema, repairs malformed output, enforces valid enum values, and ensures all parts reference real Rebrickable catalog entries.',
                color: 'from-orange-500',
              },
              {
                title: '3D Build Viewer',
                subtitle: 'React Three Fiber',
                desc: 'Interactive Three.js viewer renders color-coded bricks with step-by-step progressive build animation. Orbit controls for full 3D navigation with transparency support.',
                color: 'from-green-500',
              },
            ].map((item, i) => (
              <div key={i} className="glass-light rounded-2xl p-6 animate-slide-up" style={{ animationDelay: `${i * 0.1}s` }}>
                <div className="flex items-start gap-4">
                  <div className={`w-1 h-full min-h-[60px] rounded-full bg-gradient-to-b ${item.color} to-transparent flex-shrink-0`} />
                  <div>
                    <h3 className="text-lg font-bold text-gray-100">{item.title}</h3>
                    <p className="text-sm text-gray-500 font-medium mb-2">{item.subtitle}</p>
                    <p className="text-gray-400 text-sm leading-relaxed">{item.desc}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Tech stack */}
          <div className="mt-16 glass-light rounded-2xl p-6">
            <h2 className="text-lg font-bold mb-4 text-gray-200">Tech Stack</h2>
            <div className="flex flex-wrap gap-2">
              {[
                'PyTorch', 'BLIP-2', 'QLoRA', 'FastAPI', 'React 19', 'TypeScript',
                'Three.js', 'Tailwind CSS', 'Vite', 'Rebrickable API', 'HuggingFace',
              ].map((tech) => (
                <span key={tech} className="px-3 py-1 text-xs font-medium rounded-lg bg-white/5 text-gray-300 border border-white/5">
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default About;
