import Header from '../components/Header';
import Footer from '../components/Footer';

const About: React.FC = () => {
  const architectureCards = [
    {
      title: 'Unified AI Model',
      subtitle: 'Qwen3.5-27B + LoRA',
      desc: 'A single unified Qwen3.5-27B multimodal transformer handles both image-to-JSON and text-to-JSON generation through a shared LoRA adapter. Fine-tuned with 4-bit NF4 quantization, rank-64 LoRA on all linear layers, keeping the vision encoder frozen while adapting the language model for structured LEGO output.',
      color: 'from-blue-500',
    },
    {
      title: 'Prompt Caching System',
      subtitle: 'Three-Layer Cache Architecture',
      desc: 'A sophisticated three-layer caching system accelerates inference: KV Prefix Cache pre-computes key-value states for static system prompts, Response Cache (LRU with TTL) stores complete inference results keyed by SHA-256 hashes, and Tokenization Cache eliminates redundant chat-template processing. Together they dramatically reduce latency for repeated or similar queries.',
      color: 'from-cyan-500',
    },
    {
      title: 'Gallery & Persistence',
      subtitle: 'SQLite + Async Storage',
      desc: 'Browse, save, and revisit generated builds in a persistent gallery backed by SQLite with async I/O. Features include category filtering, search, sort by newest/top-rated/most parts, star ratings, thumbnail previews, complexity badges, and part count display.',
      color: 'from-pink-500',
    },
    {
      title: 'Structured Output',
      subtitle: 'JSON Description Schema',
      desc: 'The model generates structured JSON with part inventories, spatial relationships, color data (hex codes from Rebrickable), subassembly groupings, build complexity estimates, and dominant color analysis.',
      color: 'from-purple-500',
    },
    {
      title: 'Constraint Engine',
      subtitle: 'Validation + Repair',
      desc: 'Post-processing pipeline validates JSON schema, repairs malformed output, enforces valid enum values, checks build stability and legality, and ensures all parts reference real Rebrickable catalog entries.',
      color: 'from-orange-500',
    },
    {
      title: 'Interactive Build Guidance',
      subtitle: 'Webcam + 3D Viewer',
      desc: 'Split-screen guidance mode pairs a live webcam feed with an interactive 3D viewer for step-by-step building. Includes voice narration via Web Speech API, exploded view toggle, parts checklist with checkoff, step timer, and auto-play mode with configurable pacing.',
      color: 'from-emerald-500',
    },
    {
      title: '3D Build Viewer',
      subtitle: 'React Three Fiber',
      desc: 'Interactive Three.js viewer renders color-coded bricks with step-by-step progressive build animation. Features orbit controls, ghost brick previews, highlight effects, transparency support, and full 3D navigation.',
      color: 'from-green-500',
    },
  ];

  const techStack = [
    'PyTorch', 'Qwen3.5-27B', 'LoRA', 'BitsAndBytes', 'FastAPI',
    'React 19', 'TypeScript', 'Three.js', 'React Three Fiber',
    'Tailwind CSS', 'Vite', 'SQLite', 'aiosqlite',
    'Rebrickable API', 'HuggingFace', 'Web Speech API',
  ];

  const highlights = [
    { label: 'Unified Model', value: 'Qwen3.5-27B' },
    { label: 'Quantization', value: '4-bit NF4' },
    { label: 'LoRA Rank', value: '64 / Alpha 128' },
    { label: 'Cache Layers', value: '3 (KV + Response + Token)' },
    { label: 'Modalities', value: 'Image + Text' },
    { label: 'Output Format', value: 'Structured JSON' },
  ];

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
              An AI-powered system that generates LEGO building instructions from
              images or text prompts, built as a Final Year Project exploring
              unified multimodal transformers. Powered by a single Qwen3.5-27B
              model with LoRA adaptation and a three-layer prompt caching system
              for fast inference.
            </p>
          </div>

          {/* Key highlights */}
          <div className="mb-12 animate-slide-up" style={{ animationDelay: '0.05s' }}>
            <h2 className="text-lg font-bold mb-4 text-gray-200">Key Highlights</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {highlights.map((h) => (
                <div key={h.label} className="glass-light rounded-xl px-4 py-3 text-center">
                  <p className="text-xs text-gray-500 font-medium uppercase tracking-wider mb-1">{h.label}</p>
                  <p className="text-sm font-bold text-gray-200">{h.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Architecture cards */}
          <div className="space-y-6">
            {architectureCards.map((item, i) => (
              <div key={i} className="glass-light rounded-2xl p-6 animate-slide-up" style={{ animationDelay: `${(i + 1) * 0.1}s` }}>
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
              {techStack.map((tech) => (
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
