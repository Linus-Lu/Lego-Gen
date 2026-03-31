import { Link } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';

const FLOATING_BRICKS = [
  { color: '#C91A09', size: 'w-10 h-6', top: '15%', left: '8%', delay: '0s' },
  { color: '#0055BF', size: 'w-8 h-5', top: '25%', right: '12%', delay: '1s' },
  { color: '#F2CD37', size: 'w-12 h-7', bottom: '30%', left: '5%', delay: '2s' },
  { color: '#237841', size: 'w-6 h-4', top: '60%', right: '8%', delay: '0.5s' },
  { color: '#FE8A18', size: 'w-9 h-5', bottom: '20%', right: '15%', delay: '1.5s' },
  { color: '#A0A5A9', size: 'w-7 h-4', top: '45%', left: '12%', delay: '3s' },
];

const Home: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-gray-950 text-gray-100 font-sans">
      <Header />

      <main className="flex-grow relative overflow-hidden">
        {/* Ambient background */}
        <div className="absolute inset-0 bg-mesh pointer-events-none" />

        {/* Floating LEGO bricks */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          {FLOATING_BRICKS.map((brick, i) => (
            <div
              key={i}
              className={`absolute ${brick.size} rounded-sm opacity-15 animate-float`}
              style={{
                backgroundColor: brick.color,
                top: brick.top,
                left: brick.left,
                right: brick.right,
                bottom: brick.bottom,
                animationDelay: brick.delay,
                animationDuration: `${6 + i * 0.8}s`,
              }}
            >
              {/* Studs */}
              <div className="absolute -top-1 left-1/2 -translate-x-1/2 flex gap-1">
                <div className="w-2 h-1.5 rounded-full" style={{ backgroundColor: brick.color, filter: 'brightness(1.3)' }} />
                <div className="w-2 h-1.5 rounded-full" style={{ backgroundColor: brick.color, filter: 'brightness(1.3)' }} />
              </div>
            </div>
          ))}
        </div>

        {/* Hero section */}
        <div className="relative container mx-auto px-4 pt-24 pb-12 flex flex-col items-center justify-center text-center">
          <div className="max-w-3xl mx-auto animate-slide-up">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass-light text-xs font-medium text-gray-300 mb-8">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              AI-Powered LEGO Instruction Generator
            </div>

            <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8 leading-[1.1]">
              Imagine it.
              <br />
              <span className="text-gradient-warm">Build it.</span>
            </h1>

            <p className="text-lg md:text-xl text-gray-400 mb-12 max-w-2xl mx-auto leading-relaxed">
              Upload a photo or describe your dream model. Our AI deconstructs it into
              <span className="text-gray-200 font-medium"> step-by-step building instructions </span>
              with a full-color 3D interactive viewer.
            </p>

            <div className="flex flex-col sm:flex-row justify-center gap-4">
              <Link
                to="/build"
                className="group relative px-8 py-4 bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-2xl font-bold shadow-lg shadow-blue-600/25 hover:shadow-blue-500/40 hover:scale-[1.03] active:scale-[0.98] transition-all duration-200"
              >
                <span className="relative z-10 flex items-center justify-center gap-2">
                  Start Building
                  <svg className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}><path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>
                </span>
              </Link>
              <Link
                to="/about"
                className="px-8 py-4 glass-light text-gray-200 rounded-2xl font-bold hover:bg-white/8 hover:text-white transition-all duration-200"
              >
                How it Works
              </Link>
            </div>
          </div>

          {/* Feature cards */}
          <div className="mt-28 grid md:grid-cols-3 gap-5 w-full max-w-5xl">
            {[
              {
                icon: (
                  <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z" />
                  </svg>
                ),
                title: 'Vision to Bricks',
                desc: 'Advanced computer vision analyzes your images to deconstruct models into brick-by-brick plans.',
                accent: 'from-blue-500/20 to-blue-600/5',
                border: 'hover:border-blue-500/30',
              },
              {
                icon: (
                  <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
                  </svg>
                ),
                title: 'Prompt to Plan',
                desc: 'Describe "a red castle with a drawbridge" and watch our AI generate the full blueprint.',
                accent: 'from-purple-500/20 to-purple-600/5',
                border: 'hover:border-purple-500/30',
              },
              {
                icon: (
                  <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-2.25-1.313M21 7.5v2.25m0-2.25l-2.25 1.313M3 7.5l2.25-1.313M3 7.5l2.25 1.313M3 7.5v2.25m9 3l2.25-1.313M12 12.75l-2.25-1.313M12 12.75V15m0 6.75l2.25-1.313M12 21.75V15m0 0l-2.25-1.313M3 16.5v2.25M21 16.5v2.25M12 3v2.25m6.75 9l2.25-1.313M5.25 14.25l-2.25-1.313" />
                  </svg>
                ),
                title: '3D Interactive Guide',
                desc: 'Follow an immersive step-by-step 3D viewer with color-coded bricks, just like official manuals.',
                accent: 'from-orange-500/20 to-orange-600/5',
                border: 'hover:border-orange-500/30',
              },
            ].map((card, i) => (
              <div
                key={i}
                className={`group relative glass-light p-7 rounded-2xl ${card.border} transition-all duration-300 hover:translate-y-[-2px]`}
                style={{ animationDelay: `${i * 0.1}s` }}
              >
                {/* Gradient glow */}
                <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${card.accent} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />
                <div className="relative">
                  <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center text-gray-300 mb-5 group-hover:text-white group-hover:bg-white/10 transition-all">
                    {card.icon}
                  </div>
                  <h3 className="text-lg font-bold mb-2 text-gray-100">{card.title}</h3>
                  <p className="text-gray-400 text-sm leading-relaxed">{card.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Home;
