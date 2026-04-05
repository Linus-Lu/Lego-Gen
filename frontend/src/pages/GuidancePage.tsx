import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import Header from '../components/Header';
import GuidanceViewer from '../components/GuidanceViewer';
import StepControls from '../components/StepControls';
import PartsChecklist from '../components/PartsChecklist';
import VoiceNarrator from '../components/VoiceNarrator';
import { getGalleryBuild } from '../api/legogen';
import type { GenerateResponse, BuildStep } from '../api/legogen';

export default function GuidancePage() {
  const { buildId } = useParams<{ buildId: string }>();
  const location = useLocation();

  const [steps, setSteps] = useState<BuildStep[]>([]);
  const [currentStep, setCurrentStep] = useState(1);
  const [narratedBrickIdx, setNarratedBrickIdx] = useState(-1);
  const [exploded, setExploded] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const [buildTitle, setBuildTitle] = useState('Build');
  const [isComplete, setIsComplete] = useState(false);
  const [voiceMuted, setVoiceMuted] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const narratorRef = useRef(new VoiceNarrator());

  // Load build data from router state or gallery API
  useEffect(() => {
    async function load() {
      setLoading(true);
      setError('');

      // From BuildSession via router state
      const state = location.state as { build?: GenerateResponse } | null;
      if (state?.build) {
        setSteps(state.build.steps);
        setBuildTitle(state.build.description?.object ?? 'Build');
        setLoading(false);
        return;
      }

      // From gallery
      if (buildId && buildId !== 'new') {
        try {
          const build = await getGalleryBuild(buildId);
          const desc = JSON.parse(build.description_json);
          // Re-derive steps from description_json (same structure as GenerateResponse)
          if (desc.steps) {
            setSteps(desc.steps);
          } else if (desc.subassemblies) {
            // Convert subassemblies to build steps
            const derived: BuildStep[] = desc.subassemblies.map((sub: any, i: number) => ({
              step_number: i + 1,
              title: sub.name,
              instruction: `Build the ${sub.name} (${sub.type})`,
              parts: sub.parts ?? [],
              part_count: (sub.parts ?? []).reduce((s: number, p: any) => s + (p.quantity ?? 1), 0),
            }));
            setSteps(derived);
          }
          setBuildTitle(build.title);
        } catch (e: any) {
          setError(e.message ?? 'Failed to load build');
        }
      }
      setLoading(false);
    }
    load();
  }, [buildId, location.state]);

  // Webcam
  useEffect(() => {
    let stream: MediaStream | null = null;
    if (!navigator.mediaDevices?.getUserMedia) return;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'environment', width: 640, height: 480 } })
      .then((s) => {
        stream = s;
        setWebcamStream(s);
        if (videoRef.current) videoRef.current.srcObject = s;
      })
      .catch(() => {
        // Webcam not available — not critical
      });
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // Attach stream to video element when ref ready
  useEffect(() => {
    if (videoRef.current && webcamStream) {
      videoRef.current.srcObject = webcamStream;
    }
  }, [webcamStream]);

  const currentStepData = useMemo(() => steps[currentStep - 1] ?? null, [steps, currentStep]);

  const goNext = useCallback(() => {
    setCurrentStep((s) => {
      if (s >= steps.length) {
        setIsComplete(true);
        return s;
      }
      return s + 1;
    });
    setNarratedBrickIdx(-1);
  }, [steps.length]);

  const goPrev = useCallback(() => {
    setCurrentStep((s) => Math.max(1, s - 1));
    setNarratedBrickIdx(-1);
    setIsComplete(false);
  }, []);

  // Voice narration on step change
  useEffect(() => {
    if (!currentStepData) return;
    const narrator = narratorRef.current;
    const text = `Step ${currentStep}: ${currentStepData.title}. ${currentStepData.instruction}`;
    narrator.speak(text);
    return () => narrator.stop();
  }, [currentStep, currentStepData]);

  const handleAutoPlayTick = useCallback(() => {
    if (!currentStepData) return;
    // The useEffect on currentStep already narrates; just advance.
    goNext();
  }, [currentStepData, goNext]);

  const toggleMute = () => {
    const next = !voiceMuted;
    setVoiceMuted(next);
    narratorRef.current.setMuted(next);
  };

  if (loading) {
    return (
      <div className="flex flex-col h-screen bg-gray-950 text-gray-100">
        <Header />
        <div className="flex-grow flex items-center justify-center">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
            <span className="text-sm text-gray-500 ml-2">Loading build...</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col h-screen bg-gray-950 text-gray-100">
        <Header />
        <div className="flex-grow flex items-center justify-center">
          <div className="text-center">
            <p className="text-red-400 font-medium">{error}</p>
            <a href="/explore" className="text-blue-400 text-sm mt-2 inline-block hover:underline">Back to Explore</a>
          </div>
        </div>
      </div>
    );
  }

  if (isComplete) {
    return (
      <div className="flex flex-col h-screen bg-gray-950 text-gray-100">
        <Header />
        <div className="flex-grow flex items-center justify-center">
          <div className="text-center space-y-4 animate-fade-in">
            <div className="text-6xl">🎉</div>
            <h2 className="text-2xl font-bold text-gradient">Build Complete!</h2>
            <p className="text-gray-400">{buildTitle} — {steps.length} steps finished</p>
            <button
              onClick={() => { setIsComplete(false); setCurrentStep(1); }}
              className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-sm font-medium transition"
            >
              Start Over
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-100">
      <Header />

      {/* Title bar */}
      <div className="px-4 py-2 border-b border-white/5 flex items-center justify-between">
        <h1 className="text-sm font-semibold text-gray-300 truncate">{buildTitle}</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={toggleMute}
            className={`p-1.5 rounded-lg text-xs transition ${voiceMuted ? 'bg-red-500/10 text-red-400' : 'bg-white/5 text-gray-400 hover:text-white'}`}
            title={voiceMuted ? 'Unmute voice' : 'Mute voice'}
          >
            {voiceMuted ? '🔇' : '🔊'}
          </button>
          <button
            onClick={() => setExploded((e) => !e)}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium transition ${
              exploded ? 'bg-purple-500/15 text-purple-300 border border-purple-500/20' : 'bg-white/5 text-gray-400 hover:text-white'
            }`}
          >
            {exploded ? 'Compact' : 'Exploded'}
          </button>
        </div>
      </div>

      {/* Split screen */}
      <div className="flex-grow grid grid-cols-1 lg:grid-cols-2 gap-0 overflow-hidden">
        {/* Left: Webcam */}
        <div className="relative bg-black flex items-center justify-center overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />
          {!webcamStream && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
              <div className="text-center text-gray-600">
                <svg className="w-12 h-12 mx-auto mb-2 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z" />
                </svg>
                <p className="text-sm">Camera unavailable</p>
              </div>
            </div>
          )}
        </div>

        {/* Right: 3D Guidance Viewer */}
        <div className="h-full">
          <GuidanceViewer
            steps={steps}
            currentStep={currentStep}
            narratedBrickIdx={narratedBrickIdx}
            exploded={exploded}
          />
        </div>
      </div>

      {/* Bottom bar */}
      <div className="border-t border-white/5 bg-gray-950 px-4 py-3">
        <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Controls */}
          <div className="lg:col-span-2">
            <StepControls
              currentStep={currentStep}
              totalSteps={steps.length}
              onPrev={goPrev}
              onNext={goNext}
              onAutoPlayTick={handleAutoPlayTick}
              instructionText={currentStepData?.instruction ?? ''}
            />
          </div>

          {/* Parts checklist */}
          <div className="max-h-[160px] overflow-y-auto">
            <PartsChecklist
              parts={currentStepData?.parts ?? []}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
