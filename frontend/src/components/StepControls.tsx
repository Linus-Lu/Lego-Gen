import { useState, useEffect, useRef, useCallback } from 'react';

interface StepControlsProps {
  currentStep: number;
  totalSteps: number;
  onPrev: () => void;
  onNext: () => void;
  onAutoPlayTick?: () => void;
  instructionText: string;
  /** Called when play/pause is toggled */
  onPlayStateChange?: (playing: boolean) => void;
}

export default function StepControls({
  currentStep,
  totalSteps,
  onPrev,
  onNext,
  onAutoPlayTick,
  instructionText,
  onPlayStateChange,
}: StepControlsProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [stepElapsed, setStepElapsed] = useState(0);
  const [totalElapsed, setTotalElapsed] = useState(0);
  const [autoPlayDelay] = useState(3000); // ms between auto steps
  const autoPlayTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Total elapsed timer
  useEffect(() => {
    intervalRef.current = setInterval(() => {
      setTotalElapsed((t) => t + 1);
    }, 1000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  // Reset step timer on step change
  useEffect(() => {
    setStepElapsed(0);
    const id = setInterval(() => setStepElapsed((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [currentStep]);

  // Auto-play logic
  const scheduleNext = useCallback(() => {
    if (autoPlayTimer.current) clearTimeout(autoPlayTimer.current);
    autoPlayTimer.current = setTimeout(() => {
      if (onAutoPlayTick) {
        onAutoPlayTick();
      } else {
        onNext();
      }
    }, autoPlayDelay);
  }, [autoPlayDelay, onAutoPlayTick, onNext]);

  useEffect(() => {
    if (isPlaying && currentStep < totalSteps) {
      scheduleNext();
    }
    if (currentStep >= totalSteps && isPlaying) {
      setIsPlaying(false);
      onPlayStateChange?.(false);
    }
    return () => {
      if (autoPlayTimer.current) clearTimeout(autoPlayTimer.current);
    };
  }, [isPlaying, currentStep, totalSteps, scheduleNext, onPlayStateChange]);

  const togglePlay = () => {
    const next = !isPlaying;
    setIsPlaying(next);
    onPlayStateChange?.(next);
  };

  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const progress = totalSteps > 1 ? ((currentStep - 1) / (totalSteps - 1)) * 100 : 100;

  return (
    <div className="space-y-3">
      {/* Progress bar */}
      <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Controls row */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <button
            onClick={onPrev}
            disabled={currentStep <= 1}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 disabled:opacity-30 transition"
            aria-label="Previous step"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
            </svg>
          </button>

          <button
            onClick={togglePlay}
            className="p-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 transition active:scale-95"
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="4" width="4" height="16" rx="1" />
                <rect x="14" y="4" width="4" height="16" rx="1" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5.14v13.72a1 1 0 001.5.86l11.04-6.86a1 1 0 000-1.72L9.5 4.28A1 1 0 008 5.14z" />
              </svg>
            )}
          </button>

          <button
            onClick={onNext}
            disabled={currentStep >= totalSteps}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 disabled:opacity-30 transition"
            aria-label="Next step"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
            </svg>
          </button>
        </div>

        <div className="text-xs text-gray-400 font-mono">
          Step {currentStep}/{totalSteps}
        </div>

        <div className="flex items-center gap-3 text-xs text-gray-500 font-mono">
          <span title="Step time">{fmt(stepElapsed)}</span>
          <span className="text-gray-700">|</span>
          <span title="Total time">{fmt(totalElapsed)}</span>
        </div>
      </div>

      {/* Instruction text */}
      {instructionText && (
        <p className="text-sm text-gray-300 leading-relaxed">{instructionText}</p>
      )}

      {/* Auto-play progress bar */}
      {isPlaying && currentStep < totalSteps && (
        <div className="w-full h-0.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-400/50 rounded-full"
            style={{
              animation: `autoplay-progress ${autoPlayDelay}ms linear`,
              width: '100%',
            }}
          />
        </div>
      )}

      <style>{`
        @keyframes autoplay-progress {
          from { width: 0%; }
          to { width: 100%; }
        }
      `}</style>
    </div>
  );
}
