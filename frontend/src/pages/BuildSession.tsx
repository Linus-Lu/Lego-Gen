import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import UploadPanel from '../components/UploadPanel';
import StepList from '../components/StepList';
import StepDetail from '../components/StepDetail';
import ColorLegend from '../components/ColorLegend';
import LegoViewer from '../components/LegoViewer';
import ValidationPanel from '../components/ValidationPanel';
import { generateBuild, generateBuildFromText, createGalleryBuild } from '../api/legogen';
import type { GenerateResponse } from '../api/legogen';

interface Message {
  role: 'user' | 'assistant';
  content: React.ReactNode;
  type?: 'text' | 'manual-result' | 'error';
}

const BuildSession: React.FC = () => {
  const navigate = useNavigate();
  const [input, setInput] = useState('');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: (
        <div className="space-y-2">
          <p>Welcome to <span className="text-gradient font-semibold">LegoGen</span>.</p>
          <p className="text-gray-400">Upload an image of a LEGO set or describe what you'd like to build, and I'll generate step-by-step instructions with a 3D preview.</p>
        </div>
      ),
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [buildResult, setBuildResult] = useState<GenerateResponse | null>(null);
  const [currentStep, setCurrentStep] = useState(1);
  const [activeTab, setActiveTab] = useState<'steps' | 'validation'>('steps');

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, buildResult]);

  const handleSendMessage = async () => {
    if ((!input.trim() && !selectedFile) || isLoading) return;

    const fileToSend = selectedFile;
    const textToSend = input.trim();

    const userMessage: Message = {
      role: 'user',
      content: (
        <div className="flex flex-col gap-2">
          {fileToSend && (
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-300 text-sm w-fit">
              <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.41a2.25 2.25 0 013.182 0l2.909 2.91M3.75 21h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v13.5A1.5 1.5 0 003.75 21z" /></svg>
              {fileToSend.name}
            </div>
          )}
          {textToSend && <span className="whitespace-pre-wrap">{textToSend}</span>}
        </div>
      ),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setSelectedFile(null);
    setIsLoading(true);

    try {
      let result: GenerateResponse;
      if (fileToSend) {
        result = await generateBuild(fileToSend, textToSend || undefined);
      } else {
        result = await generateBuildFromText(textToSend);
      }
      setBuildResult(result);
      setSaved(false);
      setCurrentStep(1);

      const desc = result.description;
      const hasDescription = desc && desc.object;

      const resultMessage: Message = {
        role: 'assistant',
        type: 'manual-result',
        content: hasDescription ? (
          <div className="w-full animate-fade-in space-y-3">
            <p className="font-medium">
              Here's your build plan for{' '}
              <span className="text-gradient-warm font-bold">{desc.object}</span>
            </p>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="px-2.5 py-1 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-300">
                {desc.total_parts ?? 0} parts
              </span>
              <span className="px-2.5 py-1 rounded-lg bg-purple-500/10 border border-purple-500/20 text-purple-300">
                {desc.complexity ?? 'unknown'}
              </span>
              <span className="px-2.5 py-1 rounded-lg bg-green-500/10 border border-green-500/20 text-green-300">
                {result.steps.length} steps
              </span>
              {result.validation && (
                <span className={`px-2.5 py-1 rounded-lg border ${
                  result.validation.score >= 80
                    ? 'bg-green-500/10 border-green-500/20 text-green-300'
                    : result.validation.score >= 50
                    ? 'bg-yellow-500/10 border-yellow-500/20 text-yellow-300'
                    : 'bg-red-500/10 border-red-500/20 text-red-300'
                }`}>
                  Stability: {result.validation.score}/100
                </span>
              )}
              <span className="text-gray-600 ml-1">
                {result.metadata.generation_time_ms}ms
              </span>
            </div>
            {(desc.build_hints?.length ?? 0) > 0 && (
              <div className="text-xs text-gray-500 leading-relaxed">
                {desc.build_hints.join(' · ')}
              </div>
            )}
          </div>
        ) : (
          <div className="w-full animate-fade-in space-y-2">
            <p className="font-medium text-yellow-400">Generation produced an incomplete result</p>
            <p className="text-sm text-gray-400">
              The model output could not be parsed into a valid build description.
              {result.metadata.errors.length > 0 && (
                <span className="block mt-1 text-xs text-gray-500">
                  {result.metadata.errors.slice(0, 3).join('; ')}
                </span>
              )}
            </p>
          </div>
        ),
      };
      setMessages((prev) => [...prev, resultMessage]);
    } catch (err: unknown) {
      const errorText = err instanceof Error ? err.message : 'Failed to generate instructions.';
      const errorMessage: Message = {
        role: 'assistant',
        type: 'error',
        content: (
          <div className="flex items-start gap-3 text-red-400 animate-fade-in">
            <div className="w-8 h-8 rounded-lg bg-red-500/10 flex items-center justify-center flex-shrink-0 mt-0.5">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" /></svg>
            </div>
            <div>
              <p className="font-medium">Something went wrong</p>
              <p className="text-sm mt-1 text-red-400/70">{errorText}</p>
            </div>
          </div>
        ),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-100 font-sans">
      <Header />

      {/* Chat area */}
      <main className="flex-grow overflow-y-auto pb-40">
        <div className="absolute inset-0 bg-mesh pointer-events-none" />
        <div className="relative flex flex-col w-full items-center">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`w-full py-5 px-4 md:px-6 border-b border-white/[0.03] animate-fade-in ${
                msg.role === 'assistant' ? 'bg-white/[0.02]' : ''
              }`}
            >
              <div className="max-w-4xl mx-auto flex gap-4">
                {/* Avatar */}
                <div
                  className={`w-8 h-8 flex-shrink-0 rounded-lg flex items-center justify-center text-sm font-bold shadow-lg ${
                    msg.role === 'assistant'
                      ? 'bg-gradient-to-br from-blue-600 to-purple-600 shadow-blue-600/20'
                      : 'bg-gradient-to-br from-gray-600 to-gray-700 shadow-gray-600/10'
                  }`}
                >
                  {msg.role === 'assistant' ? 'L' : 'U'}
                </div>

                {/* Content */}
                <div className="relative flex-grow min-w-0 text-[15px] leading-7 text-gray-200 pt-0.5">
                  {msg.content}
                </div>
              </div>
            </div>
          ))}

          {/* Build viewer panel */}
          {buildResult && !isLoading && buildResult.steps.length > 0 && (
            <div className="w-full py-4 px-4 md:px-6 animate-scale-in">
              <div className="max-w-6xl mx-auto">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 h-[580px]">
                  {/* 3D Viewer */}
                  <div className="lg:col-span-2 glass rounded-2xl overflow-hidden">
                    <LegoViewer steps={buildResult.steps} currentStep={currentStep} />
                  </div>

                  {/* Side panel */}
                  <div className="flex flex-col gap-2 glass rounded-2xl p-3 overflow-hidden">
                    {/* Tab bar */}
                    <div className="flex gap-1 border-b border-white/5 pb-2 flex-shrink-0" role="tablist" aria-label="Build details">
                      <button
                        role="tab"
                        aria-selected={activeTab === 'steps'}
                        aria-controls="panel-steps"
                        onClick={() => setActiveTab('steps')}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                          activeTab === 'steps'
                            ? 'bg-blue-500/15 text-blue-300 border border-blue-500/20'
                            : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                        }`}
                      >
                        Steps
                      </button>
                      <button
                        role="tab"
                        aria-selected={activeTab === 'validation'}
                        aria-controls="panel-validation"
                        onClick={() => setActiveTab('validation')}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 ${
                          activeTab === 'validation'
                            ? 'bg-blue-500/15 text-blue-300 border border-blue-500/20'
                            : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                        }`}
                      >
                        Validation
                        {buildResult.validation && (
                          <span
                            className={`inline-block w-2 h-2 rounded-full ${
                              buildResult.validation.score >= 80 ? 'bg-green-400' :
                              buildResult.validation.score >= 50 ? 'bg-yellow-400' : 'bg-red-400'
                            }`}
                            aria-label={`Score: ${buildResult.validation.score}`}
                          />
                        )}
                      </button>
                    </div>

                    {activeTab === 'steps' ? (
                      <div id="panel-steps" role="tabpanel" aria-labelledby="tab-steps" className="contents">
                        <div className="flex-grow overflow-y-auto min-h-0">
                          <StepList
                            steps={buildResult.steps}
                            currentStep={currentStep}
                            onStepSelect={setCurrentStep}
                          />
                        </div>
                        <div className="border-t border-white/5 max-h-[200px] overflow-y-auto">
                          <StepDetail step={buildResult.steps[currentStep - 1] ?? null} />
                        </div>
                        <ColorLegend
                          dominantColors={buildResult.description.dominant_colors ?? []}
                          allParts={buildResult.steps.flatMap((s) => s.parts)}
                        />
                      </div>
                    ) : (
                      <div id="panel-validation" role="tabpanel" aria-labelledby="tab-validation" className="flex-grow overflow-y-auto min-h-0">
                        <ValidationPanel validation={buildResult.validation} />
                      </div>
                    )}
                  </div>
                </div>

                {/* Action buttons */}
                <div className="flex gap-2 mt-3">
                  <button
                    onClick={async () => {
                      if (!buildResult || saving) return;
                      setSaving(true);
                      try {
                        // Capture thumbnail from canvas
                        const canvas = document.querySelector('canvas');
                        const thumbnail = canvas ? canvas.toDataURL('image/png').split(',')[1] : '';
                        await createGalleryBuild({
                          title: buildResult.description?.object ?? 'Untitled Build',
                          description_json: JSON.stringify({
                            ...buildResult.description,
                            steps: buildResult.steps,
                          }),
                          thumbnail_b64: thumbnail,
                        });
                        setSaved(true);
                      } catch {
                        // silent
                      } finally {
                        setSaving(false);
                      }
                    }}
                    disabled={saving || saved}
                    className={`px-4 py-2 rounded-xl text-xs font-medium transition ${
                      saved
                        ? 'bg-green-500/15 text-green-300 border border-green-500/20'
                        : 'bg-white/5 hover:bg-white/10 text-gray-300'
                    }`}
                  >
                    {saved ? 'Saved!' : saving ? 'Saving...' : 'Save to Gallery'}
                  </button>
                  <button
                    onClick={() => {
                      navigate('/guide/new', { state: { build: buildResult } });
                    }}
                    className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-xs font-medium text-white transition"
                  >
                    Start Building
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Loading indicator */}
          {isLoading && (
            <div className="w-full py-5 px-4 md:px-6 bg-white/[0.02] border-b border-white/[0.03] animate-fade-in">
              <div className="max-w-4xl mx-auto flex gap-4">
                <div className="w-8 h-8 flex-shrink-0 rounded-lg bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center text-sm font-bold shadow-lg shadow-blue-600/20">
                  L
                </div>
                <div className="flex items-center gap-1.5 pt-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  <span className="text-xs text-gray-500 ml-2">Analyzing and generating instructions...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input area */}
      <div className="fixed bottom-0 left-0 right-0 bg-gradient-to-t from-gray-950 via-gray-950/95 to-transparent pt-12 pb-5 px-4 pointer-events-none">
        <div className="max-w-3xl mx-auto relative pointer-events-auto">
          {/* File preview */}
          {selectedFile && (
            <div className="absolute -top-10 left-0 inline-flex items-center gap-2 px-3 py-1.5 rounded-lg glass text-xs text-gray-200 animate-slide-up">
              <svg className="w-3.5 h-3.5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.41a2.25 2.25 0 013.182 0l2.909 2.91M3.75 21h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v13.5A1.5 1.5 0 003.75 21z" /></svg>
              <span className="max-w-[200px] truncate">{selectedFile.name}</span>
              <button
                onClick={() => setSelectedFile(null)}
                aria-label="Remove selected file"
                className="ml-1 text-gray-400 hover:text-white rounded-full w-4 h-4 flex items-center justify-center hover:bg-white/10 transition"
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </div>
          )}

          <div className="relative flex items-end gap-2 glass rounded-2xl overflow-hidden ring-offset-2 focus-within:ring-2 ring-blue-500/30 transition-shadow shadow-lg shadow-black/20">
            <div className="pl-2 py-2">
              <UploadPanel onFileSelected={setSelectedFile} />
            </div>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe a LEGO model or upload an image..."
              className="w-full max-h-[200px] py-4 pr-12 bg-transparent border-0 focus:ring-0 focus:outline-none text-white placeholder-gray-500 resize-none overflow-y-auto text-[15px]"
              rows={1}
              style={{ height: 'auto', minHeight: '56px' }}
            />

            <button
              onClick={handleSendMessage}
              disabled={isLoading || (!input.trim() && !selectedFile)}
              aria-label="Send message"
              className="absolute right-2 bottom-2 p-2.5 rounded-xl bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-30 disabled:bg-gray-600 disabled:hover:bg-gray-600 transition-all active:scale-95"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
              </svg>
            </button>
          </div>

          <p className="text-center text-[11px] text-gray-600 mt-2">
            LegoGen may produce inaccurate instructions. Verify steps before building.
          </p>
        </div>
      </div>
    </div>
  );
};

export default BuildSession;
