import React, { useState, useRef, useEffect, useMemo } from 'react';
import Header from '../components/Header';
import UploadPanel from '../components/UploadPanel';
import BrickCoordViewer from '../components/BrickCoordViewer';
import { generateBricks, parseBrickString, bricksToSteps } from '../api/legogen';
import type { BrickResponse, BrickCoord } from '../api/legogen';

interface Message {
  role: 'user' | 'assistant';
  content: React.ReactNode;
  type?: 'text' | 'brick-result' | 'error';
}

const BuildSession: React.FC = () => {
  const [input, setInput] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: (
        <div className="space-y-2">
          <p>Welcome to <span className="text-gradient font-semibold">LegoGen</span>.</p>
          <p className="text-gray-400">Upload an image or describe what you'd like to build. I'll generate a 3D LEGO model with brick coordinates.</p>
        </div>
      ),
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [brickResult, setBrickResult] = useState<BrickResponse | null>(null);
  const [bricks, setBricks] = useState<BrickCoord[]>([]);
  const [currentStep, setCurrentStep] = useState(1);

  const { steps, zLevels } = useMemo(
    () => bricksToSteps(bricks),
    [bricks],
  );

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
      const result = await generateBricks(fileToSend || undefined, textToSend || undefined);
      setBrickResult(result);
      const parsed = parseBrickString(result.bricks);
      setBricks(parsed);
      setCurrentStep(1);

      // Compute layers from parsed bricks directly to avoid stale useMemo state
      const { steps: freshSteps } = bricksToSteps(parsed);
      const layerCount = freshSteps.length;

      const resultMessage: Message = {
        role: 'assistant',
        type: 'brick-result',
        content: (
          <div className="w-full animate-fade-in space-y-3">
            <p className="font-medium">
              Built a LEGO model: <span className="text-gradient-warm font-bold">{result.caption}</span>
            </p>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="px-2.5 py-1 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-300">
                {result.brick_count} bricks
              </span>
              <span className={`px-2.5 py-1 rounded-lg border ${
                result.stable
                  ? 'bg-green-500/10 border-green-500/20 text-green-300'
                  : 'bg-yellow-500/10 border-yellow-500/20 text-yellow-300'
              }`}>
                {result.stable ? 'Stable' : 'Unstable'}
              </span>
              <span className="px-2.5 py-1 rounded-lg bg-purple-500/10 border border-purple-500/20 text-purple-300">
                {layerCount} layers
              </span>
              <span className="text-gray-600 ml-1">
                {result.metadata.generation_time_ms}ms
              </span>
            </div>
          </div>
        ),
      };
      setMessages((prev) => [...prev, resultMessage]);
    } catch (err: unknown) {
      const errorText = err instanceof Error ? err.message : 'Failed to generate model.';
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
                <div
                  className={`w-8 h-8 flex-shrink-0 rounded-lg flex items-center justify-center text-sm font-bold shadow-lg ${
                    msg.role === 'assistant'
                      ? 'bg-gradient-to-br from-blue-600 to-purple-600 shadow-blue-600/20'
                      : 'bg-gradient-to-br from-gray-600 to-gray-700 shadow-gray-600/10'
                  }`}
                >
                  {msg.role === 'assistant' ? 'L' : 'U'}
                </div>
                <div className="relative flex-grow min-w-0 text-[15px] leading-7 text-gray-200 pt-0.5">
                  {msg.content}
                </div>
              </div>
            </div>
          ))}

          {/* 3D Brick Viewer */}
          {bricks.length > 0 && !isLoading && (
            <div className="w-full py-4 px-4 md:px-6 animate-scale-in">
              <div className="max-w-6xl mx-auto">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 h-[580px]">
                  <div className="lg:col-span-2 glass rounded-2xl overflow-hidden">
                    <BrickCoordViewer bricks={bricks} zLevels={zLevels} currentStep={currentStep} />
                  </div>

                  {/* Layer steps sidebar */}
                  <div className="flex flex-col gap-2 glass rounded-2xl p-3 overflow-hidden">
                    <h3 className="text-sm font-medium text-gray-300 px-2">Build Layers</h3>
                    <div className="flex-grow overflow-y-auto min-h-0">
                      {steps.map((step) => (
                        <button
                          key={step.step_number}
                          onClick={() => setCurrentStep(step.step_number)}
                          className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                            currentStep === step.step_number
                              ? 'bg-blue-500/15 text-blue-300 border border-blue-500/20'
                              : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                          }`}
                        >
                          <div className="font-medium">{step.title}</div>
                          <div className="text-xs opacity-70">{step.instruction}</div>
                        </button>
                      ))}
                    </div>

                    {brickResult && (
                      <div className="border-t border-white/5 pt-2 px-2 text-xs text-gray-500">
                        <div>Rejections: {brickResult.metadata.rejections}</div>
                        <div>Rollbacks: {brickResult.metadata.rollbacks}</div>
                      </div>
                    )}
                  </div>
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
                  <span className="text-xs text-gray-500 ml-2">Generating LEGO model...</span>
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
            LegoGen may produce inaccurate models. Verify before building.
          </p>
        </div>
      </div>
    </div>
  );
};

export default BuildSession;
