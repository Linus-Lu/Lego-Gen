/**
 * Wrapper around the Web Speech Synthesis API for step narration.
 */
export default class VoiceNarrator {
  private synth: SpeechSynthesis;
  private rate: number = 1.0;
  private muted: boolean = false;

  constructor() {
    this.synth = window.speechSynthesis;
  }

  speak(text: string): Promise<void> {
    return new Promise((resolve) => {
      this.synth.cancel();
      if (this.muted || !text) {
        resolve();
        return;
      }
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = this.rate;
      utterance.onend = () => resolve();
      utterance.onerror = () => resolve();
      this.synth.speak(utterance);
    });
  }

  stop(): void {
    this.synth.cancel();
  }

  setRate(rate: number): void {
    this.rate = rate;
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    if (muted) this.synth.cancel();
  }

  get isMuted(): boolean {
    return this.muted;
  }

  get currentRate(): number {
    return this.rate;
  }
}
