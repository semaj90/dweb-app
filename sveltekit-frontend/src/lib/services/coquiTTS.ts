// Coqui TTS browser integration for SvelteKit
// Usage: import { speakWithCoqui } from './coquiTTS';

let tts: any = null;
let ttsReady = false;
let ttsLoading = false;

export async function loadCoquiTTS() {
  if (ttsReady || ttsLoading) return;
  ttsLoading = true;
  // @ts-ignore
  const { TTS } = await import("@coqui-ai/tts");
  tts = new TTS();
  await tts.init();
  ttsReady = true;
  ttsLoading = false;
}

export async function speakWithCoqui(text: string) {
  if (!ttsReady) await loadCoquiTTS();
  if (!tts) return;
  try {
    const audio = await tts.speak(text);
    // Play the audio buffer
    const context = new (window.AudioContext ||
      (window as any).webkitAudioContext)();
    const source = context.createBufferSource();
    source.buffer = audio;
    source.connect(context.destination);
    source.start(0);
  } catch (e) {
    console.warn("Coqui TTS failed, falling back to browser TTS.", e);
    // Fallback handled in component
  }
}
