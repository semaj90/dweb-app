// @ts-nocheck
// Coqui TTS browser integration for SvelteKit
// Usage: import { speakWithCoqui } from './coquiTTS';

export async function loadCoquiTTS() {
  // This function is now a no-op since @coqui-ai/tts is not available.
}

export async function speakWithCoqui(text: string) {
  console.warn("Coqui TTS is not available. Falling back to browser TTS.");
}
