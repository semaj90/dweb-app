
// Coqui TTS browser integration for SvelteKit
// Usage: import { speakWithCoqui } from './coquiTTS';

export async function loadCoquiTTS(): Promise<any> {
  // This function is now a no-op since @coqui-ai/tts is not available.
}

export async function speakWithCoqui(text: string): Promise<any> {
  console.warn("Coqui TTS is not available. Falling back to browser TTS.");
}
