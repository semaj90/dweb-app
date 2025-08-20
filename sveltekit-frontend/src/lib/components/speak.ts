import type { CommonProps } from '$lib/types/common-props';
// @ts-nocheck
// Simple speech synthesis helper for VoiceAssistant.svelte
export function speak(text: string) {
  if ("speechSynthesis" in window) {
    const utterance = new window.SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  }
}
