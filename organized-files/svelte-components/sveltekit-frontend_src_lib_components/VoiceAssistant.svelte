<script lang="ts">
  import { onMount } from 'svelte';
  import { speak } from "./speak";

  let isSupported = false;
  let isListening = false;
  let finalTranscript = '';
  let interimTranscript = '';
  let currentTranscript = '';
  let recognition: any

  onMount(() => {
    // Initialize speech recognition
    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
    if (SpeechRecognition) {
      isSupported = true;
      recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognition.maxAlternatives = 1;
      recognition.onstart = () => {
        isListening = true;
        speak("I'm listening. You can ask me legal questions or give voice commands.");
      };
      recognition.onresult = (event: any) => {
        let interim = '';
        let final = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            final += transcript;
          } else {
            interim += transcript;
          }
        }
        finalTranscript = final;
        interimTranscript = interim;
        currentTranscript = final + interim;
      };
      recognition.onend = () => {
        isListening = false;
        if (finalTranscript === '') {
          speak('No speech detected. Please try again.');
        }
      };
      recognition.onerror = (event: any) => {
        isListening = false;
        if (event.error === 'no-speech') {
          speak('No speech detected. Please try again.');
        } else if (event.error === 'audio-capture') {
          speak('No microphone access. Please check your microphone settings.');
        } else if (event.error === 'not-allowed') {
          speak('Permission to use microphone denied. Please enable microphone access.');
        } else {
          speak('Error occurred in recognition: ' + event.error);
        }
      };
    } else {
      isSupported = false;
    }
  });
</script>

{#if isSupported}
  <div>
    {#if isListening}
      <p>Listening...</p>
    {:else}
      <p>Click the button and start speaking.</p>
    {/if}
    <button onclick={() => {
      if (isListening) {
        recognition.stop();
        isListening = false;
      } else {
        recognition.start();
      }
    }}>
      {#if isListening}
        Stop Listening
      {:else}
        Start Listening
      {/if}
    </button>
    <p>Final Transcript: {finalTranscript}</p>
    <p>Interim Transcript: {interimTranscript}</p>
  </div>
{:else}
  <p>Speech recognition is not supported in this browser.</p>
{/if}

<style>
  /* @unocss-include */
  /* Add your styles here */
</style>
