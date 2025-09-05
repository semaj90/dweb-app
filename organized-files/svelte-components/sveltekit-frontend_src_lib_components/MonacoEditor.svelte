<script lang="ts">
import * as monaco from 'monaco-editor';
import { onDestroy, onMount } from 'svelte';
let editorContainer: HTMLDivElement
let editor: monaco.editor.IStandaloneCodeEditor;

onMount(() => {
  // SSR safety: only run in browser
  if (typeof window !== 'undefined') {
    editor = monaco.editor.create(editorContainer, {
      value: '// Type your code here...\n',
      language: 'javascript',
      theme: 'vs-dark',
      automaticLayout: true
    });
}
});

onDestroy(() => {
  if (editor) {
    editor.dispose();
}
});
</script>

<div bind:this={editorContainer} class="space-y-4" aria-label="Monaco code editor" tabindex={0}></div>

<style>
  /* @unocss-include */
/* Ensure the Monaco editor fills the container */
:global(.monaco-editor) {
  border-radius: 0.5rem;
}
</style>
