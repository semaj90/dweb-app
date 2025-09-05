<script lang="ts">
  import { onMount } from 'svelte';
  import { getAvailableModels, runInference } from "$lib/llm/tauri-llm";

  let models: string[] = [];
  let selectedModel = '';
  let prompt = '';
  let result = '';
  let loading = false;
  let error = '';

  onMount(async () => {
    try {
      models = await getAvailableModels();
      if (models.length > 0) selectedModel = models[0];
    } catch (e) {
      error = 'Failed to load models.';
    }
  });

  async function handleInference() {
    if (!selectedModel || !prompt.trim()) return;
    loading = true;
    error = '';
    result = '';
    try {
      result = await runInference(selectedModel, prompt);
    } catch (e) {
      error = 'Inference failed.';
    } finally {
      loading = false;
    }
  }
</script>

<div class="mx-auto px-4 max-w-7xl">
  <h2>Local LLM Inference (Tauri Desktop)</h2>
  <div class="mx-auto px-4 max-w-7xl">
    <label for="model">Model:</label>
    <select id="model" bind:value={selectedModel}>
      {#each models as model}
        <option value={model}>{model}</option>
      {/each}
    </select>
  </div>
  <div class="mx-auto px-4 max-w-7xl">
    <label for="prompt">Prompt:</label>
    <textarea id="prompt" rows="4" bind:value={prompt} placeholder="Enter your prompt..."></textarea>
  </div>
  <button class="mx-auto px-4 max-w-7xl" onclick={() => handleInference()} disabled={loading || !selectedModel || !prompt.trim()}>
    {loading ? 'Running...' : 'Run Inference'}
  </button>
  {#if error}
    <div class="mx-auto px-4 max-w-7xl">{error}</div>
  {/if}
  {#if result}
    <div class="mx-auto px-4 max-w-7xl">
      <h3>Result:</h3>
      <pre>{result}</pre>
    </div>
  {/if}
</div>

<style>
.llm-inference-container {
  max-width: 500px;
  margin: 2rem auto;
  padding: 2rem;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 16px rgba(0,0,0,0.08);
  font-family: 'Segoe UI', Arial, sans-serif;
}
.model-select, .prompt-input {
  margin-bottom: 1.5rem;
}
label {
  font-weight: 600;
  display: block;
  margin-bottom: 0.5rem;
}
select, textarea {
  width: 100%;
  padding: 0.75rem;
  border-radius: 6px;
  border: 1px solid #ccc;
  font-size: 1rem;
  margin-bottom: 0.5rem;
}
.run-btn {
  background: #007bff;
  color: #fff;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}
.run-btn:disabled {
  background: #b0c4de;
  cursor: not-allowed;
}
.run-btn:not(:disabled):hover {
  background: #0056b3;
}
.result {
  margin-top: 2rem;
  background: #f8f9fa;
  border-radius: 6px;
  padding: 1rem;
  font-size: 1rem;
}
.error {
  color: #b30000;
  margin-top: 1rem;
  font-weight: 600;
}
</style>
