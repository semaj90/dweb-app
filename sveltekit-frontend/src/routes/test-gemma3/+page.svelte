<script lang="ts">
  import { onMount } from "svelte";
  // $state is provided globally via src/types/svelte-helpers.d.ts

  let prompt = $state("Explain the legal concept of due process in criminal law.");
  let response = $state("");
  let isLoading = $state(false);
  let status: any = $state(null);
  let error = $state("");

  async function testGemma3() {
    if (!prompt.trim()) {
      error = "Please enter a prompt";
      return;
    }
    isLoading = true;
    error = "";
    response = "";

    try {
      const res = await fetch("/api/ai/test-gemma3", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt,
          options: {
            temperature: 0.7,
            maxTokens: 512,
          },
        }),
      });

      const data = await res.json();

      if (data.success) {
        response = data.data.response;
        console.log("Inference metadata:", data.data.metadata);
      } else {
        error = data.error;
        if (data.troubleshooting) {
          error += "\n\nTroubleshooting:\n" + data.troubleshooting.join("\n");
        }
      }
    } catch (err) {
      error = `Network error: ${err instanceof Error ? err.message : "Unknown error"}`;
    } finally {
      isLoading = false;
    }
  }

  async function checkStatus() {
    try {
      const res = await fetch("/api/ai/test-gemma3");
      const data = await res.json();
      status = data.status;
    } catch (err) {
      console.error("Status check failed:", err);
    }
  }

  onMount(() => {
    checkStatus();
  });
</script>

<svelte:head>
  <title>Gemma3 Local LLM Test</title>
</svelte:head>

<div class="container">
  <h1>🤖 Gemma3 Local LLM Test</h1>

  <div class="status-section">
    <h2>Status</h2>
    {#if status}
      <div class="status-grid">
        <div class="status-item">
          <strong>Available:</strong>
          <span
            class="status-badge"
            class:available={status.available}
            class:unavailable={!status.available}
          >
            {status.available ? "✅ Ready" : "❌ Not Available"}
          </span>
        </div>

        {#if status.currentModels}
          <div class="status-item">
            <strong>Chat Model:</strong>
            <span>{status.currentModels.chat || "None"}</span>
          </div>
          <div class="status-item">
            <strong>Embedding Model:</strong>
            <span>{status.currentModels.embedding || "None"}</span>
          </div>
        {/if}

        {#if status.models && status.models.length > 0}
          <div class="status-item">
            <strong>Available Models:</strong>
            <ul>
              {#each status.models as model}
                <li>{model.name} ({model.architecture})</li>
              {/each}
            </ul>
          </div>
        {/if}
      </div>
    {:else}
      <p>Loading status...</p>
    {/if}
  </div>

  <div class="test-section">
    <h2>Test Gemma3 Inference</h2>

    <div class="form-group">
      <label for="prompt">Prompt:</label>
      <textarea
        id="prompt"
        bind:value={prompt}
        rows="4"
        placeholder="Enter your legal query here..."
        disabled={isLoading}
      ></textarea>
    </div>

    <button
      onclick={() => testGemma3()}
      disabled={isLoading || !status?.available}
      class="test-button"
    >
      {#if isLoading}
        🔄 Processing...
      {:else}
        🚀 Test Gemma3
      {/if}
    </button>

    {#if error}
      <div class="error">
        <h3>❌ Error</h3>
        <pre>{error}</pre>
      </div>
    {/if}

    {#if response}
      <div class="response">
        <h3>✅ Gemma3 Response</h3>
        <div class="response-content">
          {response}
        </div>
      </div>
    {/if}
  </div>

  <div class="info-section">
    <h2>ℹ️ Setup Information</h2>
    <div class="info-content">
      <h3>Requirements for Local LLM:</h3>
      <ul>
        <li>Desktop application running (Tauri)</li>
        <li>Gemma3 model files downloaded</li>
        <li>Minimum 4GB RAM available</li>
        <li>Rust backend with LLM support</li>
      </ul>

      <h3>Supported Gemma3 Models:</h3>
      <ul>
        <li>Gemma 2B Instruct (Q4_K_M) - Fast, 2GB RAM</li>
        <li>Gemma 7B Instruct (Q4_K_M) - High quality, 6GB RAM</li>
        <li>Gemma 2B Instruct (Q8_0) - Precise, 3GB RAM</li>
      </ul>

      <h3>Model Download Locations:</h3>
      <ul>
        <li><code>/models/gemma-2b-it-q4_k_m.gguf</code></li>
        <li><code>/models/gemma-7b-it-q4_k_m.gguf</code></li>
        <li><code>/models/gemma-2b-it-q8_0.gguf</code></li>
      </ul>
    </div>
  </div>
</div>

<style>
  /* @unocss-include */
  .container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
    font-family:
      -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }

  h1 {
    color: #333;
    border-bottom: 2px solid #007acc;
    padding-bottom: 0.5rem;
  }

  h2 {
    color: #555;
    margin-top: 2rem;
  }

  .status-section,
  .test-section,
  .info-section {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid #e9ecef;
  }

  .status-grid {
    display: grid;
    gap: 1rem;
  }

  .status-item {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .status-badge {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.875rem;
    font-weight: 500;
  }

  .status-badge.available {
    background: #d1f2eb;
    color: #00695c;
  }

  .status-badge.unavailable {
    background: #fadbd8;
    color: #c62828;
  }

  .form-group {
    margin: 1rem 0;
  }

  label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
    color: #555;
  }

  textarea {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-family: inherit;
    resize: vertical;
  }

  .test-button {
    background: #007acc;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    transition: background 0.2s;
  }

  .test-button:hover:not(:disabled) {
    background: #005a9e;
  }

  .test-button:disabled {
    background: #ccc;
    cursor: not-allowed;
  }

  .error {
    background: #fadbd8;
    border: 1px solid #f5b7b1;
    border-radius: 4px;
    padding: 1rem;
    margin: 1rem 0;
  }

  .error h3 {
    margin: 0 0 0.5rem 0;
    color: #c62828;
  }

  .error pre {
    margin: 0;
    white-space: pre-wrap;
    font-size: 0.875rem;
  }

  .response {
    background: #d1f2eb;
    border: 1px solid #a9dfbf;
    border-radius: 4px;
    padding: 1rem;
    margin: 1rem 0;
  }

  .response h3 {
    margin: 0 0 0.5rem 0;
    color: #00695c;
  }

  .response-content {
    background: white;
    padding: 1rem;
    border-radius: 4px;
    white-space: pre-wrap;
    line-height: 1.5;
  }

  .info-content ul {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
  }

  .info-content li {
    margin: 0.25rem 0;
  }

  .info-content code {
    background: #f1f3f4;
    padding: 0.125rem 0.25rem;
    border-radius: 2px;
    font-family: monospace;
    font-size: 0.875rem;
  }
</style>
