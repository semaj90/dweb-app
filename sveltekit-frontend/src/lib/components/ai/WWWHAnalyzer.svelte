<script lang="ts">
  // Svelte 5 runes pattern
  let inputText = $state('');
  let result: string | null = $state(null);
  let loading = $state(false);
  let error: string | null = $state(null);

  async function analyzeWWWH() {
    if (!inputText.trim()) return;
    loading = true;
    error = null;
    result = null;
    try {
      const res = await fetch('/api/ai/wwwh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText })
      });
      const data = await res.json();
      if (res.ok) {
        result = data.analysis;
      } else {
        error = data.error || 'Unknown error';
      }
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
    }
  }
</script>

<div class="wwwh-analyzer uno-max-w-2xl uno-mx-auto uno-my-8 uno-p-4 uno-bg-white uno-border uno-border-gray-200 uno-rounded-lg">
  <h3 class="uno-font-bold uno-text-lg uno-mb-2">WWWH (Who, What, When, How) Analyzer</h3>
  <textarea
    bind:value={inputText}
    rows={5}
    placeholder="Paste or type text to analyze..."
    class="uno-w-full uno-p-2 uno-border uno-rounded uno-mb-2"
    aria-label="Text to analyze"
  ></textarea>
  <button
    on:onclick={analyzeWWWH}
    disabled={loading || !inputText.trim()}
    class="uno-bg-primary uno-text-white uno-px-4 uno-py-2 uno-rounded uno-font-semibold uno-shadow-sm uno-transition hover:uno-bg-primary-600 focus-visible:uno-outline focus-visible:uno-outline-2 focus-visible:uno-outline-primary"
    aria-busy={loading}
    aria-label="Analyze text"
  >
    {#if loading}
      Analyzing...
    {:else}
      Analyze
    {/if}
  </button>
  {#if error}
    <div class="uno-text-red-600 uno-mt-2" role="alert">{error}</div>
  {/if}
  {#if result}
    <div class="uno-mt-4 uno-p-3 uno-bg-gray-50 uno-border uno-rounded">
      <pre>{result}</pre>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  /* UnoCSS utility classes used above, no custom CSS needed. */
</style>


