<script lang="ts">
    let inputText = '';
  let result: string | null = null;
  let loading = false;
  let error: string | null = null;

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

<div class="wwwh-analyzer">
  <h3>WWWH (Who, What, When, How) Analyzer</h3>
  <textarea bind:value={inputText} rows={5} placeholder="Paste or type text to analyze..." class="w-full p-2 border rounded mb-2"></textarea>
  <button on:click={analyzeWWWH} disabled={loading || !inputText.trim()} class="nier-button-primary">
    {#if loading}
      Analyzing...
    {:else}
      Analyze
    {/if}
  </button>
  {#if error}
    <div class="text-red-600 mt-2">{error}</div>
  {/if}
  {#if result}
    <div class="mt-4 p-3 bg-gray-50 border rounded">
      <pre>{result}</pre>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
.wwwh-analyzer {
  max-width: 600px;
  margin: 2rem auto;
  padding: 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  background: #fff;
}
.btn.btn-primary {
  background: #6366f1;
  color: #fff;
  padding: 0.5rem 1.5rem;
  border-radius: 0.375rem;
  border: none;
  cursor: pointer;
  font-weight: 600;
}
.btn.btn-primary:disabled {
  background: #a5b4fc;
  cursor: not-allowed;
}
</style>
