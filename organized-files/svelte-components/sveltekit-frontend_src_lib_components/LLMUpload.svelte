<script lang="ts">
  import { invoke } from '@tauri-apps/api/tauri';
  let uploadResult = '';
  let error = '';
  let loading = false;

  async function handleUpload() {
    uploadResult = '';
    error = '';
    loading = true;
    try {
      // This will open the native file picker and upload the model
      const result = await invoke<string>('upload_llm_model');
      uploadResult = result;
    } catch (e) {
      error = 'Upload failed or cancelled.';
    } finally {
      loading = false;
}}
</script>

<div class="space-y-4">
  <h2>Upload Local LLM Model</h2>
  <button class="space-y-4" onclick={() => handleUpload()} disabled={loading}>
    {loading ? 'Uploading...' : 'Select & Upload Model'}
  </button>
  {#if uploadResult}
    <div class="space-y-4">{uploadResult}</div>
  {/if}
  {#if error}
    <div class="space-y-4">{error}</div>
  {/if}
</div>

<style>
  /* @unocss-include */
.llm-upload-container {
  max-width: 400px;
  margin: 2rem auto;
  padding: 2rem;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 16px rgba(0,0,0,0.08);
  font-family: 'Segoe UI', Arial, sans-serif;
  text-align: center
}
.upload-btn {
  background: #007bff;
  color: #fff;
  border: none
  padding: 0.75rem 1.5rem;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer
  transition: background 0.2s;
}
.upload-btn:disabled {
  background: #b0c4de;
  cursor: not-allowed;
}
.upload-btn:not(:disabled):hover {
  background: #0056b3;
}
.success {
  color: #218838;
  margin-top: 1.5rem;
  font-weight: 600;
}
.error {
  color: #b30000;
  margin-top: 1.5rem;
  font-weight: 600;
}
</style>
