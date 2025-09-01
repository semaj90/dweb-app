<script lang="ts">

  import UploadArea from './UploadArea.svelte';
  let uploadComponent: UploadArea = $state();
  let uploadStatus = $state('');
  let uploadedFiles: any[] = $state([]);
  let showProgress = $state(true);
  let autoUpload = $state(false);
  let maxFiles = $state(5);
  let maxFileSize = $state(10 * 1024 * 1024); // 10MB

  function handleUploadStart(event: CustomEvent) {
    uploadStatus = `Starting upload of ${event.detail.files.length} files...`;
    console.log('Upload started:', event.detail);
}
  function handleUploadProgress(event: CustomEvent) {
    uploadStatus = `Upload progress: ${Math.round(event.detail.progress)}%`;
    console.log('Upload progress:', event.detail);
}
  function handleUploadComplete(event: CustomEvent) {
    uploadStatus = `Successfully uploaded ${event.detail.files.length} files!`;
    uploadedFiles = [...uploadedFiles, ...event.detail.results];
    console.log('Upload completed:', event.detail);
}
  function handleUploadError(event: CustomEvent) {
    uploadStatus = `Upload failed: ${event.detail.error}`;
    console.error('Upload error:', event.detail);
}
  function handleFileStart(event: CustomEvent) {
    console.log('File upload started:', event.detail.file.name);
}
  function handleFileSuccess(event: CustomEvent) {
    console.log('File uploaded successfully:', event.detail.file.name);
}
  function handleFileError(event: CustomEvent) {
    console.error('File upload failed:', event.detail.file.name, event.detail.error);
}
  function handleFilesSelected(event: CustomEvent) {
    console.log('Files selected:', event.detail.files.length);
}
  function handleValidationError(event: CustomEvent) {
    console.warn('Validation errors:', event.detail.errors);
}
  function clearStatus() {
    uploadStatus = '';
    uploadedFiles = [];
}
</script>

<div class="space-y-4">
  <div class="space-y-4">
    <div class="space-y-4">
      <h3>Enhanced UploadArea Component Demo</h3>

      <!-- Configuration Controls -->
      <div class="space-y-4">
        <div class="space-y-4">
          <h5>Configuration Options</h5>
        </div>
        <div class="space-y-4">
          <div class="space-y-4">
            <div class="space-y-4">
              <div class="space-y-4">
                <input
                  class="space-y-4"
                  type="checkbox"
                  id="showProgress"
                  bind:checked={showProgress}
                >
                <label class="space-y-4" for="showProgress">
                  Show Progress
                </label>
              </div>

              <div class="space-y-4">
                <input
                  class="space-y-4"
                  type="checkbox"
                  id="autoUpload"
                  bind:checked={autoUpload}
                >
                <label class="space-y-4" for="autoUpload">
                  Auto Upload
                </label>
              </div>
            </div>

            <div class="space-y-4">
              <div class="space-y-4">
                <label for="maxFiles" class="space-y-4">Max Files:</label>
                <input
                  type="number"
                  class="space-y-4"
                  id="maxFiles"
                  bind:value={maxFiles}
                  min="1"
                  max="20"
                >
              </div>

              <div class="space-y-4">
                <label for="maxSize" class="space-y-4">Max Size (MB):</label>
                <input
                  type="number"
                  class="space-y-4"
                  id="maxSize"
                  input={(e) => {
                    const target = e.target as HTMLInputElement;
                    if (target) {
                      maxFileSize = parseInt(target.value) * 1024 * 1024;
                    }
                  }}
                  value={Math.round(maxFileSize / 1024 / 1024)}
                  min="1"
                  max="100"
                >
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Upload Component -->
      <UploadArea
        bind:this={uploadComponent}
        {maxFiles}
        {maxFileSize}
        {showProgress}
        {autoUpload}
        multiple={true}
        retryAttempts={2}
        uploadEndpoint="/api/upload/"
        acceptedTypes=".pdf,.jpg,.jpeg,.png,.mp4,.avi,.mov,.mp3,.wav"
        allowedMimeTypes={[
          'application/pdf',
          'image/jpeg', 'image/jpg', 'image/png',
          'video/mp4', 'video/avi', 'video/mov',
          'audio/mp3', 'audio/wav', 'audio/mpeg'
        ]}
        upload-start={handleUploadStart}
        upload-progress={handleUploadProgress}
        upload-complete={handleUploadComplete}
        upload-error={handleUploadError}
        file-start={handleFileStart}
        file-success={handleFileSuccess}
        file-error={handleFileError}
        files-selected={handleFilesSelected}
        validation-error={handleValidationError}
      />

      <!-- Status Display -->
      {#if uploadStatus}
        <div class="space-y-4" role="status">
          <i class="space-y-4"></i>
          {uploadStatus}
          <button
            type="button"
            class="space-y-4"
            aria-label="Clear status"
            on:onclick={() => clearStatus()}
          ></button>
        </div>
      {/if}
    </div>

    <div class="space-y-4">
      <div class="space-y-4">
        <div class="space-y-4">
          <h5>Upload Results</h5>
          {#if uploadedFiles.length > 0}
            <button
              type="button"
              class="space-y-4"
              on:onclick={() => clearStatus()}
            >
              Clear
            </button>
          {/if}
        </div>
        <div class="space-y-4">
          {#if uploadedFiles.length === 0}
            <p class="space-y-4">No files uploaded yet.</p>
          {:else}
            <div class="space-y-4">
              {#each uploadedFiles as result, index}
                <div class="space-y-4">
                  <div class="space-y-4">
                    <div>
                      <h6 class="space-y-4">{result.file?.name || `File ${index + 1}`}</h6>
                      <small class="space-y-4">
                        {result.file ? (result.file.size / 1024 / 1024).toFixed(2) + ' MB' : 'Unknown size'}
                      </small>
                    </div>
                    <span class="space-y-4">
                      <i class="space-y-4"></i>
                    </span>
                  </div>
                  {#if result.processingTime}
                    <small class="space-y-4">
                      Processed in {result.processingTime}ms
                    </small>
                  {/if}
                </div>
              {/each}
            </div>
          {/if}
        </div>
      </div>

      <!-- Feature List -->
      <div class="space-y-4">
        <div class="space-y-4">
          <h6>Enhanced Features</h6>
        </div>
        <div class="space-y-4">
          <ul class="space-y-4">
            <li><i class="space-y-4"></i>Drag & Drop with visual feedback</li>
            <li><i class="space-y-4"></i>File validation (size, type, name)</li>
            <li><i class="space-y-4"></i>Progress tracking per file</li>
            <li><i class="space-y-4"></i>Retry mechanism with backoff</li>
            <li><i class="space-y-4"></i>File preview and management</li>
            <li><i class="space-y-4"></i>Error handling and reporting</li>
            <li><i class="space-y-4"></i>Accessibility support</li>
            <li><i class="space-y-4"></i>Security validations</li>
            <li><i class="space-y-4"></i>Responsive design</li>
            <li><i class="space-y-4"></i>Customizable endpoints</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  /* @unocss-include */
  .container {
    max-width: 1200px;
  }
  .alert {
    position: relative;
  }
  .btn-close {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
  }
</style>


