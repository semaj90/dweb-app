<script lang="ts">
  import UploadArea from './UploadArea.svelte';
  
  let uploadComponent: UploadArea;
  let uploadStatus = '';
  let uploadedFiles: unknown[] = [];
  let showProgress = true;
  let autoUpload = false;
  let maxFiles = 5;
  let maxFileSize = 10 * 1024 * 1024; // 10MB
  
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

<div class="mx-auto px-4 max-w-7xl">
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">
      <h3>Enhanced UploadArea Component Demo</h3>
      
      <!-- Configuration Controls -->
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <h5>Configuration Options</h5>
        </div>
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl">
                <input 
                  class="mx-auto px-4 max-w-7xl" 
                  type="checkbox" 
                  id="showProgress" 
                  bind:checked={showProgress}
                >
                <label class="mx-auto px-4 max-w-7xl" for="showProgress">
                  Show Progress
                </label>
              </div>
              
              <div class="mx-auto px-4 max-w-7xl">
                <input 
                  class="mx-auto px-4 max-w-7xl" 
                  type="checkbox" 
                  id="autoUpload" 
                  bind:checked={autoUpload}
                >
                <label class="mx-auto px-4 max-w-7xl" for="autoUpload">
                  Auto Upload
                </label>
              </div>
            </div>
            
            <div class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl">
                <label for="maxFiles" class="mx-auto px-4 max-w-7xl">Max Files:</label>
                <input 
                  type="number" 
                  class="mx-auto px-4 max-w-7xl" 
                  id="maxFiles" 
                  bind:value={maxFiles} 
                  min="1" 
                  max="20"
                >
              </div>
              
              <div class="mx-auto px-4 max-w-7xl">
                <label for="maxSize" class="mx-auto px-4 max-w-7xl">Max Size (MB):</label>
                <input 
                  type="number" 
                  class="mx-auto px-4 max-w-7xl" 
                  id="maxSize" 
                  oninput={(e) => {
                    const target = e.target as HTMLInputElement;
                    if (target) {
                      maxFileSize = parseInt(target.value) * 1024 * 1024;
                    }
                  "
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
        on:upload-start={handleUploadStart}
        on:upload-progress={handleUploadProgress}
        on:upload-complete={handleUploadComplete}
        on:upload-error={handleUploadError}
        on:file-start={handleFileStart}
        on:file-success={handleFileSuccess}
        on:file-error={handleFileError}
        on:files-selected={handleFilesSelected}
        on:validation-error={handleValidationError}
      />
      
      <!-- Status Display -->
      {#if uploadStatus}
        <div class="mx-auto px-4 max-w-7xl" role="status">
          <i class="mx-auto px-4 max-w-7xl"></i>
          {uploadStatus}
          <button 
            type="button" 
            class="mx-auto px-4 max-w-7xl" 
            aria-label="Clear status"
            onclick={() => clearStatus()}
          ></button>
        </div>
      {/if}
    </div>
    
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <h5>Upload Results</h5>
          {#if uploadedFiles.length > 0}
            <button 
              type="button" 
              class="mx-auto px-4 max-w-7xl"
              onclick={() => clearStatus()}
            >
              Clear
            </button>
          {/if}
        </div>
        <div class="mx-auto px-4 max-w-7xl">
          {#if uploadedFiles.length === 0}
            <p class="mx-auto px-4 max-w-7xl">No files uploaded yet.</p>
          {:else}
            <div class="mx-auto px-4 max-w-7xl">
              {#each uploadedFiles as result, index}
                <div class="mx-auto px-4 max-w-7xl">
                  <div class="mx-auto px-4 max-w-7xl">
                    <div>
                      <h6 class="mx-auto px-4 max-w-7xl">{result.file?.name || `File ${index + 1}`}</h6>
                      <small class="mx-auto px-4 max-w-7xl">
                        {result.file ? (result.file.size / 1024 / 1024).toFixed(2) + ' MB' : 'Unknown size'}
                      </small>
                    </div>
                    <span class="mx-auto px-4 max-w-7xl">
                      <i class="mx-auto px-4 max-w-7xl"></i>
                    </span>
                  </div>
                  {#if result.processingTime}
                    <small class="mx-auto px-4 max-w-7xl">
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
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <h6>Enhanced Features</h6>
        </div>
        <div class="mx-auto px-4 max-w-7xl">
          <ul class="mx-auto px-4 max-w-7xl">
            <li><i class="mx-auto px-4 max-w-7xl"></i>Drag & Drop with visual feedback</li>
            <li><i class="mx-auto px-4 max-w-7xl"></i>File validation (size, type, name)</li>
            <li><i class="mx-auto px-4 max-w-7xl"></i>Progress tracking per file</li>
            <li><i class="mx-auto px-4 max-w-7xl"></i>Retry mechanism with backoff</li>
            <li><i class="mx-auto px-4 max-w-7xl"></i>File preview and management</li>
            <li><i class="mx-auto px-4 max-w-7xl"></i>Error handling and reporting</li>
            <li><i class="mx-auto px-4 max-w-7xl"></i>Accessibility support</li>
            <li><i class="mx-auto px-4 max-w-7xl"></i>Security validations</li>
            <li><i class="mx-auto px-4 max-w-7xl"></i>Responsive design</li>
            <li><i class="mx-auto px-4 max-w-7xl"></i>Customizable endpoints</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
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