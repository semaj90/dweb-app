<script lang="ts">
  import { onMount } from 'svelte';

  let pipelineStatus = $state<any>(null);
  let loading = $state(true);
  let uploadLoading = $state(false);
  let uploadResult = $state<any>(null);
  let uploadError = $state<string | null>(null);

  // Upload options
  let enableGpu = $state(true);
  let useTensorCores = $state(true);
  let quantization = $state<'4bit' | '8bit' | 'fp16' | 'fp32'>('4bit');
  let negativeLatentSpace = $state(true);
  let extractEmbeddings = $state(true);
  let processingPriority = $state<'low' | 'normal' | 'high'>('high');

  let fileInput: HTMLInputElement;
  let selectedFile: File | null = null;

  onMount(async () => {
    await loadPipelineStatus();
    setInterval(loadPipelineStatus, 10000); // Refresh every 10 seconds
  });

  async function loadPipelineStatus() {
    try {
      const response = await fetch('/api/upload/gpu-process');
      pipelineStatus = await response.json();
    } catch (error) {
      console.error('Failed to load pipeline status:', error);
      pipelineStatus = { pipeline_status: 'error', error: 'Failed to connect to pipeline' };
    } finally {
      loading = false;
    }
  }

  function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    selectedFile = target.files?.[0] || null;
    uploadResult = null;
    uploadError = null;
  }

  async function handleUpload() {
    if (!selectedFile) {
      uploadError = 'Please select a file first';
      return;
    }

    if (!pipelineStatus?.services?.upload_service?.available) {
      uploadError = 'Upload service is not available';
      return;
    }

    uploadLoading = true;
    uploadResult = null;
    uploadError = null;

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('enable_gpu', String(enableGpu));
      formData.append('use_tensor_cores', String(useTensorCores));
      formData.append('quantization', quantization);
      formData.append('negative_latent_space', String(negativeLatentSpace));
      formData.append('extract_embeddings', String(extractEmbeddings));
      formData.append('processing_priority', processingPriority);

      const response = await fetch('/api/upload/gpu-process', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (response.ok) {
        uploadResult = result;
      } else {
        uploadError = result.error || 'Upload failed';
      }

    } catch (error) {
      uploadError = error instanceof Error ? error.message : 'Upload failed';
    } finally {
      uploadLoading = false;
    }
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
</script>

<div class="min-h-screen bg-nier-bg-primary text-nier-text-primary p-golden-lg">
  <div class="container mx-auto max-w-6xl">
    <!-- Header -->
    <header class="mb-golden-xl">
      <h1 class="text-4xl font-bold text-nier-accent-warm mb-golden-md uppercase tracking-wider">
        📄 Document Upload with GPU Processing
      </h1>
      <p class="text-nier-text-secondary mb-golden-md">
        Upload legal documents for RTX tensor core processing with 4-bit quantization and negative latent space analysis
      </p>
      
      {#if loading}
        <div class="bg-yellow-500/20 text-yellow-400 border-yellow-500/30 border px-golden-md py-golden-sm rounded">
          🔄 Loading pipeline status...
        </div>
      {:else if pipelineStatus?.pipeline_status === 'healthy'}
        <div class="bg-green-500/20 text-green-400 border-green-500/30 border px-golden-md py-golden-sm rounded">
          ✅ Upload Pipeline Operational - GPU Processing Available
        </div>
      {:else}
        <div class="bg-red-500/20 text-red-400 border-red-500/30 border px-golden-md py-golden-sm rounded">
          ❌ Upload Pipeline Issues Detected
        </div>
      {/if}
    </header>

    <div class="grid grid-cols-1 xl:grid-cols-3 gap-golden-lg">
      <!-- Pipeline Status Panel -->
      <div class="xl:col-span-1">
        <div class="bg-nier-bg-secondary border border-nier-border-primary rounded p-golden-lg">
          <h2 class="text-xl font-bold text-nier-accent-warm mb-golden-md uppercase">
            🔧 Pipeline Status
          </h2>
          
          {#if pipelineStatus}
            <div class="space-y-golden-sm">
              {#if pipelineStatus.services}
                <div class="space-y-golden-xs">
                  <h3 class="font-bold text-nier-accent-cool text-sm uppercase">Services</h3>
                  
                  <div class="flex justify-between items-center">
                    <span class="text-nier-text-secondary text-sm">Upload Service:</span>
                    <span class="text-xs px-2 py-1 rounded font-mono {pipelineStatus.services.upload_service.available ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}">
                      {pipelineStatus.services.upload_service.available ? 'ONLINE' : 'OFFLINE'}
                    </span>
                  </div>
                  
                  <div class="flex justify-between items-center">
                    <span class="text-nier-text-secondary text-sm">CUDA Service:</span>
                    <span class="text-xs px-2 py-1 rounded font-mono {pipelineStatus.services.cuda_service.available ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}">
                      {pipelineStatus.services.cuda_service.available ? 'ONLINE' : 'OFFLINE'}
                    </span>
                  </div>
                </div>

                {#if pipelineStatus.features}
                  <div class="space-y-golden-xs pt-golden-sm border-t border-nier-border-muted">
                    <h3 class="font-bold text-nier-accent-cool text-sm uppercase">Features</h3>
                    
                    {#each Object.entries(pipelineStatus.features) as [feature, available]}
                      {#if typeof available === 'boolean'}
                        <div class="flex justify-between items-center">
                          <span class="text-nier-text-secondary text-xs">{feature.replace(/_/g, ' ')}:</span>
                          <span class="text-xs {available ? 'text-green-400' : 'text-red-400'}">
                            {available ? '✅' : '❌'}
                          </span>
                        </div>
                      {/if}
                    {/each}
                  </div>
                {/if}
              {/if}
            </div>
          {/if}
        </div>
      </div>

      <!-- Upload Configuration & Form -->
      <div class="xl:col-span-2">
        <div class="bg-nier-bg-secondary border border-nier-border-primary rounded p-golden-lg">
          <h2 class="text-xl font-bold text-nier-accent-warm mb-golden-md uppercase">
            ⚡ GPU-Accelerated Upload
          </h2>
          
          <!-- File Selection -->
          <div class="mb-golden-md">
            <label class="block text-nier-text-secondary mb-golden-sm font-bold text-sm uppercase">
              Select Document
            </label>
            <input
              type="file"
              bind:this={fileInput}
              on:change={handleFileSelect}
              accept=".pdf,.doc,.docx,.txt"
              class="w-full bg-nier-bg-tertiary border border-nier-border-muted rounded p-golden-md text-nier-text-primary"
            />
            {#if selectedFile}
              <p class="text-nier-text-secondary text-sm mt-golden-xs">
                📄 {selectedFile.name} ({formatFileSize(selectedFile.size)})
              </p>
            {/if}
          </div>

          <!-- GPU Processing Options -->
          <div class="mb-golden-lg">
            <h3 class="text-lg font-bold text-nier-accent-cool mb-golden-sm uppercase">
              🎯 Processing Options
            </h3>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-golden-md">
              <div class="space-y-golden-sm">
                <label class="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    bind:checked={enableGpu}
                    class="accent-nier-accent-warm"
                    disabled={!pipelineStatus?.features?.gpu_processing}
                  />
                  <span class="text-sm">Enable GPU Processing</span>
                </label>
                
                <label class="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    bind:checked={useTensorCores}
                    class="accent-nier-accent-warm"
                    disabled={!enableGpu || !pipelineStatus?.features?.tensor_cores}
                  />
                  <span class="text-sm">Use RTX Tensor Cores</span>
                </label>
                
                <label class="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    bind:checked={negativeLatentSpace}
                    class="accent-nier-accent-warm"
                    disabled={!enableGpu}
                  />
                  <span class="text-sm">Negative Latent Space</span>
                </label>
                
                <label class="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    bind:checked={extractEmbeddings}
                    class="accent-nier-accent-warm"
                    disabled={!pipelineStatus?.features?.embedding_extraction}
                  />
                  <span class="text-sm">Extract Embeddings</span>
                </label>
              </div>
              
              <div class="space-y-golden-sm">
                <div>
                  <label class="block text-nier-text-secondary text-sm mb-1">Quantization:</label>
                  <select
                    bind:value={quantization}
                    disabled={!enableGpu}
                    class="w-full bg-nier-bg-tertiary border border-nier-border-muted rounded p-2 text-nier-text-primary text-sm"
                  >
                    <option value="4bit">4-bit (75% memory reduction)</option>
                    <option value="8bit">8-bit (50% memory reduction)</option>
                    <option value="fp16">FP16 (Half precision)</option>
                    <option value="fp32">FP32 (Full precision)</option>
                  </select>
                </div>
                
                <div>
                  <label class="block text-nier-text-secondary text-sm mb-1">Priority:</label>
                  <select
                    bind:value={processingPriority}
                    class="w-full bg-nier-bg-tertiary border border-nier-border-muted rounded p-2 text-nier-text-primary text-sm"
                  >
                    <option value="low">Low Priority</option>
                    <option value="normal">Normal Priority</option>
                    <option value="high">High Priority</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          <!-- Upload Button -->
          <button
            onclick={handleUpload}
            disabled={!selectedFile || uploadLoading || pipelineStatus?.pipeline_status !== 'healthy'}
            class="w-full bg-gradient-to-r from-nier-accent-warm to-nier-accent-cool text-nier-bg-primary font-bold px-golden-lg py-golden-md rounded uppercase tracking-wide hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {#if uploadLoading}
              🔄 Processing with RTX Tensor Cores...
            {:else}
              ⚡ Upload & Process Document
            {/if}
          </button>
        </div>
      </div>
    </div>

    <!-- Results Panel -->
    {#if uploadResult || uploadError}
      <div class="mt-golden-lg bg-nier-bg-secondary border border-nier-border-primary rounded p-golden-lg">
        <h2 class="text-xl font-bold text-nier-accent-warm mb-golden-md uppercase">
          📋 Processing Results
        </h2>
        
        {#if uploadError}
          <div class="bg-red-500/20 text-red-400 border-red-500/30 border rounded p-golden-md">
            <strong>❌ Error:</strong> {uploadError}
          </div>
        {:else if uploadResult}
          <div class="space-y-golden-md">
            <!-- Upload Results -->
            {#if uploadResult.upload}
              <div class="bg-nier-bg-tertiary border border-nier-border-muted rounded p-golden-md">
                <h3 class="text-lg font-bold text-nier-accent-cool mb-golden-sm uppercase">
                  📄 Document Upload
                </h3>
                <div class="grid grid-cols-2 gap-golden-sm text-sm">
                  <div class="flex justify-between">
                    <span class="text-nier-text-secondary">Filename:</span>
                    <span class="font-mono">{uploadResult.upload.filename}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-nier-text-secondary">File Size:</span>
                    <span class="font-mono">{formatFileSize(uploadResult.upload.file_size)}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-nier-text-secondary">Content Extracted:</span>
                    <span class="font-mono {uploadResult.upload.content_extracted ? 'text-green-400' : 'text-red-400'}">
                      {uploadResult.upload.content_extracted ? 'YES' : 'NO'}
                    </span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-nier-text-secondary">Embeddings:</span>
                    <span class="font-mono {uploadResult.upload.embeddings_generated ? 'text-green-400' : 'text-red-400'}">
                      {uploadResult.upload.embeddings_generated ? 'GENERATED' : 'SKIPPED'}
                    </span>
                  </div>
                </div>
              </div>
            {/if}

            <!-- GPU Processing Results -->
            {#if uploadResult.gpu_processing?.enabled}
              <div class="bg-nier-bg-tertiary border border-nier-border-muted rounded p-golden-md">
                <h3 class="text-lg font-bold text-nier-accent-cool mb-golden-sm uppercase">
                  ⚡ GPU Processing
                </h3>
                <div class="grid grid-cols-2 gap-golden-sm text-sm">
                  <div class="flex justify-between">
                    <span class="text-nier-text-secondary">CUDA Available:</span>
                    <span class="font-mono {uploadResult.gpu_processing.cuda_available ? 'text-green-400' : 'text-red-400'}">
                      {uploadResult.gpu_processing.cuda_available ? 'YES' : 'NO'}
                    </span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-nier-text-secondary">Tensor Cores:</span>
                    <span class="font-mono {uploadResult.gpu_processing.tensor_cores_used ? 'text-green-400' : 'text-nier-text-primary'}">
                      {uploadResult.gpu_processing.tensor_cores_used ? 'ACTIVE' : 'DISABLED'}
                    </span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-nier-text-secondary">Quantization:</span>
                    <span class="font-mono text-nier-accent-warm">{uploadResult.gpu_processing.quantization_used?.toUpperCase()}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-nier-text-secondary">Latent Space:</span>
                    <span class="font-mono {uploadResult.gpu_processing.negative_latent_space_used ? 'text-green-400' : 'text-nier-text-primary'}">
                      {uploadResult.gpu_processing.negative_latent_space_used ? 'NEGATIVE' : 'STANDARD'}
                    </span>
                  </div>
                </div>

                {#if uploadResult.gpu_processing.processing_result}
                  <div class="mt-golden-md">
                    <h4 class="font-bold text-nier-accent-warm mb-golden-sm">Processing Result:</h4>
                    <pre class="bg-nier-bg-primary border border-nier-border-muted rounded p-golden-sm text-xs font-mono text-nier-text-secondary overflow-x-auto">
{JSON.stringify(uploadResult.gpu_processing.processing_result, null, 2)}
                    </pre>
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  /* Custom file input styling */
  input[type="file"] {
    cursor: pointer;
  }
  
  input[type="file"]::-webkit-file-upload-button {
    background: linear-gradient(to right, #d5b678, #c0a571);
    color: #2b2b2b;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-weight: bold;
    cursor: pointer;
    margin-right: 1rem;
  }
  
  /* Custom scrollbar for code blocks */
  pre::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  
  pre::-webkit-scrollbar-track {
    background: rgba(61, 61, 61, 0.2);
  }
  
  pre::-webkit-scrollbar-thumb {
    background: rgba(213, 182, 120, 0.5);
    border-radius: 4px;
  }
  
  pre::-webkit-scrollbar-thumb:hover {
    background: rgba(213, 182, 120, 0.8);
  }
</style>