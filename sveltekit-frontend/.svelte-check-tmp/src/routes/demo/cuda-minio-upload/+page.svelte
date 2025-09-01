<!-- Demo page for Enhanced MinIO Drag-and-Drop with CUDA acceleration -->
<script lang="ts">
  import EnhancedMinIODragDrop from '$lib/components/upload/EnhancedMinIODragDrop.svelte';
  import { onMount } from 'svelte';
  
  let uploadResults = $state<any[]>([]);
  let systemHealth = $state({
    minio: false,
    cuda: false,
    redis: false
  });
  
  onMount(async () => {
    await checkSystemHealth();
  });
  
  async function checkSystemHealth() {
    try {
      // Check MinIO health
      const minioResponse = await fetch('http://localhost:9000/minio/health/live');
      systemHealth.minio = minioResponse.ok;
      
      // Check CUDA worker health
      const cudaResponse = await fetch('/api/v1/gpu/cuda/preprocess');
      systemHealth.cuda = cudaResponse.ok;
      
      // Check Redis health
      const redisResponse = await fetch('/api/v1/redis/metrics');
      systemHealth.redis = redisResponse.ok;
      
    } catch (error) {
      console.warn('Health check failed:', error);
    }
  }
  
  function handleUploadComplete(event: CustomEvent) {
    uploadResults = [...uploadResults, ...event.detail];
  }
  
  function handleUploadError(event: CustomEvent) {
    console.error('Upload error:', event.detail);
  }
  
  function handleUploadProgress(event: CustomEvent) {
    console.log('Upload progress:', event.detail);
  }
</script>

<svelte:head>
  <title>CUDA MinIO Upload Demo - Legal AI Platform</title>
</svelte:head>

<div class="max-w-6xl mx-auto p-6">
  <header class="mb-8">
    <h1 class="text-3xl font-bold text-gray-800 mb-2">
      🚀 Enhanced MinIO Drag & Drop Demo
    </h1>
    <p class="text-gray-600">
      CUDA GPU acceleration • Clang/LLVM optimization • Visual Studio 2022 native performance
    </p>
  </header>

  <!-- System Health Status -->
  <div class="mb-6 p-4 bg-gray-50 rounded-lg">
    <h2 class="font-semibold mb-3">System Health Status</h2>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div class="flex items-center gap-2">
        <span class="w-3 h-3 rounded-full {systemHealth.minio ? 'bg-green-500' : 'bg-red-500'}"></span>
        <span class="text-sm">MinIO Storage: {systemHealth.minio ? 'Online' : 'Offline'}</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="w-3 h-3 rounded-full {systemHealth.cuda ? 'bg-green-500' : 'bg-yellow-500'}"></span>
        <span class="text-sm">CUDA Worker: {systemHealth.cuda ? 'Available' : 'Unavailable'}</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="w-3 h-3 rounded-full {systemHealth.redis ? 'bg-green-500' : 'bg-red-500'}"></span>
        <span class="text-sm">Redis Sync: {systemHealth.redis ? 'Connected' : 'Disconnected'}</span>
      </div>
    </div>
  </div>

  <!-- Enhanced Upload Component -->
  <div class="mb-8">
    <h2 class="text-xl font-semibold mb-4">File Upload with CUDA Acceleration</h2>
    <EnhancedMinIODragDrop
      caseId="demo-case-123"
      enableCudaAcceleration={true}
      enableGpuOptimization={true}
      useMsvcOptimizations={true}
      maxFileSize={100 * 1024 * 1024}
      acceptedTypes={['image/*', 'application/pdf', 'text/*', '.docx', '.xlsx', '.zip']}
      uploadComplete={handleUploadComplete}
      uploadError={handleUploadError}
      uploadProgress={handleUploadProgress}
    />
  </div>

  <!-- Upload Results -->
  {#if uploadResults.length > 0}
    <div class="mb-8">
      <h2 class="text-xl font-semibold mb-4">Upload Results</h2>
      <div class="space-y-4">
        {#each uploadResults as result}
          <div class="p-4 border rounded-lg bg-white shadow-sm">
            <div class="flex justify-between items-start mb-2">
              <div class="font-medium">{result.fileName}</div>
              <div class="text-sm text-gray-500">
                {result.processingTime}ms
              </div>
            </div>
            
            <div class="text-sm text-gray-600 space-y-1">
              <div>Size: {(result.size / (1024 * 1024)).toFixed(2)} MB</div>
              <div>MinIO Path: <code class="text-xs bg-gray-100 px-1 rounded">{result.minioPath}</code></div>
              <div>Content Type: {result.contentType}</div>
              
              {#if result.cudaOptimized}
                <div class="flex items-center gap-2 text-blue-600">
                  <span class="w-2 h-2 bg-blue-500 rounded-full"></span>
                  CUDA Optimized • Clang/LLVM compiled
                </div>
              {/if}
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Technical Details -->
  <div class="bg-gray-50 p-6 rounded-lg">
    <h2 class="text-xl font-semibold mb-4">Technical Implementation</h2>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <h3 class="font-semibold mb-2">Clang/LLVM Optimizations</h3>
        <ul class="text-sm text-gray-600 space-y-1">
          <li>• Visual Studio 2022 native compilation</li>
          <li>• MSVC compatibility mode enabled</li>
          <li>• Target: x86_64-pc-windows-msvc</li>
          <li>• MS extensions and compatibility flags</li>
        </ul>
      </div>
      
      <div>
        <h3 class="font-semibold mb-2">CUDA GPU Acceleration</h3>
        <ul class="text-sm text-gray-600 space-y-1">
          <li>• CUDA 12.8 for Clang compatibility</li>
          <li>• RTX 3060 Ti (SM 7.5) optimized</li>
          <li>• GPU memory optimization</li>
          <li>• Parallel file preprocessing</li>
        </ul>
      </div>
      
      <div>
        <h3 class="font-semibold mb-2">MinIO Integration</h3>
        <ul class="text-sm text-gray-600 space-y-1">
          <li>• Real-time Redis event publishing</li>
          <li>• Bucket management and health checks</li>
          <li>• Metadata preservation</li>
          <li>• Performance metrics tracking</li>
        </ul>
      </div>
      
      <div>
        <h3 class="font-semibold mb-2">HTML5 Features</h3>
        <ul class="text-sm text-gray-600 space-y-1">
          <li>• Native drag-and-drop API</li>
          <li>• File validation and preview</li>
          <li>• Progress tracking per file</li>
          <li>• Error handling and retry logic</li>
        </ul>
      </div>
    </div>
    
    <div class="mt-6 p-4 bg-blue-50 border border-blue-200 rounded">
      <div class="font-semibold text-blue-800 mb-2">Performance Benefits</div>
      <div class="text-sm text-blue-700">
        The Clang/LLVM + Visual Studio 2022 compilation produces native Windows binaries with 
        better optimization than MinGW, resulting in improved CUDA kernel performance and 
        reduced memory overhead for large file processing operations.
      </div>
    </div>
  </div>
</div>

<style>
  code {
    word-break: break-all;
  }
</style>