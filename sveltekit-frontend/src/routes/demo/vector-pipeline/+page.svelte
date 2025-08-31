<script lang="ts">
  import VectorPipelineDemo from '$lib/components/demo/VectorPipelineDemo.svelte';
  import type { PageData } from './$types';

  const { data } = $props() as { data: PageData };
</script>

<svelte:head>
  <title>Vector Pipeline Demo - Legal AI Platform</title>
  <meta name="description" content="Interactive demonstration of the XState vector processing pipeline orchestrating PostgreSQL → Redis Streams → Go microservice → CUDA worker → Qdrant" />
</svelte:head>

<div class="min-h-screen bg-gray-50 py-8">
  <div class="container mx-auto px-4">
    <div class="text-center mb-8">
      <h1 class="text-4xl font-bold text-gray-900 mb-4">
        Vector Pipeline Demo
      </h1>
      <p class="text-xl text-gray-600 max-w-3xl mx-auto">
        Interactive demonstration of our XState-powered vector processing pipeline that orchestrates
        data flow from PostgreSQL through Redis Streams to Go microservices, CUDA workers, and Qdrant vector database.
      </p>
    </div>

    <VectorPipelineDemo />

    <!-- Technical Details Section -->
    <div class="mt-12 max-w-4xl mx-auto">
      <div class="bg-white rounded-lg shadow-lg p-8">
        <h2 class="text-2xl font-bold mb-6">Technical Architecture</h2>

        <div class="space-y-6">
          <div>
            <h3 class="text-lg font-semibold mb-2">Pipeline Flow</h3>
            <div class="text-gray-700 space-y-2">
              <div class="flex items-center gap-2">
                <span class="w-4 h-4 bg-blue-500 rounded-full flex-shrink-0"></span>
                <span><strong>PostgreSQL</strong> - Source data and vector storage with pgvector extension</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="w-4 h-4 bg-green-500 rounded-full flex-shrink-0"></span>
                <span><strong>Redis Streams</strong> - Event streaming and job queue management</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="w-4 h-4 bg-purple-500 rounded-full flex-shrink-0"></span>
                <span><strong>Go Microservices</strong> - Enhanced RAG processing and business logic</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="w-4 h-4 bg-yellow-500 rounded-full flex-shrink-0"></span>
                <span><strong>CUDA Workers</strong> - GPU-accelerated vector computations</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="w-4 h-4 bg-red-500 rounded-full flex-shrink-0"></span>
                <span><strong>Qdrant</strong> - High-performance vector similarity search</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="w-4 h-4 bg-indigo-500 rounded-full flex-shrink-0"></span>
                <span><strong>WebGPU</strong> - Optional browser-based GPU acceleration</span>
              </div>
            </div>
          </div>

          <div>
            <h3 class="text-lg font-semibold mb-2">XState Machine States</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div class="space-y-1">
                <div><code class="bg-gray-100 px-2 py-1 rounded">initializing</code> - System startup and health checks</div>
                <div><code class="bg-gray-100 px-2 py-1 rounded">idle</code> - Ready to accept new jobs</div>
                <div><code class="bg-gray-100 px-2 py-1 rounded">processingJob</code> - Single job execution</div>
              </div>
              <div class="space-y-1">
                <div><code class="bg-gray-100 px-2 py-1 rounded">processingBatch</code> - Batch job execution</div>
                <div><code class="bg-gray-100 px-2 py-1 rounded">retrying</code> - Failed job retry logic</div>
                <div><code class="bg-gray-100 px-2 py-1 rounded">error</code> - Error state with auto-recovery</div>
              </div>
            </div>
          </div>

          <div>
            <h3 class="text-lg font-semibold mb-2">Job Types</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div class="space-y-1">
                <div><code class="bg-blue-100 px-2 py-1 rounded">upsert</code> - Create or update vector embeddings</div>
                <div><code class="bg-green-100 px-2 py-1 rounded">reembed</code> - Regenerate embeddings with new model</div>
              </div>
              <div class="space-y-1">
                <div><code class="bg-red-100 px-2 py-1 rounded">delete</code> - Remove vectors from storage</div>
                <div><code class="bg-yellow-100 px-2 py-1 rounded">batch</code> - Process multiple operations together</div>
              </div>
            </div>
          </div>

          <div>
            <h3 class="text-lg font-semibold mb-2">Owner Types</h3>
            <div class="text-sm space-y-1">
              <div><strong>Evidence:</strong> Legal evidence documents requiring vector indexing</div>
              <div><strong>Document:</strong> Legal briefs, contracts, and case files</div>
              <div><strong>Case:</strong> Complete case records with metadata</div>
              <div><strong>Report:</strong> Forensic and analytical reports</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>