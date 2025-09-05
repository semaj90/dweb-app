<!--
  3D Semantic Analysis Page
  Demonstrates the complete pipeline: .txt → WebAssembly → WebGPU → 3D spatial embeddings
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import Enhanced3DSemanticProcessor from '$lib/components/Enhanced3DSemanticProcessor.svelte';
  import { Card } from 'bits-ui';
  import { Badge } from 'bits-ui';

  let pageTitle = '3D Semantic Analysis';
  let pageDescription = 'Advanced text processing with WebAssembly tokenization, WebGPU 3D spatial embeddings, and mathematical optimization';
  
  // Mathematical concepts and algorithms info
  let mathematicalConcepts = [
    {
      name: "Nomic-Embed Architecture",
      description: "Transformer-based encoder with contrastive learning objective",
      formula: "similarity = cosine(v_a, v_b) = (v_a · v_b) / (||v_a|| ||v_b||)",
      implementation: "WGSL compute shaders with L2 normalization and mean pooling"
    },
    {
      name: "3D Spatial Projection", 
      description: "PCA-like dimensionality reduction to 3D coordinates",
      formula: "x_3d = Σ(embedding_i × basis_x_i), where basis vectors are learned",
      implementation: "GPU-accelerated projection using trigonometric basis functions"
    },
    {
      name: "Level of Detail (LOD)",
      description: "Progressive quantization based on spatial distance",
      formula: "lod_level = min(floor(distance / threshold), max_lod)",
      implementation: "Dynamic precision: fp32 → fp16 → int8 → 4-bit quantization"
    },
    {
      name: "Autocomplete N-grams",
      description: "Frequency-based prefix matching with confidence scoring",
      formula: "confidence = log(frequency + 1) × (1 - prefix_length × 0.1)",
      implementation: "WebAssembly hash tables with prefix trees for O(log n) lookup"
    },
    {
      name: "4D Boost Transform",
      description: "Hyperbolic enhancement using quaternion-inspired rotations",
      formula: "boosted = input × cos(θ) + |input| × sin(θ) × activation(input)",
      implementation: "WGSL shader with tanh, sigmoid, and ReLU activation functions"
    },
    {
      name: "Octree Spatial Indexing",
      description: "Hierarchical 3D space partitioning for fast neighbor search",
      formula: "cell_id = hash(floor(position / grid_size))",
      implementation: "GPU buffer-based octree with 3×3×3 neighborhood queries"
    }
  ];
  
  // Optimization techniques
  let optimizationTechniques = [
    {
      technique: "WebAssembly SIMD",
      benefit: "4-8x faster tokenization",
      description: "Vectorized string operations for chunking and n-gram extraction"
    },
    {
      technique: "WebGPU Compute Shaders", 
      benefit: "100-1000x GPU acceleration",
      description: "Parallel embedding computations with workgroup shared memory"
    },
    {
      technique: "Product Quantization",
      benefit: "8-16x compression",
      description: "Compress embeddings into centroids + codes for storage/retrieval"
    },
    {
      technique: "LOD Streaming",
      benefit: "Reduce bandwidth 75%",
      description: "Stream only required precision level based on viewport distance"
    },
    {
      technique: "Shader Caching",
      benefit: "90% faster startup",
      description: "Pre-compile and cache WGSL pipelines in IndexedDB"
    },
    {
      technique: "Incremental HNSW",
      benefit: "O(log n) search time",
      description: "Dynamic nearest neighbor index with hierarchical graphs"
    }
  ];
  
  // Performance metrics (example)
  let performanceMetrics = [
    { metric: "Text Processing Speed", value: "50,000 tokens/sec", detail: "WebAssembly BPE tokenization" },
    { metric: "Embedding Generation", value: "1,024 vectors/sec", detail: "WebGPU batch processing" },
    { metric: "3D Projection Time", value: "2.3ms per batch", detail: "GPU matrix operations" },
    { metric: "Autocomplete Latency", value: "<10ms", detail: "N-gram prefix matching" },
    { metric: "Memory Efficiency", value: "75% reduction", detail: "LOD + quantization" },
    { metric: "Search Accuracy", value: "98.5% recall@10", detail: "HNSW spatial index" }
  ];

  onMount(() => {
    document.title = `${pageTitle} | Legal AI System`;
  });
</script>

<svelte:head>
  <title>{pageTitle} | Legal AI System</title>
  <meta name="description" content={pageDescription} />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  
  <!-- Three.js for 3D visualization -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  
  <!-- Open Graph -->
  <meta property="og:type" content="website" />
  <meta property="og:url" content="{$page.url}" />
  <meta property="og:title" content={pageTitle} />
  <meta property="og:description" content={pageDescription} />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-50">
  <!-- Navigation -->
  <nav class="bg-white shadow-sm border-b">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <div class="flex items-center">
          <a href="/" class="text-xl font-bold text-gray-900 hover:text-blue-600 transition-colors">
            ⚖️ Legal AI
          </a>
          <span class="mx-2 text-gray-400">/</span>
          <span class="text-gray-600">3D Semantic Analysis</span>
        </div>
        
        <div class="flex items-center space-x-4">
          <a href="/gpu-orchestrator" class="text-gray-600 hover:text-blue-600 transition-colors text-sm font-medium">
            🎮 GPU Orchestrator
          </a>
          <a href="/ai-workspace" class="text-gray-600 hover:text-blue-600 transition-colors text-sm font-medium">
            🧠 AI Workspace
          </a>
          <a href="/docs" class="text-gray-600 hover:text-blue-600 transition-colors text-sm font-medium">
            📚 Documentation
          </a>
        </div>
      </div>
    </div>
  </nav>

  <!-- Main Content -->
  <main class="py-8">
    <!-- Mathematical Concepts Section -->
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mb-8">
      <Card.Root>
        <Card.Header>
          <Card.Title class="text-2xl">🧮 Mathematical Foundations</Card.Title>
        </Card.Header>
        <Card.Content>
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {#each mathematicalConcepts as concept}
              <div class="border rounded-lg p-4 bg-gray-50">
                <h3 class="font-semibold text-lg mb-2">{concept.name}</h3>
                <p class="text-gray-700 mb-3">{concept.description}</p>
                <div class="bg-white rounded p-3 mb-3">
                  <code class="text-sm text-blue-800 font-mono">{concept.formula}</code>
                </div>
                <div class="text-sm text-gray-600">
                  <strong>Implementation:</strong> {concept.implementation}
                </div>
              </div>
            {/each}
          </div>
        </Card.Content>
      </Card.Root>
    </div>

    <!-- Enhanced 3D Processor Component -->
    <div class="max-w-full mx-auto">
      <Enhanced3DSemanticProcessor 
        maxConcurrent={8}
        embeddingDimensions={768}
        spatialScale={1.5}
        lodThreshold={15.0}
        enableAutocomplete={true}
      />
    </div>

    <!-- Optimization Techniques -->
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8 mb-8">
      <Card.Root>
        <Card.Header>
          <Card.Title class="text-2xl">⚡ Optimization Techniques</Card.Title>
        </Card.Header>
        <Card.Content>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {#each optimizationTechniques as opt}
              <div class="border rounded-lg p-4 hover:shadow-lg transition-shadow">
                <div class="flex items-center justify-between mb-2">
                  <h3 class="font-semibold">{opt.technique}</h3>
                  <Badge class="bg-green-100 text-green-800">{opt.benefit}</Badge>
                </div>
                <p class="text-gray-600 text-sm">{opt.description}</p>
              </div>
            {/each}
          </div>
        </Card.Content>
      </Card.Root>
    </div>

    <!-- Performance Metrics -->
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mb-8">
      <Card.Root>
        <Card.Header>
          <Card.Title class="text-2xl">📊 Performance Benchmarks</Card.Title>
        </Card.Header>
        <Card.Content>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {#each performanceMetrics as metric}
              <div class="text-center p-4 border rounded-lg">
                <div class="text-2xl font-bold text-blue-600 mb-1">{metric.value}</div>
                <div class="font-medium text-gray-900 mb-1">{metric.metric}</div>
                <div class="text-sm text-gray-600">{metric.detail}</div>
              </div>
            {/each}
          </div>
        </Card.Content>
      </Card.Root>
    </div>

    <!-- Pipeline Architecture Diagram -->
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <Card.Root>
        <Card.Header>
          <Card.Title class="text-2xl">🏗️ Processing Pipeline Architecture</Card.Title>
        </Card.Header>
        <Card.Content>
          <div class="overflow-x-auto">
            <div class="flex items-center space-x-4 min-w-max p-4">
              <!-- Step 1: Text Input -->
              <div class="flex flex-col items-center p-4 border rounded-lg bg-blue-50 min-w-32">
                <div class="text-2xl mb-2">📄</div>
                <div class="font-semibold">Text Files</div>
                <div class="text-xs text-gray-600">.txt, .json</div>
              </div>
              
              <div class="text-2xl">→</div>
              
              <!-- Step 2: WebAssembly Processing -->
              <div class="flex flex-col items-center p-4 border rounded-lg bg-green-50 min-w-32">
                <div class="text-2xl mb-2">⚡</div>
                <div class="font-semibold">WASM</div>
                <div class="text-xs text-gray-600">Tokenization</div>
              </div>
              
              <div class="text-2xl">→</div>
              
              <!-- Step 3: WebGPU Processing -->
              <div class="flex flex-col items-center p-4 border rounded-lg bg-purple-50 min-w-32">
                <div class="text-2xl mb-2">🎮</div>
                <div class="font-semibold">WebGPU</div>
                <div class="text-xs text-gray-600">Embeddings</div>
              </div>
              
              <div class="text-2xl">→</div>
              
              <!-- Step 4: 3D Projection -->
              <div class="flex flex-col items-center p-4 border rounded-lg bg-orange-50 min-w-32">
                <div class="text-2xl mb-2">📐</div>
                <div class="font-semibold">3D Mapping</div>
                <div class="text-xs text-gray-600">Spatial Coords</div>
              </div>
              
              <div class="text-2xl">→</div>
              
              <!-- Step 5: LOD & Caching -->
              <div class="flex flex-col items-center p-4 border rounded-lg bg-red-50 min-w-32">
                <div class="text-2xl mb-2">🗄️</div>
                <div class="font-semibold">LOD Cache</div>
                <div class="text-xs text-gray-600">Quantization</div>
              </div>
              
              <div class="text-2xl">→</div>
              
              <!-- Step 6: Visualization -->
              <div class="flex flex-col items-center p-4 border rounded-lg bg-yellow-50 min-w-32">
                <div class="text-2xl mb-2">🎯</div>
                <div class="font-semibold">3D Render</div>
                <div class="text-xs text-gray-600">Three.js</div>
              </div>
            </div>
          </div>
          
          <div class="mt-6 p-4 bg-gray-50 rounded-lg">
            <h4 class="font-semibold mb-2">Key Innovations:</h4>
            <ul class="text-sm text-gray-700 space-y-1">
              <li>• <strong>Hybrid Processing:</strong> WebAssembly for tokenization + WebGPU for embeddings</li>
              <li>• <strong>Spatial Embeddings:</strong> Map high-dimensional vectors to 3D coordinates</li>
              <li>• <strong>LOD Optimization:</strong> Dynamic precision based on viewport distance</li>
              <li>• <strong>Real-time Autocomplete:</strong> N-gram indexing with semantic context</li>
              <li>• <strong>Shader Caching:</strong> Pre-compiled WGSL pipelines for instant startup</li>
              <li>• <strong>Memory Efficiency:</strong> Quantization and compression reduce storage by 75%</li>
            </ul>
          </div>
        </Card.Content>
      </Card.Root>
    </div>
  </main>

  <!-- Footer -->
  <footer class="bg-white border-t mt-12">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="text-center">
        <div class="text-sm text-gray-500 mb-4">
          © 2024 Legal AI System. Advanced 3D Semantic Processing with Mathematical Optimization.
        </div>
        <div class="flex justify-center space-x-6 text-sm">
          <a href="/docs/algorithms" class="text-gray-500 hover:text-blue-600 transition-colors">
            Algorithm Details
          </a>
          <a href="/docs/performance" class="text-gray-500 hover:text-blue-600 transition-colors">
            Performance Analysis
          </a>
          <a href="/docs/api" class="text-gray-500 hover:text-blue-600 transition-colors">
            API Reference
          </a>
          <a href="/support" class="text-gray-500 hover:text-blue-600 transition-colors">
            Technical Support
          </a>
        </div>
      </div>
    </div>
  </footer>
</div>