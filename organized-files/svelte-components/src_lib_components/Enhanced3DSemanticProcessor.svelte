<!--
  Enhanced 3D Semantic Processor - WebAssembly + WebGPU integration
  Combines text processing, 3D spatial embeddings, and autocomplete
-->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { createGPUProcessingActor, type DocumentInput } from '$lib/state/gpu-processing-machine';
  import { wasmTextProcessor } from '$lib/services/wasm-text-processor.js';
  import { Card } from 'bits-ui';
  import { Button } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { fade, fly, scale } from 'svelte/transition';
  
  // Props
  export let maxConcurrent = 5;
  export let embeddingDimensions = 768;
  export let spatialScale = 1.0;
  export let lodThreshold = 10.0;
  export let enableAutocomplete = true;
  
  // State
  let fileInput: HTMLInputElement;
  let textInput = $state('');
  let autocompleteInput = $state('');
  let autocompleteSuggestions = $state([]);
  let processedFiles = $state(new Map());
  let spatialEmbeddings = $state([]);
  let isProcessing = $state(false);
  let stats = $state({
    filesProcessed: 0,
    totalChunks: 0,
    totalTokens: 0,
    spatialPoints: 0,
    memoryUsage: 0
  });
  
  // GPU Processing Actor
  const gpuActor = createGPUProcessingActor();
  let actorState = $state(gpuActor.getSnapshot());
  
  // 3D Scene Variables
  let canvas3D: HTMLCanvasElement;
  let scene3D: any = null;
  let renderer3D: any = null;
  let pointCloud: any = null;
  let camera3D: any = null;
  let controls3D: any = null;
  
  // Processing pipeline state
  let currentLOD = $state(0);
  let viewportBounds = $state({ min: [-10, -10, -10], max: [10, 10, 10] });
  let shaderCache = $state(new Map());
  
  onMount(async () => {
    // Initialize WebAssembly text processor
    await wasmTextProcessor.initialize();
    
    // Start GPU processing actor
    gpuActor.start();
    
    // Subscribe to GPU actor state changes
    const subscription = gpuActor.subscribe((snapshot) => {
      actorState = snapshot;
    });
    
    // Initialize 3D scene
    await initialize3DScene();
    
    // Setup autocomplete if enabled
    if (enableAutocomplete) {
      setupAutocomplete();
    }
    
    return () => {
      subscription.unsubscribe();
      cleanup3DScene();
    };
  });
  
  onDestroy(() => {
    gpuActor.stop();
    cleanup3DScene();
  });
  
  // Initialize 3D scene with Three.js
  async function initialize3DScene() {
    try {
      // Import Three.js dynamically (assuming it's available)
      const THREE = (window as any).THREE;
      if (!THREE) {
        console.warn('Three.js not available, 3D visualization disabled');
        return;
      }
      
      // Setup scene
      scene3D = new THREE.Scene();
      scene3D.background = new THREE.Color(0x0a0a0a);
      
      // Setup camera
      camera3D = new THREE.PerspectiveCamera(
        75,
        canvas3D.clientWidth / canvas3D.clientHeight,
        0.1,
        1000
      );
      camera3D.position.set(20, 20, 20);
      
      // Setup renderer
      renderer3D = new THREE.WebGLRenderer({ 
        canvas: canvas3D,
        antialias: true,
        alpha: true
      });
      renderer3D.setSize(canvas3D.clientWidth, canvas3D.clientHeight);
      
      // Add controls (if OrbitControls available)
      if (THREE.OrbitControls) {
        controls3D = new THREE.OrbitControls(camera3D, canvas3D);
        controls3D.enableDamping = true;
      }
      
      // Add lighting
      const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(50, 50, 50);
      scene3D.add(ambientLight);
      scene3D.add(directionalLight);
      
      // Start render loop
      animate3D();
      
      console.log('🎮 3D Scene initialized');
    } catch (error) {
      console.error('❌ Error initializing 3D scene:', error);
    }
  }
  
  // Animation loop for 3D scene
  function animate3D() {
    if (!renderer3D || !scene3D || !camera3D) return;
    
    requestAnimationFrame(animate3D);
    
    // Update controls
    if (controls3D) {
      controls3D.update();
    }
    
    // Animate point cloud if exists
    if (pointCloud) {
      pointCloud.rotation.y += 0.005;
    }
    
    // Render the scene
    renderer3D.render(scene3D, camera3D);
  }
  
  // Handle file selection and processing
  async function handleFileSelect(event: Event) {
    const files = (event.target as HTMLInputElement).files;
    if (!files || files.length === 0) return;
    
    isProcessing = true;
    
    try {
      for (const file of Array.from(files)) {
        if (file.type === 'text/plain' || file.name.endsWith('.txt')) {
          await processTextFile(file);
        }
      }
      
      // Update 3D visualization
      updateSpatialVisualization();
      
    } catch (error) {
      console.error('❌ Error processing files:', error);
    } finally {
      isProcessing = false;
    }
  }
  
  // Process a single text file through the pipeline
  async function processTextFile(file: File) {
    console.log(`🔄 Processing file: ${file.name}`);
    
    // Step 1: WebAssembly text processing
    const textResult = await wasmTextProcessor.processTextFile(file);
    processedFiles.set(textResult.fileId, textResult);
    
    // Step 2: Get embedding requests for GPU processing
    const embeddingRequests = await wasmTextProcessor.getEmbeddingsForChunks(
      textResult.fileId, 
      embeddingDimensions
    );
    
    // Step 3: Process through GPU pipeline
    for (const request of embeddingRequests) {
      gpuActor.send({ 
        type: 'PROCESS_DOCUMENT', 
        ...request 
      });
    }
    
    // Step 4: Create spatial embeddings (simulated for now)
    const spatial = await createSpatialEmbeddings(textResult.chunks);
    spatialEmbeddings.push(...spatial);
    
    // Update stats
    stats.filesProcessed++;
    stats.totalChunks += textResult.totalChunks;
    stats.totalTokens += textResult.totalTokens;
    stats.spatialPoints = spatialEmbeddings.length;
    stats.memoryUsage = textResult.memoryUsage?.heapUsed || 0;
    
    console.log(`✅ Processed ${file.name}: ${textResult.totalChunks} chunks, ${textResult.totalTokens} tokens`);
  }
  
  // Create spatial embeddings from text chunks
  async function createSpatialEmbeddings(chunks: any[]) {
    const spatial = [];
    
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      
      // Generate 3D coordinates (simplified - in practice this would use PCA/t-SNE)
      const x = (Math.random() - 0.5) * spatialScale * 20;
      const y = (Math.random() - 0.5) * spatialScale * 20;
      const z = (Math.random() - 0.5) * spatialScale * 20;
      
      // Determine LOD based on distance from origin
      const distance = Math.sqrt(x*x + y*y + z*z);
      const lodLevel = Math.min(3, Math.floor(distance / lodThreshold));
      
      spatial.push({
        id: chunk.id,
        position: { x, y, z },
        embedding: new Float32Array(embeddingDimensions).map(() => Math.random()),
        lodLevel,
        chunkId: i,
        spriteUV: { u: (i % 16) / 16, v: Math.floor(i / 16) / 16 },
        content: chunk.content.substring(0, 100) + '...',
        tokenCount: chunk.tokens.length,
        confidence: Math.random()
      });
    }
    
    return spatial;
  }
  
  // Update 3D visualization with spatial embeddings
  function updateSpatialVisualization() {
    if (!scene3D || spatialEmbeddings.length === 0) return;
    
    try {
      const THREE = (window as any).THREE;
      if (!THREE) return;
      
      // Remove existing point cloud
      if (pointCloud) {
        scene3D.remove(pointCloud);
      }
      
      // Create geometry for points
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(spatialEmbeddings.length * 3);
      const colors = new Float32Array(spatialEmbeddings.length * 3);
      const sizes = new Float32Array(spatialEmbeddings.length);
      
      spatialEmbeddings.forEach((embedding, i) => {
        // Position
        positions[i * 3] = embedding.position.x;
        positions[i * 3 + 1] = embedding.position.y;
        positions[i * 3 + 2] = embedding.position.z;
        
        // Color based on LOD level
        const lodColors = [
          [1, 0.2, 0.2], // LOD 0: Red (high detail)
          [1, 1, 0.2],   // LOD 1: Yellow
          [0.2, 1, 0.2], // LOD 2: Green
          [0.2, 0.2, 1]  // LOD 3: Blue (low detail)
        ];
        const color = lodColors[embedding.lodLevel] || [0.5, 0.5, 0.5];
        colors[i * 3] = color[0];
        colors[i * 3 + 1] = color[1];
        colors[i * 3 + 2] = color[2];
        
        // Size based on confidence
        sizes[i] = 2 + embedding.confidence * 8;
      });
      
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
      
      // Create point cloud material
      const material = new THREE.ShaderMaterial({
        uniforms: {
          time: { value: 0.0 }
        },
        vertexShader: `
          attribute float size;
          attribute vec3 color;
          varying vec3 vColor;
          uniform float time;
          
          void main() {
            vColor = color;
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            
            // Animate based on time
            float pulse = sin(time * 2.0 + position.x * 0.1) * 0.1 + 1.0;
            gl_PointSize = size * pulse * (300.0 / -mvPosition.z);
            gl_Position = projectionMatrix * mvPosition;
          }
        `,
        fragmentShader: `
          varying vec3 vColor;
          
          void main() {
            float dist = length(gl_PointCoord - vec2(0.5));
            if (dist > 0.5) discard;
            
            float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
            gl_FragColor = vec4(vColor, alpha);
          }
        `,
        transparent: true,
        vertexColors: true
      });
      
      pointCloud = new THREE.Points(geometry, material);
      scene3D.add(pointCloud);
      
      console.log(`🎯 Updated 3D visualization with ${spatialEmbeddings.length} points`);
    } catch (error) {
      console.error('❌ Error updating 3D visualization:', error);
    }
  }
  
  // Setup autocomplete functionality
  function setupAutocomplete() {
    let debounceTimer: NodeJS.Timeout;
    
    const handleInput = async () => {
      clearTimeout(debounceTimer);
      
      debounceTimer = setTimeout(async () => {
        if (autocompleteInput.trim().length > 1) {
          const suggestions = await wasmTextProcessor.getAutocompleteSuggestions(
            autocompleteInput,
            10
          );
          autocompleteSuggestions = suggestions;
        } else {
          autocompleteSuggestions = [];
        }
      }, 200);
    };
    
    // This would be attached to an input element
    return handleInput;
  }
  
  // Apply autocomplete suggestion
  function applyAutocompleteSuggestion(suggestion: any) {
    autocompleteInput = suggestion.text;
    autocompleteSuggestions = [];
  }
  
  // Clear all data and reset
  function clearAll() {
    processedFiles.clear();
    spatialEmbeddings = [];
    autocompleteSuggestions = [];
    wasmTextProcessor.clearCache();
    
    if (pointCloud && scene3D) {
      scene3D.remove(pointCloud);
      pointCloud = null;
    }
    
    stats = {
      filesProcessed: 0,
      totalChunks: 0,
      totalTokens: 0,
      spatialPoints: 0,
      memoryUsage: 0
    };
  }
  
  // Cleanup 3D scene
  function cleanup3DScene() {
    if (renderer3D) {
      renderer3D.dispose();
    }
    if (controls3D) {
      controls3D.dispose();
    }
  }
  
  // Format bytes
  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  }
</script>

<div class="enhanced-3d-processor p-6 max-w-full mx-auto">
  <!-- Header -->
  <div class="mb-6">
    <h1 class="text-3xl font-bold text-gray-900 mb-2">
      🎮 Enhanced 3D Semantic Processor
    </h1>
    <p class="text-gray-600">
      WebAssembly text processing + WebGPU 3D spatial embeddings + Autocomplete
    </p>
  </div>

  <!-- Stats Dashboard -->
  <Card.Root class="mb-6">
    <Card.Header>
      <Card.Title>Processing Statistics</Card.Title>
    </Card.Header>
    <Card.Content>
      <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div class="text-center">
          <div class="text-2xl font-bold text-blue-600">{stats.filesProcessed}</div>
          <div class="text-sm text-gray-600">Files Processed</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-green-600">{stats.totalChunks}</div>
          <div class="text-sm text-gray-600">Text Chunks</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-purple-600">{stats.totalTokens}</div>
          <div class="text-sm text-gray-600">Total Tokens</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-orange-600">{stats.spatialPoints}</div>
          <div class="text-sm text-gray-600">3D Points</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-red-600">{formatBytes(stats.memoryUsage)}</div>
          <div class="text-sm text-gray-600">Memory Usage</div>
        </div>
      </div>
    </Card.Content>
  </Card.Root>

  <!-- File Upload & Controls -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
    <!-- File Upload -->
    <Card.Root>
      <Card.Header>
        <Card.Title>Text File Processing</Card.Title>
      </Card.Header>
      <Card.Content>
        <div class="space-y-4">
          <input
            bind:this={fileInput}
            type="file"
            accept=".txt,.json"
            multiple
            on:change={handleFileSelect}
            class="block w-full text-sm text-gray-500 
                   file:mr-4 file:py-2 file:px-4
                   file:rounded-md file:border-0
                   file:text-sm file:font-semibold
                   file:bg-blue-50 file:text-blue-700
                   hover:file:bg-blue-100"
          />
          
          {#if isProcessing}
            <div class="flex items-center space-x-2 text-blue-600" transition:fade>
              <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
              <span>Processing files...</span>
            </div>
          {/if}
          
          <div class="flex space-x-2">
            <Button.Root 
              on:click={() => fileInput?.click()}
              class="bg-blue-600 hover:bg-blue-700"
            >
              📁 Select Files
            </Button.Root>
            <Button.Root 
              on:click={clearAll}
              variant="outline"
              class="border-red-300 text-red-700 hover:bg-red-50"
            >
              🗑️ Clear All
            </Button.Root>
          </div>
        </div>
      </Card.Content>
    </Card.Root>

    <!-- Autocomplete Demo -->
    <Card.Root>
      <Card.Header>
        <Card.Title>Semantic Autocomplete</Card.Title>
      </Card.Header>
      <Card.Content>
        <div class="space-y-4">
          <div class="relative">
            <input
              bind:value={autocompleteInput}
              on:input={setupAutocomplete()}
              placeholder="Start typing to get suggestions..."
              class="w-full px-3 py-2 border border-gray-300 rounded-md 
                     focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            
            {#if autocompleteSuggestions.length > 0}
              <div class="absolute top-full left-0 right-0 bg-white border border-gray-300 
                          rounded-md shadow-lg z-10 max-h-48 overflow-y-auto" 
                   transition:fly={{ y: -10, duration: 200 }}>
                {#each autocompleteSuggestions as suggestion}
                  <button
                    on:click={() => applyAutocompleteSuggestion(suggestion)}
                    class="w-full px-3 py-2 text-left hover:bg-gray-100 
                           flex items-center justify-between"
                  >
                    <span>{suggestion.text}</span>
                    <div class="flex items-center space-x-2">
                      <Badge class="text-xs">
                        {suggestion.type}
                      </Badge>
                      <span class="text-xs text-gray-500">
                        {(suggestion.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </button>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      </Card.Content>
    </Card.Root>
  </div>

  <!-- 3D Visualization -->
  <Card.Root class="mb-6">
    <Card.Header>
      <Card.Title>3D Semantic Space Visualization</Card.Title>
    </Card.Header>
    <Card.Content>
      <div class="relative">
        <canvas
          bind:this={canvas3D}
          class="w-full h-96 bg-black rounded-lg border"
          style="max-width: 100%;"
        ></canvas>
        
        <!-- LOD Controls -->
        <div class="absolute top-4 right-4 bg-white bg-opacity-90 rounded-lg p-3">
          <div class="text-sm font-semibold mb-2">Level of Detail</div>
          <div class="space-y-1">
            <div class="flex items-center space-x-2">
              <div class="w-3 h-3 bg-red-500 rounded"></div>
              <span class="text-xs">LOD 0 (High Detail)</span>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-3 h-3 bg-yellow-500 rounded"></div>
              <span class="text-xs">LOD 1 (Medium)</span>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-3 h-3 bg-green-500 rounded"></div>
              <span class="text-xs">LOD 2 (Low)</span>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-3 h-3 bg-blue-500 rounded"></div>
              <span class="text-xs">LOD 3 (Minimal)</span>
            </div>
          </div>
        </div>
      </div>
    </Card.Content>
  </Card.Root>

  <!-- Processing Queue Info -->
  {#if actorState.context.processingQueue.length > 0 || actorState.context.activeProcessing.size > 0}
    <Card.Root>
      <Card.Header>
        <Card.Title>GPU Processing Status</Card.Title>
      </Card.Header>
      <Card.Content>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="text-center">
            <div class="text-xl font-bold text-blue-600">
              {actorState.context.processingQueue.length}
            </div>
            <div class="text-sm text-gray-600">Queued</div>
          </div>
          <div class="text-center">
            <div class="text-xl font-bold text-yellow-600">
              {actorState.context.activeProcessing.size}
            </div>
            <div class="text-sm text-gray-600">Processing</div>
          </div>
          <div class="text-center">
            <div class="text-xl font-bold text-green-600">
              {actorState.context.completedDocuments.size}
            </div>
            <div class="text-sm text-gray-600">Completed</div>
          </div>
        </div>
      </Card.Content>
    </Card.Root>
  {/if}
</div>

<style>
  .enhanced-3d-processor {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  
  canvas {
    cursor: grab;
  }
  
  canvas:active {
    cursor: grabbing;
  }
</style>