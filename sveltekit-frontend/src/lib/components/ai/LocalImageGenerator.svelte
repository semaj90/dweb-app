<!--
Local Image Generation Component
Supports Stable Diffusion WebUI, ComfyUI, and Ollama integration
Production-ready with native Windows support
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { 
    imageGenerationService, 
    imageGenerationStore,
    type ImageGenerationRequest,
    type ImageGenerationResult 
  } from '$lib/services/local-image-generation-service.js';
  
  interface Props {
    caseId?: string;
    onImageGenerated?: (result: ImageGenerationResult) => void;
    initialPrompt?: string;
    compact?: boolean;
  }
  
  let { 
    caseId = '', 
    onImageGenerated = () => {}, 
    initialPrompt = '',
    compact = false 
  }: Props = $props();

  // Component state
  let prompt = $state(initialPrompt);
  let negativePrompt = $state('blurry, low quality, distorted, text, watermark, signature');
  let selectedStyle = $state<'realistic' | 'artistic' | 'anime' | 'sketch' | 'legal-diagram' | 'evidence-recreation'>('realistic');
  let selectedProvider = $state<'stable-diffusion-webui' | 'comfyui' | 'ollama-vision' | 'fallback'>('fallback');
  let advancedMode = $state(false);
  
  // Advanced parameters
  let width = $state(512);
  let height = $state(512);
  let steps = $state(20);
  let cfgScale = $state(7.5);
  let seed = $state(-1);
  
  // UI state
  let showHistory = $state(false);
  let selectedImage = $state<ImageGenerationResult | null>(null);
  let generationHistory = $state<ImageGenerationResult[]>([]);
  
  // Provider status
  let providerStatus = $state<Map<string, string>>(new Map());

  onMount(() => {
    // Load provider status
    providerStatus = imageGenerationService.getProviderStatus();
    
    // Load generation history
    loadHistory();
  });

  async function loadHistory() {
    try {
      generationHistory = await imageGenerationService.getGenerationHistory();
    } catch (error) {
      console.error('Failed to load generation history:', error);
    }
  }

  async function generateImage() {
    if (!prompt.trim()) {
      alert('Please enter a prompt');
      return;
    }

    try {
      const request: ImageGenerationRequest = {
        prompt: prompt.trim(),
        negativePrompt: negativePrompt.trim() || undefined,
        width,
        height,
        steps,
        cfgScale,
        seed: seed === -1 ? undefined : seed,
        style: selectedStyle,
        provider: selectedProvider
      };

      const result = await imageGenerationService.generateImage(request);
      
      // Update history
      generationHistory = [result, ...generationHistory];
      selectedImage = result;
      
      // Notify parent component
      onImageGenerated(result);
      
    } catch (error) {
      console.error('Image generation failed:', error);
      alert(`Image generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  function useImageAsEvidence(result: ImageGenerationResult) {
    if (caseId && onImageGenerated) {
      // Create evidence record for the generated image
      const evidence = {
        id: `generated_${result.id}`,
        title: `AI Generated: ${result.prompt.substring(0, 50)}...`,
        description: `Generated image from prompt: ${result.prompt}`,
        evidenceType: 'image',
        fileUrl: result.imageUrl,
        metadata: {
          aiGenerated: true,
          provider: result.provider,
          parameters: result.parameters,
          generatedAt: result.timestamp
        },
        tags: ['ai-generated', result.provider, selectedStyle]
      };
      
      onImageGenerated(result);
    }
  }

  async function regenerateWithSeed(result: ImageGenerationResult) {
    prompt = result.prompt;
    if (result.metadata.seed !== -1) {
      seed = result.metadata.seed;
    }
    selectedStyle = (result.parameters.style as any) || 'realistic';
    width = result.metadata.size.width;
    height = result.metadata.size.height;
    
    await generateImage();
  }

  function copyPrompt(text: string) {
    navigator.clipboard.writeText(text).then(() => {
      // Could add a toast notification here
    });
  }

  // Legal/evidence specific prompts
  const legalPromptTemplates = [
    { name: 'Crime Scene Recreation', prompt: 'detailed crime scene recreation, professional forensic photography style, accurate lighting, evidence markers' },
    { name: 'Suspect Identification', prompt: 'police sketch style, facial composite, professional law enforcement illustration' },
    { name: 'Traffic Accident Diagram', prompt: 'traffic accident scene diagram, aerial view, clear road markings, vehicle positions, technical illustration' },
    { name: 'Property Damage Documentation', prompt: 'property damage documentation, insurance photo style, clear details, professional lighting' },
    { name: 'Evidence Visualization', prompt: 'forensic evidence visualization, scientific illustration, detailed analysis, laboratory setting' },
    { name: 'Legal Diagram', prompt: 'legal process diagram, flowchart style, professional presentation, clear annotations' }
  ];
</script>

<div class="image-generator nes-container is-rounded {compact ? 'compact' : 'full'}">
  <div class="generator-header">
    <h3>🎨 AI Image Generation</h3>
    <div class="provider-status">
      {#each Array.from(providerStatus.entries()) as [provider, status]}
        <span class="provider-badge nes-badge {status !== 'internal' ? 'is-success' : 'is-warning'}">
          {provider}: {status !== 'internal' ? '✓' : '⚠️'}
        </span>
      {/each}
    </div>
  </div>

  <div class="generation-controls">
    <!-- Prompt Input -->
    <div class="input-group">
      <label class="nes-text">Prompt:</label>
      <textarea 
        class="nes-textarea" 
        bind:value={prompt} 
        placeholder="Describe the image you want to generate..."
        rows="3"
      ></textarea>
    </div>

    <!-- Legal Templates -->
    <div class="template-section">
      <label class="nes-text">Legal Templates:</label>
      <div class="template-buttons">
        {#each legalPromptTemplates as template}
          <button 
            class="template-btn nes-btn is-primary"
            on:on:onclick={() => prompt = template.prompt}
          >
            {template.name}
          </button>
        {/each}
      </div>
    </div>

    <!-- Style and Provider Selection -->
    <div class="selection-row">
      <div class="select-group">
        <label class="nes-text">Style:</label>
        <div class="nes-select">
          <select bind:value={selectedStyle}>
            <option value="realistic">Realistic</option>
            <option value="artistic">Artistic</option>
            <option value="anime">Anime</option>
            <option value="sketch">Sketch</option>
            <option value="legal-diagram">Legal Diagram</option>
            <option value="evidence-recreation">Evidence Recreation</option>
          </select>
        </div>
      </div>

      <div class="select-group">
        <label class="nes-text">Provider:</label>
        <div class="nes-select">
          <select bind:value={selectedProvider}>
            {#each Array.from(providerStatus.keys()) as provider}
              <option value={provider}>
                {provider} {providerStatus.get(provider) !== 'internal' ? '(Available)' : '(Fallback)'}
              </option>
            {/each}
          </select>
        </div>
      </div>
    </div>

    <!-- Advanced Controls -->
    <div class="advanced-toggle">
      <label class="nes-checkbox">
        <input type="checkbox" bind:checked={advancedMode}>
        <span>Advanced Settings</span>
      </label>
    </div>

    {#if advancedMode}
      <div class="advanced-controls nes-container is-dark">
        <div class="input-group">
          <label class="nes-text">Negative Prompt:</label>
          <textarea 
            class="nes-textarea" 
            bind:value={negativePrompt} 
            placeholder="What to avoid in the image..."
            rows="2"
          ></textarea>
        </div>

        <div class="parameter-row">
          <div class="param-group">
            <label class="nes-text">Width:</label>
            <input class="nes-input" type="number" bind:value={width} min="256" max="1024" step="64">
          </div>
          <div class="param-group">
            <label class="nes-text">Height:</label>
            <input class="nes-input" type="number" bind:value={height} min="256" max="1024" step="64">
          </div>
          <div class="param-group">
            <label class="nes-text">Steps:</label>
            <input class="nes-input" type="number" bind:value={steps} min="1" max="100">
          </div>
          <div class="param-group">
            <label class="nes-text">CFG Scale:</label>
            <input class="nes-input" type="number" bind:value={cfgScale} min="1" max="30" step="0.5">
          </div>
          <div class="param-group">
            <label class="nes-text">Seed (-1 for random):</label>
            <input class="nes-input" type="number" bind:value={seed} min="-1" max="999999999">
          </div>
        </div>
      </div>
    {/if}

    <!-- Generation Button and Status -->
    <div class="generate-section">
      <button 
        class="generate-btn nes-btn is-success"
        on:on:onclick={generateImage}
        disabled={$imageGenerationStore.status.isGenerating || !prompt.trim()}
      >
        {#if $imageGenerationStore.status.isGenerating}
          <span class="spinner"></span>
          Generating... ({Math.round($imageGenerationStore.status.progress)}%)
        {:else}
          🎨 Generate Image
        {/if}
      </button>

      {#if $imageGenerationStore.status.isGenerating}
        <div class="progress-info nes-container is-rounded">
          <div class="nes-progress is-primary">
            <progress class="progress" value={$imageGenerationStore.status.progress} max="100">
              {$imageGenerationStore.status.progress}%
            </progress>
          </div>
          <p>{$imageGenerationStore.status.currentStep}</p>
        </div>
      {/if}

      {#if $imageGenerationStore.status.error}
        <div class="error-message nes-container is-error">
          <p>❌ {$imageGenerationStore.status.error}</p>
        </div>
      {/if}
    </div>
  </div>

  <!-- Generated Image Display -->
  {#if $imageGenerationStore.currentGeneration}
    <div class="current-generation nes-container is-rounded">
      <h4>Latest Generation</h4>
      <div class="image-result">
        <img 
          src={$imageGenerationStore.currentGeneration.imageUrl} 
          alt={$imageGenerationStore.currentGeneration.prompt}
          class="generated-image"
        >
        <div class="image-actions">
          <button 
            class="nes-btn is-primary"
            on:on:onclick={() => copyPrompt($imageGenerationStore.currentGeneration!.prompt)}
          >
            📋 Copy Prompt
          </button>
          <button 
            class="nes-btn is-warning"
            on:on:onclick={() => regenerateWithSeed($imageGenerationStore.currentGeneration!)}
          >
            🔄 Regenerate
          </button>
          {#if caseId}
            <button 
              class="nes-btn is-success"
              on:on:onclick={() => useImageAsEvidence($imageGenerationStore.currentGeneration!)}
            >
              📁 Use as Evidence
            </button>
          {/if}
        </div>
        <div class="image-metadata nes-container is-dark">
          <p><strong>Provider:</strong> {$imageGenerationStore.currentGeneration.provider}</p>
          <p><strong>Size:</strong> {$imageGenerationStore.currentGeneration.metadata.size.width}×{$imageGenerationStore.currentGeneration.metadata.size.height}</p>
          <p><strong>Processing Time:</strong> {$imageGenerationStore.currentGeneration.processingTime}ms</p>
          {#if $imageGenerationStore.currentGeneration.metadata.seed !== -1}
            <p><strong>Seed:</strong> {$imageGenerationStore.currentGeneration.metadata.seed}</p>
          {/if}
        </div>
      </div>
    </div>
  {/if}

  <!-- History Section -->
  <div class="history-section">
    <div class="history-header">
      <button 
        class="nes-btn is-normal"
        on:on:onclick={() => showHistory = !showHistory}
      >
        📚 History ({generationHistory.length})
      </button>
      {#if generationHistory.length > 0}
        <button 
          class="nes-btn is-error"
          on:on:onclick={() => { 
            imageGenerationService.clearHistory(); 
            generationHistory = []; 
          }}
        >
          🗑️ Clear
        </button>
      {/if}
    </div>

    {#if showHistory}
      <div class="history-grid">
        {#each generationHistory as result}
          <div class="history-item nes-container is-rounded">
            <img 
              src={result.imageUrl} 
              alt={result.prompt}
              class="history-thumbnail"
              on:on:onclick={() => selectedImage = result}
            >
            <div class="history-info">
              <p class="history-prompt">{result.prompt.substring(0, 50)}...</p>
              <p class="history-meta">{result.provider} • {new Date(result.timestamp).toLocaleTimeString()}</p>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Selected Image Modal -->
  {#if selectedImage}
    <div class="modal-overlay" on:on:onclick={() => selectedImage = null}>
      <div class="modal-content nes-container is-rounded" on:on:onclick={(e) => e.stopPropagation()}>
        <div class="modal-header">
          <h4>Generated Image Details</h4>
          <button class="nes-btn is-error" on:on:onclick={() => selectedImage = null}>×</button>
        </div>
        <div class="modal-body">
          <img src={selectedImage.imageUrl} alt={selectedImage.prompt} class="modal-image">
          <div class="modal-info">
            <p><strong>Prompt:</strong> {selectedImage.prompt}</p>
            <p><strong>Style:</strong> {selectedImage.parameters.style || 'realistic'}</p>
            <p><strong>Provider:</strong> {selectedImage.provider}</p>
            <p><strong>Generated:</strong> {selectedImage.timestamp.toLocaleString()}</p>
          </div>
          <div class="modal-actions">
            <button 
              class="nes-btn is-primary"
              on:on:onclick={() => {
                prompt = selectedImage!.prompt;
                selectedImage = null;
              }}
            >
              Use Prompt
            </button>
            <button 
              class="nes-btn is-warning"
              on:on:onclick={() => regenerateWithSeed(selectedImage!)}
            >
              Regenerate
            </button>
            {#if caseId}
              <button 
                class="nes-btn is-success"
                on:on:onclick={() => {
                  useImageAsEvidence(selectedImage!);
                  selectedImage = null;
                }}
              >
                Use as Evidence
              </button>
            {/if}
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .image-generator {
    max-width: 100%;
    margin: 1rem 0;
  }

  .image-generator.compact {
    max-width: 600px;
  }

  .generator-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .provider-status {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .provider-badge {
    font-size: 0.75rem;
  }

  .generation-controls {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .input-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .template-section {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .template-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .template-btn {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
  }

  .selection-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }

  .select-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    flex: 1;
    min-width: 150px;
  }

  .advanced-toggle {
    margin: 0.5rem 0;
  }

  .advanced-controls {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 1rem;
  }

  .parameter-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }

  .param-group {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    flex: 1;
    min-width: 100px;
  }

  .generate-section {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    align-items: center;
  }

  .generate-btn {
    padding: 1rem 2rem;
    font-size: 1.1rem;
    min-width: 200px;
  }

  .progress-info {
    width: 100%;
    max-width: 400px;
    text-align: center;
  }

  .spinner {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid transparent;
    border-top: 2px solid currentColor;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-right: 0.5rem;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .current-generation {
    margin: 1rem 0;
  }

  .image-result {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .generated-image {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }

  .image-actions {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    justify-content: center;
  }

  .image-metadata {
    font-size: 0.875rem;
    padding: 0.5rem;
  }

  .history-section {
    margin-top: 2rem;
  }

  .history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .history-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
  }

  .history-item {
    cursor: pointer;
    transition: transform 0.2s ease;
  }

  .history-item:hover {
    transform: translateY(-2px);
  }

  .history-thumbnail {
    width: 100%;
    height: 120px;
    object-fit: cover;
    border-radius: 4px;
  }

  .history-info {
    margin-top: 0.5rem;
  }

  .history-prompt {
    font-size: 0.75rem;
    font-weight: bold;
    margin: 0;
  }

  .history-meta {
    font-size: 0.7rem;
    color: #666;
    margin: 0;
  }

  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
    padding: 1rem;
  }

  .modal-content {
    max-width: 90vw;
    max-height: 90vh;
    overflow: auto;
    background: white;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .modal-image {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    margin-bottom: 1rem;
  }

  .modal-info {
    margin-bottom: 1rem;
  }

  .modal-actions {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    justify-content: center;
  }

  .error-message {
    color: #d32f2f;
    text-align: center;
  }

  @media (max-width: 768px) {
    .selection-row,
    .parameter-row {
      flex-direction: column;
    }
    
    .history-grid {
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    }
    
    .modal-content {
      margin: 0.5rem;
      max-width: calc(100vw - 1rem);
    }
  }
</style>