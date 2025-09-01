<!--
AI Image Generation Gallery
Dedicated page for image generation, management, and diffusion controls
-->
<script lang="ts">
  import { onMount } from "svelte";
  import { page } from "$app/stores";
  import LocalImageGenerator from "$lib/components/ai/LocalImageGenerator.svelte";
  import { imageGenerationStore } from "$lib/services/local-image-generation-service.js";
  
  const caseId = $page.params.id;
  let caseTitle = $state("Case Images");
  
  // Gallery state
  let viewMode = $state<'grid' | 'list' | 'detail'>('grid');
  let filterProvider = $state<'all' | 'stable-diffusion-webui' | 'comfyui' | 'ollama-vision' | 'fallback'>('all');
  let sortOrder = $state<'newest' | 'oldest' | 'prompt'>('newest');
  let searchQuery = $state('');
  
  // Filtered and sorted images
  let filteredImages = $derived(() => {
    let images = $imageGenerationStore.history;
    
    // Filter by provider
    if (filterProvider !== 'all') {
      images = images.filter(img => img.provider === filterProvider);
    }
    
    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      images = images.filter(img => 
        img.prompt.toLowerCase().includes(query) ||
        img.provider.toLowerCase().includes(query) ||
        (img.parameters.style && img.parameters.style.toLowerCase().includes(query))
      );
    }
    
    // Sort images
    images = [...images].sort((a, b) => {
      switch (sortOrder) {
        case 'newest':
          return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
        case 'oldest':
          return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
        case 'prompt':
          return a.prompt.localeCompare(b.prompt);
        default:
          return 0;
      }
    });
    
    return images;
  });

  onMount(async () => {
    // Load case title
    try {
      const response = await fetch(`/api/cases/${caseId}`);
      if (response.ok) {
        const caseData = await response.json();
        caseTitle = `${caseData.title} - AI Generated Images`;
      }
    } catch (error) {
      console.error('Failed to load case data:', error);
    }
  });

  function handleImageGenerated(result: any) {
    // The image is automatically added to the store by the service
    console.log('New image generated:', result);
  }

  function exportImages() {
    // Export all generated images as a ZIP file
    const data = {
      caseId,
      images: filteredImages().map(img => ({
        ...img,
        // Convert base64 to download-ready format if needed
      }))
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `case-${caseId}-generated-images-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function shareImage(image: any) {
    if (navigator.share) {
      navigator.share({
        title: `AI Generated Image - ${image.prompt.substring(0, 50)}...`,
        text: `Generated with ${image.provider}: ${image.prompt}`,
        files: [] // Could convert base64 to file if needed
      });
    } else {
      // Fallback to clipboard
      navigator.clipboard.writeText(`AI Generated Image: ${image.prompt}\nProvider: ${image.provider}\nGenerated: ${image.timestamp}`);
    }
  }
</script>

<svelte:head>
  <title>{caseTitle}</title>
</svelte:head>

<div class="image-gallery-page">
  <!-- Header -->
  <header class="gallery-header nes-container is-rounded">
    <div class="header-content">
      <div>
        <h1>🎨 AI Image Generation</h1>
        <p class="case-info">Case ID: {caseId}</p>
        <p class="image-count">{filteredImages().length} generated images</p>
      </div>
      <div class="header-actions">
        <a href="/cases/{caseId}/enhanced" class="nes-btn is-normal">
          ← Back to Case
        </a>
        <button class="nes-btn is-success" on:on:onclick={exportImages}>
          📁 Export Images
        </button>
      </div>
    </div>
  </header>

  <!-- Generation Status -->
  {#if $imageGenerationStore.status.isGenerating}
    <div class="generation-status nes-container is-rounded">
      <div class="status-content">
        <div class="nes-progress is-primary">
          <progress class="progress" value={$imageGenerationStore.status.progress} max="100">
            {$imageGenerationStore.status.progress}%
          </progress>
        </div>
        <p>🎨 {$imageGenerationStore.status.currentStep}</p>
      </div>
    </div>
  {/if}

  <div class="gallery-layout">
    <!-- Image Generator Sidebar -->
    <aside class="generator-sidebar">
      <div class="nes-container is-dark with-title">
        <p class="title">Image Generator</p>
        <LocalImageGenerator 
          {caseId} 
          onImageGenerated={handleImageGenerated}
          compact={false}
        />
      </div>
    </aside>

    <!-- Main Gallery -->
    <main class="gallery-main">
      <!-- Gallery Controls -->
      <div class="gallery-controls nes-container is-rounded">
        <div class="controls-row">
          <!-- Search -->
          <div class="search-group">
            <input 
              type="text" 
              class="nes-input" 
              placeholder="Search prompts, providers, styles..."
              bind:value={searchQuery}
            >
          </div>

          <!-- Filters -->
          <div class="filter-group">
            <label class="nes-text">Provider:</label>
            <div class="nes-select">
              <select bind:value={filterProvider}>
                <option value="all">All Providers</option>
                <option value="stable-diffusion-webui">Stable Diffusion WebUI</option>
                <option value="comfyui">ComfyUI</option>
                <option value="ollama-vision">Ollama Vision</option>
                <option value="fallback">Fallback</option>
              </select>
            </div>
          </div>

          <!-- Sort -->
          <div class="sort-group">
            <label class="nes-text">Sort:</label>
            <div class="nes-select">
              <select bind:value={sortOrder}>
                <option value="newest">Newest First</option>
                <option value="oldest">Oldest First</option>
                <option value="prompt">By Prompt</option>
              </select>
            </div>
          </div>

          <!-- View Mode -->
          <div class="view-mode-group">
            <button 
              class="nes-btn {viewMode === 'grid' ? 'is-primary' : 'is-normal'}"
              on:on:onclick={() => viewMode = 'grid'}
            >
              ⊞ Grid
            </button>
            <button 
              class="nes-btn {viewMode === 'list' ? 'is-primary' : 'is-normal'}"
              on:on:onclick={() => viewMode = 'list'}
            >
              ☰ List
            </button>
          </div>
        </div>
      </div>

      <!-- Gallery Content -->
      <div class="gallery-content">
        {#if filteredImages().length === 0}
          <div class="empty-state nes-container is-rounded">
            <h3>No Images Generated Yet</h3>
            <p>Use the generator on the left to create your first AI-generated image for this case.</p>
            <div class="empty-tips">
              <h4>Tips for Legal Image Generation:</h4>
              <ul>
                <li>• Use the "Evidence Recreation" style for crime scene reconstructions</li>
                <li>• Try "Legal Diagram" for process visualizations</li>
                <li>• Use specific, descriptive prompts for better results</li>
                <li>• Consider multiple angles and perspectives</li>
              </ul>
            </div>
          </div>
        {:else if viewMode === 'grid'}
          <div class="image-grid">
            {#each filteredImages() as image}
              <div class="image-card nes-container is-rounded">
                <div class="image-container">
                  <img src={image.imageUrl} alt={image.prompt} class="gallery-image">
                  <div class="image-overlay">
                    <button 
                      class="nes-btn is-primary overlay-btn"
                      on:on:onclick={() => shareImage(image)}
                    >
                      📤
                    </button>
                  </div>
                </div>
                
                <div class="image-info">
                  <p class="image-prompt">{image.prompt.substring(0, 80)}...</p>
                  <div class="image-meta">
                    <span class="provider-badge nes-badge is-{image.provider === 'stable-diffusion-webui' ? 'success' : image.provider === 'fallback' ? 'warning' : 'primary'}">
                      {image.provider}
                    </span>
                    <span class="timestamp">{new Date(image.timestamp).toLocaleDateString()}</span>
                  </div>
                  
                  {#if image.parameters.style}
                    <div class="style-info">
                      <span class="nes-badge is-normal">{image.parameters.style}</span>
                    </div>
                  {/if}
                  
                  <div class="technical-info">
                    <small>
                      {image.metadata.size.width}×{image.metadata.size.height} • 
                      {image.processingTime}ms
                      {#if image.metadata.seed !== -1}
                        • Seed: {image.metadata.seed}
                      {/if}
                    </small>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {:else}
          <div class="image-list">
            {#each filteredImages() as image}
              <div class="list-item nes-container is-rounded">
                <div class="list-image">
                  <img src={image.imageUrl} alt={image.prompt} class="list-thumbnail">
                </div>
                <div class="list-content">
                  <h4 class="list-prompt">{image.prompt}</h4>
                  <div class="list-meta">
                    <span class="provider-badge nes-badge is-{image.provider === 'stable-diffusion-webui' ? 'success' : image.provider === 'fallback' ? 'warning' : 'primary'}">
                      {image.provider}
                    </span>
                    {#if image.parameters.style}
                      <span class="nes-badge is-normal">{image.parameters.style}</span>
                    {/if}
                    <span class="timestamp">{new Date(image.timestamp).toLocaleString()}</span>
                  </div>
                  <div class="list-technical">
                    {image.metadata.size.width}×{image.metadata.size.height} • 
                    Processing: {image.processingTime}ms
                    {#if image.metadata.seed !== -1}
                      • Seed: {image.metadata.seed}
                    {/if}
                  </div>
                </div>
                <div class="list-actions">
                  <button 
                    class="nes-btn is-primary"
                    on:on:onclick={() => shareImage(image)}
                  >
                    📤 Share
                  </button>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </main>
  </div>
</div>

<style>
  .image-gallery-page {
    min-height: 100vh;
    background: #f5f5f5;
    padding: 1rem;
  }

  .gallery-header {
    margin-bottom: 2rem;
  }

  .header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 2rem;
  }

  .case-info {
    color: #666;
    margin: 0.5rem 0;
  }

  .image-count {
    font-weight: bold;
    color: #333;
    margin: 0;
  }

  .header-actions {
    display: flex;
    gap: 1rem;
  }

  .generation-status {
    margin-bottom: 1rem;
  }

  .status-content {
    text-align: center;
  }

  .gallery-layout {
    display: grid;
    grid-template-columns: 400px 1fr;
    gap: 2rem;
  }

  .generator-sidebar {
    height: fit-content;
    position: sticky;
    top: 1rem;
  }

  .gallery-controls {
    margin-bottom: 2rem;
  }

  .controls-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    align-items: end;
  }

  .search-group {
    flex: 1;
    min-width: 200px;
  }

  .filter-group,
  .sort-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .view-mode-group {
    display: flex;
    gap: 0.5rem;
  }

  .empty-state {
    text-align: center;
    padding: 3rem 2rem;
  }

  .empty-tips {
    text-align: left;
    margin-top: 2rem;
  }

  .empty-tips ul {
    padding-left: 1rem;
  }

  /* Grid View */
  .image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.5rem;
  }

  .image-card {
    overflow: hidden;
  }

  .image-container {
    position: relative;
    width: 100%;
    height: 200px;
    overflow: hidden;
  }

  .gallery-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.3s ease;
  }

  .image-overlay {
    position: absolute;
    top: 0;
    right: 0;
    padding: 0.5rem;
    opacity: 0;
    transition: opacity 0.3s ease;
  }

  .image-card:hover .image-overlay {
    opacity: 1;
  }

  .image-card:hover .gallery-image {
    transform: scale(1.05);
  }

  .overlay-btn {
    padding: 0.25rem 0.5rem;
  }

  .image-info {
    padding: 1rem;
  }

  .image-prompt {
    font-weight: bold;
    margin-bottom: 0.5rem;
    line-height: 1.4;
  }

  .image-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .timestamp {
    font-size: 0.8rem;
    color: #666;
  }

  .style-info {
    margin-bottom: 0.5rem;
  }

  .technical-info {
    color: #666;
  }

  /* List View */
  .image-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .list-item {
    display: flex;
    gap: 1rem;
    align-items: center;
    padding: 1rem;
  }

  .list-image {
    flex-shrink: 0;
  }

  .list-thumbnail {
    width: 100px;
    height: 100px;
    object-fit: cover;
    border-radius: 4px;
  }

  .list-content {
    flex: 1;
  }

  .list-prompt {
    margin: 0 0 0.5rem 0;
    font-size: 1rem;
  }

  .list-meta {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .list-technical {
    font-size: 0.8rem;
    color: #666;
  }

  .list-actions {
    flex-shrink: 0;
  }

  .provider-badge {
    font-size: 0.7rem;
  }

  /* Responsive Design */
  @media (max-width: 1200px) {
    .gallery-layout {
      grid-template-columns: 1fr;
    }
    
    .generator-sidebar {
      position: static;
    }
  }

  @media (max-width: 768px) {
    .header-content {
      flex-direction: column;
      align-items: flex-start;
    }
    
    .controls-row {
      flex-direction: column;
      align-items: stretch;
    }
    
    .search-group,
    .filter-group,
    .sort-group {
      min-width: auto;
    }
    
    .image-grid {
      grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    }
    
    .list-item {
      flex-direction: column;
      text-align: center;
    }
  }
</style>