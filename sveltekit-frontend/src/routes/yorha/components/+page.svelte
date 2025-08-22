<!-- YoRHa 3D Components Gallery -->
<script lang="ts">
  // $state runtime rune is provided globally via src/types/svelte-helpers.d.ts
  import { onMount } from 'svelte';
  import { yorhaAPI } from '$lib/components/three/yorha-ui/api/YoRHaAPIClient';
  // Import types only to avoid 3D dependencies for now
  import type {
    YoRHaButton3DOptions,
    YoRHaPanel3DOptions,
    YoRHaInput3DOptions,
    YoRHaModal3DOptions
  } from '$lib/components/three/yorha-ui/api/YoRHaAPIClient';
  import {
    Gamepad2,
    Monitor,
    Cpu,
    Settings,
    Eye,
    Code,
    Palette,
    Layers
  } from 'lucide-svelte';

  // Component instances and options
  let selectedComponent = $state('button');
  let previewMode = $state('3d');
  let isLoading = $state(false);

  // Component configurations
  let buttonConfig = $state<YoRHaButton3DOptions>({
    text: 'YoRHa Button',
    variant: 'primary',
    size: 'medium',
    icon: 'terminal',
    loading: false,
    disabled: false,
    glowEffect: true,
    hoverAnimation: true
  });

  let panelConfig = $state<YoRHaPanel3DOptions>({
    title: 'YoRHa Panel',
    variant: 'default',
    width: 400,
    height: 300,
    scrollable: true,
    collapsible: true,
    glitchEffect: false,
    borderGlow: true
  });

  let inputConfig = $state<YoRHaInput3DOptions>({
    placeholder: 'Enter command...',
    type: 'text',
    variant: 'default',
    value: '',
    error: false,
    focused: false,
    scanlineEffect: true,
    terminalMode: true
  });

  let modalConfig = $state<YoRHaModal3DOptions>({
    title: 'YoRHa Modal',
    variant: 'default',
    size: 'medium',
    closable: true,
    open: false,
    backdropBlur: true,
    hologramEffect: true
  });

  // UI state
  let yorhaUI: any | null = null;
  let canvasContainer: HTMLElement;

  // Component variants and options
  const componentTypes = [
    { id: 'button', label: 'Button 3D', icon: Gamepad2, description: '3D interactive buttons with hover effects' },
    { id: 'panel', label: 'Panel 3D', icon: Monitor, description: 'Floating 3D panels with content areas' },
    { id: 'input', label: 'Input 3D', icon: Code, description: 'Terminal-style 3D input fields' },
    { id: 'modal', label: 'Modal 3D', icon: Layers, description: 'Holographic modal dialogs' }
  ];

  const previewModes = [
    { id: '3d', label: '3D View', icon: Eye },
    { id: 'code', label: 'Code', icon: Code },
    { id: 'config', label: 'Config', icon: Settings }
  ];

  onMount(async () => {
    // Initialize 3D UI (placeholder for now)
    if (canvasContainer) {
      // TODO: Initialize 3D UI when Three.js dependencies are resolved
      console.log('3D UI container ready:', canvasContainer);
      updatePreview();
    }

    // Load configurations from API
    try {
      await loadComponentConfigs();
    } catch (error) {
      console.warn('API configurations not available, using defaults');
    }
  });

  async function loadComponentConfigs() {
    isLoading = true;
    try {
      const [button, panel, input, modal] = await Promise.all([
        yorhaAPI.createButtonFromAPI('demo-button'),
        yorhaAPI.createPanelFromAPI('demo-panel'),
        yorhaAPI.createInputFromAPI('demo-input'),
        yorhaAPI.createModalFromAPI('demo-modal')
      ]);

      buttonConfig = button;
      panelConfig = panel;
      inputConfig = input;
      modalConfig = modal;

      updatePreview();
    } catch (error) {
      console.error('Failed to load component configs:', error);
    } finally {
      isLoading = false;
    }
  }

  function updatePreview() {
    if (!yorhaUI) {
      console.log('Preview updated for:', selectedComponent, getCurrentConfig());
      return;
    }

    // Clear existing components
    yorhaUI.clearComponents();

    // Add selected component
    switch (selectedComponent) {
      case 'button':
        yorhaUI.addButton('preview-button', buttonConfig);
        break;
      case 'panel':
        yorhaUI.addPanel('preview-panel', panelConfig);
        break;
      case 'input':
        yorhaUI.addInput('preview-input', inputConfig);
        break;
      case 'modal':
        yorhaUI.addModal('preview-modal', modalConfig);
        break;
    }

    yorhaUI.render();
  }

  function onComponentChange(componentId: string) {
    selectedComponent = componentId;
    updatePreview();
  }

  function onConfigChange() {
    updatePreview();
  }

  function exportConfig() {
    const configs = {
      button: buttonConfig,
      panel: panelConfig,
      input: inputConfig,
      modal: modalConfig
    };

    const blob = new Blob([JSON.stringify(configs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'yorha-components-config.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  async function saveConfig() {
    try {
      await yorhaAPI.updateComponentConfig(`demo-${selectedComponent}`, getCurrentConfig());
      console.log('Configuration saved');
    } catch (error) {
      console.error('Failed to save configuration:', error);
    }
  }

  function getCurrentConfig() {
    switch (selectedComponent) {
      case 'button': return buttonConfig;
      case 'panel': return panelConfig;
      case 'input': return inputConfig;
      case 'modal': return modalConfig;
      default: return {};
    }
  }

  function resetConfig() {
    switch (selectedComponent) {
      case 'button':
        buttonConfig = {
          text: 'YoRHa Button',
          variant: 'primary',
          size: 'medium',
          icon: 'terminal',
          loading: false,
          disabled: false,
          glowEffect: true,
          hoverAnimation: true
        };
        break;
      case 'panel':
        panelConfig = {
          title: 'YoRHa Panel',
          variant: 'default',
          width: 400,
          height: 300,
          scrollable: true,
          collapsible: true,
          glitchEffect: false,
          borderGlow: true
        };
        break;
      case 'input':
        inputConfig = {
          placeholder: 'Enter command...',
          type: 'text',
          variant: 'default',
          value: '',
          error: false,
          focused: false,
          scanlineEffect: true,
          terminalMode: true
        };
        break;
      case 'modal':
        modalConfig = {
          title: 'YoRHa Modal',
          variant: 'default',
          size: 'medium',
          closable: true,
          open: false,
          backdropBlur: true,
          hologramEffect: true
        };
        break;
    }
    updatePreview();
  }
</script>

<svelte:head>
  <title>YoRHa Components - 3D UI Gallery</title>
</svelte:head>

<div class="yorha-components-page">
  <!-- Page Header -->
  <header class="yorha-page-header">
    <div class="yorha-header-content">
      <div class="yorha-header-title">
        <Gamepad2 size={48} />
        <h1>3D COMPONENT GALLERY</h1>
        <div class="yorha-header-subtitle">INTERACTIVE YORHA UI ELEMENTS</div>
      </div>
    </div>
  </header>

  <div class="yorha-components-layout">
    <!-- Sidebar Controls -->
    <aside class="yorha-controls-sidebar">
      <!-- Component Selection -->
      <section class="yorha-control-section">
        <h3 class="yorha-control-title">
          <Layers size={20} />
          COMPONENTS
        </h3>
        <div class="yorha-component-list">
          {#each componentTypes as component}
            <button
              class="yorha-component-btn"
              class:yorha-component-active={selectedComponent === component.id}
              onclick={() => onComponentChange(component.id)}
            >
              <svelte:component this={component.icon} size={18} />
              <div class="yorha-component-info">
                <span class="yorha-component-label">{component.label}</span>
                <span class="yorha-component-desc">{component.description}</span>
              </div>
            </button>
          {/each}
        </div>
      </section>

      <!-- Preview Mode -->
      <section class="yorha-control-section">
        <h3 class="yorha-control-title">
          <Eye size={20} />
          VIEW MODE
        </h3>
        <div class="yorha-mode-buttons">
          {#each previewModes as mode}
            <button
              class="yorha-mode-btn"
              class:yorha-mode-active={previewMode === mode.id}
              onclick={() => previewMode = mode.id}
            >
              <svelte:component this={mode.icon} size={16} />
              <span>{mode.label}</span>
            </button>
          {/each}
        </div>
      </section>

      <!-- Configuration Panel -->
      {#if previewMode === 'config'}
        <section class="yorha-control-section">
          <h3 class="yorha-control-title">
            <Settings size={20} />
            CONFIGURATION
          </h3>

          <div class="yorha-config-form">
            {#if selectedComponent === 'button'}
              <div class="yorha-config-group">
                <label>Text</label>
                <input type="text" bind:value={buttonConfig.text} oninput={onConfigChange} />
              </div>
              <div class="yorha-config-group">
                <label>Variant</label>
                <select bind:value={buttonConfig.variant} onchange={onConfigChange}>
                  <option value="primary">Primary</option>
                  <option value="secondary">Secondary</option>
                  <option value="danger">Danger</option>
                  <option value="ghost">Ghost</option>
                </select>
              </div>
              <div class="yorha-config-group">
                <label>Size</label>
                <select bind:value={buttonConfig.size} onchange={onConfigChange}>
                  <option value="small">Small</option>
                  <option value="medium">Medium</option>
                  <option value="large">Large</option>
                </select>
              </div>
              <div class="yorha-config-group">
                <label class="yorha-checkbox">
                  <input type="checkbox" bind:checked={buttonConfig.loading} onchange={onConfigChange} />
                  <span>Loading</span>
                </label>
              </div>
              <div class="yorha-config-group">
                <label class="yorha-checkbox">
                  <input type="checkbox" bind:checked={buttonConfig.disabled} onchange={onConfigChange} />
                  <span>Disabled</span>
                </label>
              </div>
              <div class="yorha-config-group">
                <label class="yorha-checkbox">
                  <input type="checkbox" bind:checked={buttonConfig.glowEffect} onchange={onConfigChange} />
                  <span>Glow Effect</span>
                </label>
              </div>
            {/if}

            {#if selectedComponent === 'panel'}
              <div class="yorha-config-group">
                <label>Title</label>
                <input type="text" bind:value={panelConfig.title} oninput={onConfigChange} />
              </div>
              <div class="yorha-config-group">
                <label>Width</label>
                <input type="number" bind:value={panelConfig.width} oninput={onConfigChange} />
              </div>
              <div class="yorha-config-group">
                <label>Height</label>
                <input type="number" bind:value={panelConfig.height} oninput={onConfigChange} />
              </div>
              <div class="yorha-config-group">
                <label class="yorha-checkbox">
                  <input type="checkbox" bind:checked={panelConfig.scrollable} onchange={onConfigChange} />
                  <span>Scrollable</span>
                </label>
              </div>
              <div class="yorha-config-group">
                <label class="yorha-checkbox">
                  <input type="checkbox" bind:checked={panelConfig.collapsible} onchange={onConfigChange} />
                  <span>Collapsible</span>
                </label>
              </div>
              <div class="yorha-config-group">
                <label class="yorha-checkbox">
                  <input type="checkbox" bind:checked={panelConfig.borderGlow} onchange={onConfigChange} />
                  <span>Border Glow</span>
                </label>
              </div>
            {/if}

            {#if selectedComponent === 'input'}
              <div class="yorha-config-group">
                <label>Placeholder</label>
                <input type="text" bind:value={inputConfig.placeholder} oninput={onConfigChange} />
              </div>
              <div class="yorha-config-group">
                <label>Type</label>
                <select bind:value={inputConfig.type} onchange={onConfigChange}>
                  <option value="text">Text</option>
                  <option value="password">Password</option>
                  <option value="email">Email</option>
                  <option value="number">Number</option>
                </select>
              </div>
              <div class="yorha-config-group">
                <label>Value</label>
                <input type="text" bind:value={inputConfig.value} oninput={onConfigChange} />
              </div>
              <div class="yorha-config-group">
                <label class="yorha-checkbox">
                  <input type="checkbox" bind:checked={inputConfig.error} onchange={onConfigChange} />
                  <span>Error State</span>
                </label>
              </div>
              <div class="yorha-config-group">
                <label class="yorha-checkbox">
                  <input type="checkbox" bind:checked={inputConfig.scanlineEffect} onchange={onConfigChange} />
                  <span>Scanline Effect</span>
                </label>
              </div>
            {/if}

            {#if selectedComponent === 'modal'}
              <div class="yorha-config-group">
                <label>Title</label>
                <input type="text" bind:value={modalConfig.title} oninput={onConfigChange} />
              </div>
              <div class="yorha-config-group">
                <label>Size</label>
                <select bind:value={modalConfig.size} onchange={onConfigChange}>
                  <option value="small">Small</option>
                  <option value="medium">Medium</option>
                  <option value="large">Large</option>
                  <option value="fullscreen">Fullscreen</option>
                </select>
              </div>
              <div class="yorha-config-group">
                <label class="yorha-checkbox">
                  <input type="checkbox" bind:checked={modalConfig.open} onchange={onConfigChange} />
                  <span>Open</span>
                </label>
              </div>
              <div class="yorha-config-group">
                <label class="yorha-checkbox">
                  <input type="checkbox" bind:checked={modalConfig.closable} onchange={onConfigChange} />
                  <span>Closable</span>
                </label>
              </div>
              <div class="yorha-config-group">
                <label class="yorha-checkbox">
                  <input type="checkbox" bind:checked={modalConfig.hologramEffect} onchange={onConfigChange} />
                  <span>Hologram Effect</span>
                </label>
              </div>
            {/if}
          </div>

          <!-- Config Actions -->
          <div class="yorha-config-actions">
            <button class="yorha-config-btn yorha-btn-save" onclick={saveConfig}>
              SAVE
            </button>
            <button class="yorha-config-btn yorha-btn-reset" onclick={resetConfig}>
              RESET
            </button>
            <button class="yorha-config-btn yorha-btn-export" onclick={exportConfig}>
              EXPORT
            </button>
          </div>
        </section>
      {/if}
    </aside>

    <!-- Main Preview Area -->
    <main class="yorha-preview-area">
      {#if previewMode === '3d'}
        <div class="yorha-3d-container">
          <div bind:this={canvasContainer} class="yorha-canvas-container"></div>
          {#if isLoading}
            <div class="yorha-3d-loading">
              <div class="yorha-loading-spinner"></div>
              <span>INITIALIZING 3D PREVIEW...</span>
            </div>
          {/if}
        </div>
      {:else if previewMode === 'code'}
        <div class="yorha-code-view">
          <h3>Component Configuration</h3>
          <pre class="yorha-code-block">{JSON.stringify(getCurrentConfig(), null, 2)}</pre>
        </div>
      {:else if previewMode === 'config'}
        <div class="yorha-config-preview">
          <h3>Live Configuration</h3>
          <div class="yorha-config-display">
            <p>Use the sidebar controls to modify the {selectedComponent} component configuration in real-time.</p>
            <div class="yorha-config-summary">
              <h4>Current Configuration:</h4>
              <pre>{JSON.stringify(getCurrentConfig(), null, 2)}</pre>
            </div>
          </div>
        </div>
      {/if}

      <!-- Component Info -->
      <div class="yorha-component-info-panel">
        {#each componentTypes as component}
          {#if component.id === selectedComponent}
            <div class="yorha-info-content">
              <svelte:component this={component.icon} size={24} />
              <h4>{component.label}</h4>
              <p>{component.description}</p>
            </div>
          {/if}
        {/each}
      </div>
    </main>
  </div>
</div>

<style>
  .yorha-components-page {
    @apply min-h-screen;
  }

  /* Page Header */
  .yorha-page-header {
    @apply py-12 px-6 border-b border-amber-400 border-opacity-30;
    background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(255, 191, 0, 0.05) 100%);
  }

  .yorha-header-content {
    @apply max-w-6xl mx-auto text-center;
  }

  .yorha-header-title h1 {
    @apply text-3xl md:text-4xl font-bold tracking-wider text-amber-400 flex items-center justify-center gap-4;
    text-shadow: 0 0 20px rgba(255, 191, 0, 0.5);
  }

  .yorha-header-subtitle {
    @apply text-lg text-amber-300 tracking-wide opacity-80 mt-2;
  }

  /* Layout */
  .yorha-components-layout {
    @apply flex min-h-screen;
  }

  /* Sidebar */
  .yorha-controls-sidebar {
    @apply w-80 bg-gray-900 border-r border-amber-400 border-opacity-30 p-6 space-y-6 overflow-y-auto;
    background: linear-gradient(135deg, rgba(0, 0, 0, 0.95) 0%, rgba(255, 191, 0, 0.05) 100%);
  }

  .yorha-control-section {
    @apply space-y-4;
  }

  .yorha-control-title {
    @apply text-lg font-bold text-amber-400 tracking-wider flex items-center gap-2;
  }

  /* Component List */
  .yorha-component-list {
    @apply space-y-2;
  }

  .yorha-component-btn {
    @apply w-full p-4 text-left flex items-center gap-3 border border-amber-400 border-opacity-30;
    @apply hover:border-opacity-60 hover:bg-amber-400 hover:bg-opacity-10 transition-all;
  }

  .yorha-component-active {
    @apply border-amber-400 bg-amber-400 bg-opacity-20;
  }

  .yorha-component-info {
    @apply flex-1 min-w-0;
  }

  .yorha-component-label {
    @apply block font-semibold text-amber-400;
  }

  .yorha-component-desc {
    @apply block text-xs text-amber-300 opacity-60 truncate;
  }

  /* Mode Buttons */
  .yorha-mode-buttons {
    @apply grid grid-cols-3 gap-2;
  }

  .yorha-mode-btn {
    @apply px-3 py-2 text-xs font-mono border border-amber-400 border-opacity-30;
    @apply hover:border-opacity-60 hover:bg-amber-400 hover:bg-opacity-10 transition-all;
    @apply flex items-center justify-center gap-2;
  }

  .yorha-mode-active {
    @apply border-amber-400 bg-amber-400 bg-opacity-20 text-amber-400;
  }

  /* Configuration Form */
  .yorha-config-form {
    @apply space-y-4;
  }

  .yorha-config-group {
    @apply space-y-2;
  }

  .yorha-config-group label {
    @apply block text-sm font-semibold text-amber-400;
  }

  .yorha-config-group input,
  .yorha-config-group select {
    @apply w-full px-3 py-2 bg-black border border-amber-400 border-opacity-30 text-amber-300;
    @apply focus:border-opacity-60 focus:outline-none font-mono text-sm;
  }

  .yorha-checkbox {
    @apply flex items-center gap-2 cursor-pointer;
  }

  .yorha-checkbox input[type="checkbox"] {
    @apply w-auto;
  }

  /* Config Actions */
  .yorha-config-actions {
    @apply grid grid-cols-3 gap-2 pt-4 border-t border-amber-400 border-opacity-30;
  }

  .yorha-config-btn {
    @apply px-3 py-2 text-xs font-mono border transition-all;
  }

  .yorha-btn-save {
    @apply border-green-400 text-green-400 hover:bg-green-400 hover:text-black;
  }

  .yorha-btn-reset {
    @apply border-orange-400 text-orange-400 hover:bg-orange-400 hover:text-black;
  }

  .yorha-btn-export {
    @apply border-blue-400 text-blue-400 hover:bg-blue-400 hover:text-black;
  }

  /* Preview Area */
  .yorha-preview-area {
    @apply flex-1 relative;
  }

  .yorha-3d-container {
    @apply h-full relative;
  }

  .yorha-canvas-container {
    @apply w-full h-full min-h-screen;
  }

  .yorha-3d-loading {
    @apply absolute inset-0 flex flex-col items-center justify-center bg-black bg-opacity-50 space-y-4;
  }

  .yorha-loading-spinner {
    @apply w-8 h-8 border-2 border-amber-400 border-t-transparent rounded-full;
    animation: spin 1s linear infinite;
  }

  /* Code View */
  .yorha-code-view {
    @apply p-6 h-full;
  }

  .yorha-code-view h3 {
    @apply text-xl font-bold text-amber-400 mb-4;
  }

  .yorha-code-block {
    @apply bg-black border border-amber-400 border-opacity-30 p-4 text-amber-300 text-sm font-mono;
    @apply whitespace-pre-wrap overflow-auto max-h-96;
  }

  /* Config Preview */
  .yorha-config-preview {
    @apply p-6 h-full;
  }

  .yorha-config-preview h3 {
    @apply text-xl font-bold text-amber-400 mb-4;
  }

  .yorha-config-display p {
    @apply text-amber-300 mb-4;
  }

  .yorha-config-summary h4 {
    @apply text-lg font-semibold text-amber-400 mb-2;
  }

  .yorha-config-summary pre {
    @apply bg-black border border-amber-400 border-opacity-30 p-4 text-amber-300 text-sm font-mono;
    @apply whitespace-pre-wrap overflow-auto max-h-64;
  }

  /* Component Info Panel */
  .yorha-component-info-panel {
    @apply absolute bottom-6 right-6 bg-gray-900 border border-amber-400 border-opacity-30 p-4;
    @apply backdrop-blur-sm;
  }

  .yorha-info-content {
    @apply flex items-center gap-3;
  }

  .yorha-info-content h4 {
    @apply font-semibold text-amber-400;
  }

  .yorha-info-content p {
    @apply text-xs text-amber-300 opacity-80;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Responsive */
  @media (max-width: 1024px) {
    .yorha-components-layout {
      @apply flex-col;
    }

    .yorha-controls-sidebar {
      @apply w-full border-r-0 border-b;
    }
  }
</style>