<script lang="ts">
  interface Props {
    onsave?: (event?: any) => void;
  }
  let {
    content = '',
    title = 'Untitled Document',
    autoSave = true,
    autoSaveInterval = 5000
  } = $props();



    import { writable } from 'svelte/store';
  import { Save, Maximize, Minimize, Eye, EyeOff, FileText, Keyboard } from 'lucide-svelte';

  // Props
        let { autoSaveInterval = $bindable() } = $props(); // number = 5000; // 5 seconds

  // State management
  let editorElement: HTMLDivElement
  let isFullscreen = false;
  let isFocusMode = false;
  let showShortcuts = false;
  let lastSaved = new Date();
  let wordCount = 0;
  let charCount = 0;
  let readingTime = 0;
  let hasUnsavedChanges = false;

  
  // Auto-save functionality
  let autoSaveTimer: NodeJS.Timeout;
  
  function startAutoSave() {
    if (autoSaveTimer) clearInterval(autoSaveTimer);
    autoSaveTimer = setInterval(() => {
      if (hasUnsavedChanges) {
        saveDocument();
      }
    }, autoSaveInterval);
  }

  function stopAutoSave() {
    if (autoSaveTimer) clearInterval(autoSaveTimer);
  }

  function saveDocument() {
    onsave?.();
    hasUnsavedChanges = false;
    lastSaved = new Date();
  }

  // Document statistics
  function updateStatistics() {
    const text = editorElement?.textContent || '';
    wordCount = text.trim().split(/\s+/).filter(word => word.length > 0).length;
    charCount = text.length;
    readingTime = Math.ceil(wordCount / 200); // Assuming 200 WPM reading speed
    hasUnsavedChanges = true;
  }

  // Keyboard shortcuts
  function handleKeydown(event: KeyboardEvent) {
    const isCtrl = event.ctrlKey || event.metaKey;
    
    // Save (Ctrl+S)
    if (isCtrl && event.key === 's') {
      event.preventDefault();
      saveDocument();
      return;
    }
    
    // Bold (Ctrl+B)
    if (isCtrl && event.key === 'b') {
      event.preventDefault();
      document.execCommand('bold');
      return;
    }
    
    // Italic (Ctrl+I)
    if (isCtrl && event.key === 'i') {
      event.preventDefault();
      document.execCommand('italic');
      return;
    }
    
    // Show shortcuts (Ctrl+/)
    if (isCtrl && event.key === '/') {
      event.preventDefault();
      showShortcuts = !showShortcuts;
      return;
    }
    
    // Fullscreen (F11)
    if (event.key === 'F11') {
      event.preventDefault();
      toggleFullscreen();
      return;
    }
    
    // Focus mode (F10)
    if (event.key === 'F10') {
      event.preventDefault();
      toggleFocusMode();
      return;
    }
  }

  // Fullscreen functionality
  function toggleFullscreen() {
    isFullscreen = !isFullscreen;
    if (isFullscreen) {
      document.documentElement.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  }

  // Focus mode
  function toggleFocusMode() {
    isFocusMode = !isFocusMode;
  }

  // Text formatting
  function formatText(command: string, value?: string) {
    document.execCommand(command, false, value);
    editorElement.focus();
    updateStatistics();
  }

  onMount(() => {
    if (autoSave) startAutoSave();
    
    // Listen for fullscreen changes
    document.addEventListener('fullscreenchange', () => {
      isFullscreen = !!document.fullscreenElement;
    });

    return () => {
      stopAutoSave();
    };
  });

  // Shortcuts data
  const shortcuts = [
    { key: 'Ctrl + S', action: 'Save document' },
    { key: 'Ctrl + B', action: 'Bold text' },
    { key: 'Ctrl + I', action: 'Italic text' },
    { key: 'F11', action: 'Toggle full-screen' },
    { key: 'F10', action: 'Toggle focus mode' },
    { key: 'Ctrl + /', action: 'Show shortcuts' },
    { key: 'Ctrl + Z', action: 'Undo' },
    { key: 'Ctrl + Y', action: 'Redo' },
  ];
</script>

<svelte:window onkeydown={handleKeydown} />

<div 
  class="professional-editor {isFullscreen ? 'fullscreen' : ''} {isFocusMode ? 'focus-mode' : ''}"
  class:yorha-card={!isFullscreen}
>
  <!-- Header -->
  <header class="editor-header" class:dimmed={isFocusMode}>
    <div class="title-section">
      <FileText class="h-5 w-5 text-yorha-primary" />
      <input 
        bind:value={title}
        class="title-input yorha-input"
        placeholder="Document title..."
      />
      {#if hasUnsavedChanges}
        <span class="unsaved-indicator">•</span>
      {/if}
    </div>
    
    <div class="header-actions">
      <button 
        class="action-btn yorha-btn yorha-btn-secondary"
        onclick={() => showShortcuts = !showShortcuts}
        title="Keyboard shortcuts (Ctrl+/)"
      >
        <Keyboard class="h-4 w-4" />
      </button>
      
      <button 
        class="action-btn yorha-btn yorha-btn-secondary"
        onclick={toggleFocusMode}
        title="Focus mode (F10)"
      >
        {#if isFocusMode}
          <EyeOff class="h-4 w-4" />
        {:else}
          <Eye class="h-4 w-4" />
        {/if}
      </button>
      
      <button 
        class="action-btn yorha-btn yorha-btn-secondary"
        onclick={toggleFullscreen}
        title="Fullscreen (F11)"
      >
        {#if isFullscreen}
          <Minimize class="h-4 w-4" />
        {:else}
          <Maximize class="h-4 w-4" />
        {/if}
      </button>
      
      <button 
        class="action-btn yorha-btn yorha-btn-primary"
        onclick={saveDocument}
        title="Save document (Ctrl+S)"
      >
        <Save class="h-4 w-4" />
        Save
      </button>
    </div>
  </header>

  <!-- Toolbar -->
  <div class="editor-toolbar" class:dimmed={isFocusMode}>
    <div class="format-group">
      <button 
        class="format-btn yorha-btn yorha-btn-secondary"
        onclick={() => formatText('bold')}
        title="Bold (Ctrl+B)"
      >
        <strong>B</strong>
      </button>
      
      <button 
        class="format-btn yorha-btn yorha-btn-secondary"
        onclick={() => formatText('italic')}
        title="Italic (Ctrl+I)"
      >
        <em>I</em>
      </button>
      
      <button 
        class="format-btn yorha-btn yorha-btn-secondary"
        onclick={() => formatText('underline')}
        title="Underline"
      >
        <u>U</u>
      </button>
    </div>
    
    <div class="format-group">
      <button 
        class="format-btn yorha-btn yorha-btn-secondary"
        onclick={() => formatText('justifyLeft')}
        title="Align left"
      >
        ⟸
      </button>
      
      <button 
        class="format-btn yorha-btn yorha-btn-secondary"
        onclick={() => formatText('justifyCenter')}
        title="Center"
      >
        ▤
      </button>
      
      <button 
        class="format-btn yorha-btn yorha-btn-secondary"
        onclick={() => formatText('justifyRight')}
        title="Align right"
      >
        ⟹
      </button>
    </div>
    
    <div class="format-group">
      <button 
        class="format-btn yorha-btn yorha-btn-secondary"
        onclick={() => formatText('insertUnorderedList')}
        title="Bullet list"
      >
        ⋯
      </button>
      
      <button 
        class="format-btn yorha-btn yorha-btn-secondary"
        onclick={() => formatText('insertOrderedList')}
        title="Numbered list"
      >
        ①
      </button>
    </div>
  </div>

  <!-- Editor -->
  <div class="editor-container">
    <div 
      bind:this={editorElement}
      class="editor-content"
      contenteditable="true"
      oninput={updateStatistics}
      bind:innerHTML={content}
      placeholder="Start writing your document..."
    ></div>
  </div>

  <!-- Status Bar -->
  <footer class="status-bar" class:dimmed={isFocusMode}>
    <div class="status-left">
      <span class="stat-item">
        Words: {wordCount.toLocaleString()}
      </span>
      <span class="stat-item">
        Characters: {charCount.toLocaleString()}
      </span>
      <span class="stat-item">
        Reading time: {readingTime} min
      </span>
    </div>
    
    <div class="status-right">
      {#if autoSave}
        <span class="auto-save-status">
          Auto-saved {lastSaved.toLocaleTimeString()}
        </span>
      {/if}
    </div>
  </footer>
</div>

<!-- Keyboard Shortcuts Modal -->
{#if showShortcuts}
  <div class="shortcuts-overlay" onclick={() => showShortcuts = false}>
    <div class="shortcuts-modal yorha-card" on:click|stopPropagation>
      <h3 class="shortcuts-title gradient-text-primary">
        Keyboard Shortcuts
      </h3>
      
      <div class="shortcuts-grid">
        {#each shortcuts as shortcut}
          <div class="shortcut-item">
            <kbd class="shortcut-key">{shortcut.key}</kbd>
            <span class="shortcut-action">{shortcut.action}</span>
          </div>
        {/each}
      </div>
      
      <button 
        class="close-shortcuts yorha-btn yorha-btn-primary"
        onclick={() => showShortcuts = false}
      >
        Close
      </button>
    </div>
  </div>
{/if}

<style>
  .professional-editor {
    display: flex
    flex-direction: column
    height: 80vh;
    min-height: 600px;
    background: #f4f1ea;
    border: 1px solid #ada895;
    border-radius: 8px;
    overflow: hidden
    font-family: 'Georgia', 'Times New Roman', serif;
    transition: all 0.3s ease;
  }

  .professional-editor.fullscreen {
    position: fixed
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 9999;
    height: 100vh;
    border-radius: 0;
    border: none
  }

  .professional-editor.focus-mode .dimmed {
    opacity: 0.3;
    transition: opacity 0.3s ease;
  }

  .professional-editor.focus-mode .dimmed:hover {
    opacity: 1;
  }

  /* Header */
  .editor-header {
    display: flex
    align-items: center
    justify-content: space-between;
    padding: 1rem;
    background: #faf8f3;
    border-bottom: 1px solid #ada895;
  }

  .title-section {
    display: flex
    align-items: center
    gap: 0.75rem;
    flex: 1;
  }

  .title-input {
    border: none
    background: transparent
    font-size: 1.125rem;
    font-weight: 600;
    color: #3a372f;
    max-width: 400px;
  }

  .title-input:focus {
    outline: none
    border-bottom: 2px solid #3a372f;
  }

  .unsaved-indicator {
    color: #ef4444;
    font-size: 1.5rem;
    font-weight: bold
  }

  .header-actions {
    display: flex
    gap: 0.5rem;
  }

  .action-btn {
    display: flex
    align-items: center
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
  }

  /* Toolbar */
  .editor-toolbar {
    display: flex
    align-items: center
    gap: 1rem;
    padding: 0.75rem 1rem;
    background: #faf8f3;
    border-bottom: 1px solid #ada895;
  }

  .format-group {
    display: flex
    gap: 0.25rem;
  }

  .format-btn {
    width: 2.5rem;
    height: 2.5rem;
    display: flex
    align-items: center
    justify-content: center
    font-size: 0.875rem;
    font-weight: bold
  }

  /* Editor */
  .editor-container {
    flex: 1;
    overflow-y: auto
    padding: 2rem;
    background: #f4f1ea;
  }

  .editor-content {
    min-height: 100%;
    outline: none
    line-height: 1.8;
    font-size: 1.125rem;
    color: #3a372f;
    font-family: 'Georgia', 'Times New Roman', serif;
  }

  .editor-content:empty::before {
    content: attr(placeholder);
    color: rgba(58, 55, 47, 0.5);
    font-style: italic
  }

  /* Professional typography */
  .editor-content h1 {
    font-size: 2.25rem;
    font-weight: 700;
    margin: 2rem 0 1rem 0;
    color: #3a372f;
  }

  .editor-content h2 {
    font-size: 1.875rem;
    font-weight: 600;
    margin: 1.5rem 0 0.75rem 0;
    color: #75726a;
  }

  .editor-content h3 {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 1.25rem 0 0.5rem 0;
  }

  .editor-content p {
    margin: 1rem 0;
    text-align: justify
  }

  .editor-content blockquote {
    border-left: 4px solid #3a372f;
    padding-left: 1rem;
    margin: 1.5rem 0;
    font-style: italic
    color: rgba(58, 55, 47, 0.8);
  }

  .editor-content ul, .editor-content ol {
    margin: 1rem 0;
    padding-left: 2rem;
  }

  .editor-content li {
    margin: 0.5rem 0;
  }

  /* Status Bar */
  .status-bar {
    display: flex
    align-items: center
    justify-content: space-between;
    padding: 0.75rem 1rem;
    background: #faf8f3;
    border-top: 1px solid #ada895;
    font-size: 0.875rem;
    color: rgba(58, 55, 47, 0.7);
  }

  .status-left {
    display: flex
    gap: 1.5rem;
  }

  .stat-item {
    font-family: 'Consolas', 'Monaco', monospace;
  }

  .auto-save-status {
    font-style: italic
    color: #10b981;
  }

  /* Shortcuts Modal */
  .shortcuts-overlay {
    position: fixed
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex
    align-items: center
    justify-content: center
    z-index: 10000;
  }

  .shortcuts-modal {
    background: #f4f1ea;
    padding: 2rem;
    border-radius: 12px;
    max-width: 500px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto
  }

  .shortcuts-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    text-align: center
  }

  .shortcuts-grid {
    display: grid
    gap: 0.75rem;
    margin-bottom: 2rem;
  }

  .shortcut-item {
    display: flex
    align-items: center
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(173, 168, 149, 0.3);
  }

  .shortcut-key {
    background: #faf8f3;
    border: 1px solid #ada895;
    border-radius: 4px;
    padding: 0.25rem 0.5rem;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.875rem;
    color: #3a372f;
  }

  .shortcut-action {
    color: #3a372f;
  }

  .close-shortcuts {
    width: 100%;
    justify-content: center
    padding: 0.75rem;
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .editor-header {
      flex-direction: column
      gap: 1rem;
      align-items: stretch
    }

    .header-actions {
      justify-content: center
    }

    .editor-toolbar {
      flex-wrap: wrap
      justify-content: center
    }

    .status-bar {
      flex-direction: column
      gap: 0.5rem;
      align-items: center
    }

    .editor-container {
      padding: 1rem;
    }
  }

  /* Focus indicators for accessibility */
  .action-btn:focus,
  .format-btn:focus {
    outline: 2px solid #3a372f;
    outline-offset: 2px;
  }

  /* Smooth animations */
  .professional-editor * {
    transition: opacity 0.3s ease, transform 0.3s ease;
  }
</style>