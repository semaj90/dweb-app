<!-- @migration-task Error while migrating Svelte code: A component can have a single top-level `<script lang="ts">
` element and/or a single top-level `<script module>` element
https://svelte.dev/e/script_duplicate -->
<!-- Tiptap Editor with AI Assistant Integration -->
<!-- Real-time suggestions, auto-save, and CrewAI inline recommendations -->

<script lang="ts">
  import { $props, $state, $derived } from 'svelte';
  import { onMount, onDestroy, tick } from 'svelte';
  import { Editor } from '@tiptap/core';
  import StarterKit from '@tiptap/starter-kit';
  import { Collaboration } from '@tiptap/extension-collaboration';
  import { CollaborationCursor } from '@tiptap/extension-collaboration-cursor';
  import { useMachine } from '@xstate/svelte';
  import { crewAIOrchestrationMachine } from '$lib/state/crewAIOrchestrationMachine';
  import { slide, fade } from 'svelte/transition';

  // Props
  let {
    documentId,
    initialContent = '',
    placeholder = 'Start typing your legal document...',
    autoSave = true,
    showAIAssistant = true,
    enableInlineSuggestions = true,
    readOnly = false
  } = $props();

  // State management
  // Rename to avoid confusion with Svelte runes naming ($state function)
  const { state: machineState, send } = useMachine(crewAIOrchestrationMachine);

  // Component state
  let editor: Editor | null = null;
  let editorElement: HTMLElement;
  let showSuggestions = $state(false);
  let currentSuggestions = $state([]);
  let userTyping = $state(false);
  let lastSaveTime = $state<Date | null>(null);
  let wordCount = $state(0);
  let aiAssistantVisible = $state(false);
  let currentRecommendation = $state<string | null>(null);
  let recommendationPosition = $state({ x: 0, y: 0 });

  // Auto-save timer
  let autoSaveTimer: NodeJS.Timeout | null = null;
  let idleTimer: NodeJS.Timeout | null = null;

  // Derived state
  const isProcessing = $derived(machineState.matches('orchestrating'));
  const hasRecommendations = $derived(machineState.context.currentRecommendations.length > 0);
  const userIntent = $derived(machineState.context.userIntent);
  const focusSchema = $derived(machineState.context.focusSchema);

  // Slash command + Mermaid lazy load support ---------------------------------
  let mermaidLoaded = $state(false);
  async function loadMermaidIfNeeded(content?: string){
    if (mermaidLoaded) return;
    if (typeof window === 'undefined') return;
    if (content && !content.includes('```mermaid')) return; // only load when needed
    try {
      const m = await import('mermaid');
      m.initialize({ startOnLoad: false });
      mermaidLoaded = true;
    } catch (e) {
      console.warn('Mermaid failed to load (optional):', e);
    }
  }

  // Real slash menu implementation -----------------------------------------
  interface SlashCommand { id: string; label: string; description: string; action: () => void; keywords?: string[] }
  let slashActive = $state(false);
  let slashQuery = $state('');
  let slashCoords = $state<{x:number;y:number}>({x:0,y:0});
  let commandIndex = $state(0);
  let liveRegionEl: HTMLElement | null = null;

  function announce(msg: string){
    if (!liveRegionEl) return; liveRegionEl.textContent = ''; // force reflow for some SRs
    requestAnimationFrame(()=>{ if (liveRegionEl) liveRegionEl.textContent = msg; });
  }

  function insertHeading(level: number){
    editor?.chain().focus().setNode('heading', { level }).run();
  }
  function insertBulletList(){ editor?.chain().focus().toggleBulletList().run(); }
  function insertMermaidBlock(){
    const template = '```mermaid\ngraph TD;\n  A[Start] --> B{Decision};\n  B -->|Yes| C[Outcome 1];\n  B -->|No| D[Outcome 2];\n```';
    editor?.chain().focus().insertContent(template + '\n').run();
    loadMermaidIfNeeded(template);
  }
  function triggerAISuggest(){
    aiAssistantVisible = true;
    showSuggestions = true;
  }
  const slashCommands: SlashCommand[] = [
    { id:'h1', label:'Heading 1', description:'Insert H1 heading', action: () => insertHeading(1), keywords:['title','h1'] },
    { id:'h2', label:'Heading 2', description:'Insert H2 heading', action: () => insertHeading(2), keywords:['subtitle','h2'] },
    { id:'bullet', label:'Bullet List', description:'Toggle bullet list', action: insertBulletList, keywords:['list','ul','bullet'] },
    { id:'mermaid', label:'Mermaid Diagram', description:'Insert mermaid fenced block', action: insertMermaidBlock, keywords:['diagram','graph','flow'] },
    { id:'ai', label:'AI Suggest', description:'Open AI assistant suggestions', action: triggerAISuggest, keywords:['ai','assist','suggest'] }
  ];
  const filteredSlashCommands = $derived(() => {
    const q = slashQuery.toLowerCase();
    if (!q) return slashCommands.slice(0,5);
    return slashCommands.filter(c => c.label.toLowerCase().includes(q) || c.keywords?.some(k => k.includes(q))).slice(0,7);
  });

  function openSlash(position?: {x:number;y:number}){
    slashActive = true; slashQuery=''; commandIndex=0;
    if (position) slashCoords = position;
  // announce open after tick
  setTimeout(()=> announce('Command menu opened. Type to filter, arrow keys to navigate.'), 0);
  }
  function closeSlash(){ slashActive = false; slashQuery=''; }
  function runSelected(){
    const cmd = filteredSlashCommands[commandIndex];
  if (cmd){ announce(`${cmd.label} executed`); closeSlash(); cmd.action(); }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const SlashMenuExtension: any = {
    name: 'slashMenu',
    addKeyboardShortcuts(){
      return {
        '/': () => {
          // Determine caret position for menu placement
          const sel = window.getSelection();
            if (sel && sel.rangeCount > 0) {
              const range = sel.getRangeAt(0).cloneRange();
              const rect = range.getBoundingClientRect();
              openSlash({x: rect.left, y: rect.bottom + window.scrollY});
            } else {
              openSlash();
            }
            return true;
        },
        Escape: () => { if (slashActive){ closeSlash(); return true; } return false; },
        ArrowDown: () => { if (slashActive){ commandIndex = (commandIndex + 1) % filteredSlashCommands.length; return true; } return false; },
        ArrowUp: () => { if (slashActive){ commandIndex = (commandIndex - 1 + filteredSlashCommands.length) % filteredSlashCommands.length; return true; } return false; },
        Enter: () => { if (slashActive){ runSelected(); return true; } return false; }
      };
    },
    addProseMirrorPlugins(){
      return [];
    },
  };

  onMount(async () => {
    await initializeEditor();
    setupEventListeners();
  // live region for a11y
  liveRegionEl = document.getElementById('editor-live-region');
  });

  onDestroy(() => {
    if (editor) {
      editor.destroy();
    }
    if (autoSaveTimer) {
      clearTimeout(autoSaveTimer);
    }
    if (idleTimer) {
      clearTimeout(idleTimer);
    }
  });

  // ============================================================================
  // EDITOR INITIALIZATION
  // ============================================================================

  async function initializeEditor() {
  editor = new Editor({
      element: editorElement,
      extensions: [
        StarterKit.configure({
          history: false, // We'll handle our own history with collaboration
        }),
    SlashMenuExtension,
        // Add collaboration extensions if needed
        // Collaboration.configure({
        //   document: yDoc,
        // }),
        // CollaborationCursor.configure({
        //   provider,
        // }),
      ],
      content: initialContent,
      editable: !readOnly,
      editorProps: {
        attributes: {
          class: 'tiptap-editor prose prose-lg max-w-none focus:outline-none',
          'aria-activedescendant': () => slashActive && filteredSlashCommands[commandIndex] ? `slash-cmd-${filteredSlashCommands[commandIndex].id}` : undefined,
          'aria-expanded': () => slashActive ? 'true' : 'false',
          'aria-owns': 'slash-command-list',
          'aria-autocomplete': 'list',
        },
        handleKeyDown: (view, event) => {
          handleKeyDown(event);
          return false;
        },
      },
      onUpdate: ({ editor }) => {
        handleContentUpdate(editor.getHTML());
      },
      onSelectionUpdate: ({ editor }) => {
        handleSelectionUpdate(editor);
      },
      onFocus: () => {
        handleEditorFocus();
      },
      onBlur: () => {
        handleEditorBlur();
      },
    });

    // Update word count
  updateWordCount();
  await loadMermaidIfNeeded(initialContent);
  }

  // ============================================================================
  // EVENT HANDLERS
  // ============================================================================

  function handleKeyDown(event: KeyboardEvent) {
    userTyping = true;

    // Send user activity to state machine
    send({ type: 'USER_ACTIVITY', activity: 'typing' });

    // Reset typing flag after short delay
    setTimeout(() => {
      userTyping = false;
    }, 1000);

    // Handle special key combinations
    if (event.ctrlKey || event.metaKey) {
      switch (event.key) {
        case 's':
          event.preventDefault();
          handleManualSave();
          break;
        case '/':
          event.preventDefault();
          toggleAIAssistant();
          break;
        case 'Enter':
          if (event.shiftKey) {
            event.preventDefault();
            showInlineSuggestions();
          }
          break;
      }
    }

    // Handle escape key
    if (event.key === 'Escape') {
      hideAllSuggestions();
    }
  }

  function handleContentUpdate(content: string) {
    updateWordCount();

    if (autoSave) {
      scheduleAutoSave();
    }

    // Generate contextual suggestions based on content
    if (enableInlineSuggestions && !userTyping) {
      generateInlineSuggestions(content);
    }
  }

  function handleSelectionUpdate(editor: Editor) {
    const selection = editor.state.selection;
    const pos = editor.view.coordsAtPos(selection.from);
    recommendationPosition = { x: pos.left, y: pos.top };

    // If slash menu active, adjust coords for viewport bounds
    if (slashActive){
      adjustSlashMenuPosition();
    }

    // Check if selection contains recommended text
    checkForRecommendationAtSelection(selection);
  }

  function handleEditorFocus() {
    send({ type: 'FOCUS_CHANGED', schema: 'document_edit' });
    resetIdleTimer();
  }

  function handleEditorBlur() {
    // Don't immediately change focus if user is interacting with suggestions
    setTimeout(() => {
      if (!aiAssistantVisible && !showSuggestions) {
        send({ type: 'USER_IDLE' });
      }
    }, 1000);
  }

  // ============================================================================
  // AUTO-SAVE & IDLE DETECTION
  // ============================================================================

  function scheduleAutoSave() {
    if (autoSaveTimer) {
      clearTimeout(autoSaveTimer);
    }

    autoSaveTimer = setTimeout(() => {
      handleAutoSave();
    }, 3000); // 3 second delay
  }

  async function handleAutoSave() {
    if (!editor) return;

    const content = editor.getHTML();

    try {
      // This would integrate with your document update system
      await saveDocument(content);
      lastSaveTime = new Date();

      // Send auto-save event to state machine
      send({ type: 'AUTO_SAVE_TRIGGERED' });

    } catch (error) {
      console.error('Auto-save failed:', error);
    }
  }

  async function handleManualSave() {
    if (!editor) return;

    const content = editor.getHTML();
    await saveDocument(content);
    lastSaveTime = new Date();

    // Show save confirmation
    showNotification('Document saved', 'success');
  }

  function resetIdleTimer() {
    if (idleTimer) {
      clearTimeout(idleTimer);
    }

    idleTimer = setTimeout(() => {
      send({ type: 'USER_IDLE' });
    }, 300000); // 5 minutes
  }

  // ============================================================================
  // AI ASSISTANT & SUGGESTIONS
  // ============================================================================

  function toggleAIAssistant() {
    aiAssistantVisible = !aiAssistantVisible;

    if (aiAssistantVisible) {
      send({ type: 'FOCUS_CHANGED', schema: 'analysis_mode' });
    } else {
      send({ type: 'FOCUS_CHANGED', schema: 'document_edit' });
    }
  }

  async function generateInlineSuggestions(content: string) {
    if (!enableInlineSuggestions || content.length < 100) return;

    // This would integrate with your AI suggestion system
    try {
      const suggestions = await fetchInlineSuggestions(content);
      currentSuggestions = suggestions;

      if (suggestions.length > 0) {
        showSuggestions = true;
      }
    } catch (error) {
      console.error('Failed to generate suggestions:', error);
    }
  }

  async function startCrewAIReview() {
    if (!editor || !documentId) return;

    const content = editor.getText();

    try {
      const response = await fetch('/api/crewai/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          documentId,
          reviewType: 'comprehensive',
          priority: 'medium',
          assignedAgents: ['compliance_specialist', 'risk_analyst', 'legal_editor'],
          context: {
            userIntent: 'comprehensive_review'
          }
        })
      });

      const result = await response.json();

      if (result.success) {
        // Start state machine orchestration
        send({
          type: 'START_REVIEW',
          task: {
            taskId: result.data.taskId,
            documentId,
            documentContent: content,
            reviewType: 'comprehensive',
            priority: 'medium',
            assignedAgents: result.data.assignedAgents.map((a: any) => a.id)
          }
        });

        showNotification('CrewAI review started', 'info');
      }
    } catch (error) {
      console.error('Failed to start CrewAI review:', error);
      showNotification('Failed to start review', 'error');
    }
  }

  function applySuggestion(suggestion: any) {
    if (!editor) return;

    // Apply the suggestion to the editor
    if (suggestion.position !== undefined) {
      editor.commands.setTextSelection({
        from: suggestion.position,
        to: suggestion.position + suggestion.length
      });
      editor.commands.insertContent(suggestion.suggestedText);
    }

    // Accept recommendation in state machine
    send({ type: 'ACCEPT_RECOMMENDATION', recommendationId: suggestion.id });

    showNotification('Suggestion applied', 'success');
  }

  function rejectSuggestion(suggestion: any) {
    send({ type: 'REJECT_RECOMMENDATION', recommendationId: suggestion.id });
    showNotification('Suggestion rejected', 'info');
  }

  function hideAllSuggestions() {
    showSuggestions = false;
    aiAssistantVisible = false;
    currentRecommendation = null;
  }

  function showInlineSuggestions() {
    if (!editor) return;

    const selection = editor.state.selection;
    const selectedText = editor.state.doc.textBetween(selection.from, selection.to);

    if (selectedText.length > 0) {
      generateContextualSuggestion(selectedText);
    }
  }

  // ============================================================================
  // HELPER FUNCTIONS
  // ============================================================================

  async function saveDocument(content: string): Promise<void> {
    // This would integrate with your document save API
    await fetch(`/api/documents/${documentId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content })
    });
  }

  async function fetchInlineSuggestions(content: string): Promise<unknown[]> {
    // This would call your AI suggestion API
    const response = await fetch('/api/ai/suggestions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, type: 'inline' })
    });

    const result = await response.json();
    return result.suggestions || [];
  }

  async function generateContextualSuggestion(selectedText: string): Promise<void> {
    // Generate suggestion for selected text
    const suggestions = await fetchInlineSuggestions(selectedText);
    if (suggestions.length > 0) {
      currentRecommendation = suggestions[0].text;
      showSuggestions = true;
    }
  }

  function checkForRecommendationAtSelection(selection: any): void {
    // Check if current selection contains any pending recommendations
  const recommendations = machineState.context.currentRecommendations;

    for (const rec of recommendations) {
      if (rec.position && selection.from <= rec.position && selection.to >= rec.position) {
        currentRecommendation = rec.text;
        break;
      }
    }
  }

  function updateWordCount(): void {
    if (editor) {
      const text = editor.getText();
      wordCount = text.split(/\s+/).filter(word => word.length > 0).length;
    }
  }

  function showNotification(message: string, type: 'success' | 'error' | 'info'): void {
    // This would integrate with your notification system
    console.log(`${type.toUpperCase()}: ${message}`);
  }

  function setupEventListeners(): void {
    // Global keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === '/') {
        e.preventDefault();
        toggleAIAssistant();
      }
    });
    window.addEventListener('resize', () => { if (slashActive) adjustSlashMenuPosition(true); });
    window.addEventListener('scroll', () => { if (slashActive) adjustSlashMenuPosition(true); }, { passive: true });
  }

  function adjustSlashMenuPosition(force = false){
    // Use current coords, adjust if near viewport edges
    const margin = 8;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const estimatedWidth = 260; // palette width basis
    const estimatedHeight = 280; // approx height
    let { x, y } = slashCoords;
    if (x + estimatedWidth + margin > vw){ x = Math.max(margin, vw - estimatedWidth - margin); }
    if (y + estimatedHeight + margin > vh + window.scrollY){ y = window.scrollY + Math.max(margin, vh - estimatedHeight - margin); }
    if (x !== slashCoords.x || y !== slashCoords.y || force){ slashCoords = { x, y }; }
  }
</script>

<!-- Editor Container -->
<div class="tiptap-container relative">

  <!-- Live region for screen reader announcements -->
  <div id="editor-live-region" class="sr-only" aria-live="polite" aria-atomic="true"></div>

  {#if slashActive && filteredSlashCommands.length}
    <!-- Slash Command Palette -->
    <div
      class="slash-menu absolute z-30 w-64 bg-white border border-gray-300 rounded-lg shadow-lg overflow-hidden"
      style="left: {slashCoords.x}px; top: {slashCoords.y + 24}px;"
      role="listbox"
      aria-label="Slash command menu"
      id="slash-command-list"
    >
      <ul class="max-h-72 overflow-y-auto">
        {#each filteredSlashCommands as cmd, i (cmd.id)}
          <li
            id={"slash-cmd-" + cmd.id}
            class="px-3 py-2 text-sm cursor-pointer flex items-start space-x-2 transition-colors"
            class:bg-blue-50={i === commandIndex}
            class:text-blue-700={i === commandIndex}
            role="option"
            aria-selected={i === commandIndex}
            tabindex="-1"
            on:mousedown|preventDefault={() => { commandIndex = i; runSelected(); }}
          >
            <div class="flex-shrink-0 w-5 text-center">
              {#if cmd.icon}{cmd.icon}{/if}
            </div>
            <div class="flex-1 min-w-0">
              <div class="font-medium leading-tight">{cmd.label}</div>
              {#if cmd.description}
                <div class="text-[11px] text-gray-500 mt-0.5 line-clamp-2">{cmd.description}</div>
              {/if}
            </div>
            {#if cmd.keywords?.length}
              <div class="hidden md:block text-[10px] text-gray-400 uppercase tracking-wide">
                {cmd.keywords.slice(0,2).join(' ')}{cmd.keywords.length > 2 ? '…' : ''}
              </div>
            {/if}
          </li>
        {/each}
      </ul>
      {#if slashQuery && !filteredSlashCommands.length}
        <div class="px-3 py-2 text-xs text-gray-500">No matches for "{slashQuery}"</div>
      {/if}
      <div class="slash-hint border-t border-gray-200 px-3 py-1.5 text-[10px] text-gray-500 flex justify-between">
        <span>↑↓ navigate • Enter run</span>
        <span>Esc close</span>
      </div>
    </div>
  {/if}

  <!-- Editor Element -->
  <div
    bind:this={editorElement}
    class="tiptap-editor-wrapper min-h-96 border border-gray-300 rounded-lg p-4 focus-within:border-blue-500 transition-colors"
    class:opacity-50={readOnly}
  />

  <!-- Status Bar -->
  <div class="status-bar flex items-center justify-between mt-2 text-sm text-gray-500">
    <div class="flex items-center space-x-4">
      <span>{wordCount} words</span>

      {#if lastSaveTime}
        <span>Saved {formatTime(lastSaveTime)}</span>
      {/if}

      {#if isProcessing}
        <div class="flex items-center space-x-2 text-blue-600">
          <div class="animate-spin w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
          <span>AI reviewing...</span>
        </div>
      {/if}
    </div>

    <div class="flex items-center space-x-2">
      {#if userIntent === 'idle'}
        <span class="text-yellow-600">💤 Idle</span>
      {:else if userIntent === 'editing'}
        <span class="text-green-600">✏️ Editing</span>
      {:else if userIntent === 'reviewing'}
        <span class="text-blue-600">🔍 Reviewing</span>
      {/if}
    </div>
  </div>

  <!-- AI Assistant Panel -->
  {#if aiAssistantVisible}
    <div
      class="ai-assistant-panel absolute top-0 right-0 w-80 bg-white border border-gray-300 rounded-lg shadow-lg p-4 z-10"
      transitislide={{ axis: 'x', duration: 200 }}
    >
      <div class="flex items-center justify-between mb-4">
        <h3 class="font-semibold text-gray-800">AI Assistant</h3>
          <button
          click={() => aiAssistantVisible = false}
          class="text-gray-500 hover:text-gray-700"
        >
          ✕
        </button>
      </div>

      <!-- Quick Actions -->
      <div class="space-y-2 mb-4">
        <button
          click={startCrewAIReview}
          class="w-full bg-blue-600 text-white px-3 py-2 rounded text-sm hover:bg-blue-700 transition-colors"
          disabled={isProcessing}
        >
          {isProcessing ? 'Review in Progress...' : 'Start CrewAI Review'}
        </button>

        <button
          click={() => generateInlineSuggestions(editor?.getHTML() || '')}
          class="w-full bg-green-600 text-white px-3 py-2 rounded text-sm hover:bg-green-700 transition-colors"
        >
          Generate Suggestions
        </button>
      </div>

      <!-- Current Recommendations -->
      {#if hasRecommendations}
        <div class="recommendations">
          <h4 class="font-medium text-gray-700 mb-2">Recommendations</h4>

          {#each machineState.context.currentRecommendations as rec (rec.id)}
            <div
              class="recommendation-item p-2 border border-gray-200 rounded mb-2"
              transitifade={{ duration: 150 }}
            >
              <div class="flex items-start justify-between">
                <div class="flex-1">
                  <div class="text-sm text-gray-800">{rec.text}</div>
                  <div class="text-xs text-gray-500 mt-1">
                    {rec.type} • {Math.round(rec.confidence * 100)}% confidence
                  </div>
                </div>

                <div class="flex space-x-1 ml-2">
                  <button
                    click={() => applySuggestion(rec)}
                    class="text-green-600 hover:text-green-800 text-xs px-2 py-1 rounded"
                    title="Accept"
                  >
                    ✓
                  </button>
                  <button
                    click={() => rejectSuggestion(rec)}
                    class="text-red-600 hover:text-red-800 text-xs px-2 py-1 rounded"
                    title="Reject"
                  >
                    ✕
                  </button>
                </div>
              </div>
            </div>
          {/each}
        </div>
      {/if}

      <!-- Focus Schema Indicator -->
      <div class="mt-4 pt-4 border-t border-gray-200">
        <div class="text-xs text-gray-500">
          Focus: <span class="font-medium">{focusSchema.replace('_', ' ')}</span>
        </div>
      </div>
    </div>
  {/if}

  <!-- Inline Suggestions Popup -->
  {#if showSuggestions && currentRecommendation}
    <div
      class="inline-suggestion absolute bg-yellow-50 border border-yellow-300 rounded-lg p-3 shadow-lg z-20 max-w-xs"
      style="left: {recommendationPosition.x}px; top: {recommendationPosition.y + 25}px;"
      transitifade={{ duration: 150 }}
    >
      <div class="text-sm text-gray-800 mb-2">{currentRecommendation}</div>

      <div class="flex justify-end space-x-2">
        <button
          click={() => showSuggestions = false}
          class="text-xs text-gray-500 hover:text-gray-700 px-2 py-1"
        >
          Dismiss
        </button>
        <button
          click={() => applySuggestion({ text: currentRecommendation })}
          class="text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700"
        >
          Apply
        </button>
      </div>
    </div>
  {/if}

  <!-- Keyboard Shortcuts Help -->
  <div class="keyboard-shortcuts text-xs text-gray-400 mt-2">
    <span>Ctrl+S: Save</span> •
    <span>Ctrl+/: AI Assistant</span> •
    <span>Shift+Enter: Suggestions</span> •
    <span>Esc: Hide suggestions</span>
  </div>
</div>

<style>
  .tiptap-editor {
    outline: none;
  }

  .tiptap-editor :global(.ProseMirror) {
    outline: none;
    min-height: 200px;
  }

  .tiptap-editor :global(.ProseMirror p.is-editor-empty:first-child::before) {
    content: attr(data-placeholder);
    float: left;
    color: #9ca3af;
    pointer-events: none;
    height: 0;
  }

  .ai-assistant-panel {
    max-height: 500px;
    overflow-y: auto;
  }

  .inline-suggestion {
    animation: slideInUp 0.2s ease-out;
  }

  @keyframes slideInUp {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>

  /* Slash menu styles */
  .slash-menu {
    animation: fadeIn 90ms ease-out;
  }
  :global(.dark) .slash-menu { background:#1f2937; border-color:#374151; }
  :global(.dark) .slash-menu li { color:#d1d5db; }
  :global(.dark) .slash-menu li.bg-blue-50 { background:#1e3a8a; }
  :global(.dark) .slash-menu .slash-hint { border-color:#374151; color:#6b7280; }
  :global(.sr-only){ position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); border:0; }
  .slash-menu ul::-webkit-scrollbar { width: 6px; }
  .slash-menu ul::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
  .slash-menu ul::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
  .slash-menu li {
    outline: none;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }

  // Merged from second script block (formatTime helper)
  function formatTime(date: Date): string {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    if (diff < 60000) return 'just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    return `${Math.floor(diff / 3600000)}h ago`;
  }

  // END script
</script>

