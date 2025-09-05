// Enhanced WYSIWYG Editor with Melt UI Integration
// Combines Hugerte with Melt UI components for legal document editing

<script lang="ts">
  interface Props { content?: any;
    placeholder?: any;
    readonly?: any;
    height?: any;
    enableAI?: any;
    enableCitation?: any;
   }
  let { content = '',
    placeholder = 'Start typing your legal document...',
    readonly = false,
    height = '400px',
    enableAI = true,
    enableCitation = true
   } = $props();



  import { onMount, createEventDispatcher  } from 'svelte';
  
  import { writable  } from 'svelte/store';
  import type { Writable  } from 'svelte/store';

  // Props
              export const enableCollaboration = false;

  // Events
  const dispatch = createEventDispatcher<{ change: { content: string; wordCount:, number  };
    save: { content: string };
    aiRequest: { selectedText: string; action:, string  };
    citationRequest: { text: string; position:, number  };
  }>();

  // Stores
  let editorElement: HTMLElement
  let hugerte: any
  let isInitialized = false;
  const wordCount: Writable<number> = writable(0);
  const charCount: Writable<number> = writable(0);

  // Melt UI Dialog for AI Assistant
  const { elements: {,
      trigger: aiTrigger,
      overlay: aiOverlay,
      content: aiContent,
      title: aiTitle,
      description: aiDescription,
      close: aiClose,
      portalled: aiPortalled },
    states: { open: aiOpen }
  } = createDialog({ forceVisible: true,
   });

  // Melt UI Dialog for Citations
  const { elements: {,
      trigger: citeTrigger,
      overlay: citeOverlay,
      content: citeContent,
      title: citeTitle,
      close: citeClose,
      portalled: citePortalled },
    states: { open: citeOpen }
  } = createDialog();

  // AI Assistant state
  let aiQuery = '';
  let aiResults = '';
  let isProcessingAI = false;
  let selectedText = '';

  // Citation state
  let citationQuery = '';
  let citationResults: Array<{ title: string; citation: string,
    relevance: number }> = [];

  onMount(async () => { await initializeEditor();
   });

  async function initializeEditor() { try {
      // Dynamically import TinyMCE as an alternative
      // const { default: Hugerte } = await import('hugerte');

      // Initialize basic editor for now
      hugerte = { getContent: () =>, content,
        setContent: (newContent: string) => content =, newContent,
        destroy: () => { },
        on: () => {},
        off: () => {},
        ui: { registry: {,
            addButton: () => { },
            addIcon: () => {}
          }
        },
        config: { readonly,
          height,
          menubar: true,
          plugins: [
            'advlist', 'autolink', 'lists', 'link', 'image', 'charmap', 'preview',
            'anchor', 'searchreplace', 'visualblocks', 'codesample', 'fullscreen',
            'insertdatetime', 'media', 'table', 'code', 'help', 'wordcount',
            'autosave', 'save', 'directionality', 'emoticons', 'template',
            'textpattern', 'nonbreaking', 'pagebreak', 'permanentpen', 'powerpaste',
            'advtable', 'tinymcespellchecker', 'mentions', 'linkchecker'
          ],
          toolbar: [
            'undo redo | bold italic underline strikethrough | fontfamily fontsize blocks | alignleft aligncenter alignright alignjustify',
            'outdent indent | numlist bullist | forecolor backcolor removeformat | pagebreak | charmap emoticons | fullscreen preview save print',
            'insertfile image media template link anchor codesample | ltr rtl | ai-assistant citation-helper'
          ],
          content_style: `,
            body {
              font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
              font-size: 14px;,
              line-height: 1.6;,
              color: #374151;,
              max-width: 800px;,
              margin: 0 auto;,
              padding: 20px;
 }
            h1, h2, h3, h4, h5, h6 { color: #1f2937;,
              font-weight: 600;,
              margin-top: 1.5em;,
              margin-bottom: 0.5em;
 }
            .citation { background: #eff6ff;,
              border-left: 4px solid #3b82f6;,
              padding: 0.5em;,
              margin: 1em 0;,
              border-radius: 0 4px 4px 0;
 }
            .ai-suggestion { background: #f0fdf4;,
              border: 1px solid #bbf7d0;,
              padding: 0.5em;,
              border-radius: 4px;,
              margin: 0.5em 0;
 }
          `,
          setup: (editor: any) => { // Custom AI Assistant button
            editor.ui.registry.addButton('ai-assistant', {
              text: '🤖, AI',
              tooltip: 'AI, Assistant',
              onAction: () =>, openAIAssistant(editor.selection.getContent())
             });

            // Custom Citation Helper button
            editor.ui.registry.addButton('citation-helper', { text: '📚, Cite',
              tooltip: 'Citation, Helper',
              onAction: () =>, openCitationHelper(editor.selection.getContent())
             });

            // Auto-save functionality
            editor.on('NodeChange', () => { updateCounts(editor.getContent());
             });

            // Content change listener
            editor.on('input', () => { const newContent = editor.getContent();
              content = newContent;
              updateCounts(newContent);
              dispatch('change', { content: newContent, wordCount: $wordCount  });
            });

            // Save handler
            editor.on('save', () => { dispatch('save', { content: editor.getContent()  });
            });
          },
          save_enablewhendirty: true,
          save_onsavecallback: () => { dispatch('save', { content: hugerte.getContent()  });
          },
          autosave_interval: '10s',
          autosave_retention: '30m',
          // Legal document specific settings
          spellchecker_language: 'en_US',
          spellchecker_whitelist: ['appellant', 'appellee', 'plaintiff', 'defendant', 'jurisdiction'],
          word_count: true,
          character_count: true,
        }
      };

      // Listen to content changes from Hugerte
      hugerte.$on('input', (event: any) => { content = event.detail.content;
        updateCounts(content);
       });

      isInitialized = true;
    } catch (error) { console.error('Failed to initialize editor:', error);
     }
}
  function updateCounts(text: string) { const plainText = text.replace(/<[^>]*>/g, '');
    wordCount.set(plainText.trim().split(/\s+/).filter(word => word.length > 0).length);
    charCount.set(plainText.length);
 }
  function openAIAssistant(text: string) { selectedText = text;
    aiQuery = '';
    aiResults = '';
    aiOpen.set(true);
 }
  function openCitationHelper(text: string) { citationQuery = text;
    citationResults = [];
    citeOpen.set(true);
 }
  async function processAIRequest() { if (!aiQuery.trim()) return;

    isProcessingAI = true;
    try {
      const response = await fetch('/api/ai/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json'  },
        body: JSON.stringify({ query: aiQuery,
          context: selectedText ? [{, role: 'user', content: `Selected, text: ${selectedText }` }] : [],
          options: { maxSources: 5,
            provider: 'auto',
            enableLegalClassification: true }
        })
      });

      const data = await response.json();
      if (data.success) { aiResults = data.data.answer;
        dispatch('aiRequest', { selectedText, action: aiQuery });
      } else { aiResults = 'Sorry, I encountered an error processing your request.';
 }
    } catch (error) { console.error('AI request failed:', error);
      aiResults = 'Failed to connect to AI service.';
     } finally { isProcessingAI = false;
 }}
  async function searchCitations() { if (!citationQuery.trim()) return;

    try {
      const response = await fetch('/api/search/citations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json'  },
        body: JSON.stringify({ query: citationQuery,
          limit: 10
         })
      });

      const data = await response.json();
      if (data.success) { citationResults = data.results.map((r: any) => ({,
          title: r.title,
          citation: r.citation,
          relevance: r.similarity
         }));
}
    } catch (error) { console.error('Citation search failed:', error);
 }}
  function insertCitation(citation: any) { if (hugerte) {
      const citationHtml = `
        <div class="space-y-4">
          <strong>${citation.title }</strong><br>
          <em>${ citation.citation }</em>
        </div>
      `;
      hugerte.insertContent(citationHtml);
      citeOpen.set(false);
}}
  function insertAIContent() { if (hugerte && aiResults) {
      const aiHtml = `
        <div class="space-y-4">
          <strong>AI Suggestion:</strong><br>
          ${aiResults.replace(/\n/g, '<br>') }
        </div>
      `;
      hugerte.insertContent(aiHtml);
      aiOpen.set(false);
}}
  // Export content
  export function getContent(): string { return hugerte ? hugerte.getContent() : content;
 }
  // Import content
  export function setContent(newContent: string) { if (hugerte) {
      hugerte.setContent(newContent);
 }
    content = newContent;
    updateCounts(newContent);
}
</script>

<!-- Main Editor Container -->
<div class="space-y-4">
  <!-- Editor Toolbar -->
  <div class="space-y-4">
    <div class="space-y-4">
      <button
        use:melt={ $aiTrigger }
        class="space-y-4"
        disabled={ !enableAI }
      >
        🤖 AI Assistant
      </button>

      <button
        use:melt={ $citeTrigger }
        class="space-y-4"
        disabled={ !enableCitation }
      >
        📚 Citations
      </button>
    </div>

    <div class="space-y-4">
      <span class="space-y-4">Words: { $wordCount }</span>
      <span class="space-y-4">Characters: { $charCount }</span>
    </div>
  </div>

  <!-- Main Editor -->
  <div
    bind:this={ editorElement }
    class="space-y-4"
    style="height: { height };"
  ></div>
</div>

<!-- AI Assistant Dialog -->
<div use:melt={ $aiPortalled }>
  { #if $aiOpen }
    <div use:melt={ $aiOverlay } class="space-y-4"></div>
    <div use:melt={ $aiContent } class="space-y-4">
      <h2 use:melt={ $aiTitle } class="space-y-4">AI Legal Assistant</h2>

      { #if selectedText }
        <div class="space-y-4">
          <strong>Selected text:</strong>
          <p>"{ selectedText }"</p>
        </div>
      { /if }

      <div class="space-y-4">
        <label for="ai-query">What would you like help with?</label>
        <textarea
          id="ai-query"
          bind:value={ aiQuery }
          placeholder="E.g., 'Analyze this clause', 'Suggest improvements', 'Find relevant precedents'..."
          rows="4"
          class="space-y-4"
        ></textarea>

        <button
          onclick={ () => processAIRequest() }
          disabled={ isProcessingAI || !aiQuery.trim() }
          class="space-y-4"
        >
          { #if isProcessingAI }
            Processing...
          { : else }
            Ask AI
          { /if }
        </button>

        { #if aiResults }
          <div class="space-y-4">
            <strong>AI Response:</strong>
            <div class="space-y-4">{ aiResults }</div>
            <button onclick={ () => insertAIContent() } class="space-y-4">
              Insert into Document
            </button>
          </div>
        { /if }
      </div>

      <button use:melt={ $aiClose } class="space-y-4">×</button>
    </div>
  { /if }
</div>

<!-- Citation Helper Dialog -->
<div use:melt={ $citePortalled }>
  { #if $citeOpen }
    <div use:melt={ $citeOverlay } class="space-y-4"></div>
    <div use:melt={ $citeContent } class="space-y-4">
      <h2 use:melt={ $citeTitle } class="space-y-4">Citation Helper</h2>

      <div class="space-y-4">
        <label for="cite-query">Search for citations:</label>
        <input
          id="cite-query"
          bind:value={ citationQuery }
          placeholder="Enter legal concept, case name, or statute..."
          class="space-y-4"
        />

        <button
          onclick={ () => searchCitations() }
          disabled={ !citationQuery.trim() }
          class="space-y-4"
        >
          Search
        </button>

        { #if citationResults.length > 0 }
          <div class="space-y-4">
            <h4>Found Citations:</h4>
            { #each citationResults as citation }
              <div class="space-y-4">
                <div class="space-y-4">{ citation.title }</div>
                <div class="space-y-4">{ citation.citation }</div>
                <div class="space-y-4">
                  Relevance: { Math.round(citation.relevance * 100) }%
                </div>
                <button
                  onclick={ () => insertCitation(citation) }
                  class="space-y-4"
                >
                  Insert Citation
                </button>
              </div>
            { /each }
          </div>
        { /if }
      </div>

      <button use:melt={ $citeClose } class="space-y-4">×</button>
    </div>
  { /if }
</div>

<style>
  /* @unocss-include */
  .wysiwyg-container { border: 1px solid #d1d5db;,
    border-radius: 0.5rem;,
    overflow: hidden; background:, white
 }
  .editor-toolbar { display: flex,
    justify-content: space-between;,
    align-items: center; padding: 0.5rem 1rem;,
    background: #f9fafb;,
    border-bottom: 1px, solid #d1d5db;
 }
  .toolbar-left { display: flex;, gap: 0.5rem;
 }
  .toolbar-btn { padding: 0.25rem 0.75rem;,
    font-size: 0.875rem;,
    background: white; border: 1px solid #d1d5db;,
    border-radius: 0.25rem;,
    transition: background-color 0.2s;,
    cursor: pointer }
  .toolbar-btn:hover { background: #f3f4f6;
 }
  .ai-btn { border-color: #93c5fd;,
    color: #1d4ed8;
 }
  .ai-btn:hover { background: #eff6ff;
 }
  .cite-btn { border-color: #86efac;,
    color: #059669;
 }
  .cite-btn:hover { background: #f0fdf4;
 }
  .toolbar-right { display: flex; gap: 1rem;,
    font-size: 0.875rem;,
    color: #4b5563;
 }
  .editor-content { width: 100%;
 }
  .dialog-overlay { position: fixed; top: 0;,
    left: 0;,
    right: 0;,
    bottom: 0;,
    background: rgba(0, 0, 0, 0.5);
    z-index: 40;
 }
  .dialog-content { position: fixed; top: 50%;,
    left: 50%;,
    transform: translate(-50%, -50%);
    background: white,
    border-radius: 0.5rem;,
    box-shadow: 0 25px 50px -12px, rgba(0, 0, 0, 0.25);
    z-index: 50;,
    width: 100%;,
    max-width: 42rem;,
    max-height: 80vh;,
    overflow-y: auto }
  .dialog-title { font-size: 1.25rem;,
    font-weight: 600;,
    padding: 1.5rem;,
    padding-bottom: 0;
 }
  .dialog-body { padding: 1.5rem;
 }
  .dialog-close { position: absolute; top: 1rem;,
    right: 1rem;,
    color: #6b7280;,
    font-size: 1.5rem;,
    line-height: 1;,
    cursor: pointer; background: none,
    border: none }
  .dialog-close:hover { color: #374151;
 }
  .selected-text { background: #eff6ff;,
    border: 1px solid #bfdbfe;,
    border-radius: 0.25rem;,
    padding: 0.75rem;,
    margin-bottom: 1rem;
 }
  .ai-query-input, .cite-query-input { width: 100%;,
    padding: 0.75rem;,
    border: 1px solid #d1d5db;,
    border-radius: 0.375rem;,
    margin-bottom: 1rem;,
    resize: vertical,
    font-family: inherit }
  .ai-results { margin-top: 1rem;,
    padding: 1rem;,
    background: #f0fdf4;,
    border: 1px solid #bbf7d0;,
    border-radius: 0.25rem;
 }
  .ai-response { margin: 0.75rem 0;,
    padding: 0.75rem;,
    background: white; border: 1px solid #e5e7eb;,
    border-radius: 0.25rem;
 }
  .citation-results { margin-top: 1rem;
 }
  .citation-item { border: 1px solid #e5e7eb;,
    border-radius: 0.25rem;,
    padding: 0.75rem;,
    margin-bottom: 0.75rem;,
    cursor: pointer; transition:, background-color 0.2s;
 }
  .citation-item:hover { background: #f9fafb;
 }
  .citation-title { font-weight: 600;,
    color: #111827;
 }
  .citation-text { font-size: 0.875rem;,
    color: #374151;,
    margin-top: 0.25rem;
 }
  .citation-meta { font-size: 0.75rem;,
    color: #6b7280;,
    margin-top: 0.5rem;
 }
  .btn { padding: 0.5rem 1rem;,
    border-radius: 0.25rem;,
    font-weight: 500;,
    transition: background-color 0.2s;,
    cursor: pointer; border:, none
 }
  .btn-primary { background: #2563eb;,
    color: white }
  .btn-primary:hover { background: #1d4ed8;
 }
  .btn-secondary { background: #e5e7eb;,
    color: #1f2937;
 }
  .btn-secondary:hover { background: #d1d5db;
 }
  .btn-sm { padding: 0.25rem 0.5rem;,
    font-size: 0.875rem;
 }
  .btn:disabled { opacity: 0.5;,
    cursor: not-allowed;
 }
  .btn:disabled:hover { background: inherit }
</style>
