<!--
  Enhanced Claude Inline Suggestion Loop
  Real-time AI-powered editing with mini text box suggestions
-->
<script lang="ts">
  import { onMount, onDestroy, tick  } from 'svelte';
  import { createActor  } from 'xstate';
  import { aiProcessingMachine, createAITask, aiTaskCreators  } from '$lib/stores/machines';
  import { enhancedRAGStore  } from '$lib/stores';
  import { debounce  } from 'lodash-es';

  // Props
  let { value = $bindable(''),
    placeholder = 'Start typing to get AI suggestions...',
    aiModel = 'gemma3-legal',
    enableAutoComplete = true,
    enableGrammarCheck = true,
    enableSemanticSuggestions = true,
    minCharactersForSuggestion = 10,
    suggestionDelay = 800,
    maxSuggestions = 3,
    className = ''
   } = $props();

  // State management
  let editorElement: HTMLDivElement
  let suggestionPopup: HTMLDivElement
  let isShowingSuggestions = $state(false);
  let currentSuggestions = $state<AISuggestion[]>([]);
  let selectedSuggestionIndex = $state(-1);
  let cursorPosition = $state({ x: 0, y: 0  });
  let isProcessing = $state(false);
  let lastProcessedText = '';

  // XState actor for AI processing
  const aiActor = createActor(aiProcessingMachine);
  aiActor.start();

  // AI suggestion types
  interface AISuggestion { id: string; type: 'completion' | 'grammar' | 'semantic' | 'legal_term';,
    text: string,
    replacement?: string;
    confidence: number; reasoning: string,
    range?: { start: number; end:, number  };
  }

  // Debounced suggestion generation
  const generateSuggestions = debounce(async (text: string, cursorPos: number) => { if (text.length < minCharactersForSuggestion || text === lastProcessedText) {
      return;
     }

    lastProcessedText = text;
    isProcessing = true;

    try { // Get context around cursor
      const contextBefore = text.slice(Math.max(0, cursorPos - 100), cursorPos);
      const contextAfter = text.slice(cursorPos, Math.min(text.length, cursorPos + 50));
      
      // Generate AI suggestions using the enhanced RAG system
      const suggestions = await generateAISuggestions({
        text,
        contextBefore,
        contextAfter,
        cursorPosition: cursorPos });

      currentSuggestions = suggestions.slice(0, maxSuggestions);
      
      if (currentSuggestions.length > 0) { await tick();
        updateSuggestionPopupPosition();
        isShowingSuggestions = true;
        selectedSuggestionIndex = 0;
       }
    } catch (error) { console.error('Failed to generate AI suggestions:', error);
     } finally { isProcessing = false;
     }
  }, suggestionDelay);

  // Generate AI suggestions using multiple techniques
  async function generateAISuggestions(context: { text: string; contextBefore: string,
    contextAfter: string; cursorPosition:, number
   }): Promise<AISuggestion[]> { const suggestions: AISuggestion[] = [];

    // 1. Auto-completion suggestions
    if (enableAutoComplete) {
      const completionTask = aiTaskCreators.analyzeDocument(
        `Complete this text naturally: "${context.contextBefore }[CURSOR]${ context.contextAfter }"
        
        Provide 2-3 natural completions for the text at [CURSOR]. Focus on:
        - Legal terminology accuracy
        - Contextual relevance
        - Natural language flow
        
        Return JSON array with completions.`,
        aiModel,
        'json'
      );

      aiActor.send({ type: 'START_PROCESSING', task: completionTask });
      
      const result = await waitForAIResult(completionTask.id);
      if (result?.success && result.result?.completions) { suggestions.push(...result.result.completions.map((completion: string, index: number) => ({,
          id: `completion_${index }`,
          type: 'completion' as const,
          text: completion,
          confidence: 0.8,
          reasoning: 'AI-generated text completion'
        })));
      }
    }

    // 2. Grammar and style suggestions
    if (enableGrammarCheck) { const grammarTask = aiTaskCreators.analyzeDocument(
        `Analyze this text for grammar, style, and legal writing improvements: "${context.text }"
        
        Focus on:
        - Grammar errors
        - Legal writing style
        - Clarity improvements
        - Professional tone
        
        Return JSON with specific suggestions and replacements.`,
        aiModel,
        'json'
      );

      aiActor.send({ type: 'START_PROCESSING', task: grammarTask });
      
      const result = await waitForAIResult(grammarTask.id);
      if (result?.success && result.result?.suggestions) { suggestions.push(...result.result.suggestions.map((suggestion: any, index: number) => ({,
          id: `grammar_${index }`,
          type: 'grammar' as const,
          text: suggestion.text,
          replacement: suggestion.replacement,
          confidence: suggestion.confidence || 0.7,
          reasoning: suggestion.reasoning || 'Grammar/style improvement',
          range: suggestion.range
        })));
      }
    }

    // 3. Semantic and legal term suggestions
    if (enableSemanticSuggestions) { const semanticTask = createAITask('embed', {
        text: context.contextBefore,
        model: 'nomic-embed-text'
       }, { priority: 'medium'  });

      aiActor.send({ type: 'START_PROCESSING', task: semanticTask });
      
      const embeddingResult = await waitForAIResult(semanticTask.id);
      if (embeddingResult?.success) { // Use RAG to find related legal terms and concepts
        const ragResults = await enhancedRAGStore.queryRAG(
          context.contextBefore,
          { 
            userId: 'inline-editor',
            sessionId: Date.now().toString()
           },
          { topK: 5,
            useEnhancedMode: true,
            filters: {, confidenceThreshold: 0.7  }
          }
        );

        if (ragResults.results?.length > 0) { suggestions.push(...ragResults.results.slice(0, 2).map((result, index) => ({
            id: `semantic_${index }`,
            type: 'legal_term' as const,
            text: result.summary || result.content.slice(0, 100),
            confidence: result.confidence,
            reasoning: `Related legal concept: ${ result.metadata?.type || 'case law' }`
          })));
        }
      }
    }

    return suggestions;
  }

  // Wait for AI task completion
  function waitForAIResult(taskId: string): Promise<any> { return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('AI task timeout')), 10000);
      
      const subscription = aiActor.subscribe((state) => {
        if (state.context.result?.taskId === taskId) {
          clearTimeout(timeout);
          resolve(state.context.result);
         }
        if (state.context.error && state.context.task?.id === taskId) { clearTimeout(timeout);
          reject(new Error(state.context.error));
         }
      });
    });
  }

  // Update suggestion popup position
  function updateSuggestionPopupPosition() { if (!suggestionPopup || !editorElement) return;

    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    const editorRect = editorElement.getBoundingClientRect();

    cursorPosition = {
      x: rect.left -, editorRect.left,
      y: rect.bottom -, editorRect.top + 5
     };
  }

  // Handle input events
  function handleInput(event: InputEvent) { const target = event.target as HTMLDivElement;
    value = target.textContent || '';
    
    const selection = window.getSelection();
    const cursorPos = selection ? selection.anchorOffset : 0;
    
    generateSuggestions(value, cursorPos);
   }

  // Handle keyboard navigation in suggestions
  function handleKeyDown(event: KeyboardEvent) { if (!isShowingSuggestions || currentSuggestions.length === 0) return;

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        selectedSuggestionIndex = Math.min(
          selectedSuggestionIndex + 1,
          currentSuggestions.length - 1
        );
        break;
      
      case 'ArrowUp':
        event.preventDefault();
        selectedSuggestionIndex = Math.max(selectedSuggestionIndex - 1, 0);
        break;
      
      case 'Tab':
      case 'Enter':
        if (selectedSuggestionIndex >= 0) {
          event.preventDefault();
          applySuggestion(currentSuggestions[selectedSuggestionIndex]);
         }
        break;
      
      case 'Escape':
        event.preventDefault();
        hideSuggestions();
        break;
    }
  }

  // Apply selected suggestion
  function applySuggestion(suggestion: AISuggestion) { if (!editorElement) return;

    const selection = window.getSelection();
    if (!selection) return;

    if (suggestion.type === 'completion') {
      // Insert completion at cursor
      const range = selection.getRangeAt(0);
      const textNode = document.createTextNode(suggestion.text);
      range.insertNode(textNode);
      range.setStartAfter(textNode);
      range.setEndAfter(textNode);
      selection.removeAllRanges();
      selection.addRange(range);
     } else if (suggestion.replacement && suggestion.range) { // Replace specific text range
      const textContent = editorElement.textContent || '';
      const before = textContent.slice(0, suggestion.range.start);
      const after = textContent.slice(suggestion.range.end);
      editorElement.textContent = before + suggestion.replacement + after;
     }

    value = editorElement.textContent || '';
    hideSuggestions();
    editorElement.focus();
  }

  // Hide suggestions
  function hideSuggestions() { isShowingSuggestions = false;
    currentSuggestions = [];
    selectedSuggestionIndex = -1;
   }

  // Handle clicks outside to hide suggestions
  function handleClickOutside(event: MouseEvent) { if (suggestionPopup && !suggestionPopup.contains(event.target as Node)) {
      hideSuggestions();
     }
  }

  onMount(() => { document.addEventListener('click', handleClickOutside);
    return () => {
      document.removeEventListener('click', handleClickOutside);
     };
  });

  onDestroy(() => { aiActor.stop();
    generateSuggestions.cancel();
   });
</script>

<div class="enhanced-inline-editor { className }" class:processing={ isProcessing }>
  <!-- Main editor -->
  <div
    bind:this={ editorElement }
    contenteditable="true"
    role="textbox"
    aria-label="AI-enhanced text editor"
    aria-multiline="true"
    class="editor-content"
    { placeholder }
    oninput={ handleInput }
    onkeydown={ handleKeyDown }
    onfocus={ () => generateSuggestions(value, 0) }
  >
    { value }
  </div>

  <!-- Processing indicator -->
  { #if isProcessing }
    <div class="processing-indicator">
      <div class="spinner"></div>
      <span>AI analyzing...</span>
    </div>
  { /if }

  <!-- Suggestion popup -->
  { #if isShowingSuggestions && currentSuggestions.length > 0 }
    <div
      bind:this={ suggestionPopup }
      class="suggestion-popup"
      style="left: { cursorPosition.x }px; top: { cursorPosition.y }px;"
    >
      <div class="suggestion-header">
        <span class="suggestion-title">AI Suggestions</span>
        <button class="close-btn" onclick={ hideSuggestions }>×</button>
      </div>
      
      <div class="suggestions-list">
        { #each currentSuggestions as suggestion, index }
          <button
            class="suggestion-item"
            class:selected={ index === selectedSuggestionIndex }
            onclick={ () => applySuggestion(suggestion) }
          >
            <div class="suggestion-content">
              <div class="suggestion-text">{ suggestion.text }</div>
              <div class="suggestion-meta">
                <span class="suggestion-type">{ suggestion.type }</span>
                <span class="suggestion-confidence">{ Math.round(suggestion.confidence * 100) }%</span>
              </div>
              { #if suggestion.reasoning }
                <div class="suggestion-reasoning">{ suggestion.reasoning }</div>
              { /if }
            </div>
          </button>
        { /each }
      </div>
      
      <div class="suggestion-footer">
        <span class="keyboard-hint">Tab or Enter to apply • ↑↓ to navigate</span>
      </div>
    </div>
  { /if }
</div>

<style>
  .enhanced-inline-editor { position: relative,
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont;
   }

  .editor-content { min-height: 120px;,
    max-height: 400px;,
    overflow-y: auto; padding: 12px 16px;,
    border: 2px solid #e5e7eb;,
    border-radius: 8px;,
    background: white,
    font-size: 14px;,
    line-height: 1.5;,
    outline: none; transition: border-color 0.2s, ease;
   }

  .editor-content:focus { border-color: #3b82f6;,
    box-shadow: 0 0 0 3px, rgba(59, 130, 246, 0.1);
   }

  .editor-content:empty:before { content: attr(placeholder);,
    color: #9ca3af;,
    pointer-events: none }

  .processing-indicator { position: absolute; top: 8px;,
    right: 12px;,
    display: flex,
    align-items: center; gap: 8px;,
    color: #6b7280;,
    font-size: 12px;
   }

  .spinner { width: 12px;,
    height: 12px;,
    border: 2px solid #e5e7eb;,
    border-top: 2px solid #3b82f6;,
    border-radius: 50%;,
    animation: spin 1s linear, infinite;
   }

  @keyframes spin { to { transform: rotate(360deg);  }
  }

  .suggestion-popup { position: absolute,
    z-index: 1000;,
    background: white; border: 1px solid #e5e7eb;,
    border-radius: 8px;,
    box-shadow: 0 10px 25px, rgba(0, 0, 0, 0.15);
    max-width: 400px;,
    min-width: 300px;
   }

  .suggestion-header { display: flex,
    justify-content: space-between;,
    align-items: center; padding: 8px 12px;,
    border-bottom: 1px solid #f3f4f6;,
    background: #f9fafb;,
    border-radius: 8px 8px 0 0;
   }

  .suggestion-title { font-size: 12px;,
    font-weight: 600;,
    color: #374151;
   }

  .close-btn { background: none; border: none,
    font-size: 16px;,
    color: #9ca3af;,
    cursor: pointer; padding: 0;,
    width: 20px;,
    height: 20px;,
    display: flex,
    align-items: center,
    justify-content: center }

  .close-btn:hover { color: #374151;
   }

  .suggestions-list { max-height: 300px;,
    overflow-y: auto }

  .suggestion-item { width: 100%;,
    padding: 12px;,
    border: none; background: white,
    text-align: left; cursor: pointer,
    border-bottom: 1px solid #f3f4f6;,
    transition: background-color 0.15s, ease;
   }

  .suggestion-item:hover,
  .suggestion-item.selected { background: #f3f4f6;
   }

  .suggestion-item.selected { background: #eff6ff;,
    border-left: 3px, solid #3b82f6;
   }

  .suggestion-content { display: flex,
    flex-direction: column;, gap: 4px;
   }

  .suggestion-text { font-size: 14px;,
    color: #111827;,
    font-weight: 500;,
    line-height: 1.3;
   }

  .suggestion-meta { display: flex,
    justify-content: space-between;,
    align-items: center }

  .suggestion-type { font-size: 11px;,
    text-transform: uppercase,
    font-weight: 600;,
    color: #6b7280;,
    background: #f3f4f6;,
    padding: 2px 6px;,
    border-radius: 4px;
   }

  .suggestion-confidence { font-size: 11px;,
    color: #059669;,
    font-weight: 600;
   }

  .suggestion-reasoning { font-size: 12px;,
    color: #6b7280;,
    font-style: italic,
    line-height: 1.3;
   }

  .suggestion-footer { padding: 8px 12px;,
    border-top: 1px solid #f3f4f6;,
    background: #f9fafb;,
    border-radius: 0 0 8px 8px;
   }

  .keyboard-hint { font-size: 11px;,
    color: #9ca3af;
   }

  .processing { opacity: 0.9;
   }

  /* Responsive adjustments */
  @media (max-width: 640px) { .suggestion-popup {
      max-width: 280px;,
      min-width: 240px;
     }
    
    .editor-content { font-size: 16px; /* Prevent zoom on, iOS */
     }
  }
</style>