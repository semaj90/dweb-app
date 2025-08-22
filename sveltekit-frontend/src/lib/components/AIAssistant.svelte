<script lang="ts">
  import { $state, $derived } from 'svelte';
  import { createMachine, assign } from 'xstate';
  import { useMachine } from '@xstate/svelte';
  import { createToaster, melt } from 'melt';
  
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Textarea } from '$lib/components/ui/textarea';
  import EnhancedButton from '$lib/components/ui/EnhancedButton.svelte';
  
  // Legal AI Assistant State Machine (XState Best Practices)
  const legalAIMachine = createMachine({
    id: 'legalAI',
    initial: 'idle',
    context: {
      prompt: '',
      response: '',
      error: null,
      conversationHistory: []
    },
    states: {
      idle: {
        on: {
          QUERY: {
            target: 'querying',
            guard: ({ event }) => !!event.prompt?.trim(),
            actions: assign({
              prompt: ({ event }) => event.prompt,
              error: null
            })
          }
        }
      },
      querying: {
        invoke: {
          src: 'queryGemma3Legal',
          input: ({ context }) => ({ prompt: context.prompt }),
          onDone: {
            target: 'success',
            actions: assign({
              response: ({ event }) => event.output.response,
              conversationHistory: ({ context, event }) => [
                ...context.conversationHistory,
                { prompt: context.prompt, response: event.output.response, timestamp: Date.now() }
              ]
            })
          },
          onError: {
            target: 'error',
            actions: assign({
              error: ({ event }) => event.error?.message || 'Failed to connect to Legal AI'
            })
          }
        }
      },
      success: {
        on: {
          QUERY: {
            target: 'querying',
            guard: ({ event }) => !!event.prompt?.trim(),
            actions: assign({
              prompt: ({ event }) => event.prompt,
              error: null
            })
          },
          CLEAR: {
            target: 'idle',
            actions: assign({
              prompt: '',
              response: '',
              error: null
            })
          }
        }
      },
      error: {
        on: {
          RETRY: {
            target: 'querying'
          },
          QUERY: {
            target: 'querying',
            guard: ({ event }) => !!event.prompt?.trim(),
            actions: assign({
              prompt: ({ event }) => event.prompt,
              error: null
            })
          }
        }
      }
    }
  }, {
    actors: {
      queryGemma3Legal: async ({ input }: { input: { prompt: string } }) => {
        const response = await fetch('http://localhost:11434/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'gemma3-legal:latest', // Updated to latest model
            prompt: `As a legal AI assistant, please provide accurate and helpful information about: ${input.prompt}`,
            stream: false,
            options: {
              temperature: 0.3, // Lower temperature for more consistent legal advice
              max_tokens: 2048,  // Increased for detailed legal responses
              top_p: 0.9,
              frequency_penalty: 0.0,
              presence_penalty: 0.0
            }
          })
        });
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        return { response: data.response };
      }
    }
  });

  // Initialize XState machine
  const { snapshot, send } = useMachine(legalAIMachine);
  
  // Reactive state (Svelte 5 best practices)
  let promptInput = $state('');
  let isLoading = $derived(snapshot.matches('querying'));
  let currentResponse = $derived(snapshot.context.response);
  let errorMessage = $derived(snapshot.context.error);
  let canSubmit = $derived(promptInput.trim().length > 0 && !isLoading);
  
  // Toast notifications for user feedback
  const {
    elements: { content: toastContent, title: toastTitle, description: toastDescription, close: toastClose },
    helpers: { addToast },
    states: { toasts }
  } = createToaster({
    placement: 'top-right',
    closeDelay: 5000
  });

  // Enhanced query function with error handling
  function handleQuery() {
    if (!canSubmit) return;
    
    send({ type: 'QUERY', prompt: promptInput });
    
    // Add toast notification
    addToast({
      data: {
        title: 'Legal AI Query',
        description: 'Processing your legal question...',
        color: 'bg-blue-500'
      }
    });
  }

  function handleRetry() {
    send({ type: 'RETRY' });
  }

  function handleClear() {
    send({ type: 'CLEAR' });
    promptInput = '';
  }

  // Keyboard shortcuts (best practices)
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      handleQuery();
    }
  }
</script>

<!-- Toast Notifications -->
{#each $toasts as toast (toast.id)}
  <div use:melt={$toastContent(toast)} class="toast yorha-toast">
    <div use:melt={$toastTitle(toast)}>{toast.data.title}</div>
    <div use:melt={$toastDescription(toast)}>{toast.data.description}</div>
    <button use:melt={$toastClose(toast)} class="toast-close">&times;</button>
  </div>
{/each}

<Card class="w-full max-w-4xl yorha-card">
  <CardHeader class="yorha-header">
    <CardTitle class="flex items-center gap-2">
      <div class="w-3 h-3 rounded-full" class:bg-green-500={snapshot.matches('idle')} 
           class:bg-yellow-500={isLoading} class:bg-red-500={snapshot.matches('error')}></div>
      YoRHa Legal AI Assistant - Gemma3 Legal Latest
    </CardTitle>
    <div class="text-sm text-muted-foreground">
      {#if isLoading}
        Processing legal query with Gemma3-Legal model...
      {:else if snapshot.matches('error')}
        Connection error - Please check Ollama service
      {:else}
        Ready for legal questions • Press Ctrl+Enter to submit
      {/if}
    </div>
  </CardHeader>
  
  <CardContent class="space-y-6">
    <!-- Input Section -->
    <div class="space-y-2">
      <label for="legal-prompt" class="text-sm font-medium">Legal Question</label>
      <Textarea 
        id="legal-prompt"
        bind:value={promptInput}
        onkeydown={handleKeydown}
        placeholder="Ask a legal question (e.g., 'What are the key elements of a valid contract?', 'Explain force majeure clauses', etc.)"
        rows={4}
        class="yorha-textarea"
        disabled={isLoading}
      />
      <div class="flex justify-between text-xs text-muted-foreground">
        <span>Characters: {promptInput.length}</span>
        <span>Ctrl+Enter to submit</span>
      </div>
    </div>

    <!-- Action Buttons -->
    <div class="flex gap-2">
      <EnhancedButton 
        variant="legal" 
        onclick={handleQuery} 
        disabled={!canSubmit}
        loading={isLoading}
        loadingText="Analyzing..."
        useMelt={true}
        class="flex-1"
      >
        {isLoading ? 'Processing Legal Query...' : 'Ask Legal AI'}
      </EnhancedButton>
      
      {#if snapshot.matches('error')}
        <EnhancedButton 
          variant="outline" 
          onclick={handleRetry}
          useMelt={true}
        >
          Retry
        </EnhancedButton>
      {/if}
      
      {#if currentResponse}
        <EnhancedButton 
          variant="ghost" 
          onclick={handleClear}
          useMelt={true}
        >
          Clear
        </EnhancedButton>
      {/if}
    </div>

    <!-- Response Section -->
    {#if errorMessage}
      <div class="p-4 bg-red-50 border border-red-200 rounded-lg yorha-error">
        <div class="flex items-center gap-2 text-red-700">
          <div class="w-4 h-4 text-red-500">⚠️</div>
          <span class="font-medium">Error</span>
        </div>
        <p class="mt-2 text-sm text-red-600">{errorMessage}</p>
        <p class="mt-1 text-xs text-red-500">
          Please ensure Ollama is running with gemma3-legal:latest model
        </p>
      </div>
    {/if}

    {#if currentResponse}
      <div class="space-y-4">
        <div class="flex items-center justify-between">
          <h3 class="text-sm font-medium">Legal AI Response</h3>
          <div class="flex items-center gap-2 text-xs text-muted-foreground">
            <div class="w-2 h-2 bg-green-500 rounded-full"></div>
            Gemma3-Legal Latest
          </div>
        </div>
        
        <div class="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg yorha-response">
          <div class="prose max-w-none">
            <p class="whitespace-pre-wrap text-sm leading-relaxed">{currentResponse}</p>
          </div>
        </div>
        
        <!-- Response Actions -->
        <div class="flex gap-2 text-xs">
          <button class="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full transition-colors">
            Copy Response
          </button>
          <button class="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full transition-colors">
            Save to Case
          </button>
          <button class="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full transition-colors">
            Follow-up Question
          </button>
        </div>
      </div>
    {/if}

    <!-- Conversation History Preview -->
    {#if snapshot.context.conversationHistory.length > 0}
      <details class="mt-6">
        <summary class="text-sm font-medium cursor-pointer hover:text-blue-600">
          Conversation History ({snapshot.context.conversationHistory.length} queries)
        </summary>
        <div class="mt-4 space-y-3 max-h-40 overflow-y-auto">
          {#each snapshot.context.conversationHistory.slice(-3) as item}
            <div class="p-3 bg-gray-50 rounded border-l-4 border-blue-500">
              <div class="text-xs text-gray-500 mb-1">
                {new Date(item.timestamp).toLocaleTimeString()}
              </div>
              <div class="text-sm font-medium mb-1">Q: {item.prompt.slice(0, 100)}...</div>
              <div class="text-xs text-gray-600">A: {item.response.slice(0, 150)}...</div>
            </div>
          {/each}
        </div>
      </details>
    {/if}
  </CardContent>
</Card>

<style>
  /* YoRHa Legal AI Assistant Styling */
  :global(.yorha-card) {
    background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
    border: 2px solid #e5e5e5;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  }
  
  :global(.yorha-header) {
    background: linear-gradient(45deg, #ffbf00, #ffd700);
    color: #000;
    border-bottom: 2px solid #ffbf00;
  }
  
  :global(.yorha-textarea) {
    background: #ffffff;
    border: 2px solid #e5e5e5;
    transition: all 0.2s ease;
    font-family: 'JetBrains Mono', monospace;
  }
  
  :global(.yorha-textarea:focus) {
    border-color: #ffbf00;
    box-shadow: 0 0 0 3px rgba(255, 191, 0, 0.1);
  }
  
  :global(.yorha-response) {
    position: relative;
    overflow: hidden;
  }
  
  :global(.yorha-response::before) {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ffbf00, transparent);
    animation: shimmer 2s infinite;
  }
  
  :global(.yorha-error) {
    border-left: 4px solid #ef4444;
    background: #fef2f2;
  }
  
  /* Toast Styling */
  :global(.yorha-toast) {
    background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
    color: #ffd700;
    border: 1px solid #ffbf00;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    min-width: 300px;
  }
  
  :global(.toast-close) {
    background: none;
    border: none;
    color: #ffd700;
    font-size: 20px;
    cursor: pointer;
    padding: 0;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  @keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
  }
  
  /* Responsive Design */
  @media (max-width: 768px) {
    :global(.yorha-card) {
      max-width: 100%;
      margin: 0 16px;
    }
  }
</style>
