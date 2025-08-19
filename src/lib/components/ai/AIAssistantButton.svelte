<!-- AI Assistant Button with Backend Integration and NES-Style State Caching -->
<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import { writable, derived } from 'svelte/store';
  import { fade, fly, scale } from 'svelte/transition';
  import { 
    MessageSquare, 
    Brain, 
    Zap, 
    Settings, 
    Minimize2, 
    Maximize2,
    Send,
    Loader2,
    CheckCircle,
    AlertTriangle,
    Layers3,
    ExternalLink
  } from 'lucide-svelte';
  import { goto } from '$app/navigation';
  import { createMachine, interpret, assign } from 'xstate';
  import type { AIAssistantState, ChatMessage, AICapabilities } from '$lib/types/ai';

  // Props
  interface Props {
    position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
    size?: 'sm' | 'md' | 'lg';
    theme?: 'light' | 'dark' | 'yorha';
    enableGPUAcceleration?: boolean;
    enableRealTimeAnalysis?: boolean;
    enableContextSwitching?: boolean;
    enableEnhancedMode?: boolean;
    showDemoAccess?: boolean;
    className?: string;
  }

  let {
    position = 'bottom-right',
    size = 'md',
    theme = 'yorha',
    enableGPUAcceleration = true,
    enableRealTimeAnalysis = true,
    enableContextSwitching = true,
    enableEnhancedMode = false,
    showDemoAccess = true,
    className = ''
  }: Props = $props();

  // Event dispatcher
  const dispatch = createEventDispatcher<{
    open: { timestamp: number };
    close: { timestamp: number };
    message: { message: ChatMessage };
    error: { error: string };
    statusChange: { status: string };
  }>();

  // XState Machine for AI Assistant
  const aiAssistantMachine = createMachine({
    id: 'aiAssistant',
    initial: 'collapsed',
    context: {
      isConnected: false,
      capabilities: [] as AICapabilities[],
      messages: [] as ChatMessage[],
      currentInput: '',
      isProcessing: false,
      error: null as string | null,
      connectionProtocol: 'QUIC' as 'REST' | 'gRPC' | 'QUIC',
      gpuStatus: 'unknown' as 'available' | 'unavailable' | 'unknown'
    },
    states: {
      collapsed: {
        on: {
          EXPAND: 'expanded',
          CONNECT: 'connecting'
        }
      },
      connecting: {
        invoke: {
          id: 'connectToBackend',
          src: 'connectToBackend',
          onDone: {
            target: 'expanded',
            actions: assign({
              isConnected: true,
              capabilities: (_, event) => event.data.capabilities,
              connectionProtocol: (_, event) => event.data.protocol
            })
          },
          onError: {
            target: 'error',
            actions: assign({
              error: (_, event) => event.data.message
            })
          }
        }
      },
      expanded: {
        initial: 'idle',
        states: {
          idle: {
            on: {
              SEND_MESSAGE: 'processing',
              COLLAPSE: '#aiAssistant.collapsed'
            }
          },
          processing: {
            invoke: {
              id: 'processMessage',
              src: 'processMessage',
              onDone: {
                target: 'idle',
                actions: assign({
                  messages: (context, event) => [
                    ...context.messages,
                    event.data.userMessage,
                    event.data.aiResponse
                  ],
                  currentInput: '',
                  isProcessing: false
                })
              },
              onError: {
                target: 'idle',
                actions: assign({
                  error: (_, event) => event.data.message,
                  isProcessing: false
                })
              }
            },
            entry: assign({ isProcessing: true }),
            exit: assign({ isProcessing: false })
          }
        },
        on: {
          COLLAPSE: 'collapsed',
          DISCONNECT: 'collapsed'
        }
      },
      error: {
        on: {
          RETRY: 'connecting',
          COLLAPSE: 'collapsed'
        }
      }
    }
  }, {
    services: {
      connectToBackend: async () => {
        // Multi-protocol backend connection with context switching
        const protocols = ['QUIC', 'gRPC', 'REST'];
        let lastError;

        for (const protocol of protocols) {
          try {
            const response = await fetch('/api/ai/connect', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                protocol,
                capabilities: [
                  'legal_analysis',
                  'document_summarization',
                  'case_research',
                  'contract_review',
                  'real_time_chat',
                  'gpu_acceleration'
                ],
                gpuAcceleration: enableGPUAcceleration,
                realTimeAnalysis: enableRealTimeAnalysis
              })
            });

            if (response.ok) {
              const data = await response.json();
              return {
                capabilities: data.capabilities,
                protocol,
                gpuStatus: data.gpuStatus,
                modelInfo: data.modelInfo
              };
            }
          } catch (error) {
            lastError = error;
            console.warn(`Connection failed with ${protocol}, trying next protocol...`);
          }
        }

        throw lastError || new Error('All connection protocols failed');
      },

      processMessage: async (context, event) => {
        const userMessage: ChatMessage = {
          id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          role: 'user',
          content: event.input,
          timestamp: Date.now(),
          metadata: {
            protocol: context.connectionProtocol,
            gpuAccelerated: enableGPUAcceleration
          }
        };

        // Send to enhanced RAG pipeline with context switching
        const response = await fetch('/api/ai/chat', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'X-Protocol': context.connectionProtocol,
            'X-GPU-Acceleration': enableGPUAcceleration.toString()
          },
          body: JSON.stringify({
            message: userMessage,
            context: {
              previousMessages: context.messages.slice(-5), // Last 5 messages for context
              sessionId: getSessionId(),
              capabilities: context.capabilities,
              preferences: {
                responseLength: 'detailed',
                includeReferences: true,
                legalCitations: true
              }
            },
            pipeline: {
              useRAG: true,
              useRealtimeAnalysis: enableRealTimeAnalysis,
              useSemanticSearch: true,
              useIntentDetection: true
            }
          })
        });

        if (!response.ok) {
          throw new Error(`AI processing failed: ${response.statusText}`);
        }

        const data = await response.json();
        
        const aiResponse: ChatMessage = {
          id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          role: 'assistant',
          content: data.response,
          timestamp: Date.now(),
          metadata: {
            processingTime: data.processingTime,
            protocol: context.connectionProtocol,
            confidence: data.confidence,
            sources: data.sources,
            reasoning: data.reasoning
          }
        };

        return { userMessage, aiResponse };
      }
    }
  });

  // XState service
  let aiService: any;
  let currentState = $state(aiAssistantMachine.initialState);
  let currentInput = $state('');

  // NES-Style State Caching
  let stateCache: Map<string, any> = new Map();
  let animationWorker: Worker | null = null;

  // Connection status
  let connectionStatus = $derived(() => {
    if (currentState.matches('connecting')) return 'connecting';
    if (currentState.matches('error')) return 'error';
    if (currentState.context.isConnected) return 'connected';
    return 'disconnected';
  });

  // UI State
  let isExpanded = $derived(() => currentState.matches('expanded'));
  let isProcessing = $derived(() => currentState.context.isProcessing);
  let messages = $derived(() => currentState.context.messages);

  onMount(async () => {
    // Initialize XState service
    aiService = interpret(aiAssistantMachine);
    aiService.subscribe((state: any) => {
      currentState = state;
      cacheState(state);
      
      // Dispatch status changes
      dispatch('statusChange', { status: connectionStatus() });
    });
    aiService.start();

    // Initialize NES-style state caching
    await initializeStateCaching();

    // Auto-connect if enabled
    if (enableContextSwitching) {
      setTimeout(() => aiService.send('CONNECT'), 1000);
    }

    return () => {
      aiService?.stop();
      animationWorker?.terminate();
    };
  });

  async function initializeStateCaching() {
    try {
      // Initialize animation worker for smooth state transitions
      animationWorker = new Worker('/workers/nes-state-animator.js');
      
      animationWorker.onmessage = (event) => {
        const { type, data } = event.data;
        
        switch (type) {
          case 'STATE_TRANSITION_READY':
            applyStateTransition(data);
            break;
          case 'CACHE_OPTIMIZED':
            console.log('State cache optimized:', data.stats);
            break;
        }
      };

      // Pre-cache common UI states
      const commonStates = [
        'collapsed_idle',
        'expanded_chat',
        'processing_loading',
        'error_display'
      ];

      commonStates.forEach(stateId => {
        preLoadState(stateId);
      });

    } catch (error) {
      console.warn('State caching initialization failed:', error);
    }
  }

  function cacheState(state: any) {
    const stateKey = generateStateKey(state);
    stateCache.set(stateKey, {
      value: state.value,
      context: state.context,
      timestamp: Date.now()
    });

    // Send to animation worker for predictive caching
    if (animationWorker) {
      animationWorker.postMessage({
        type: 'CACHE_STATE',
        stateKey,
        state: {
          value: state.value,
          context: state.context
        }
      });
    }
  }

  function generateStateKey(state: any): string {
    return `${JSON.stringify(state.value)}_${state.context.isConnected}_${state.context.isProcessing}`;
  }

  function preLoadState(stateId: string) {
    // Pre-load common state transitions for instant UI updates
    if (animationWorker) {
      animationWorker.postMessage({
        type: 'PRELOAD_STATE',
        stateId
      });
    }
  }

  function applyStateTransition(transitionData: any) {
    // Apply smooth state transitions using cached data
    // This creates the "instant sprite switching" effect
    const cachedState = stateCache.get(transitionData.targetStateKey);
    if (cachedState) {
      // Apply transition instantly using cached state
      console.log('Applying cached state transition:', transitionData.targetStateKey);
    }
  }

  function handleExpand() {
    if (currentState.matches('collapsed')) {
      aiService.send('EXPAND');
      dispatch('open', { timestamp: Date.now() });
    }
  }

  function handleCollapse() {
    if (currentState.matches('expanded')) {
      aiService.send('COLLAPSE');
      dispatch('close', { timestamp: Date.now() });
    }
  }

  function handleSendMessage() {
    if (currentInput.trim() && !isProcessing) {
      const message = currentInput.trim();
      aiService.send({ type: 'SEND_MESSAGE', input: message });
      
      dispatch('message', { 
        message: {
          id: `temp_${Date.now()}`,
          role: 'user',
          content: message,
          timestamp: Date.now()
        }
      });
      
      currentInput = '';
    }
  }

  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  }

  function handleRetry() {
    aiService.send('RETRY');
  }

  function getSessionId(): string {
    let sessionId = sessionStorage.getItem('ai_session_id');
    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      sessionStorage.setItem('ai_session_id', sessionId);
    }
    return sessionId;
  }

  function handleEnhancedMode() {
    // Navigate to the high-performance AI assistant demo
    goto('/ai-assistant-demo');
    dispatch('open', { timestamp: Date.now(), mode: 'enhanced' });
  }

  function toggleEnhancedMode() {
    enableEnhancedMode = !enableEnhancedMode;
  }

  // Position classes
  const positionClasses = {
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'top-right': 'top-4 right-4',
    'top-left': 'top-4 left-4'
  };

  // Size classes
  const sizeClasses = {
    sm: 'w-80 h-96',
    md: 'w-96 h-[32rem]',
    lg: 'w-[28rem] h-[36rem]'
  };

  // Theme classes
  const themeClasses = {
    light: 'bg-white border-gray-200 text-gray-900',
    dark: 'bg-gray-800 border-gray-600 text-gray-100',
    yorha: 'bg-yorha-surface border-yorha-border text-yorha-text'
  };
</script>

<div 
  class="fixed z-50 {positionClasses[position]} {className}"
  style="font-family: 'Share Tech Mono', monospace;"
>
  {#if isExpanded}
    <!-- Expanded Chat Interface -->
    <div 
      class="border shadow-2xl rounded-lg {sizeClasses[size]} {themeClasses[theme]} flex flex-col"
      transition:scale={{ duration: 300, start: 0.8 }}
    >
      <!-- Header -->
      <div class="flex items-center justify-between p-4 border-b {theme === 'yorha' ? 'border-yorha-border' : 'border-gray-200 dark:border-gray-600'}">
        <div class="flex items-center space-x-3">
          <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <Brain class="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 class="font-semibold text-sm">Legal AI Assistant</h3>
            <div class="flex items-center space-x-2 text-xs {theme === 'yorha' ? 'text-yorha-muted' : 'text-gray-500'}">
              <div class="w-2 h-2 rounded-full {
                connectionStatus() === 'connected' ? 'bg-green-500' :
                connectionStatus() === 'connecting' ? 'bg-yellow-500' :
                'bg-red-500'
              }"></div>
              <span>{connectionStatus()}</span>
              {#if currentState.context.connectionProtocol}
                <span>• {currentState.context.connectionProtocol}</span>
              {/if}
              {#if enableGPUAcceleration && currentState.context.gpuStatus === 'available'}
                <Zap class="w-3 h-3 text-purple-500" />
              {/if}
            </div>
          </div>
        </div>
        
        <div class="flex items-center space-x-2">
          {#if showDemoAccess}
            <button
              onclick={handleEnhancedMode}
              class="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors group"
              aria-label="Open Enhanced 3D Mode"
              title="Launch High-Performance AI Assistant with 3D Visualization"
            >
              <Layers3 class="w-4 h-4 group-hover:text-purple-500 transition-colors" />
            </button>
          {/if}
          
          <button
            onclick={handleCollapse}
            class="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label="Minimize chat"
          >
            <Minimize2 class="w-4 h-4" />
          </button>
        </div>
      </div>

      <!-- Messages Area -->
      <div class="flex-1 overflow-y-auto p-4 space-y-4">
        {#if messages.length === 0}
          <div class="text-center py-8">
            <Brain class="w-12 h-12 {theme === 'yorha' ? 'text-yorha-muted' : 'text-gray-400'} mx-auto mb-4" />
            <h4 class="font-medium mb-2">Legal AI Assistant Ready</h4>
            <p class="text-sm {theme === 'yorha' ? 'text-yorha-muted' : 'text-gray-500'}">
              Ask questions about legal matters, contracts, case analysis, or document review.
            </p>
          </div>
        {:else}
          {#each messages as message (message.id)}
            <div 
              class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}"
              transition:fly={{ y: 20, duration: 300 }}
            >
              <div 
                class="max-w-[80%] rounded-lg px-3 py-2 {
                  message.role === 'user' 
                    ? 'bg-blue-600 text-white' 
                    : theme === 'yorha'
                      ? 'bg-yorha-accent/10 border border-yorha-border'
                      : 'bg-gray-100 dark:bg-gray-700'
                }"
              >
                <div class="text-sm whitespace-pre-wrap">{message.content}</div>
                
                {#if message.metadata}
                  <div class="mt-2 pt-2 border-t border-opacity-20 border-gray-300 text-xs opacity-70">
                    {#if message.metadata.processingTime}
                      <span>Processed in {message.metadata.processingTime}ms</span>
                    {/if}
                    {#if message.metadata.confidence}
                      <span> • Confidence: {Math.round(message.metadata.confidence * 100)}%</span>
                    {/if}
                    {#if message.metadata.sources?.length}
                      <span> • {message.metadata.sources.length} sources</span>
                    {/if}
                  </div>
                {/if}
              </div>
            </div>
          {/each}
        {/if}

        {#if isProcessing}
          <div class="flex justify-start" transition:fade>
            <div class="max-w-[80%] rounded-lg px-3 py-2 {theme === 'yorha' ? 'bg-yorha-accent/10 border border-yorha-border' : 'bg-gray-100 dark:bg-gray-700'}">
              <div class="flex items-center space-x-2">
                <Loader2 class="w-4 h-4 animate-spin" />
                <span class="text-sm">Analyzing with AI...</span>
              </div>
            </div>
          </div>
        {/if}

        {#if currentState.matches('error')}
          <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3" transition:fade>
            <div class="flex items-center space-x-2 text-red-700 dark:text-red-300">
              <AlertTriangle class="w-4 h-4" />
              <span class="text-sm font-medium">Connection Error</span>
            </div>
            <p class="text-sm text-red-600 dark:text-red-400 mt-1">
              {currentState.context.error}
            </p>
            <button
              onclick={handleRetry}
              class="mt-2 text-sm text-red-700 dark:text-red-300 hover:text-red-800 dark:hover:text-red-200 underline"
            >
              Retry Connection
            </button>
          </div>
        {/if}
      </div>

      <!-- Input Area -->
      <div class="border-t {theme === 'yorha' ? 'border-yorha-border' : 'border-gray-200 dark:border-gray-600'} p-4">
        <div class="flex space-x-2">
          <textarea
            bind:value={currentInput}
            onkeydown={handleKeyPress}
            placeholder="Ask a legal question..."
            disabled={!currentState.context.isConnected || isProcessing}
            class="flex-1 resize-none rounded-lg border {theme === 'yorha' ? 'border-yorha-border bg-yorha-surface' : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700'} px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
            rows="2"
          ></textarea>
          
          <button
            onclick={handleSendMessage}
            disabled={!currentInput.trim() || !currentState.context.isConnected || isProcessing}
            class="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg transition-colors flex items-center justify-center"
            aria-label="Send message"
          >
            {#if isProcessing}
              <Loader2 class="w-4 h-4 animate-spin" />
            {:else}
              <Send class="w-4 h-4" />
            {/if}
          </button>
        </div>
        
        <div class="mt-2 flex items-center justify-between">
          {#if enableGPUAcceleration && currentState.context.gpuStatus === 'available'}
            <div class="flex items-center space-x-2 text-xs {theme === 'yorha' ? 'text-yorha-muted' : 'text-gray-500'}">
              <Zap class="w-3 h-3 text-purple-500" />
              <span>GPU-accelerated processing enabled</span>
            </div>
          {/if}
          
          {#if showDemoAccess}
            <button
              onclick={handleEnhancedMode}
              class="flex items-center space-x-1 text-xs {theme === 'yorha' ? 'text-yorha-accent hover:text-yorha-text' : 'text-purple-600 hover:text-purple-800'} transition-colors"
              aria-label="Launch Enhanced 3D Mode"
            >
              <Layers3 class="w-3 h-3" />
              <span>3D Mode</span>
              <ExternalLink class="w-2 h-2" />
            </button>
          {/if}
        </div>
      </div>
    </div>
  {:else}
    <!-- Collapsed Button -->
    <div class="relative">
      <button
        onclick={handleExpand}
        on:dblclick={showDemoAccess ? handleEnhancedMode : undefined}
        class="w-14 h-14 bg-gradient-to-br from-blue-600 to-purple-700 hover:from-blue-700 hover:to-purple-800 text-white rounded-full shadow-lg hover:shadow-xl transition-all duration-300 flex items-center justify-center group"
        transition:scale={{ duration: 300 }}
        aria-label="Open AI Assistant"
        title={showDemoAccess ? "Click to open chat • Double-click for 3D mode" : "Click to open AI Assistant"}
      >
      <div class="relative">
        <MessageSquare class="w-6 h-6 group-hover:scale-110 transition-transform duration-200" />
        
        {#if connectionStatus() === 'connected'}
          <div class="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white"></div>
        {:else if connectionStatus() === 'connecting'}
          <div class="absolute -top-1 -right-1 w-3 h-3 bg-yellow-500 rounded-full border-2 border-white animate-pulse"></div>
        {:else if connectionStatus() === 'error'}
          <div class="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full border-2 border-white"></div>
        {/if}

        {#if enableGPUAcceleration}
          <div class="absolute -bottom-1 -right-1 w-4 h-4 bg-purple-600 rounded-full border-2 border-white flex items-center justify-center">
            <Zap class="w-2 h-2 text-white" />
          </div>
        {/if}
      </div>
      </button>
      
      {#if showDemoAccess}
        <!-- Enhanced Mode Quick Access -->
        <button
          onclick={handleEnhancedMode}
          class="absolute -top-2 -right-2 w-6 h-6 bg-gradient-to-br from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white rounded-full shadow-lg hover:shadow-xl transition-all duration-200 flex items-center justify-center opacity-0 group-hover:opacity-100"
          aria-label="Launch 3D Mode"
          title="High-Performance 3D AI Assistant"
        >
          <Layers3 class="w-3 h-3" />
        </button>
      {/if}
    </div>
  {/if}
</div>

<style>
  /* YoRHa theme custom properties */
  :global(.yorha-surface) {
    background-color: #1a1a1a;
  }
  
  :global(.yorha-border) {
    border-color: #333;
  }
  
  :global(.yorha-text) {
    color: #e0e0e0;
  }
  
  :global(.yorha-muted) {
    color: #999;
  }
  
  :global(.yorha-accent) {
    color: #00bcd4;
  }

  /* Custom scrollbar */
  :global(.overflow-y-auto::-webkit-scrollbar) {
    width: 4px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-track) {
    background: transparent;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb) {
    background: rgba(156, 163, 175, 0.5);
    border-radius: 2px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb:hover) {
    background: rgba(156, 163, 175, 0.8);
  }

  /* Animations */
  @keyframes pulse-slow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .animate-pulse-slow {
    animation: pulse-slow 2s infinite;
  }
</style>