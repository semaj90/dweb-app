<!-- AI Assistant Modal - Simple Implementation -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  
  const dispatch = createEventDispatcher();
  
  interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
  }
  
  let messages: ChatMessage[] = [];
  let messageInput = '';
  let isLoading = false;
  let messagesContainer: HTMLDivElement;

  // Auto-scroll to bottom when new messages arrive
  $: if (messages.length > 0) {
    setTimeout(() => {
      if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }
    }, 50);
  }

  async function handleSendMessage() {
    if (!messageInput.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: messageInput.trim(),
      timestamp: new Date()
    };

    messages = [...messages, userMessage];
    const currentMessage = messageInput;
    messageInput = '';
    isLoading = true;

    try {
      // Call Ollama API directly
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: currentMessage,
          stream: false,
          system: "You are a helpful YoRHa AI legal assistant. You help investigators analyze cases, evidence, and provide legal guidance. Respond in a professional but slightly futuristic tone, as if you're part of the YoRHa investigation unit. Keep responses concise and actionable."
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      const aiMessage: ChatMessage = {
        id: `ai-${Date.now()}`,
        role: 'assistant',
        content: data.response || 'No response received',
        timestamp: new Date()
      };

      messages = [...messages, aiMessage];
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
        timestamp: new Date()
      };
      messages = [...messages, errorMessage];
    } finally {
      isLoading = false;
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  }

  function handleClose() {
    dispatch('close');
  }

  function formatTime(timestamp: Date): string {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    }).format(timestamp);
  }

  function usePredefinedMessage(message: string) {
    messageInput = message;
    handleSendMessage();
  }
</script>

<!-- Overlay -->
<div
  class="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
  transition:fade={{ duration: 150 }}
  on:click={handleClose}
  on:keydown={(e) => e.key === 'Escape' && handleClose()}
  role="button"
  tabindex="0"
  aria-label="Close modal overlay"
></div>

<!-- Modal Content -->
<div
  class="fixed left-1/2 top-1/2 z-50 max-h-[85vh] w-[90vw] max-w-4xl -translate-x-1/2 -translate-y-1/2 
         bg-stone-900 border border-stone-600 shadow-2xl font-mono"
  transition:fly={{ y: -50, duration: 200 }}
  on:click|stopPropagation
  on:keydown={(e) => e.key === 'Escape' && handleClose()}
  role="dialog"
  aria-modal="true"
  aria-labelledby="modal-title"
  tabindex="-1"
>
  <!-- Header -->
  <div class="flex items-center justify-between border-b border-stone-600 p-4 bg-stone-800">
    <div>
      <h2 id="modal-title" class="text-xl font-bold text-stone-100 tracking-wider">
        AI ASSISTANT
      </h2>
      <p class="text-sm text-stone-400">
        YoRHa Legal Intelligence • Status: {isLoading ? 'Processing' : 'Ready'}
      </p>
    </div>
    <div class="flex items-center gap-3">
      <!-- Connection Status -->
      <div class="flex items-center gap-2">
        <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
        <span class="text-xs text-stone-400">ONLINE</span>
      </div>
      <!-- Close Button -->
      <button
        class="p-2 text-stone-400 hover:text-stone-100 hover:bg-stone-700 transition-colors"
        on:click={handleClose}
      >
        ✕
      </button>
    </div>
  </div>

  <!-- Chat Area -->
  <div class="flex flex-col h-[70vh]">
    <!-- Messages Container -->
    <div 
      bind:this={messagesContainer}
      class="flex-1 overflow-y-auto p-4 space-y-4 bg-stone-900"
    >
      {#if messages.length === 0}
        <!-- Welcome Screen -->
        <div class="flex flex-col items-center justify-center h-full text-center">
          <div class="text-6xl mb-4">🤖</div>
          <h3 class="text-xl text-stone-100 mb-2">YoRHa AI Assistant</h3>
          <p class="text-stone-400 mb-6 max-w-md">
            I can help you with case analysis, legal research, evidence evaluation, 
            and investigation planning. What would you like to know?
          </p>
          <div class="space-y-2 w-full max-w-lg">
            <button 
              on:click={() => usePredefinedMessage('Analyze the current active cases')}
              class="w-full p-3 bg-stone-800 border border-stone-600 hover:bg-stone-700 text-stone-200 text-left"
            >
              📊 Analyze current active cases
            </button>
            <button 
              on:click={() => usePredefinedMessage('Help me search for evidence patterns')}
              class="w-full p-3 bg-stone-800 border border-stone-600 hover:bg-stone-700 text-stone-200 text-left"
            >
              🔍 Search for evidence patterns
            </button>
            <button 
              on:click={() => usePredefinedMessage('Generate investigation timeline')}
              class="w-full p-3 bg-stone-800 border border-stone-600 hover:bg-stone-700 text-stone-200 text-left"
            >
              📅 Generate investigation timeline
            </button>
          </div>
        </div>
      {:else}
        <!-- Chat Messages -->
        {#each messages as message}
          <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
            <div class="max-w-[80%] {message.role === 'user' ? 'bg-blue-600' : 'bg-stone-700'} 
                       text-white p-4 border border-stone-600">
              <!-- Message Header -->
              <div class="flex items-center justify-between mb-2 text-xs opacity-80">
                <span>{message.role === 'user' ? '👤 INVESTIGATOR' : '🤖 AI ASSISTANT'}</span>
                <span>{formatTime(message.timestamp)}</span>
              </div>
              
              <!-- Message Content -->
              <div class="text-sm leading-relaxed whitespace-pre-wrap">
                {message.content}
              </div>
            </div>
          </div>
        {/each}
        
        <!-- Typing Indicator -->
        {#if isLoading}
          <div class="flex justify-start">
            <div class="max-w-[80%] bg-stone-700 text-white p-4 border border-stone-600">
              <div class="flex items-center gap-2">
                <div class="flex gap-1">
                  <div class="w-2 h-2 bg-stone-400 rounded-full animate-bounce"></div>
                  <div class="w-2 h-2 bg-stone-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                  <div class="w-2 h-2 bg-stone-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                </div>
                <span class="text-sm text-stone-300">AI is analyzing...</span>
              </div>
            </div>
          </div>
        {/if}
      {/if}
    </div>

    <!-- Input Area -->
    <div class="border-t border-stone-600 p-4 bg-stone-800">
      <div class="flex gap-3">
        <input
          type="text"
          bind:value={messageInput}
          on:keydown={handleKeydown}
          placeholder="Ask the AI assistant..."
          disabled={isLoading}
          class="flex-1 p-3 bg-stone-900 border border-stone-600 text-stone-100 placeholder-stone-500
                 focus:outline-none focus:border-blue-500 disabled:opacity-50"
        />
        <button
          on:click={handleSendMessage}
          disabled={!messageInput.trim() || isLoading}
          class="px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-stone-600 disabled:cursor-not-allowed
                 text-white transition-colors"
        >
          {#if isLoading}
            ⏳
          {:else}
            SEND
          {/if}
        </button>
      </div>
      
      <div class="flex justify-between items-center mt-2 text-xs text-stone-500">
        <span>Press Enter to send</span>
        <span>
          {#if messageInput.length > 0}
            {messageInput.length} chars
          {/if}
        </span>
      </div>
    </div>
  </div>
</div>

<style>
  :global(.modal-overlay) {
    backdrop-filter: blur(4px);
  }
</style>