<!-- AI Assistant Modal - YoRHa Theme with XState -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import { useMachine } from '@xstate/svelte';
  import { createChatMachine } from '$lib/state/chatMachine';
  
  const dispatch = createEventDispatcher();
  
  // XState Chat Machine
  const chatMachine = createChatMachine();
  const { state, send, context } = useMachine(chatMachine);
  
  let messageInput = '';
  let messagesContainer: HTMLDivElement;

  // Auto-scroll to bottom when new messages arrive
  $: if ($context.messages.length > 0) {
    setTimeout(() => {
      if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }
    }, 50);
  }

  function handleSendMessage() {
    if (messageInput.trim()) {
      send({ type: 'SEND_MESSAGE', message: messageInput.trim() });
      messageInput = '';
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

  onMount(() => {
    // Initialize chat session
    send({ type: 'INITIALIZE' });
  });
</script>

<!-- Overlay -->
<div
  class="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
  transition:fade={{ duration: 150 }}
  on:click={handleClose}
/>

<!-- Modal Content -->
<div
  class="fixed left-1/2 top-1/2 z-50 max-h-[85vh] w-[90vw] max-w-4xl -translate-x-1/2 -translate-y-1/2 
         bg-stone-900 border border-stone-600 shadow-2xl font-mono"
  transition:fly={{ y: -50, duration: 200 }}
  on:click|stopPropagation
>
  <!-- Header -->
  <div class="flex items-center justify-between border-b border-stone-600 p-4 bg-stone-800">
    <div>
      <h2 class="text-xl font-bold text-stone-100 tracking-wider">
        AI ASSISTANT
      </h2>
      <p class="text-sm text-stone-400">
        YoRHa Legal Intelligence • Status: {$state.value}
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
          {#if $context.messages.length === 0}
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
                  on:click={() => send({ type: 'SEND_MESSAGE', message: 'Analyze the current active cases' })}
                  class="w-full p-3 bg-stone-800 border border-stone-600 hover:bg-stone-700 text-stone-200 text-left"
                >
                  📊 Analyze current active cases
                </button>
                <button 
                  on:click={() => send({ type: 'SEND_MESSAGE', message: 'Help me search for evidence patterns' })}
                  class="w-full p-3 bg-stone-800 border border-stone-600 hover:bg-stone-700 text-stone-200 text-left"
                >
                  🔍 Search for evidence patterns
                </button>
                <button 
                  on:click={() => send({ type: 'SEND_MESSAGE', message: 'Generate investigation timeline' })}
                  class="w-full p-3 bg-stone-800 border border-stone-600 hover:bg-stone-700 text-stone-200 text-left"
                >
                  📅 Generate investigation timeline
                </button>
              </div>
            </div>
          {:else}
            <!-- Chat Messages -->
            {#each $context.messages as message, index}
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

                  <!-- Message Metadata -->
                  {#if message.role === 'assistant' && message.metadata}
                    <div class="mt-3 pt-3 border-t border-stone-600 text-xs opacity-60">
                      {#if message.metadata.confidence}
                        <span>Confidence: {Math.round(message.metadata.confidence * 100)}%</span>
                      {/if}
                      {#if message.metadata.responseTime}
                        <span class="ml-3">Response: {message.metadata.responseTime}ms</span>
                      {/if}
                    </div>
                  {/if}
                </div>
              </div>
            {/each}
            
            <!-- Typing Indicator -->
            {#if $state.matches('sending')}
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

        <!-- Error Display -->
        {#if $context.error}
          <div class="mx-4 p-3 bg-red-900/50 border border-red-600 text-red-200">
            <div class="flex items-center justify-between">
              <span class="text-sm">⚠️ {$context.error}</span>
              <button 
                on:click={() => send({ type: 'CLEAR_ERROR' })}
                class="text-red-400 hover:text-red-200"
              >
                ✕
              </button>
            </div>
          </div>
        {/if}

        <!-- Input Area -->
        <div class="border-t border-stone-600 p-4 bg-stone-800">
          <div class="flex gap-3">
            <input
              type="text"
              bind:value={messageInput}
              on:keydown={handleKeydown}
              placeholder="Ask the AI assistant..."
              disabled={$state.matches('sending')}
              class="flex-1 p-3 bg-stone-900 border border-stone-600 text-stone-100 placeholder-stone-500
                     focus:outline-none focus:border-blue-500 disabled:opacity-50"
            />
            <button
              on:click={handleSendMessage}
              disabled={!messageInput.trim() || $state.matches('sending')}
              class="px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-stone-600 disabled:cursor-not-allowed
                     text-white transition-colors"
            >
              {#if $state.matches('sending')}
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
  </div>

<style>
  :global(.modal-overlay) {
    backdrop-filter: blur(4px);
  }
</style>