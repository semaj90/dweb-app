<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from 'bits-ui';
  import { ollama, MODELS } from '$lib/ai/ollama';
  import type { ChatMessage } from '$lib/ai/types';
  
  // Svelte 5 runes
  let messages = $state<ChatMessage[]>([]);
  let input = $state('');
  let isLoading = $state(false);
  let isStreaming = $state(false);
  let currentResponse = $state('');
  let selectedModel = $state(MODELS.LEGAL_DETAILED);
  
  // Props
  let { 
    caseId = null,
    systemPrompt = 'You are an expert legal AI assistant.',
    onMessage = null
  }: {
    caseId?: string | null;
    systemPrompt?: string;
    onMessage?: ((message: ChatMessage) => void) | null;
  } = $props();

  // Reactive computations
  const canSend = $derived(input.trim().length > 0 && !isLoading);
  
  // Mount effect
  onMount(() => {
    // Load chat history if caseId provided
    if (caseId) {
      loadChatHistory();
    }
  });

  async function loadChatHistory() {
    // Implementation depends on your backend
    // const history = await fetch(`/api/cases/${caseId}/chat`).then(r => r.json());
    // messages = history;
  }

  async function sendMessage() {
    if (!canSend) return;
    
    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };
    
    messages = [...messages, userMessage];
    input = '';
    isLoading = true;
    isStreaming = true;
    currentResponse = '';
    
    // Add empty assistant message
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content: '',
      timestamp: new Date()
    };
    messages = [...messages, assistantMessage];
    
    try {
      // Stream the response
      const stream = ollama.generateStream(selectedModel, userMessage.content, {
        system: systemPrompt,
        onToken: (token) => {
          currentResponse += token;
          // Update the last message
          messages[messages.length - 1].content = currentResponse;
          messages = messages; // Trigger reactivity
        }
      });
      
      for await (const token of stream) {
        // Token already handled in onToken callback
      }
      
      // Save to database if caseId provided
      if (caseId && onMessage) {
        onMessage(userMessage);
        onMessage(messages[messages.length - 1]);
      }
      
    } catch (error) {
      console.error('Error generating response:', error);
      messages[messages.length - 1].content = 'Error: Failed to generate response.';
    } finally {
      isLoading = false;
      isStreaming = false;
      currentResponse = '';
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }
</script>

<div class="flex flex-col h-full bg-white dark:bg-gray-900 rounded-lg shadow-lg">
  <!-- Header -->
  <div class="flex items-center justify-between p-4 border-b dark:border-gray-700">
    <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
      Legal AI Assistant
    </h3>
    <select 
      bind:value={selectedModel}
      class="px-3 py-1 text-sm border rounded-md dark:bg-gray-800 dark:border-gray-600"
    >
      <option value={MODELS.LEGAL_DETAILED}>Detailed Analysis</option>
      <option value={MODELS.LEGAL_QUICK}>Quick Response</option>
    </select>
  </div>

  <!-- Messages -->
  <div class="flex-1 overflow-y-auto p-4 space-y-4">
    {#if messages.length === 0}
      <div class="text-center text-gray-500 dark:text-gray-400 mt-8">
        <div class="i-carbon-chat-bot text-4xl mx-auto mb-4"></div>
        <p>Start a conversation with your legal AI assistant</p>
        <p class="text-sm mt-2">Ask about contracts, legal terms, or get document analysis</p>
      </div>
    {/if}
    
    {#each messages as message}
      <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
        <div class="max-w-[80%] {message.role === 'user' 
          ? 'bg-blue-600 text-white' 
          : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white'} 
          rounded-lg px-4 py-3 shadow-sm">
          
          {#if message.role === 'assistant'}
            <div class="flex items-center gap-2 mb-2">
              <div class="i-carbon-bot text-sm"></div>
              <span class="text-xs font-medium">AI Assistant</span>
            </div>
          {/if}
          
          <div class="prose prose-sm dark:prose-invert max-w-none">
            {#if message.role === 'assistant' && isStreaming && message === messages[messages.length - 1]}
              <!-- Show streaming content with cursor -->
              {@html message.content}<span class="animate-pulse">▊</span>
            {:else}
              {@html message.content}
            {/if}
          </div>
          
          {#if message.timestamp}
            <div class="text-xs mt-2 opacity-70">
              {new Date(message.timestamp).toLocaleTimeString()}
            </div>
          {/if}
        </div>
      </div>
    {/each}
    
    {#if isLoading && !isStreaming}
      <div class="flex justify-start">
        <div class="bg-gray-100 dark:bg-gray-800 rounded-lg px-4 py-3">
          <div class="flex items-center gap-2">
            <div class="i-carbon-circle-dash animate-spin"></div>
            <span class="text-sm">Thinking...</span>
          </div>
        </div>
      </div>
    {/if}
  </div>

  <!-- Input -->
  <div class="border-t dark:border-gray-700 p-4">
    <div class="flex gap-2">
      <textarea
        bind:value={input}
        onkeydown={handleKeydown}
        placeholder="Ask a legal question..."
        disabled={isLoading}
        class="flex-1 px-4 py-2 border rounded-lg resize-none
               dark:bg-gray-800 dark:border-gray-600 dark:text-white
               focus:outline-none focus:ring-2 focus:ring-blue-500
               disabled:opacity-50"
        rows="2"
      />
      <Button
        onclick={sendMessage}
        disabled={!canSend}
        class="px-4 py-2 bg-blue-600 text-white rounded-lg
               hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed
               transition-colors duration-200"
      >
        {#if isLoading}
          <div class="i-carbon-send-alt animate-pulse"></div>
        {:else}
          <div class="i-carbon-send-alt"></div>
        {/if}
      </Button>
    </div>
    
    <div class="mt-2 text-xs text-gray-500 dark:text-gray-400">
      Press Enter to send, Shift+Enter for new line
    </div>
  </div>
</div>

<style>
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  .animate-pulse {
    animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }
</style>
