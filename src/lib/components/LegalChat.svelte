<script lang="ts">
  import { onMount } from 'svelte';
  import { ollama, MODELS } from '$lib/ai/ollama';

  // Svelte 5 runes
  let messages = $state<any[]>([]);
  let input = $state('');
  let isLoading = $state(false);
  let isStreaming = $state(false);
  let currentResponse = $state('');
  <script lang="ts">
    import { onMount } from 'svelte';
    import { ollama, MODELS } from '$lib/ai/ollama';

    type ChatMessage = {
      role: 'user' | 'assistant';
      content: string;
      timestamp: Date | string;
    };

    let messages = $state<ChatMessage[]>([]);
    let input = $state('');
    let isLoading = $state(false);
    let isStreaming = $state(false);
    let currentResponse = $state('');
    let selectedModel = $state<string>(MODELS.LEGAL_DETAILED);

    export let caseId: string | null = null;
    export let systemPrompt: string = 'You are an expert legal AI assistant.';
    export let onMessage: ((message: ChatMessage) => void) | null = null;

    const canSend = $derived(input.trim().length > 0 && !isLoading);

    onMount(() => {
      // optionally load history
    });

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

      const assistantMessage: ChatMessage = { role: 'assistant', content: '', timestamp: new Date() };
      messages = [...messages, assistantMessage];

      try {
        const stream = ollama.generateStream(selectedModel, userMessage.content, {
          system: systemPrompt,
          onToken: (token) => {
            currentResponse += token;
            messages[messages.length - 1].content = currentResponse;
            messages = messages;
          }
        });
        for await (const _ of stream) {}
        if (caseId && onMessage) {
          onMessage(userMessage);
          onMessage(messages[messages.length - 1]);
        }
      } catch (e) {
        console.error('Chat error:', e);
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

  <div class="space-y-4">
    <div class="flex items-center justify-between">
      <h2 class="text-lg font-semibold">Legal AI Assistant</h2>
      <select bind:value={selectedModel} class="px-3 py-1 text-sm border rounded-md dark:bg-gray-800 dark:border-gray-600">
        <option value={MODELS.LEGAL_DETAILED}>Detailed Analysis</option>
        <option value={MODELS.LEGAL_QUICK}>Quick Response</option>
      </select>
    </div>

    {#if messages.length === 0}
      <div class="text-sm text-gray-600 dark:text-gray-300">Start a conversation with your legal AI assistant. Ask about contracts, legal terms, or get document analysis.</div>
    {/if}

    <div class="space-y-3">
      {#each messages as message}
        <div class={`max-w-[80%] rounded-lg px-4 py-3 shadow-sm ${message.role === 'user' ? 'bg-blue-600 text-white ml-auto' : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white mr-auto'}`}>
          {#if message.role === 'assistant'}
            <div class="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">AI Assistant</div>
          {/if}
          {#if message.role === 'assistant' && isStreaming && message === messages[messages.length - 1]}
            {@html message.content}▊
          {:else}
            {@html message.content}
          {/if}
          {#if message.timestamp}
            <div class="mt-2 text-[10px] opacity-70">{new Date(message.timestamp).toLocaleTimeString()}</div>
          {/if}
        </div>
      {/each}
    </div>

    {#if isLoading && !isStreaming}
      <div class="text-sm text-gray-500 animate-pulse">Thinking...</div>
    {/if}

    <div class="flex gap-2 items-end">
      <textarea bind:value={input} on:keydown={handleKeydown} placeholder="Ask a legal question..." disabled={isLoading} class="flex-1 px-4 py-2 border rounded-lg resize-none dark:bg-gray-800 dark:border-gray-600 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50" rows="2"></textarea>
      <button on:click={sendMessage} disabled={!canSend} class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200">
        {#if isLoading}Sending...{:else}Send{/if}
      </button>
    </div>

    <div class="text-xs text-gray-500">Press Enter to send, Shift+Enter for new line</div>
  </div>

  <style>
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    .animate-pulse { animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
  </style>
  </style>

