<script lang="ts">
  import { onMount } from 'svelte';
  import { $props, $state, $derived, $effect } from 'svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Badge } from '$lib/components/ui/badge/index.js';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card/index.js';
  import { Textarea } from '$lib/components/ui/textarea/index.js';
  import { Separator } from '$lib/components/ui/separator/index.js';
  import { Select } from '$lib/components/ui/select/index.js';
  import { Slider } from '$lib/components/ui/slider/index.js';
  import { Switch } from '$lib/components/ui/switch/index.js';
  import { Dialog } from 'bits-ui';
  import { 
    aiAssistantManager,
    isAIActive,
    isProcessing,
    currentResponse,
    conversationHistory,
    currentModel,
    currentTemperature,
    aiError,
    clusterHealth,
    context7Analysis,
    aiUsage
  } from '$lib/stores/aiAssistant.svelte.js';

  // Component props
  interface Props {
    height?: string;
    showSettings?: boolean;
    enableContext7?: boolean;
    autoFocus?: boolean;
  }

  let {
    height = '600px',
    showSettings = true,
    enableContext7 = true,
    autoFocus = true
  }: Props = $props();

  // Reactive state using Svelte 5 runes
  let currentMessage = $state('');
  let useContext7 = $state(false);
  let showSettingsDialog = $state(false);
  let showExportDialog = $state(false);
  let messageInput = $state<HTMLTextAreaElement>();
  let chatContainer = $state<HTMLDivElement>();
  let availableModels = $state<string[]>(['gemma3-legal', 'nomic-embed-text', 'deeds-web']);

  // Derived state for UI
  let canSend = $derived(() => currentMessage.trim().length > 0 && !isProcessing());
  let hasConversation = $derived(() => conversationHistory().length > 0);
  let clusterStatus = $derived(() => {
    const health = clusterHealth();
    const healthyCount = Object.values(health).filter(Boolean).length;
    return {
      healthy: healthyCount === Object.keys(health).length,
      count: healthyCount,
      total: Object.keys(health).length
    };
  });

  // Component lifecycle
  onMount(async () => {
    // Focus input if enabled
    if (autoFocus && messageInput) {
      messageInput.focus();
    }

    // Load available models
    try {
      availableModels = await aiAssistantManager.getAvailableModels();
    } catch (error) {
      console.warn('Failed to load available models:', error);
    }

    // Check cluster health
    aiAssistantManager.checkClusterHealth();
  });

  // Effect to scroll to bottom when conversation updates
  $effect(() => {
    if (conversationHistory().length > 0 && chatContainer) {
      setTimeout(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }, 100);
    }
  });

  // Send message to AI
  async function sendMessage() {
    if (!canSend) return;

    const message = currentMessage.trim();
    currentMessage = '';

    try {
      await aiAssistantManager.sendMessage(message, {
        useContext7: enableContext7 && useContext7,
        model: currentModel(),
        temperature: currentTemperature()
      });
    } catch (error) {
      console.error('Failed to send message:', error);
    }

    // Focus back to input
    if (messageInput) {
      messageInput.focus();
    }
  }

  // Handle keyboard shortcuts
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  // Format timestamp
  function formatTime(date: Date): string {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    }).format(date);
  }

  // Get message role color
  function getRoleColor(role: string): string {
    switch (role) {
      case 'user': return 'bg-blue-100 border-blue-200';
      case 'assistant': return 'bg-green-100 border-green-200';
      case 'system': return 'bg-gray-100 border-gray-200';
      default: return 'bg-gray-100 border-gray-200';
    }
  }

  // Clear conversation
  function clearConversation() {
    aiAssistantManager.clearConversation();
  }

  // Export conversation
  function exportConversation() {
    aiAssistantManager.exportConversation();
  }

  // Stop generation
  function stopGeneration() {
    aiAssistantManager.stopGeneration();
  }

  // Retry last message
  function retryLast() {
    aiAssistantManager.retryLast();
  }
</script>

<Card class="w-full" style="height: {height};">
  <CardHeader class="p-4 border-b">
    <div class="flex items-center justify-between">
      <CardTitle class="flex items-center gap-2">
        <span class="text-lg font-semibold">Legal AI Assistant</span>
        <Badge variant={clusterStatus.healthy ? 'default' : 'destructive'} class="text-xs">
          {clusterStatus.healthy ? 'Online' : `${clusterStatus.count}/${clusterStatus.total} Healthy`}
        </Badge>
      </CardTitle>
      
      <div class="flex items-center gap-2">
        {#if showSettings}
          <Button 
            variant="ghost" 
            size="sm"
            onclick={() => showSettingsDialog = true}
          >
            <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </Button>
        {/if}

        {#if hasConversation}
          <Button 
            variant="ghost" 
            size="sm"
            onclick={() => showExportDialog = true}
          >
            <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </Button>
        {/if}
      </div>
    </div>
  </CardHeader>

  <CardContent class="p-0 flex flex-col" style="height: calc({height} - 80px);">
    <!-- Chat Messages -->
    <div 
      bind:this={chatContainer}
      class="flex-1 overflow-y-auto p-4 space-y-4"
    >
      {#if conversationHistory().length === 0}
        <div class="text-center text-gray-500 mt-8">
          <div class="text-lg font-medium mb-2">Welcome to Legal AI Assistant</div>
          <p class="text-sm">Ask me about legal documents, case analysis, or get help with your legal research.</p>
          {#if enableContext7}
            <div class="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div class="text-sm font-medium text-blue-800 mb-1">Context7 Enhanced</div>
              <p class="text-xs text-blue-600">Get suggestions powered by the latest documentation and best practices.</p>
            </div>
          {/if}
        </div>
      {:else}
        {#each conversationHistory() as entry (entry.id)}
          <div class="flex flex-col gap-1">
            <div class="flex items-center gap-2 text-xs text-gray-500">
              <span class="font-medium capitalize">{entry.type}</span>
              <span>{formatTime(entry.timestamp)}</span>
              {#if entry.metadata?.model}
                <Badge variant="outline" class="text-xs">{entry.metadata.model}</Badge>
              {/if}
              {#if entry.metadata?.context7Used}
                <Badge variant="secondary" class="text-xs">Context7</Badge>
              {/if}
            </div>
            <div class="p-3 rounded-lg border {getRoleColor(entry.type)}">
              <div class="whitespace-pre-wrap text-sm">{entry.content}</div>
              {#if entry.metadata?.responseTime}
                <div class="text-xs text-gray-500 mt-2">
                  Response time: {entry.metadata.responseTime}ms
                  {#if entry.metadata.tokenCount}
                    • Tokens: {entry.metadata.tokenCount}
                  {/if}
                </div>
              {/if}
            </div>
          </div>
        {/each}
      {/if}

      {#if isProcessing()}
        <div class="flex items-center gap-2 text-gray-500">
          <div class="animate-spin h-4 w-4 border-2 border-gray-300 border-t-blue-600 rounded-full"></div>
          <span class="text-sm">AI is thinking...</span>
        </div>
      {/if}

      {#if aiError()}
        <div class="p-3 bg-red-50 border border-red-200 rounded-lg">
          <div class="text-sm text-red-800 font-medium">Error</div>
          <div class="text-sm text-red-600 mt-1">{aiError()}</div>
          <Button 
            variant="outline" 
            size="sm" 
            class="mt-2"
            onclick={retryLast}
          >
            Retry
          </Button>
        </div>
      {/if}
    </div>

    <!-- Input Area -->
    <div class="border-t p-4">
      {#if enableContext7}
        <div class="flex items-center gap-2 mb-3">
          <Switch 
            bind:checked={useContext7}
            disabled={isProcessing()}
          />
          <label class="text-sm font-medium">
            Use Context7 Enhancement
          </label>
          {#if context7Analysis() && useContext7}
            <Badge variant="secondary" class="text-xs">
              Confidence: {Math.round((context7Analysis()?.confidence || 0) * 100)}%
            </Badge>
          {/if}
        </div>
      {/if}

      <div class="flex gap-2">
        <Textarea
          bind:this={messageInput}
          bind:value={currentMessage}
          onkeydown={handleKeydown}
          placeholder="Ask about legal documents, cases, or research..."
          disabled={isProcessing()}
          class="flex-1 min-h-[40px] max-h-[120px] resize-none"
        />
        
        <div class="flex flex-col gap-1">
          <Button 
            onclick={sendMessage}
            disabled={!canSend}
            class="px-4"
          >
            {#if isProcessing()}
              <div class="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
            {:else}
              <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            {/if}
          </Button>
          
          {#if isProcessing()}
            <Button 
              variant="outline"
              size="sm"
              onclick={stopGeneration}
            >
              <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" />
              </svg>
            </Button>
          {/if}
        </div>
      </div>

      <!-- Usage Stats -->
      {#if aiUsage().totalQueries > 0}
        <div class="flex items-center gap-4 mt-2 text-xs text-gray-500">
          <span>Queries: {aiUsage().totalQueries}</span>
          <span>Tokens: {aiUsage().totalTokens.toLocaleString()}</span>
          <span>Avg Response: {Math.round(aiUsage().averageResponseTime)}ms</span>
        </div>
      {/if}
    </div>
  </CardContent>
</Card>

<!-- Settings Dialog -->
<Dialog.Root open={showSettingsDialog} onopenchange={(open) => showSettingsDialog = open}>
  <Dialog.Portal>
    <Dialog.Overlay class="fixed inset-0 z-50 bg-black/80" />
    <Dialog.Content class="fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 sm:rounded-lg">
      <Dialog.Title class="text-lg font-semibold">AI Assistant Settings</Dialog.Title>
      
      <div class="space-y-4">
        <!-- Model Selection -->
        <div class="space-y-2">
          <label class="text-sm font-medium">Model</label>
          <Select.Root bind:value={currentModel()}>
            <Select.Trigger>
              <Select.Value placeholder="Select model" />
            </Select.Trigger>
            <Select.Content>
              {#each availableModels as model}
                <Select.Item value={model}>{model}</Select.Item>
              {/each}
            </Select.Content>
          </Select.Root>
        </div>

        <!-- Temperature Slider -->
        <div class="space-y-2">
          <label class="text-sm font-medium">
            Temperature: {currentTemperature()}
          </label>
          <Slider
            bind:value={[currentTemperature()]}
            min={0}
            max={2}
            step={0.1}
            onvaluechange={(values) => aiAssistantManager.setTemperature(values[0])}
          />
          <p class="text-xs text-gray-500">
            Lower values make responses more focused, higher values more creative
          </p>
        </div>

        <!-- Cluster Health -->
        <div class="space-y-2">
          <label class="text-sm font-medium">Cluster Health</label>
          <div class="grid grid-cols-3 gap-2">
            <div class="flex items-center gap-2">
              <div class={`w-2 h-2 rounded-full ${clusterHealth().primary ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span class="text-sm">Primary</span>
            </div>
            <div class="flex items-center gap-2">
              <div class={`w-2 h-2 rounded-full ${clusterHealth().secondary ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span class="text-sm">Secondary</span>
            </div>
            <div class="flex items-center gap-2">
              <div class={`w-2 h-2 rounded-full ${clusterHealth().embeddings ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span class="text-sm">Embeddings</span>
            </div>
          </div>
          <Button 
            variant="outline" 
            size="sm"
            onclick={() => aiAssistantManager.checkClusterHealth()}
          >
            Refresh Health
          </Button>
        </div>
      </div>

      <div class="flex justify-between gap-2">
        <Button 
          variant="destructive"
          onclick={clearConversation}
          disabled={!hasConversation}
        >
          Clear Chat
        </Button>
        <Button onclick={() => showSettingsDialog = false}>
          Close
        </Button>
      </div>
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>

<!-- Export Dialog -->
<Dialog.Root open={showExportDialog} onopenchange={(open) => showExportDialog = open}>
  <Dialog.Portal>
    <Dialog.Overlay class="fixed inset-0 z-50 bg-black/80" />
    <Dialog.Content class="fixed left-[50%] top-[50%] z-50 grid w-full max-w-md translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 sm:rounded-lg">
      <Dialog.Title class="text-lg font-semibold">Export Conversation</Dialog.Title>
      
      <div class="space-y-4">
        <p class="text-sm text-gray-600">
          Export your conversation history as a JSON file for backup or analysis.
        </p>
        
        <div class="space-y-2">
          <div class="text-sm">
            <strong>Messages:</strong> {conversationHistory().length}
          </div>
          <div class="text-sm">
            <strong>Total Queries:</strong> {aiUsage().totalQueries}
          </div>
          <div class="text-sm">
            <strong>Total Tokens:</strong> {aiUsage().totalTokens.toLocaleString()}
          </div>
        </div>
      </div>

      <div class="flex justify-end gap-2">
        <Button 
          variant="outline"
          onclick={() => showExportDialog = false}
        >
          Cancel
        </Button>
        <Button onclick={() => {
          exportConversation();
          showExportDialog = false;
        }}>
          Export JSON
        </Button>
      </div>
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>

<style>
  /* Component-specific styles */
  :global(.animate-spin) {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>