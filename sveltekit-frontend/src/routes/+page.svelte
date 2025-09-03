<!-- YoRHa Detective Interface - Command Center Dashboard -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import DetectiveLayout from '$lib/components/DetectiveLayout.svelte';
  import StatsCard from '$lib/components/ui/StatsCard.svelte';
  import CaseItem from '$lib/components/ui/CaseItem.svelte';
  import SystemStatusCard from '$lib/components/ui/SystemStatusCard.svelte';
  import QuickActionButton from '$lib/components/ui/QuickActionButton.svelte';
  import AIAssistantModal from '$lib/components/ai/AIAssistantModal.svelte';

  // YoRHa Detective Interface State
  let currentTime = writable('');
  let systemStatus = writable('Operational');
  let showAIAssistant = false;

  // Dashboard Statistics
  let stats = {
    activeCases: 3,
    evidenceItems: 27,
    personsOfInterest: 8,
    recentActivity: 12
  };

  // Active Cases Data
  let activeCases = [
    {
      id: 'CORP-001',
      title: 'CORPORATE ESPIONAGE INVESTIGATION',
      items: 8,
      timeAgo: '2 hours ago',
      priority: 'high',
      status: 'active'
    },
    {
      id: 'MISS-002', 
      title: 'MISSING PERSON: DR. SARAH CHEN',
      items: 15,
      timeAgo: '4 hours ago',
      priority: 'high',
      status: 'active'
    },
    {
      id: 'FRAUD-003',
      title: 'FINANCIAL FRAUD ANALYSIS',
      items: 6,
      timeAgo: '1 day ago',
      priority: 'medium',
      status: 'pending'
    }
  ];

  // System Status Alerts
  let systemAlerts = [
    {
      type: 'success',
      message: 'System backup completed successfully',
      time: '10 minutes ago'
    },
    {
      type: 'warning',
      message: 'Evidence analysis queue processing slowly',
      time: '1 hour ago'
    },
    {
      type: 'success',
      message: 'New facial recognition matches found',
      time: '2 hours ago'
    }
  ];

  // Update current time
  onMount(() => {
    const updateTime = () => {
      const now = new Date();
      currentTime.set(`${now.toLocaleDateString()} ${now.toLocaleTimeString()}`);
    };
    
    updateTime();
    const interval = setInterval(updateTime, 1000);
    
    return () => clearInterval(interval);
  });

  // Quick Actions
  const quickActions = [
    { label: 'EVIDENCE BOARD', icon: 'folder', action: () => console.log('Evidence Board') },
    { label: 'TIMELINE ANALYSIS', icon: 'clock', action: () => console.log('Timeline') },
    { label: 'TERMINAL ACCESS', icon: 'terminal', action: () => console.log('Terminal') }
  ];

  function openAIAssistant() {
    showAIAssistant = true;
  }

  function closeAIAssistant() {
    showAIAssistant = false;
  }
</script>

<svelte:head>
  <title>AI Legal Assistant - Enhanced RAG Chat</title>
  <meta name="description" content="AI-powered legal assistant with advanced reasoning and retrieval capabilities" />
</svelte:head>

<div class="flex flex-col h-screen bg-gray-50">
  <!-- Header -->
  <header class="bg-white border-b border-gray-200 p-4 shadow-sm">
    <div class="max-w-6xl mx-auto flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
          <span class="text-white font-bold text-lg">🤖</span>
        </div>
        <div>
          <h1 class="text-xl font-semibold text-gray-900">AI Legal Assistant</h1>
          <p class="text-sm text-gray-600">Enhanced RAG with Local LLM</p>
        </div>
      </div>

      <div class="flex items-center space-x-4">
        <!-- Connection Status -->
        <div class="flex items-center space-x-2">
          <div class="w-3 h-3 rounded-full {$isAIConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse"></div>
          <span class="text-sm {statusColor} font-medium">{connectionStatus}</span>
        </div>

        <!-- System Health -->
        <div class="text-sm text-gray-600">
          Health: <span class="font-medium capitalize">{$systemHealth}</span>
        </div>

        <!-- Actions -->
        <div class="flex space-x-2">
          <button class="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50" on:click={clearChat}>
            Clear Chat
          </button>
          {#if !$isAIConnected}
            <button class="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700" on:click={retryConnection}>
              Reconnect
            </button>
          {/if}
        </div>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <div class="flex-1 flex max-w-6xl mx-auto w-full">
    <!-- Chat Area -->
    <div class="flex-1 flex flex-col">
      <!-- Messages Container -->
      <div
        bind:this={chatContainer}
        class="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {#if $currentConversation.length === 0}
          <!-- Welcome Screen -->
          <div class="flex flex-col items-center justify-center h-full text-center">
            <div class="mb-8">
              <div class="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mb-4 mx-auto">
                <span class="text-3xl">⚖️</span>
              </div>
              <h2 class="text-2xl font-semibold text-gray-900 mb-2">Welcome to AI Legal Assistant</h2>
              <p class="text-gray-600 max-w-md">
                Ask me about legal concepts, case analysis, evidence review, or any legal questions you have.
                I'm powered by local LLMs with enhanced retrieval capabilities.
              </p>
            </div>

            <!-- Sample Queries -->
            {#if sampleQueries.length > 0}
              <div class="w-full max-w-2xl">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Try asking:</h3>
                <div class="grid gap-2">
                  {#each sampleQueries as query}
                    <button
                      on:click={() => useSampleQuery(query)}
                      class="text-left p-3 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-blue-300 transition-colors"
                    >
                      <span class="text-gray-700">{query}</span>
                    </button>
                  {/each}
                </div>
              </div>
            {/if}
          </div>
        {:else}
          <!-- Conversation Messages -->
          {#each $currentConversation as message, index (message.id)}
            <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
              <div class={getMessageClasses(message.role)}>
                <!-- Message Header -->
                <div class="flex items-center justify-between mb-2">
                  <div class="flex items-center space-x-2">
                    <span class="text-xs font-medium">
                      {message.role === 'user' ? '👤 You' : '🤖 AI Assistant'}
                    </span>
                    <span class="text-xs opacity-70">
                      {formatTimestamp(message.timestamp)}
                    </span>
                  </div>

                  {#if message.metadata?.model}
                    <span class="text-xs opacity-70">
                      {message.metadata.model}
                    </span>
                  {/if}
                </div>

                <!-- Message Content -->
                <div class="prose prose-sm max-w-none">
                  <p class="whitespace-pre-wrap">{message.content}</p>
                </div>

                <!-- Message Metadata -->
                {#if message.role === 'assistant' && message.metadata}
                  <div class="mt-3 pt-3 border-t border-gray-200 text-xs opacity-70">
                    <div class="flex flex-wrap gap-2">
                      {#if message.metadata.executionTime}
                        <span>⏱️ {message.metadata.executionTime}ms</span>
                      {/if}
                      {#if message.metadata.confidence}
                        <span>🎯 {Math.round(message.metadata.confidence * 100)}% confidence</span>
                      {/if}
                      {#if message.metadata.tokensUsed}
                        <span>📊 {message.metadata.tokensUsed} tokens</span>
                      {/if}
                    </div>
                  </div>
                {/if}

                <!-- Sources (if available) -->
                {#if message.sources && message.sources.length > 0}
                  <div class="mt-3 pt-3 border-t border-gray-200">
                    <p class="text-xs font-medium mb-2">📚 Sources:</p>
                    <div class="space-y-1">
                      {#each message.sources.slice(0, 3) as source}
                        <div class="text-xs p-2 bg-gray-50 rounded">
                          <span class="font-medium">{source.type}</span>
                          <span class="opacity-70 ml-2">Score: {(source.score * 100).toFixed(0)}%</span>
                        </div>
                      {/each}
                    </div>
                  </div>
                {/if}
              </div>
            </div>
          {/each}

          <!-- Loading Indicator -->
          {#if $chatState.isLoading}
            <div class="flex justify-start">
              <div class="max-w-[80%] p-4 bg-gray-100 rounded-lg">
                <div class="flex items-center space-x-2">
                  <div class="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
                  <span class="text-gray-600">AI is thinking...</span>
                </div>
              </div>
            </div>
          {/if}
        {/if}
      </div>

      <!-- Error Display -->
      {#if $chatState.error}
        <div class="mx-4 mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-2">
              <span class="text-red-600">⚠️</span>
              <span class="text-red-800 text-sm">{$chatState.error}</span>
            </div>
            <button
              on:click={() => chatState.update(s => ({ ...s, error: null }))}
              class="text-red-600 hover:text-red-800"
            >
              ✕
            </button>
          </div>
        </div>
      {/if}

      <!-- Input Area -->
      <div class="border-t border-gray-200 bg-white p-4">
        <div class="flex space-x-3">
          <div class="flex-1">
            <textarea
              bind:this={messageInput}
              bind:value={$chatState.message}
              onkeydown={handleKeyDown}
              placeholder="Ask me about legal matters, case analysis, evidence review..."
              class="w-full p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows="2"
              disabled={$chatState.isLoading || !$isAIConnected}
            ></textarea>
          </div>

          <div class="flex flex-col space-y-2">
            <button
              on:click={sendMessage}
              disabled={!canSend}
              class="px-6 py-3 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {#if $chatState.isLoading}
                <div class="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
              {:else}
                Send
              {/if}
            </button>

            {#if $chatState.message.trim().length > 0}
              <div class="text-xs text-gray-500 text-center">
                {$chatState.message.trim().length} chars
              </div>
            {/if}
          </div>
        </div>

        <!-- Quick Actions -->
        <div class="mt-2 flex justify-between items-center text-xs text-gray-500">
          <span>Press Shift+Enter for new line, Enter to send</span>
          <div class="flex space-x-4">
            <a href="/test" class="hover:text-blue-600">Test Page</a>
            <a href="/api/ai/health" target="_blank" class="hover:text-blue-600">API Health</a>
          </div>
        </div>
      </div>
    </div>

    <!-- Sidebar (Status & Info) -->
    <div class="w-80 border-l border-gray-200 bg-white p-4 overflow-y-auto">
      <h3 class="font-semibold text-gray-900 mb-4">System Status</h3>

      <!-- Connection Info -->
      <div class="space-y-3 mb-6">
        <div class="flex justify-between">
          <span class="text-sm text-gray-600">Connection:</span>
          <span class="text-sm font-medium {statusColor}">{connectionStatus}</span>
        </div>
        <div class="flex justify-between">
          <span class="text-sm text-gray-600">Health:</span>
          <span class="text-sm font-medium capitalize">{$systemHealth}</span>
        </div>
        <div class="flex justify-between">
          <span class="text-sm text-gray-600">Messages:</span>
          <span class="text-sm font-medium">{$currentConversation.length}</span>
        </div>
        {#if $chatState.lastActivity}
          <div class="flex justify-between">
            <span class="text-sm text-gray-600">Last Activity:</span>
            <span class="text-sm font-medium">{formatTimestamp($chatState.lastActivity)}</span>
          </div>
        {/if}
      </div>

      <!-- Features -->
      <div class="mb-6">
        <h4 class="font-medium text-gray-900 mb-3">Features</h4>
        <div class="space-y-2 text-sm">
          <div class="flex items-center space-x-2">
            <span class="text-green-500">✓</span>
            <span>Local LLM Integration</span>
          </div>
          <div class="flex items-center space-x-2">
            <span class="text-green-500">✓</span>
            <span>Enhanced RAG Pipeline</span>
          </div>
          <div class="flex items-center space-x-2">
            <span class="text-green-500">✓</span>
            <span>Real-time Responses</span>
          </div>
          <div class="flex items-center space-x-2">
            <span class="text-green-500">✓</span>
            <span>Error Recovery</span>
          </div>
          <div class="flex items-center space-x-2">
            <span class="text-green-500">✓</span>
            <span>Source Attribution</span>
          </div>
        </div>
      </div>

      <!-- Quick Actions -->
      <div>
        <h4 class="font-medium text-gray-900 mb-3">Quick Actions</h4>
        <div class="space-y-2">
          <button class="w-full px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50" on:click={clearChat}>
            🗑️ Clear Conversation
          </button>

          {#if !$isAIConnected}
            <button class="w-full px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700" on:click={retryConnection}>
              🔄 Retry Connection
            </button>
          {/if}

          <a href="/test" class="block">
            <button class="w-full px-3 py-1 text-sm text-gray-700 rounded hover:bg-gray-100">
              🧪 Test Interface
            </button>
          </a>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  /* Custom scrollbar */
  :global(.overflow-y-auto::-webkit-scrollbar) {
    width: 6px;
  }

  :global(.overflow-y-auto::-webkit-scrollbar-track) {
    background: #f1f1f1;
  }

  :global(.overflow-y-auto::-webkit-scrollbar-thumb) {
    background: #c1c1c1;
    border-radius: 3px;
  }

  :global(.overflow-y-auto::-webkit-scrollbar-thumb:hover) {
    background: #a8a8a8;
  }
</style>
