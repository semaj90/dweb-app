<!--
  Enhanced AI Assistant Button Example
  Demonstrates integration with High-Performance AI Assistant
-->

<script lang="ts">
  import AIAssistantButton from '$lib/components/ai/AIAssistantButton.svelte';
  import { onMount } from 'svelte';
  import { page } from '$app/stores';

  let assistantEvents: Array<{ type: string; data: any; timestamp: number }> = [];
  let showEventLog = false;

  function handleAssistantEvent(event: CustomEvent) {
    const { type, detail } = event;
    assistantEvents = [
      { type, data: detail, timestamp: Date.now() },
      ...assistantEvents.slice(0, 19) // Keep last 20 events
    ];
  }

  onMount(() => {
    console.log('🚀 AI Assistant Button Example loaded');
  });
</script>

<svelte:head>
  <title>Enhanced AI Assistant Button Example</title>
  <meta name="description" content="Example usage of the enhanced AI Assistant Button with 3D mode integration" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-8">
  <div class="max-w-4xl mx-auto">
    <!-- Header -->
    <div class="text-center mb-12">
      <h1 class="text-4xl font-bold text-white mb-4">Enhanced AI Assistant Button</h1>
      <p class="text-blue-200 text-lg max-w-2xl mx-auto">
        Experience the next-generation AI assistant with integrated 3D visualization and high-performance features.
      </p>
    </div>

    <!-- Feature Cards -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
      <!-- Standard Mode -->
      <div class="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
        <div class="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center mb-4">
          <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-3.582 8-8 8a8.001 8.001 0 01-7.93-6.93c-.42-.79-.42-1.7 0-2.48A8.001 8.001 0 0113 4c4.418 0 8 3.582 8 8z"></path>
          </svg>
        </div>
        <h3 class="text-xl font-semibold text-white mb-2">Standard Chat</h3>
        <p class="text-blue-200 text-sm">
          Traditional chat interface with AI-powered legal analysis and document processing.
        </p>
        <ul class="mt-4 text-sm text-blue-300 space-y-1">
          <li>• Multi-protocol support (REST/gRPC/QUIC)</li>
          <li>• XState-powered interactions</li>
          <li>• Real-time analysis</li>
        </ul>
      </div>

      <!-- Enhanced 3D Mode -->
      <div class="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-purple-500/30 relative">
        <div class="absolute -top-2 -right-2 bg-purple-600 text-white text-xs px-2 py-1 rounded-full">
          NEW
        </div>
        <div class="w-12 h-12 bg-purple-600 rounded-lg flex items-center justify-center mb-4">
          <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
          </svg>
        </div>
        <h3 class="text-xl font-semibold text-white mb-2">3D Visualization</h3>
        <p class="text-purple-200 text-sm">
          Immersive 3D document visualization with Self-Organizing Maps and GPU acceleration.
        </p>
        <ul class="mt-4 text-sm text-purple-300 space-y-1">
          <li>• Three.js 3D canvas</li>
          <li>• WebGPU acceleration</li>
          <li>• SOM topic modeling</li>
          <li>• Proactive caching</li>
        </ul>
      </div>

      <!-- Integration Features -->
      <div class="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
        <div class="w-12 h-12 bg-green-600 rounded-lg flex items-center justify-center mb-4">
          <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
          </svg>
        </div>
        <h3 class="text-xl font-semibold text-white mb-2">Smart Integration</h3>
        <p class="text-green-200 text-sm">
          Seamless integration with interaction tracking and predictive user interface.
        </p>
        <ul class="mt-4 text-sm text-green-300 space-y-1">
          <li>• User behavior analysis</li>
          <li>• Context-aware responses</li>
          <li>• Performance monitoring</li>
        </ul>
      </div>
    </div>

    <!-- Usage Examples -->
    <div class="bg-white/5 rounded-lg p-8 mb-8">
      <h2 class="text-2xl font-bold text-white mb-6">Usage Examples</h2>
      
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <!-- Basic Usage -->
        <div>
          <h3 class="text-lg font-semibold text-white mb-4">Basic Implementation</h3>
          <pre class="bg-black/30 rounded p-4 text-sm text-green-400 overflow-x-auto"><code>{`<script>
  import AIAssistantButton from '$lib/components/ai/AIAssistantButton.svelte';
</script>

<!-- Basic usage with enhanced mode -->
<AIAssistantButton 
  enableGPUAcceleration={true}
  showDemoAccess={true}
  theme="yorha"
  position="bottom-right"
/>`}</code></pre>
        </div>

        <!-- Advanced Usage -->
        <div>
          <h3 class="text-lg font-semibold text-white mb-4">Advanced Configuration</h3>
          <pre class="bg-black/30 rounded p-4 text-sm text-blue-400 overflow-x-auto"><code>{`<AIAssistantButton 
  enableGPUAcceleration={true}
  enableRealTimeAnalysis={true}
  enableContextSwitching={true}
  enableEnhancedMode={true}
  showDemoAccess={true}
  size="lg"
  theme="yorha"
  position="bottom-right"
  on:open={handleOpen}
  on:close={handleClose}
  on:message={handleMessage}
  on:statusChange={handleStatusChange}
/>`}</code></pre>
        </div>
      </div>
    </div>

    <!-- Live Demo Controls -->
    <div class="bg-white/5 rounded-lg p-6 mb-8">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-xl font-bold text-white">Live Demo</h2>
        <button 
          class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
          onclick={() => showEventLog = !showEventLog}
        >
          {showEventLog ? 'Hide' : 'Show'} Event Log
        </button>
      </div>
      
      <p class="text-blue-200 mb-4">
        The AI Assistant Button is active in the bottom-right corner. Try these actions:
      </p>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div class="space-y-2">
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span class="text-white">Single click to open standard chat</span>
          </div>
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 bg-purple-500 rounded-full"></div>
            <span class="text-white">Double-click for 3D mode</span>
          </div>
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 bg-green-500 rounded-full"></div>
            <span class="text-white">Click 3D icon in chat header</span>
          </div>
        </div>
        <div class="space-y-2">
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 bg-yellow-500 rounded-full"></div>
            <span class="text-white">Hover for enhanced mode indicator</span>
          </div>
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 bg-red-500 rounded-full"></div>
            <span class="text-white">GPU acceleration indicators</span>
          </div>
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 bg-pink-500 rounded-full"></div>
            <span class="text-white">Real-time status monitoring</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Event Log -->
    {#if showEventLog}
      <div class="bg-black/50 rounded-lg p-6" transition:slide={{ duration: 300 }}>
        <h3 class="text-lg font-semibold text-white mb-4">Event Log</h3>
        <div class="max-h-64 overflow-y-auto space-y-2">
          {#each assistantEvents as event (event.timestamp)}
            <div class="flex items-center justify-between text-sm p-2 bg-white/5 rounded">
              <div class="flex items-center space-x-3">
                <span class="text-green-400 font-mono">{event.type}</span>
                <span class="text-gray-300">{JSON.stringify(event.data)}</span>
              </div>
              <span class="text-gray-500 text-xs">
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
            </div>
          {/each}
          
          {#if assistantEvents.length === 0}
            <div class="text-center text-gray-500 py-8">
              No events yet. Interact with the AI Assistant to see events.
            </div>
          {/if}
        </div>
      </div>
    {/if}

    <!-- Technical Details -->
    <div class="mt-12 bg-white/5 rounded-lg p-6">
      <h2 class="text-xl font-bold text-white mb-4">Technical Implementation</h2>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
        <div>
          <h4 class="font-semibold text-blue-400 mb-2">Frontend Stack</h4>
          <ul class="text-gray-300 space-y-1">
            <li>• SvelteKit 2 + Svelte 5</li>
            <li>• Three.js for 3D rendering</li>
            <li>• XState for state management</li>
            <li>• WebGPU for acceleration</li>
            <li>• TypeScript end-to-end</li>
          </ul>
        </div>
        <div>
          <h4 class="font-semibold text-purple-400 mb-2">AI & Processing</h4>
          <ul class="text-gray-300 space-y-1">
            <li>• Self-Organizing Maps</li>
            <li>• GPU-accelerated embeddings</li>
            <li>• Real-time interaction tracking</li>
            <li>• Proactive caching system</li>
            <li>• Multi-dimensional encoding</li>
          </ul>
        </div>
        <div>
          <h4 class="font-semibold text-green-400 mb-2">Backend Integration</h4>
          <ul class="text-gray-300 space-y-1">
            <li>• Go microservices</li>
            <li>• PostgreSQL + pgvector</li>
            <li>• Redis caching</li>
            <li>• Multi-protocol APIs</li>
            <li>• Real-time WebSockets</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Enhanced AI Assistant Button with all features enabled -->
<AIAssistantButton 
  enableGPUAcceleration={true}
  enableRealTimeAnalysis={true}
  enableContextSwitching={true}
  enableEnhancedMode={true}
  showDemoAccess={true}
  size="md"
  theme="yorha"
  position="bottom-right"
  on:open={handleAssistantEvent}
  on:close={handleAssistantEvent}
  on:message={handleAssistantEvent}
  on:statusChange={handleAssistantEvent}
/>

<style>
  :global(body) {
    margin: 0;
    font-family: 'Inter', system-ui, sans-serif;
  }
</style>