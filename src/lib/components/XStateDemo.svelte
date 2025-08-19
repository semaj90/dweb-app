<!-- XState Management Demo Component -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { fade, fly, scale } from 'svelte/transition';
  import { 
    Activity, 
    MessageSquare, 
    Search, 
    Upload, 
    User, 
    Settings,
    Play,
    Pause,
    RotateCcw,
    BarChart3,
    CheckCircle,
    AlertTriangle,
    Clock,
    Zap,
    Database,
    Brain
  } from 'lucide-svelte';

  // Import XState stores and helpers
  import { 
    globalAppState, 
    performanceMetrics,
    xstateManager,
    xstateHelpers,
    appState,
    chatState,
    searchState,
    documentUploadState
  } from '../stores/xstate-store-manager';

  // Component state
  let selectedDemo = $state<'app' | 'chat' | 'search' | 'upload' | 'performance'>('app');
  let isRunning = $state(true);
  let testMessage = $state('What are the key elements of a valid contract?');
  let testQuery = $state('contract law precedents');
  let testFiles = $state<File[]>([]);

  // Reactive state access
  let globalState = $state($globalAppState);
  let metrics = $state($performanceMetrics);

  // Subscribe to store updates
  globalAppState.subscribe(state => globalState = state);
  performanceMetrics.subscribe(m => metrics = m);

  onMount(async () => {
    console.log('🎭 XState Demo component mounted');
    
    // Ensure XState manager is initialized
    if (!xstateManager.getService('app')) {
      await xstateManager.initialize();
    }
  });

  // Demo functions
  function runAppDemo() {
    // Simulate user login
    xstateHelpers.login({
      email: 'demo@legalai.com',
      password: 'demo123'
    });

    // Add some notifications
    setTimeout(() => {
      xstateHelpers.addNotification({
        type: 'success',
        title: 'Welcome!',
        message: 'XState demo is running successfully'
      });
    }, 1000);
  }

  function runChatDemo() {
    // Create a new chat session
    xstateHelpers.createChatSession(
      'Legal AI Demo Chat',
      { demoMode: true, legalContext: 'contract-law' }
    );

    // Send a test message
    setTimeout(() => {
      xstateHelpers.sendChatMessage(testMessage);
    }, 1000);
  }

  function runSearchDemo() {
    // Perform a search
    xstateHelpers.performSearch(testQuery);

    // Apply some filters after search
    setTimeout(() => {
      xstateHelpers.applySearchFilters({
        documentTypes: ['contract', 'case-law'],
        dateRange: {
          start: Date.now() - (365 * 24 * 60 * 60 * 1000), // Last year
          end: Date.now()
        }
      });
    }, 2000);
  }

  function runUploadDemo() {
    // Create demo files
    const demoFiles = [
      new File(['Demo contract content...'], 'sample-contract.txt', { type: 'text/plain' }),
      new File(['Demo case law content...'], 'case-precedent.txt', { type: 'text/plain' })
    ];

    xstateHelpers.uploadFiles(demoFiles);
  }

  function runCombinedDemo() {
    // Demonstrate cross-machine communication
    xstateHelpers.searchAndChat('employment contract termination clauses');
  }

  function toggleSystem() {
    if (isRunning) {
      xstateManager.stopAll();
      isRunning = false;
    } else {
      xstateManager.restart();
      isRunning = true;
    }
  }

  function resetSystem() {
    xstateManager.restart();
  }

  function getStateIcon(machineName: string) {
    const state = globalState?.[machineName];
    if (!state) return AlertTriangle;
    
    if (typeof state === 'object' && state.context) {
      if (state.context.isAuthenticated || 
          state.context.isProcessing || 
          state.context.isSearching || 
          state.context.currentFile) {
        return Activity;
      }
    }
    
    return CheckCircle;
  }

  function getStateColor(machineName: string) {
    const state = globalState?.[machineName];
    if (!state) return 'text-red-500';
    
    if (typeof state === 'object' && state.context) {
      if (state.context.isAuthenticated || 
          state.context.isProcessing || 
          state.context.isSearching || 
          state.context.currentFile) {
        return 'text-blue-500';
      }
    }
    
    return 'text-green-500';
  }

  function formatStateValue(value: any): string {
    if (typeof value === 'object') {
      return JSON.stringify(value, null, 2);
    }
    return String(value);
  }
</script>

<div class="xstate-demo bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 min-h-screen p-6">
  <!-- Header -->
  <div class="max-w-7xl mx-auto mb-8">
    <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-4">
          <div class="w-12 h-12 bg-gradient-to-br from-purple-500 to-blue-600 rounded-lg flex items-center justify-center">
            <Brain class="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 class="text-3xl font-bold text-slate-900 dark:text-slate-100">XState Management Demo</h1>
            <p class="text-slate-600 dark:text-slate-400">Reactive State Machines for Legal AI Platform</p>
          </div>
        </div>
        
        <!-- System Controls -->
        <div class="flex items-center space-x-3">
          <button
            onclick={toggleSystem}
            class="px-4 py-2 {isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'} text-white rounded-lg transition-colors flex items-center space-x-2"
          >
            {#if isRunning}
              <Pause class="w-4 h-4" />
              <span>Stop System</span>
            {:else}
              <Play class="w-4 h-4" />
              <span>Start System</span>
            {/if}
          </button>
          
          <button
            onclick={resetSystem}
            class="px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition-colors flex items-center space-x-2"
          >
            <RotateCcw class="w-4 h-4" />
            <span>Reset</span>
          </button>
        </div>
      </div>
      
      <!-- System Status -->
      <div class="mt-6 grid grid-cols-4 gap-4">
        {#each ['app', 'chat', 'search', 'upload'] as machine}
          {@const StateIcon = getStateIcon(machine)}
          <div class="flex items-center space-x-3 p-3 bg-slate-50 dark:bg-slate-700 rounded-lg">
            <StateIcon class="w-5 h-5 {getStateColor(machine)}" />
            <div>
              <p class="font-medium text-slate-900 dark:text-slate-100 capitalize">{machine}</p>
              <p class="text-sm text-slate-600 dark:text-slate-400">
                {globalState?.[machine]?.value || 'idle'}
              </p>
            </div>
          </div>
        {/each}
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
    <!-- Demo Controls -->
    <div class="lg:col-span-1 space-y-6">
      <!-- Demo Selection -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Demo Controls</h3>
        
        <div class="space-y-3">
          {#each [
            { id: 'app', label: 'App State', icon: User, desc: 'Authentication & notifications' },
            { id: 'chat', label: 'Chat System', icon: MessageSquare, desc: 'AI conversations' },
            { id: 'search', label: 'Search Engine', icon: Search, desc: 'Document search' },
            { id: 'upload', label: 'File Upload', icon: Upload, desc: 'Document processing' },
            { id: 'performance', label: 'Performance', icon: BarChart3, desc: 'System metrics' }
          ] as demo}
            <button
              onclick={() => selectedDemo = demo.id}
              class="w-full flex items-center space-x-3 p-3 rounded-lg transition-colors {
                selectedDemo === demo.id
                  ? 'bg-blue-100 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
                  : 'hover:bg-slate-100 dark:hover:bg-slate-700'
              }"
            >
              <svelte:component this={demo.icon} class="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <div class="text-left">
                <p class="font-medium text-slate-900 dark:text-slate-100">{demo.label}</p>
                <p class="text-sm text-slate-600 dark:text-slate-400">{demo.desc}</p>
              </div>
            </button>
          {/each}
        </div>
      </div>

      <!-- Quick Actions -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Quick Actions</h3>
        
        <div class="space-y-3">
          <button
            onclick={runAppDemo}
            class="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            Run App Demo
          </button>
          
          <button
            onclick={runChatDemo}
            class="w-full px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
          >
            Run Chat Demo
          </button>
          
          <button
            onclick={runSearchDemo}
            class="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
          >
            Run Search Demo
          </button>
          
          <button
            onclick={runUploadDemo}
            class="w-full px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors"
          >
            Run Upload Demo
          </button>
          
          <button
            onclick={runCombinedDemo}
            class="w-full px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-lg transition-colors"
          >
            Combined Demo
          </button>
        </div>
      </div>

      <!-- Test Inputs -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Test Inputs</h3>
        
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Chat Message
            </label>
            <textarea
              bind:value={testMessage}
              class="w-full rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
              rows="3"
            ></textarea>
          </div>
          
          <div>
            <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Search Query
            </label>
            <input
              bind:value={testQuery}
              class="w-full rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Main Demo Area -->
    <div class="lg:col-span-2 space-y-6">
      <!-- Selected Demo Content -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4 capitalize">
          {selectedDemo} State Machine
        </h3>
        
        {#if selectedDemo === 'app'}
          <div class="space-y-4">
            <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
              <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-2">Authentication Status</h4>
              <p class="text-sm text-slate-600 dark:text-slate-400">
                Status: {globalState?.isAuthenticated ? 'Authenticated' : 'Not Authenticated'}
              </p>
              {#if globalState?.currentUser}
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  User: {globalState.currentUser.email}
                </p>
              {/if}
            </div>
            
            {#if globalState?.app?.notifications?.length > 0}
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-2">Notifications</h4>
                <div class="space-y-2">
                  {#each globalState.app.notifications.slice(0, 3) as notification}
                    <div class="p-2 bg-white dark:bg-slate-600 rounded border-l-4 border-{notification.type === 'success' ? 'green' : 'blue'}-500">
                      <p class="text-sm font-medium text-slate-900 dark:text-slate-100">{notification.title}</p>
                      <p class="text-xs text-slate-600 dark:text-slate-400">{notification.message}</p>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
          </div>
        {:else if selectedDemo === 'chat'}
          <div class="space-y-4">
            {#if globalState?.chat?.currentSession}
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-2">Current Session</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  Title: {globalState.chat.currentSession.title}
                </p>
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  Messages: {globalState.chat.currentSession.messages?.length || 0}
                </p>
              </div>
              
              {#if globalState.chat.currentSession.messages?.length > 0}
                <div class="space-y-2 max-h-64 overflow-y-auto">
                  {#each globalState.chat.currentSession.messages as message}
                    <div class="p-3 {message.role === 'user' ? 'bg-blue-50 dark:bg-blue-900/20' : 'bg-gray-50 dark:bg-gray-700'} rounded-lg">
                      <p class="text-sm font-medium text-slate-900 dark:text-slate-100 capitalize">{message.role}</p>
                      <p class="text-sm text-slate-600 dark:text-slate-400">{message.content}</p>
                      <p class="text-xs text-slate-500 dark:text-slate-500 mt-1">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  {/each}
                </div>
              {/if}
            {:else}
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg text-center">
                <p class="text-slate-600 dark:text-slate-400">No active chat session</p>
              </div>
            {/if}
          </div>
        {:else if selectedDemo === 'search'}
          <div class="space-y-4">
            {#if globalState?.search?.query}
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-2">Current Search</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  Query: "{globalState.search.query}"
                </p>
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  Results: {globalState.search.totalResults}
                </p>
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  Type: {globalState.search.searchType}
                </p>
              </div>
              
              {#if globalState.search.results?.length > 0}
                <div class="space-y-2 max-h-64 overflow-y-auto">
                  {#each globalState.search.results.slice(0, 5) as result}
                    <div class="p-3 bg-slate-50 dark:bg-slate-700 rounded-lg">
                      <p class="text-sm font-medium text-slate-900 dark:text-slate-100">{result.title}</p>
                      <p class="text-xs text-slate-600 dark:text-slate-400">Score: {result.score?.toFixed(2)}</p>
                      <p class="text-xs text-slate-600 dark:text-slate-400">{result.snippet}</p>
                    </div>
                  {/each}
                </div>
              {/if}
            {:else}
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg text-center">
                <p class="text-slate-600 dark:text-slate-400">No active search</p>
              </div>
            {/if}
          </div>
        {:else if selectedDemo === 'upload'}
          <div class="space-y-4">
            {#if globalState?.upload?.currentFile}
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-2">Current Upload</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  File: {globalState.upload.currentFile.name}
                </p>
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  Stage: {globalState.upload.stages[globalState.upload.currentStageIndex]?.name}
                </p>
                <div class="mt-2 w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                  <div 
                    class="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style="width: {globalState.upload.stages[globalState.upload.currentStageIndex]?.progress || 0}%"
                  ></div>
                </div>
              </div>
            {:else if globalState?.upload?.results?.length > 0}
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-2">Upload Results</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  Completed: {globalState.upload.results.filter(r => r.success).length}
                </p>
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  Failed: {globalState.upload.results.filter(r => !r.success).length}
                </p>
              </div>
            {:else}
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg text-center">
                <p class="text-slate-600 dark:text-slate-400">No active uploads</p>
              </div>
            {/if}
          </div>
        {:else if selectedDemo === 'performance'}
          <div class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
              {#each Object.entries(metrics || {}).filter(([key]) => key !== 'timestamp') as [machine, perf]}
                <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                  <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-2 capitalize">{machine}</h4>
                  {#if typeof perf === 'object' && perf !== null}
                    {#each Object.entries(perf).slice(0, 3) as [key, value]}
                      <p class="text-sm text-slate-600 dark:text-slate-400">
                        {key}: {typeof value === 'number' ? value.toFixed(2) : String(value)}
                      </p>
                    {/each}
                  {/if}
                </div>
              {/each}
            </div>
          </div>
        {/if}
      </div>

      <!-- Machine States -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Machine States</h3>
        
        <div class="grid grid-cols-2 gap-4">
          {#each Object.entries(globalState || {}).filter(([key]) => ['app', 'chat', 'search', 'upload'].includes(key)) as [machine, state]}
            <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
              <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-2 capitalize">{machine}</h4>
              {#if state && typeof state === 'object' && 'value' in state}
                <p class="text-sm text-slate-600 dark:text-slate-400 font-mono">
                  {formatStateValue(state.value)}
                </p>
              {:else}
                <p class="text-sm text-slate-600 dark:text-slate-400">No state data</p>
              {/if}
            </div>
          {/each}
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .xstate-demo {
    font-family: system-ui, -apple-system, sans-serif;
  }
</style>