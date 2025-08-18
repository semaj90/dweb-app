<!-- LangChain.js with Event Listeners Demo Page -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { fade, fly } from 'svelte/transition';
  import { Brain, Cpu, Zap, MessageSquare, Tool, Activity, BarChart3 } from 'lucide-svelte';
  
  // Import our complete LangChain implementation
  import LangChainDemo from '$lib/components/LangChainDemo.svelte';
  import { 
    getLangChainService, 
    createLangChainService,
    langchainServiceStatus,
    langchainSessions,
    langchainMetrics
  } from '$lib/langchain/langchain-service';

  // Component state
  let serviceReady = $state(false);
  let demoVisible = $state(false);
  let initializationLogs = $state<string[]>([]);

  // Reactive store access
  let serviceStatus = $state($langchainServiceStatus);
  let sessions = $state($langchainSessions);
  let metrics = $state($langchainMetrics);

  // Subscribe to store updates
  langchainServiceStatus.subscribe(status => serviceStatus = status);
  langchainSessions.subscribe(s => sessions = s);
  langchainMetrics.subscribe(m => metrics = m);

  onMount(async () => {
    console.log('🚀 LangChain AI Demo Page mounted');
    
    addLog('Initializing LangChain.js with Event Listeners...');
    
    try {
      // Get or create LangChain service
      let service = getLangChainService();
      
      if (!service) {
        addLog('Creating new LangChain service instance...');
        service = createLangChainService({
          llmProvider: 'ollama',
          model: 'gemma3:legal-latest',
          enableTools: true,
          enableMemory: true,
          enableStreaming: true,
          enableEventLogging: true
        });

        addLog('Setting up event listeners...');
        setupServiceEventListeners(service);
        
        addLog('Initializing service...');
        const success = await service.initialize();
        
        if (success) {
          addLog('✓ LangChain service initialized successfully');
          serviceReady = true;
          
          // Show demo after brief delay
          setTimeout(() => {
            demoVisible = true;
          }, 1000);
        } else {
          addLog('❌ Service initialization failed');
        }
      } else {
        addLog('✓ Using existing LangChain service');
        serviceReady = service.isReady;
        if (serviceReady) {
          setTimeout(() => {
            demoVisible = true;
          }, 500);
        }
      }
    } catch (error) {
      addLog(`❌ Error: ${error.message}`);
      console.error('LangChain initialization error:', error);
    }
  });

  function setupServiceEventListeners(service: any) {
    // Service-level events
    service.on('service:initialized', (data: any) => {
      addLog('🎉 Service initialized with config');
      console.log('Service initialized:', data);
    });

    service.on('service:error', (data: any) => {
      addLog(`❌ Service error: ${data.error}`);
    });

    // Session events
    service.on('session:created', (data: any) => {
      addLog(`📝 New session created: ${data.session.title}`);
    });

    service.on('session:deleted', (data: any) => {
      addLog(`🗑️ Session deleted: ${data.sessionId}`);
    });

    // Message events
    service.on('message:sending', (data: any) => {
      addLog(`📤 Sending message to session ${data.sessionId}`);
    });

    service.on('message:received', (data: any) => {
      addLog(`📥 Message processed for session ${data.sessionId}`);
    });

    service.on('message:error', (data: any) => {
      addLog(`❌ Message error: ${data.error}`);
    });

    // Streaming events
    service.on('streaming:started', (data: any) => {
      addLog(`🌊 Streaming started for session ${data.sessionId}`);
    });

    service.on('streaming:chunk', (data: any) => {
      // Don't log every chunk, just track them
    });

    service.on('streaming:completed', (data: any) => {
      addLog(`✓ Streaming completed for session ${data.sessionId}`);
    });

    service.on('streaming:cancelled', (data: any) => {
      addLog(`⏹️ Streaming cancelled for session ${data.sessionId}`);
    });

    // Tool events
    service.on('tool:executing', (data: any) => {
      addLog(`🔧 Executing tool: ${data.toolName}`);
    });

    service.on('tool:executed', (data: any) => {
      addLog(`✓ Tool executed: ${data.toolName} (${data.executionTime}ms)`);
    });

    service.on('tool:error', (data: any) => {
      addLog(`❌ Tool error: ${data.toolName} - ${data.error}`);
    });

    // Execution events
    service.on('execution:completed', (data: any) => {
      addLog(`✅ Execution completed: ${data.execution.type}`);
    });

    service.on('execution:failed', (data: any) => {
      addLog(`❌ Execution failed: ${data.error}`);
    });

    // Memory events
    service.on('memory:saved', (data: any) => {
      addLog(`💾 Memory saved: ${data.size} bytes`);
    });

    service.on('memory:loaded', (data: any) => {
      addLog(`📂 Memory loaded: ${data.size} bytes`);
    });

    service.on('memory:cleared', (data: any) => {
      addLog(`🧹 Memory cleared for session ${data.sessionId}`);
    });
  }

  function addLog(message: string) {
    initializationLogs = [...initializationLogs, `${new Date().toLocaleTimeString()} - ${message}`];
    
    // Keep only last 20 logs
    if (initializationLogs.length > 20) {
      initializationLogs = initializationLogs.slice(-20);
    }
  }

  // Demo features showcase
  const features = [
    {
      icon: Brain,
      title: 'Event-Driven Architecture',
      description: 'Real-time event listeners for all LangChain operations with comprehensive logging and monitoring'
    },
    {
      icon: MessageSquare,
      title: 'Conversational AI',
      description: 'Multi-session chat with memory, context preservation, and legal specialization'
    },
    {
      icon: Tool,
      title: 'Legal AI Tools',
      description: 'Specialized tools for legal search, case analysis, document drafting, and citation checking'
    },
    {
      icon: Activity,
      title: 'Streaming Responses',
      description: 'Real-time streaming with cancellation support and chunk-by-chunk processing'
    },
    {
      icon: Cpu,
      title: 'Multi-Protocol Integration',
      description: 'Seamless integration with enhanced RAG pipeline and multi-protocol routing'
    },
    {
      icon: BarChart3,
      title: 'Performance Metrics',
      description: 'Real-time monitoring of executions, token usage, latency, and tool utilization'
    }
  ];
</script>

<svelte:head>
  <title>LangChain.js with Event Listeners - Legal AI Demo</title>
  <meta name="description" content="Complete LangChain.js implementation with event-driven architecture, legal AI tools, and real-time streaming" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800">
  <!-- Hero Section -->
  <div class="relative overflow-hidden bg-white dark:bg-slate-900 shadow-xl">
    <div class="absolute inset-0 bg-gradient-to-r from-purple-600/10 to-blue-600/10"></div>
    <div class="relative max-w-7xl mx-auto px-6 py-16">
      <div class="text-center">
        <div class="flex justify-center mb-6">
          <div class="w-20 h-20 bg-gradient-to-br from-purple-500 to-blue-600 rounded-2xl flex items-center justify-center transform rotate-3">
            <Brain class="w-10 h-10 text-white" />
          </div>
        </div>
        
        <h1 class="text-4xl md:text-6xl font-bold text-slate-900 dark:text-slate-100 mb-6">
          LangChain.js
          <span class="bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
            Event Architecture
          </span>
        </h1>
        
        <p class="text-xl text-slate-600 dark:text-slate-400 mb-8 max-w-3xl mx-auto">
          Complete implementation with event-driven architecture, real-time streaming, legal AI tools, 
          and comprehensive monitoring for production-ready legal AI applications.
        </p>

        <!-- Service Status -->
        <div class="flex items-center justify-center space-x-4 mb-8">
          <div class="flex items-center space-x-2 px-4 py-2 bg-slate-100 dark:bg-slate-800 rounded-full">
            <div class="w-3 h-3 rounded-full {serviceReady ? 'bg-green-500' : 'bg-yellow-500'}"></div>
            <span class="text-sm font-medium text-slate-700 dark:text-slate-300">
              {serviceReady ? 'LangChain Ready' : 'Initializing...'}
            </span>
          </div>
          
          {#if serviceStatus.isReady}
            <div class="flex items-center space-x-4 text-sm text-slate-600 dark:text-slate-400">
              <span>{serviceStatus.sessions} Sessions</span>
              <span>{serviceStatus.activeStreams} Streaming</span>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>

  <!-- Features Grid -->
  <div class="max-w-7xl mx-auto px-6 py-16">
    <div class="text-center mb-12">
      <h2 class="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
        Complete LangChain.js Implementation
      </h2>
      <p class="text-lg text-slate-600 dark:text-slate-400">
        Production-ready features with event-driven architecture and legal AI specialization
      </p>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
      {#each features as feature, index}
        <div 
          class="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-sm border border-slate-200 dark:border-slate-700 hover:shadow-md transition-all duration-300"
          in:fly={{ y: 20, delay: index * 100 }}
        >
          <div class="w-12 h-12 bg-gradient-to-br from-purple-500 to-blue-600 rounded-lg flex items-center justify-center mb-4">
            <svelte:component this={feature.icon} class="w-6 h-6 text-white" />
          </div>
          
          <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-3">
            {feature.title}
          </h3>
          
          <p class="text-slate-600 dark:text-slate-400">
            {feature.description}
          </p>
        </div>
      {/each}
    </div>

    <!-- Initialization Logs -->
    {#if initializationLogs.length > 0}
      <div class="bg-slate-900 rounded-xl p-6 mb-8" in:fade>
        <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
          <Activity class="w-5 h-5 mr-2" />
          Event Listeners Activity Log
        </h3>
        
        <div class="bg-black rounded-lg p-4 max-h-64 overflow-y-auto">
          <div class="space-y-1 font-mono text-sm">
            {#each initializationLogs as log}
              <div class="text-green-400">{log}</div>
            {/each}
          </div>
        </div>
      </div>
    {/if}

    <!-- Live Demo -->
    {#if demoVisible && serviceReady}
      <div in:fade={{ delay: 300 }}>
        <div class="text-center mb-8">
          <h2 class="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            Interactive Demo
          </h2>
          <p class="text-lg text-slate-600 dark:text-slate-400">
            Experience the complete LangChain.js implementation with real-time event monitoring
          </p>
        </div>

        <LangChainDemo />
      </div>
    {:else if !serviceReady}
      <div class="text-center py-16">
        <div class="w-16 h-16 bg-gradient-to-br from-purple-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6 animate-pulse">
          <Zap class="w-8 h-8 text-white" />
        </div>
        <p class="text-xl text-slate-600 dark:text-slate-400">
          Initializing LangChain.js with Event Listeners...
        </p>
      </div>
    {/if}
  </div>

  <!-- Technical Implementation Details -->
  <div class="bg-slate-100 dark:bg-slate-800 py-16">
    <div class="max-w-7xl mx-auto px-6">
      <div class="text-center mb-12">
        <h2 class="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
          Technical Architecture
        </h2>
        <p class="text-lg text-slate-600 dark:text-slate-400">
          Event-driven design with comprehensive monitoring and integration
        </p>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-12">
        <!-- Event System -->
        <div class="bg-white dark:bg-slate-900 rounded-xl p-8 shadow-sm">
          <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-6">
            Event-Driven Architecture
          </h3>
          
          <div class="space-y-4">
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Service Events</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Initialization, errors, status changes</p>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Session Management</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Create, delete, switch conversations</p>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Message Processing</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Send, receive, stream, error handling</p>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-orange-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Tool Execution</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Legal search, analysis, drafting, citations</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Integration -->
        <div class="bg-white dark:bg-slate-900 rounded-xl p-8 shadow-sm">
          <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-6">
            System Integration
          </h3>
          
          <div class="space-y-4">
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-cyan-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Enhanced RAG Pipeline</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Multi-protocol routing with WebGPU acceleration</p>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-pink-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">XState Management</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">State machines for complex workflows</p>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-indigo-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Ollama Integration</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Local LLM with gemma3:legal-latest model</p>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Memory & Persistence</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Conversation memory with serialization</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  :global(.langchain-demo) {
    margin-top: 0;
    padding-top: 0;
  }
</style>