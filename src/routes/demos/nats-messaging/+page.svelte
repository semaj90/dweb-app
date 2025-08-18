<!-- NATS Messaging System Demo Page -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { fade, fly } from 'svelte/transition';
  import { Network, Zap, MessageSquare, Activity, Brain, CheckCircle } from 'lucide-svelte';
  
  // Import NATS components and services
  import NATSIntegrationDemo from '$lib/components/NATSIntegrationDemo.svelte';
  import { 
    createNATSService, 
    getNATSService,
    natsStatus,
    natsMetrics,
    NATS_SUBJECTS
  } from '$lib/services/nats-messaging-service';

  // Component state
  let natsService: any = null;
  let serviceReady = $state(false);
  let initializationLogs = $state<string[]>([]);

  // Reactive store access
  let status = $state($natsStatus);
  let metrics = $state($natsMetrics);

  // Subscribe to store updates
  natsStatus.subscribe(s => status = s);
  natsMetrics.subscribe(m => metrics = m);

  onMount(async () => {
    console.log('🚀 NATS Messaging Demo Page mounted');
    
    addLog('Initializing NATS messaging system...');
    
    try {
      // Get or create NATS service
      natsService = getNATSService();
      
      if (!natsService) {
        addLog('Creating new NATS service instance...');
        natsService = createNATSService({
          servers: ['ws://localhost:4222'],
          name: 'LegalAI-Demo-Client',
          enableLegalChannels: true,
          enableDocumentStreaming: true,
          enableRealTimeAnalysis: true,
          enableCaseUpdates: true
        });

        addLog('Setting up event listeners...');
        setupNATSEventListeners(natsService);
        
        addLog('Initializing NATS service...');
        const success = await natsService.initialize();
        
        if (success) {
          addLog('✓ NATS service initialized successfully');
          serviceReady = true;
        } else {
          addLog('❌ NATS service initialization failed');
        }
      } else {
        addLog('✓ Using existing NATS service');
        serviceReady = natsService.isReady;
      }
    } catch (error) {
      addLog(`❌ Error: ${error.message}`);
      console.error('NATS initialization error:', error);
    }
  });

  function setupNATSEventListeners(service: any) {
    // Connection events
    service.on('nats:connected', () => {
      addLog('🔗 NATS connected to server');
    });

    service.on('nats:disconnected', () => {
      addLog('🔌 NATS disconnected from server');
    });

    service.on('nats:error', (data: any) => {
      addLog(`❌ NATS error: ${data.error}`);
    });

    // Publishing events
    service.on('nats:published', (data: any) => {
      addLog(`📤 Published message to ${data.subject}`);
    });

    service.on('nats:publish_failed', (data: any) => {
      addLog(`❌ Failed to publish to ${data.subject}: ${data.error}`);
    });

    // Subscription events
    service.on('nats:subscribed', (data: any) => {
      addLog(`📡 Subscribed to ${data.subject}`);
    });

    service.on('nats:unsubscribed', (data: any) => {
      addLog(`❌ Unsubscribed from ${data.subject}`);
    });

    // Legal channel events
    service.on('message:' + NATS_SUBJECTS.CASE_CREATED, (message: any) => {
      addLog(`📋 Case created: ${message.data.title || message.data.caseId}`);
    });

    service.on('message:' + NATS_SUBJECTS.DOCUMENT_UPLOADED, (message: any) => {
      addLog(`📄 Document uploaded: ${message.data.name || message.data.documentId}`);
    });

    service.on('message:' + NATS_SUBJECTS.AI_ANALYSIS_COMPLETED, (message: any) => {
      addLog(`🧠 AI analysis completed for ${message.data.caseId || 'unknown case'}`);
    });

    service.on('message:' + NATS_SUBJECTS.SEARCH_QUERY, (message: any) => {
      addLog(`🔍 Search query: ${message.data.query}`);
    });

    service.on('message:' + NATS_SUBJECTS.CHAT_MESSAGE, (message: any) => {
      addLog(`💬 Chat message in session ${message.data.sessionId}`);
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
      icon: Network,
      title: 'Distributed Messaging',
      description: 'High-performance NATS messaging with legal AI subject patterns and real-time event distribution'
    },
    {
      icon: MessageSquare,
      title: 'Legal Channels',
      description: 'Pre-configured channels for case management, document processing, and AI analysis workflows'
    },
    {
      icon: Brain,
      title: 'LangChain Integration',
      description: 'Seamless integration with LangChain service for AI-powered message processing and responses'
    },
    {
      icon: Activity,
      title: 'Real-time Analytics',
      description: 'Live monitoring of message flow, subscription health, and system performance metrics'
    },
    {
      icon: Zap,
      title: 'Request-Reply Pattern',
      description: 'Synchronous communication with timeout handling and correlation ID tracking'
    },
    {
      icon: CheckCircle,
      title: 'Multi-Protocol RAG',
      description: 'Integration with enhanced RAG pipeline for intelligent document retrieval and analysis'
    }
  ];

  // NATS subjects overview
  const subjectCategories = [
    {
      category: 'Case Management',
      subjects: [
        'legal.case.created',
        'legal.case.updated', 
        'legal.case.closed'
      ]
    },
    {
      category: 'Document Processing',
      subjects: [
        'legal.document.uploaded',
        'legal.document.processed',
        'legal.document.analyzed',
        'legal.document.indexed'
      ]
    },
    {
      category: 'AI Operations',
      subjects: [
        'legal.ai.analysis.started',
        'legal.ai.analysis.completed',
        'legal.search.query',
        'legal.chat.message'
      ]
    },
    {
      category: 'System Events',
      subjects: [
        'system.health',
        'system.metrics',
        'system.alerts'
      ]
    }
  ];
</script>

<svelte:head>
  <title>NATS Messaging System - Legal AI Demo</title>
  <meta name="description" content="High-performance distributed messaging system with LangChain integration for legal AI applications" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800">
  <!-- Hero Section -->
  <div class="relative overflow-hidden bg-white dark:bg-slate-900 shadow-xl">
    <div class="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-purple-600/10"></div>
    <div class="relative max-w-7xl mx-auto px-6 py-16">
      <div class="text-center">
        <div class="flex justify-center mb-6">
          <div class="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center transform -rotate-3">
            <Network class="w-10 h-10 text-white" />
          </div>
        </div>
        
        <h1 class="text-4xl md:text-6xl font-bold text-slate-900 dark:text-slate-100 mb-6">
          NATS Messaging
          <span class="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            System
          </span>
        </h1>
        
        <p class="text-xl text-slate-600 dark:text-slate-400 mb-8 max-w-3xl mx-auto">
          High-performance distributed messaging system with real-time event handling, 
          LangChain integration, and specialized legal AI communication patterns.
        </p>

        <!-- Service Status -->
        <div class="flex items-center justify-center space-x-4 mb-8">
          <div class="flex items-center space-x-2 px-4 py-2 bg-slate-100 dark:bg-slate-800 rounded-full">
            <div class="w-3 h-3 rounded-full {serviceReady ? 'bg-green-500' : 'bg-yellow-500'}"></div>
            <span class="text-sm font-medium text-slate-700 dark:text-slate-300">
              {serviceReady ? 'NATS Ready' : 'Initializing...'}
            </span>
          </div>
          
          {#if status.connected}
            <div class="flex items-center space-x-4 text-sm text-slate-600 dark:text-slate-400">
              <span>{status.subscriptions} Subscriptions</span>
              <span>{status.publishedMessages} Published</span>
              <span>{status.receivedMessages} Received</span>
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
        Enterprise Messaging Features
      </h2>
      <p class="text-lg text-slate-600 dark:text-slate-400">
        Production-ready messaging infrastructure for legal AI applications
      </p>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
      {#each features as feature, index}
        <div 
          class="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-sm border border-slate-200 dark:border-slate-700 hover:shadow-md transition-all duration-300"
          in:fly={{ y: 20, delay: index * 100 }}
        >
          <div class="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center mb-4">
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

    <!-- NATS Subjects Overview -->
    <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-8 mb-16">
      <h2 class="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-6 text-center">
        Legal AI Subject Patterns
      </h2>
      
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {#each subjectCategories as category}
          <div class="space-y-3">
            <h3 class="font-semibold text-slate-900 dark:text-slate-100 text-lg border-b border-slate-200 dark:border-slate-600 pb-2">
              {category.category}
            </h3>
            <div class="space-y-2">
              {#each category.subjects as subject}
                <div class="text-sm font-mono text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-700 px-3 py-2 rounded">
                  {subject}
                </div>
              {/each}
            </div>
          </div>
        {/each}
      </div>
    </div>

    <!-- Activity Logs -->
    {#if initializationLogs.length > 0}
      <div class="bg-slate-900 rounded-xl p-6 mb-8" in:fade>
        <h3 class="text-lg font-semibold text-white mb-4 flex items-center">
          <Activity class="w-5 h-5 mr-2" />
          NATS Activity Log
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

    <!-- Interactive Demo -->
    {#if serviceReady}
      <div in:fade={{ delay: 300 }}>
        <div class="text-center mb-8">
          <h2 class="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            Interactive NATS Demo
          </h2>
          <p class="text-lg text-slate-600 dark:text-slate-400">
            Test messaging, subscriptions, LangChain integration, and real-time analytics
          </p>
        </div>

        <NATSIntegrationDemo />
      </div>
    {:else}
      <div class="text-center py-16">
        <div class="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 animate-pulse">
          <Network class="w-8 h-8 text-white" />
        </div>
        <p class="text-xl text-slate-600 dark:text-slate-400">
          Initializing NATS messaging system...
        </p>
      </div>
    {/if}
  </div>

  <!-- Architecture Overview -->
  <div class="bg-slate-100 dark:bg-slate-800 py-16">
    <div class="max-w-7xl mx-auto px-6">
      <div class="text-center mb-12">
        <h2 class="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
          System Architecture
        </h2>
        <p class="text-lg text-slate-600 dark:text-slate-400">
          Distributed messaging with comprehensive legal AI integration
        </p>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-12">
        <!-- Messaging Patterns -->
        <div class="bg-white dark:bg-slate-900 rounded-xl p-8 shadow-sm">
          <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-6">
            Messaging Patterns
          </h3>
          
          <div class="space-y-4">
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Publish-Subscribe</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Event-driven communication with wildcard subscriptions</p>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Request-Reply</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Synchronous communication with timeout handling</p>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Streaming</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Real-time data streaming for document processing</p>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-orange-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Queue Groups</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Load balancing and message distribution</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Integration Points -->
        <div class="bg-white dark:bg-slate-900 rounded-xl p-8 shadow-sm">
          <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-6">
            System Integration
          </h3>
          
          <div class="space-y-4">
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-cyan-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">LangChain Service</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">AI processing triggered by NATS messages</p>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-pink-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Enhanced RAG Pipeline</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Document search and retrieval via messaging</p>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-indigo-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">WebGPU Acceleration</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">GPU-accelerated processing integration</p>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
              <div>
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Real-time Analytics</h4>
                <p class="text-sm text-slate-600 dark:text-slate-400">Performance monitoring and metrics collection</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>