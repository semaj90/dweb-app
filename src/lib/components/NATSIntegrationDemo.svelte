<!-- NATS Messaging Integration Demo Component -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade, fly, scale } from 'svelte/transition';
  import { 
    Activity, 
    MessageSquare, 
    Send, 
    Settings, 
    Monitor, 
    Zap,
    Database,
    Network,
    Users,
    Bell,
    Search,
    FileText,
    Brain,
    BarChart3,
    CheckCircle,
    AlertTriangle,
    Loader2,
    Play,
    Pause,
    RotateCcw
  } from 'lucide-svelte';

  // Import NATS services
  import { 
    createNATSService, 
    getNATSService,
    natsStatus,
    natsMetrics,
    NATS_SUBJECTS,
    type NATSMessage
  } from '$lib/services/nats-messaging-service';

  // Import LangChain integration
  import { getLangChainService } from '$lib/langchain/langchain-service';

  // Component state
  let selectedDemo = $state<'publish' | 'subscribe' | 'chat' | 'analytics' | 'integration'>('publish');
  let natsService: any = null;
  let langchainService: any = null;
  
  // Publishing demo
  let publishSubject = $state('legal.case.created');
  let publishData = $state('{"caseId": "case_001", "title": "Personal Injury Case", "status": "active"}');
  let publishPriority = $state<'low' | 'normal' | 'high' | 'critical'>('normal');
  let isPublishing = $state(false);
  
  // Subscription demo
  let subscribeSubject = $state('legal.document.*');
  let activeSubscriptions = $state<Array<{ id: string; subject: string; messageCount: number }>>([]);
  let receivedMessages = $state<NATSMessage[]>([]);
  
  // Chat integration demo
  let chatMessage = $state('Analyze the liability issues in the Johnson v. Smith case');
  let chatSession = $state('session_001');
  let chatHistory = $state<Array<{ type: 'sent' | 'received'; content: string; timestamp: number }>>([]);
  
  // Real-time analytics
  let analyticsData = $state({
    messagesPerSecond: 0,
    totalMessages: 0,
    activeSubscriptions: 0,
    connectionHealth: 'healthy',
    lastActivity: null
  });

  // Reactive store access
  let status = $state($natsStatus);
  let metrics = $state($natsMetrics);

  // Subscribe to store updates
  natsStatus.subscribe(s => status = s);
  natsMetrics.subscribe(m => metrics = m);

  // Sample subjects for demonstration
  const sampleSubjects = [
    'legal.case.created',
    'legal.case.updated',
    'legal.document.uploaded',
    'legal.document.processed',
    'legal.ai.analysis.completed',
    'legal.search.query',
    'legal.chat.message',
    'legal.evidence.added',
    'system.health',
    'legal.notification.send'
  ];

  onMount(async () => {
    console.log('🚀 NATS Integration Demo mounted');
    
    // Get or create NATS service
    natsService = getNATSService();
    if (!natsService) {
      natsService = createNATSService({
        enableLegalChannels: true,
        enableDocumentStreaming: true,
        enableRealTimeAnalysis: true
      });
      await natsService.initialize();
    }

    // Get LangChain service for integration
    langchainService = getLangChainService();

    // Setup NATS event listeners
    setupNATSEventListeners();
    
    // Setup LangChain integration
    if (langchainService) {
      setupLangChainIntegration();
    }

    // Start analytics monitoring
    startAnalyticsMonitoring();
  });

  onDestroy(() => {
    // Cleanup subscriptions
    activeSubscriptions.forEach(sub => {
      if (natsService) {
        natsService.unsubscribe(sub.id);
      }
    });
  });

  function setupNATSEventListeners() {
    if (!natsService) return;

    natsService.on('nats:published', (data: any) => {
      console.log('📤 NATS Published:', data);
      updateAnalytics();
    });

    natsService.on('nats:subscribed', (data: any) => {
      console.log('📡 NATS Subscribed:', data);
      updateActiveSubscriptions();
    });

    natsService.on('message:' + NATS_SUBJECTS.CHAT_MESSAGE, (message: NATSMessage) => {
      console.log('💬 Chat message received:', message);
      addChatMessage('received', message.data.content || JSON.stringify(message.data));
    });

    natsService.on('message:' + NATS_SUBJECTS.AI_ANALYSIS_COMPLETED, (message: NATSMessage) => {
      console.log('🧠 AI Analysis completed:', message);
      addReceivedMessage(message);
    });

    natsService.on('message:' + NATS_SUBJECTS.DOCUMENT_PROCESSED, (message: NATSMessage) => {
      console.log('📄 Document processed:', message);
      addReceivedMessage(message);
    });
  }

  function setupLangChainIntegration() {
    if (!langchainService || !natsService) return;

    // Forward LangChain events to NATS
    langchainService.on('message:received', async (data: any) => {
      await natsService.publishChatMessage({
        sessionId: data.sessionId,
        message: data.message,
        response: data.response,
        timestamp: Date.now()
      }, data.sessionId);
    });

    langchainService.on('tool:executed', async (data: any) => {
      await natsService.publishAIAnalysisEvent('completed', {
        toolName: data.toolName,
        input: data.input,
        output: data.output,
        executionTime: data.executionTime,
        timestamp: Date.now()
      });
    });

    // Listen for NATS messages to trigger LangChain operations
    natsService.on('message:' + NATS_SUBJECTS.SEARCH_QUERY, async (message: NATSMessage) => {
      if (langchainService.isReady) {
        try {
          const result = await langchainService.executeTool('legal_search', JSON.stringify(message.data));
          
          await natsService.publish(NATS_SUBJECTS.SEARCH_RESULTS, {
            queryId: message.messageId,
            results: result,
            timestamp: Date.now()
          });
        } catch (error) {
          console.error('LangChain search execution failed:', error);
        }
      }
    });
  }

  function startAnalyticsMonitoring() {
    setInterval(() => {
      updateAnalytics();
    }, 1000);
  }

  function updateAnalytics() {
    if (!natsService) return;

    const connectionMetrics = natsService.getConnectionMetrics();
    const messageStats = natsService.messageStats;
    
    analyticsData = {
      messagesPerSecond: Math.random() * 10, // Simulated for demo
      totalMessages: messageStats.published + messageStats.received,
      activeSubscriptions: connectionMetrics.subscriptions,
      connectionHealth: connectionMetrics.connected ? 'healthy' : 'disconnected',
      lastActivity: connectionMetrics.lastConnected
    };
  }

  function updateActiveSubscriptions() {
    if (!natsService) return;
    activeSubscriptions = natsService.getSubscriptionStats();
  }

  function addReceivedMessage(message: NATSMessage) {
    receivedMessages = [message, ...receivedMessages].slice(0, 50); // Keep last 50
  }

  function addChatMessage(type: 'sent' | 'received', content: string) {
    chatHistory = [...chatHistory, {
      type,
      content,
      timestamp: Date.now()
    }].slice(-20); // Keep last 20
  }

  // ============ Demo Actions ============

  async function publishMessage() {
    if (!natsService || isPublishing) return;

    isPublishing = true;
    
    try {
      let data: any;
      try {
        data = JSON.parse(publishData);
      } catch {
        data = { message: publishData };
      }

      await natsService.publish(publishSubject, data, {
        metadata: {
          source: 'nats-demo',
          priority: publishPriority
        }
      });

      console.log('✓ Message published successfully');
    } catch (error) {
      console.error('❌ Publishing failed:', error);
    } finally {
      isPublishing = false;
    }
  }

  async function subscribeToSubject() {
    if (!natsService || !subscribeSubject.trim()) return;

    try {
      const subscriptionId = await natsService.subscribe(subscribeSubject, (message: NATSMessage) => {
        console.log(`📨 Received message on ${subscribeSubject}:`, message);
        addReceivedMessage(message);
      });

      updateActiveSubscriptions();
      console.log(`✓ Subscribed to ${subscribeSubject}`);
    } catch (error) {
      console.error('❌ Subscription failed:', error);
    }
  }

  async function unsubscribeFromSubject(subscriptionId: string) {
    if (!natsService) return;

    try {
      await natsService.unsubscribe(subscriptionId);
      updateActiveSubscriptions();
      console.log('✓ Unsubscribed successfully');
    } catch (error) {
      console.error('❌ Unsubscription failed:', error);
    }
  }

  async function sendChatMessage() {
    if (!natsService || !chatMessage.trim()) return;

    const message = chatMessage.trim();
    chatMessage = '';

    try {
      // Add to local chat history
      addChatMessage('sent', message);

      // Publish to NATS
      await natsService.publishChatMessage({
        content: message,
        sessionId: chatSession,
        timestamp: Date.now()
      }, chatSession);

      // If LangChain is available, process the message
      if (langchainService && langchainService.isReady) {
        const session = langchainService.getSession(chatSession) || 
                       langchainService.createSession(`NATS Chat ${chatSession}`);
        
        const result = await langchainService.sendMessage(session.id, message);
        
        // Add response to chat history
        addChatMessage('received', result.response);
      }

    } catch (error) {
      console.error('❌ Chat message failed:', error);
    }
  }

  function getStatusIcon(isConnected: boolean) {
    return isConnected ? CheckCircle : AlertTriangle;
  }

  function getStatusColor(isConnected: boolean) {
    return isConnected ? 'text-green-500' : 'text-red-500';
  }

  function formatTimestamp(timestamp: number): string {
    return new Date(timestamp).toLocaleTimeString();
  }

  function clearReceivedMessages() {
    receivedMessages = [];
  }

  function clearChatHistory() {
    chatHistory = [];
  }

  // Test integration with different message types
  async function testIntegration() {
    if (!natsService) return;

    const testMessages = [
      {
        subject: NATS_SUBJECTS.CASE_CREATED,
        data: { caseId: 'test_001', title: 'Test Case', type: 'litigation' }
      },
      {
        subject: NATS_SUBJECTS.DOCUMENT_UPLOADED,
        data: { documentId: 'doc_001', name: 'evidence.pdf', caseId: 'test_001' }
      },
      {
        subject: NATS_SUBJECTS.SEARCH_QUERY,
        data: { query: 'negligence standards', filters: { jurisdiction: 'federal' } }
      }
    ];

    for (const msg of testMessages) {
      await natsService.publish(msg.subject, msg.data);
      await new Promise(resolve => setTimeout(resolve, 500)); // Delay between messages
    }
  }
</script>

<div class="nats-integration-demo bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 min-h-screen p-6">
  <!-- Header -->
  <div class="max-w-7xl mx-auto mb-8">
    <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-4">
          <div class="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <Network class="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 class="text-3xl font-bold text-slate-900 dark:text-slate-100">NATS Messaging Demo</h1>
            <p class="text-slate-600 dark:text-slate-400">Real-time distributed messaging for Legal AI</p>
          </div>
        </div>
        
        <!-- Connection Status -->
        <div class="flex items-center space-x-4">
          {@const StatusIcon = getStatusIcon(status.connected)}
          <div class="flex items-center space-x-2 px-4 py-2 bg-slate-100 dark:bg-slate-700 rounded-lg">
            <StatusIcon class="w-5 h-5 {getStatusColor(status.connected)}" />
            <div>
              <p class="text-sm font-medium text-slate-900 dark:text-slate-100">
                {status.connected ? 'NATS Connected' : 'Disconnected'}
              </p>
              <p class="text-xs text-slate-600 dark:text-slate-400">
                {status.subscriptions} subscriptions, {status.publishedMessages} published
              </p>
            </div>
          </div>
          
          <button
            on:click={testIntegration}
            class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center space-x-2"
          >
            <Play class="w-4 h-4" />
            <span>Test Integration</span>
          </button>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-4 gap-8">
    <!-- Demo Controls -->
    <div class="lg:col-span-1 space-y-6">
      <!-- Demo Selection -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">NATS Demos</h3>
        
        <div class="space-y-3">
          {#each [
            { id: 'publish', label: 'Message Publishing', icon: Send, desc: 'Publish messages to NATS subjects' },
            { id: 'subscribe', label: 'Subscriptions', icon: Bell, desc: 'Subscribe to message channels' },
            { id: 'chat', label: 'Chat Integration', icon: MessageSquare, desc: 'LangChain + NATS chat' },
            { id: 'analytics', label: 'Real-time Analytics', icon: BarChart3, desc: 'Message flow analytics' },
            { id: 'integration', label: 'System Integration', icon: Brain, desc: 'Full system integration' }
          ] as demo}
            <button
              on:click={() => selectedDemo = demo.id}
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

      <!-- Real-time Analytics -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Live Metrics</h3>
        
        <div class="space-y-4">
          <div class="flex justify-between items-center">
            <span class="text-sm text-slate-600 dark:text-slate-400">Messages/sec</span>
            <span class="font-bold text-slate-900 dark:text-slate-100">
              {analyticsData.messagesPerSecond.toFixed(1)}
            </span>
          </div>
          
          <div class="flex justify-between items-center">
            <span class="text-sm text-slate-600 dark:text-slate-400">Total Messages</span>
            <span class="font-bold text-slate-900 dark:text-slate-100">
              {analyticsData.totalMessages}
            </span>
          </div>
          
          <div class="flex justify-between items-center">
            <span class="text-sm text-slate-600 dark:text-slate-400">Subscriptions</span>
            <span class="font-bold text-slate-900 dark:text-slate-100">
              {analyticsData.activeSubscriptions}
            </span>
          </div>
          
          <div class="flex justify-between items-center">
            <span class="text-sm text-slate-600 dark:text-slate-400">Health</span>
            <span class="font-bold {analyticsData.connectionHealth === 'healthy' ? 'text-green-600' : 'text-red-600'}">
              {analyticsData.connectionHealth}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Main Demo Area -->
    <div class="lg:col-span-3 space-y-6">
      <!-- Selected Demo Content -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4 capitalize">
          {selectedDemo} Demo
        </h3>
        
        {#if selectedDemo === 'publish'}
          <div class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  NATS Subject
                </label>
                <select
                  bind:value={publishSubject}
                  class="w-full rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
                >
                  {#each sampleSubjects as subject}
                    <option value={subject}>{subject}</option>
                  {/each}
                </select>
              </div>
              
              <div>
                <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Priority
                </label>
                <select
                  bind:value={publishPriority}
                  class="w-full rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
                >
                  <option value="low">Low</option>
                  <option value="normal">Normal</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>
            </div>
            
            <div>
              <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Message Data (JSON)
              </label>
              <textarea
                bind:value={publishData}
                class="w-full h-32 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
                placeholder="Enter JSON message data..."
              ></textarea>
            </div>
            
            <button
              on:click={publishMessage}
              disabled={isPublishing || !status.connected}
              class="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-slate-400 text-white rounded-lg transition-colors flex items-center space-x-2"
            >
              {#if isPublishing}
                <Loader2 class="w-4 h-4 animate-spin" />
              {:else}
                <Send class="w-4 h-4" />
              {/if}
              <span>Publish Message</span>
            </button>
          </div>

        {:else if selectedDemo === 'subscribe'}
          <div class="space-y-4">
            <div class="flex space-x-4">
              <input
                bind:value={subscribeSubject}
                placeholder="Enter subject pattern (e.g., legal.document.*)"
                class="flex-1 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
              />
              <button
                on:click={subscribeToSubject}
                disabled={!status.connected}
                class="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white rounded-lg transition-colors"
              >
                Subscribe
              </button>
            </div>
            
            <!-- Active Subscriptions -->
            <div class="border border-slate-200 dark:border-slate-600 rounded-lg">
              <div class="bg-slate-50 dark:bg-slate-700 px-4 py-2 border-b border-slate-200 dark:border-slate-600 flex justify-between items-center">
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Active Subscriptions</h4>
                <span class="text-sm text-slate-600 dark:text-slate-400">{activeSubscriptions.length}</span>
              </div>
              
              <div class="max-h-48 overflow-y-auto">
                {#each activeSubscriptions as subscription}
                  <div class="flex justify-between items-center p-3 border-b border-slate-100 dark:border-slate-600 last:border-b-0">
                    <div>
                      <p class="font-medium text-slate-900 dark:text-slate-100 text-sm">{subscription.subject}</p>
                      <p class="text-xs text-slate-600 dark:text-slate-400">{subscription.messageCount} messages</p>
                    </div>
                    <button
                      on:click={() => unsubscribeFromSubject(subscription.id)}
                      class="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-sm"
                    >
                      Unsubscribe
                    </button>
                  </div>
                {:else}
                  <div class="p-4 text-center text-slate-500 dark:text-slate-400">
                    No active subscriptions
                  </div>
                {/each}
              </div>
            </div>
            
            <!-- Received Messages -->
            <div class="border border-slate-200 dark:border-slate-600 rounded-lg">
              <div class="bg-slate-50 dark:bg-slate-700 px-4 py-2 border-b border-slate-200 dark:border-slate-600 flex justify-between items-center">
                <h4 class="font-medium text-slate-900 dark:text-slate-100">Received Messages</h4>
                <button
                  on:click={clearReceivedMessages}
                  class="text-sm text-blue-600 hover:text-blue-700"
                >
                  Clear
                </button>
              </div>
              
              <div class="max-h-64 overflow-y-auto">
                {#each receivedMessages as message}
                  <div class="p-3 border-b border-slate-100 dark:border-slate-600 last:border-b-0">
                    <div class="flex justify-between items-start mb-2">
                      <span class="font-medium text-slate-900 dark:text-slate-100 text-sm">{message.subject}</span>
                      <span class="text-xs text-slate-500 dark:text-slate-400">{formatTimestamp(message.timestamp)}</span>
                    </div>
                    <pre class="text-xs text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-700 p-2 rounded overflow-x-auto">
{JSON.stringify(message.data, null, 2)}
                    </pre>
                  </div>
                {:else}
                  <div class="p-4 text-center text-slate-500 dark:text-slate-400">
                    No messages received yet
                  </div>
                {/each}
              </div>
            </div>
          </div>

        {:else if selectedDemo === 'chat'}
          <div class="space-y-4">
            <!-- Chat Messages -->
            <div class="h-96 border border-slate-200 dark:border-slate-600 rounded-lg p-4 overflow-y-auto space-y-3">
              {#each chatHistory as message}
                <div class="flex {message.type === 'sent' ? 'justify-end' : 'justify-start'}">
                  <div class="max-w-xs lg:max-w-md px-4 py-2 rounded-lg {
                    message.type === 'sent' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-slate-100 dark:bg-slate-700 text-slate-900 dark:text-slate-100'
                  }">
                    <p class="text-sm">{message.content}</p>
                    <p class="text-xs mt-1 opacity-70">{formatTimestamp(message.timestamp)}</p>
                  </div>
                </div>
              {:else}
                <div class="text-center text-slate-500 dark:text-slate-400 py-8">
                  <MessageSquare class="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Start a conversation with the Legal AI assistant</p>
                  <p class="text-sm">Messages are routed through NATS and processed by LangChain</p>
                </div>
              {/each}
            </div>
            
            <!-- Chat Input -->
            <div class="flex space-x-3">
              <input
                bind:value={chatMessage}
                placeholder="Ask a legal question..."
                class="flex-1 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
                on:keydown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendChatMessage();
                  }
                }}
              />
              <button
                on:click={sendChatMessage}
                disabled={!chatMessage.trim() || !status.connected}
                class="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white rounded-lg transition-colors"
              >
                <Send class="w-4 h-4" />
              </button>
              <button
                on:click={clearChatHistory}
                class="px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition-colors"
              >
                <RotateCcw class="w-4 h-4" />
              </button>
            </div>
            
            <div class="text-sm text-slate-600 dark:text-slate-400">
              Session ID: {chatSession} | NATS + LangChain Integration
            </div>
          </div>

        {:else if selectedDemo === 'analytics'}
          <div class="space-y-6">
            <!-- Real-time Charts -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div class="bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg p-4 text-white">
                <div class="flex items-center justify-between">
                  <div>
                    <p class="text-blue-100 text-sm">Messages/sec</p>
                    <p class="text-2xl font-bold">{analyticsData.messagesPerSecond.toFixed(1)}</p>
                  </div>
                  <Activity class="w-8 h-8 text-blue-200" />
                </div>
              </div>
              
              <div class="bg-gradient-to-r from-green-500 to-green-600 rounded-lg p-4 text-white">
                <div class="flex items-center justify-between">
                  <div>
                    <p class="text-green-100 text-sm">Total Messages</p>
                    <p class="text-2xl font-bold">{analyticsData.totalMessages}</p>
                  </div>
                  <MessageSquare class="w-8 h-8 text-green-200" />
                </div>
              </div>
              
              <div class="bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg p-4 text-white">
                <div class="flex items-center justify-between">
                  <div>
                    <p class="text-purple-100 text-sm">Subscriptions</p>
                    <p class="text-2xl font-bold">{analyticsData.activeSubscriptions}</p>
                  </div>
                  <Bell class="w-8 h-8 text-purple-200" />
                </div>
              </div>
              
              <div class="bg-gradient-to-r from-orange-500 to-orange-600 rounded-lg p-4 text-white">
                <div class="flex items-center justify-between">
                  <div>
                    <p class="text-orange-100 text-sm">Health</p>
                    <p class="text-xl font-bold">{analyticsData.connectionHealth}</p>
                  </div>
                  <Monitor class="w-8 h-8 text-orange-200" />
                </div>
              </div>
            </div>
            
            <!-- Connection Details -->
            <div class="bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
              <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-3">Connection Details</h4>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span class="text-slate-600 dark:text-slate-400">Status:</span>
                  <span class="ml-2 font-medium text-slate-900 dark:text-slate-100">
                    {status.connected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                <div>
                  <span class="text-slate-600 dark:text-slate-400">Published:</span>
                  <span class="ml-2 font-medium text-slate-900 dark:text-slate-100">{status.publishedMessages}</span>
                </div>
                <div>
                  <span class="text-slate-600 dark:text-slate-400">Received:</span>
                  <span class="ml-2 font-medium text-slate-900 dark:text-slate-100">{status.receivedMessages}</span>
                </div>
                <div>
                  <span class="text-slate-600 dark:text-slate-400">Reconnects:</span>
                  <span class="ml-2 font-medium text-slate-900 dark:text-slate-100">{status.reconnectAttempts}</span>
                </div>
              </div>
            </div>
          </div>

        {:else if selectedDemo === 'integration'}
          <div class="space-y-6">
            <div class="text-center">
              <h4 class="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                Complete System Integration
              </h4>
              <p class="text-slate-600 dark:text-slate-400">
                NATS messaging seamlessly integrated with LangChain, RAG pipeline, and WebGPU acceleration
              </p>
            </div>
            
            <!-- Integration Flow -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
                <div class="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center mb-4">
                  <Network class="w-6 h-6 text-white" />
                </div>
                <h5 class="font-semibold text-slate-900 dark:text-slate-100 mb-2">NATS Messaging</h5>
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  High-performance message routing with legal AI subject patterns and real-time event distribution
                </p>
              </div>
              
              <div class="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border border-green-200 dark:border-green-800">
                <div class="w-12 h-12 bg-green-600 rounded-lg flex items-center justify-center mb-4">
                  <Brain class="w-6 h-6 text-white" />
                </div>
                <h5 class="font-semibold text-slate-900 dark:text-slate-100 mb-2">LangChain Integration</h5>
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  AI processing triggered by NATS messages with automatic response publishing and tool execution
                </p>
              </div>
              
              <div class="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800">
                <div class="w-12 h-12 bg-purple-600 rounded-lg flex items-center justify-center mb-4">
                  <Zap class="w-6 h-6 text-white" />
                </div>
                <h5 class="font-semibold text-slate-900 dark:text-slate-100 mb-2">Multi-Protocol RAG</h5>
                <p class="text-sm text-slate-600 dark:text-slate-400">
                  Enhanced RAG pipeline with WebGPU acceleration responding to NATS search queries
                </p>
              </div>
            </div>
            
            <!-- Live Integration Status -->
            <div class="bg-slate-50 dark:bg-slate-700 rounded-lg p-6">
              <h5 class="font-semibold text-slate-900 dark:text-slate-100 mb-4">Integration Status</h5>
              
              <div class="space-y-3">
                <div class="flex items-center justify-between">
                  <span class="text-slate-700 dark:text-slate-300">NATS Connection</span>
                  <div class="flex items-center space-x-2">
                    <div class="w-2 h-2 rounded-full {status.connected ? 'bg-green-500' : 'bg-red-500'}"></div>
                    <span class="text-sm font-medium">{status.connected ? 'Connected' : 'Disconnected'}</span>
                  </div>
                </div>
                
                <div class="flex items-center justify-between">
                  <span class="text-slate-700 dark:text-slate-300">LangChain Service</span>
                  <div class="flex items-center space-x-2">
                    <div class="w-2 h-2 rounded-full {langchainService?.isReady ? 'bg-green-500' : 'bg-red-500'}"></div>
                    <span class="text-sm font-medium">{langchainService?.isReady ? 'Ready' : 'Not Ready'}</span>
                  </div>
                </div>
                
                <div class="flex items-center justify-between">
                  <span class="text-slate-700 dark:text-slate-300">Event Forwarding</span>
                  <div class="flex items-center space-x-2">
                    <div class="w-2 h-2 rounded-full bg-green-500"></div>
                    <span class="text-sm font-medium">Active</span>
                  </div>
                </div>
                
                <div class="flex items-center justify-between">
                  <span class="text-slate-700 dark:text-slate-300">Message Processing</span>
                  <div class="flex items-center space-x-2">
                    <div class="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div>
                    <span class="text-sm font-medium">Processing</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>

<style>
  .nats-integration-demo {
    font-family: system-ui, -apple-system, sans-serif;
  }
</style>