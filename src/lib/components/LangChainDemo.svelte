<!-- LangChain Integration Demo Component -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade, fly, scale } from 'svelte/transition';
  import { 
    MessageSquare, 
    Brain, 
    Tool, 
    Search, 
    FileText, 
    Users,
    Activity, 
    BarChart3, 
    Clock,
    Zap,
    CheckCircle,
    AlertTriangle,
    Play,
    Pause,
    RotateCcw,
    Send,
    Loader2,
    Bot,
    User,
    Settings,
    Download,
    Upload
  } from 'lucide-svelte';

  // Import LangChain services and stores
  import { 
    createLangChainService, 
    getLangChainService,
    langchainServiceStatus,
    langchainSessions,
    langchainMetrics,
    type ConversationSession,
    type StreamingResponse
  } from '../langchain/langchain-service';

  // Component state
  let selectedDemo = $state<'chat' | 'tools' | 'analysis' | 'streaming' | 'metrics'>('chat');
  let currentSession: ConversationSession | null = $state(null);
  let isProcessing = $state(false);
  let isStreaming = $state(false);
  
  // Chat interface
  let messageInput = $state('What are the key elements of a valid contract under contract law?');
  let chatMessages = $state<Array<{
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: number;
    metadata?: any;
  }>>([]);
  let streamingContent = $state('');

  // Tool testing
  let selectedTool = $state('legal_search');
  let toolInput = $state('personal injury negligence cases');
  let toolResults = $state<any>(null);

  // Document analysis
  let analysisText = $state(`SUPREME COURT OF THE UNITED STATES

Smith v. Jones, 123 U.S. 456 (2023)

FACTS: The plaintiff alleges that defendant failed to exercise reasonable care in maintaining the premises, resulting in a slip and fall accident. The defendant argues that the plaintiff was contributorily negligent.

HOLDING: The court holds that property owners have a duty to maintain their premises in a reasonably safe condition for invitees.`);
  let analysisType = $state<'summary' | 'precedents' | 'facts' | 'holding' | 'reasoning'>('summary');
  let analysisResults = $state<any>(null);

  // Reactive state access
  let serviceStatus = $state($langchainServiceStatus);
  let sessions = $state($langchainSessions);
  let metrics = $state($langchainMetrics);

  // LangChain service instance
  let langchainService: any = null;

  // Subscribe to store updates
  langchainServiceStatus.subscribe(status => serviceStatus = status);
  langchainSessions.subscribe(s => sessions = s);
  langchainMetrics.subscribe(m => metrics = m);

  onMount(async () => {
    console.log('🧠 LangChain Demo component mounted');
    
    // Get or create LangChain service
    langchainService = getLangChainService();
    
    if (!langchainService) {
      langchainService = createLangChainService({
        llmProvider: 'ollama',
        model: 'gemma3:legal-latest',
        enableTools: true,
        enableMemory: true,
        enableStreaming: true,
        enableEventLogging: true
      });
      
      await langchainService.initialize();
    }

    // Setup event listeners
    setupEventListeners();
    
    // Create initial session
    createNewSession();
  });

  onDestroy(() => {
    if (langchainService) {
      // Cancel any active streams
      if (currentSession && isStreaming) {
        langchainService.cancelStream(currentSession.id);
      }
    }
  });

  function setupEventListeners() {
    if (!langchainService) return;

    langchainService.on('message:received', (data: any) => {
      console.log('💬 Message received:', data);
      updateSessionsList();
    });

    langchainService.on('streaming:chunk', (response: StreamingResponse) => {
      streamingContent += response.chunk;
    });

    langchainService.on('streaming:completed', (data: any) => {
      console.log('✓ Streaming completed');
      isStreaming = false;
      
      // Add final message to chat
      if (streamingContent.trim()) {
        chatMessages = [...chatMessages, {
          role: 'assistant',
          content: streamingContent,
          timestamp: Date.now(),
          metadata: { streamed: true }
        }];
        streamingContent = '';
      }
    });

    langchainService.on('tool:executed', (data: any) => {
      console.log('🔧 Tool executed:', data);
    });
  }

  // ============ Session Management ============

  function createNewSession() {
    if (!langchainService) return;

    currentSession = langchainService.createSession(
      `Legal AI Chat ${new Date().toLocaleTimeString()}`,
      {
        legalContext: 'general-legal-assistance',
        userRole: 'legal-professional'
      }
    );

    chatMessages = [{
      role: 'system',
      content: 'Legal AI Assistant ready. How can I help you today?',
      timestamp: Date.now()
    }];

    updateSessionsList();
  }

  function selectSession(session: ConversationSession) {
    currentSession = session;
    // In a full implementation, you would load the chat history
    chatMessages = [{
      role: 'system',
      content: `Switched to session: ${session.title}`,
      timestamp: Date.now()
    }];
  }

  function updateSessionsList() {
    if (!langchainService) return;
    const allSessions = langchainService.getSessions();
    langchainSessions.set(allSessions);
  }

  // ============ Chat Interface ============

  async function sendMessage() {
    if (!langchainService || !currentSession || !messageInput.trim() || isProcessing) return;

    const message = messageInput.trim();
    messageInput = '';
    isProcessing = true;

    // Add user message to chat
    chatMessages = [...chatMessages, {
      role: 'user',
      content: message,
      timestamp: Date.now()
    }];

    try {
      const result = await langchainService.sendMessage(currentSession.id, message, {
        chainType: 'conversation',
        enableTools: true,
        context: { legalContext: true }
      });

      // Add assistant response to chat
      chatMessages = [...chatMessages, {
        role: 'assistant',
        content: result.response,
        timestamp: Date.now(),
        metadata: result.execution.metadata
      }];

      updateSessionsList();

    } catch (error) {
      console.error('Message failed:', error);
      chatMessages = [...chatMessages, {
        role: 'system',
        content: `Error: ${error.message}`,
        timestamp: Date.now()
      }];
    } finally {
      isProcessing = false;
    }
  }

  async function sendStreamingMessage() {
    if (!langchainService || !currentSession || !messageInput.trim() || isStreaming) return;

    const message = messageInput.trim();
    messageInput = '';
    isStreaming = true;
    streamingContent = '';

    // Add user message to chat
    chatMessages = [...chatMessages, {
      role: 'user',
      content: message,
      timestamp: Date.now()
    }];

    try {
      const stream = langchainService.sendStreamingMessage(currentSession.id, message, {
        chainType: 'conversation',
        enableTools: true,
        context: { legalContext: true }
      });

      for await (const response of stream) {
        if (response.isComplete) {
          isStreaming = false;
          break;
        }
      }

      updateSessionsList();

    } catch (error) {
      console.error('Streaming failed:', error);
      isStreaming = false;
      chatMessages = [...chatMessages, {
        role: 'system',
        content: `Streaming error: ${error.message}`,
        timestamp: Date.now()
      }];
    }
  }

  function cancelStreaming() {
    if (langchainService && currentSession && isStreaming) {
      langchainService.cancelStream(currentSession.id);
      isStreaming = false;
    }
  }

  // ============ Tool Testing ============

  async function executeTool() {
    if (!langchainService || !toolInput.trim()) return;

    isProcessing = true;
    toolResults = null;

    try {
      const result = await langchainService.executeTool(selectedTool, toolInput);
      toolResults = {
        tool: result.toolName,
        input: result.input,
        output: JSON.parse(result.output),
        executionTime: result.executionTime
      };
    } catch (error) {
      console.error('Tool execution failed:', error);
      toolResults = {
        error: error.message,
        tool: selectedTool,
        input: toolInput
      };
    } finally {
      isProcessing = false;
    }
  }

  // ============ Document Analysis ============

  async function analyzeDocument() {
    if (!langchainService || !analysisText.trim()) return;

    isProcessing = true;
    analysisResults = null;

    try {
      const result = await langchainService.analyzeLegalDocument(
        analysisText,
        analysisType,
        { jurisdiction: 'federal' }
      );
      analysisResults = result;
    } catch (error) {
      console.error('Document analysis failed:', error);
      analysisResults = { error: error.message };
    } finally {
      isProcessing = false;
    }
  }

  // ============ Helper Functions ============

  function getStatusIcon(isReady: boolean) {
    return isReady ? CheckCircle : AlertTriangle;
  }

  function getStatusColor(isReady: boolean) {
    return isReady ? 'text-green-500' : 'text-red-500';
  }

  function formatTimestamp(timestamp: number): string {
    return new Date(timestamp).toLocaleTimeString();
  }

  function exportSession() {
    if (!currentSession) return;

    const exportData = {
      session: currentSession,
      messages: chatMessages,
      exportedAt: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `langchain_session_${currentSession.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function resetDemo() {
    chatMessages = [];
    toolResults = null;
    analysisResults = null;
    streamingContent = '';
    isProcessing = false;
    isStreaming = false;
    createNewSession();
  }
</script>

<div class="langchain-demo bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 min-h-screen p-6">
  <!-- Header -->
  <div class="max-w-7xl mx-auto mb-8">
    <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-4">
          <div class="w-12 h-12 bg-gradient-to-br from-purple-500 to-blue-600 rounded-lg flex items-center justify-center">
            <Brain class="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 class="text-3xl font-bold text-slate-900 dark:text-slate-100">LangChain AI Demo</h1>
            <p class="text-slate-600 dark:text-slate-400">Event-Driven Legal AI Assistant</p>
          </div>
        </div>
        
        <!-- Service Status -->
        <div class="flex items-center space-x-4">
          {@const StatusIcon = getStatusIcon(serviceStatus.isReady)}
          <div class="flex items-center space-x-2 px-4 py-2 bg-slate-100 dark:bg-slate-700 rounded-lg">
            <StatusIcon class="w-5 h-5 {getStatusColor(serviceStatus.isReady)}" />
            <div>
              <p class="text-sm font-medium text-slate-900 dark:text-slate-100">
                {serviceStatus.isReady ? 'LangChain Ready' : 'Initializing...'}
              </p>
              <p class="text-xs text-slate-600 dark:text-slate-400">
                {serviceStatus.sessions} sessions, {serviceStatus.activeStreams} streaming
              </p>
            </div>
          </div>
          
          <button
            onclick={resetDemo}
            class="px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition-colors flex items-center space-x-2"
          >
            <RotateCcw class="w-4 h-4" />
            <span>Reset</span>
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
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">LangChain Demos</h3>
        
        <div class="space-y-3">
          {#each [
            { id: 'chat', label: 'AI Chat', icon: MessageSquare, desc: 'Conversational AI with memory' },
            { id: 'tools', label: 'Legal Tools', icon: Tool, desc: 'Specialized legal functions' },
            { id: 'analysis', label: 'Document Analysis', icon: FileText, desc: 'Case and statute analysis' },
            { id: 'streaming', label: 'Real-time Streaming', icon: Activity, desc: 'Live response streaming' },
            { id: 'metrics', label: 'Performance Metrics', icon: BarChart3, desc: 'System analytics' }
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

      <!-- Session Management -->
      {#if serviceStatus.isReady}
        <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100">Sessions</h3>
            <button
              onclick={createNewSession}
              class="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm"
            >
              New Session
            </button>
          </div>
          
          <div class="space-y-2 max-h-64 overflow-y-auto">
            {#each sessions as session}
              <button
                onclick={() => selectSession(session)}
                class="w-full p-3 text-left rounded-lg transition-colors {
                  currentSession?.id === session.id
                    ? 'bg-blue-100 dark:bg-blue-900/20'
                    : 'hover:bg-slate-100 dark:hover:bg-slate-700'
                }"
              >
                <p class="font-medium text-slate-900 dark:text-slate-100 text-sm">{session.title}</p>
                <p class="text-xs text-slate-600 dark:text-slate-400">
                  {session.messageCount} messages • {formatTimestamp(session.lastActivity)}
                </p>
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Quick Actions -->
      {#if currentSession}
        <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
          <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Actions</h3>
          
          <div class="space-y-3">
            <button
              onclick={exportSession}
              class="w-full px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center justify-center space-x-2"
            >
              <Download class="w-4 h-4" />
              <span>Export Session</span>
            </button>
          </div>
        </div>
      {/if}
    </div>

    <!-- Main Demo Area -->
    <div class="lg:col-span-3 space-y-6">
      <!-- Selected Demo Content -->
      <div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
        <h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4 capitalize">
          {selectedDemo} Demo
        </h3>
        
        {#if selectedDemo === 'chat'}
          <div class="space-y-4">
            <!-- Chat Messages -->
            <div class="h-96 border border-slate-200 dark:border-slate-600 rounded-lg p-4 overflow-y-auto space-y-3">
              {#each chatMessages as message}
                <div class="flex space-x-3 {message.role === 'user' ? 'justify-end' : 'justify-start'}">
                  <div class="flex items-start space-x-3 max-w-3xl">
                    {#if message.role !== 'user'}
                      <div class="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                        {#if message.role === 'assistant'}
                          <Bot class="w-4 h-4 text-white" />
                        {:else}
                          <Settings class="w-4 h-4 text-white" />
                        {/if}
                      </div>
                    {/if}
                    
                    <div class="flex-1">
                      <div class="p-3 rounded-lg {
                        message.role === 'user' 
                          ? 'bg-blue-600 text-white ml-auto'
                          : message.role === 'assistant'
                          ? 'bg-slate-100 dark:bg-slate-700 text-slate-900 dark:text-slate-100'
                          : 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200'
                      }">
                        <p class="text-sm">{message.content}</p>
                        {#if message.metadata}
                          <p class="text-xs mt-2 opacity-70">
                            {message.metadata.tokens || 0} tokens • {formatTimestamp(message.timestamp)}
                          </p>
                        {/if}
                      </div>
                    </div>
                    
                    {#if message.role === 'user'}
                      <div class="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                        <User class="w-4 h-4 text-white" />
                      </div>
                    {/if}
                  </div>
                </div>
              {/each}
              
              <!-- Streaming Content -->
              {#if isStreaming && streamingContent}
                <div class="flex space-x-3 justify-start">
                  <div class="flex items-start space-x-3 max-w-3xl">
                    <div class="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                      <Bot class="w-4 h-4 text-white" />
                    </div>
                    <div class="flex-1">
                      <div class="p-3 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-900 dark:text-slate-100">
                        <p class="text-sm">{streamingContent}<span class="animate-pulse">|</span></p>
                      </div>
                    </div>
                  </div>
                </div>
              {/if}
            </div>
            
            <!-- Chat Input -->
            <div class="flex space-x-3">
              <textarea
                bind:value={messageInput}
                placeholder="Ask a legal question..."
                disabled={isProcessing || isStreaming}
                class="flex-1 resize-none rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100 placeholder-slate-500 dark:placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
                rows="2"
                onkeydown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
              ></textarea>
              
              <div class="flex flex-col space-y-2">
                <button
                  onclick={sendMessage}
                  disabled={!messageInput.trim() || isProcessing || isStreaming}
                  class="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white rounded-lg transition-colors flex items-center justify-center"
                >
                  {#if isProcessing}
                    <Loader2 class="w-4 h-4 animate-spin" />
                  {:else}
                    <Send class="w-4 h-4" />
                  {/if}
                </button>
                
                <button
                  onclick={isStreaming ? cancelStreaming : sendStreamingMessage}
                  disabled={!messageInput.trim() && !isStreaming || isProcessing}
                  class="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-400 text-white rounded-lg transition-colors flex items-center justify-center"
                >
                  {#if isStreaming}
                    <Pause class="w-4 h-4" />
                  {:else}
                    <Activity class="w-4 h-4" />
                  {/if}
                </button>
              </div>
            </div>
          </div>
        {:else if selectedDemo === 'tools'}
          <div class="space-y-4">
            <!-- Tool Selection -->
            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Select Tool
                </label>
                <select
                  bind:value={selectedTool}
                  class="w-full rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
                >
                  <option value="legal_search">Legal Search</option>
                  <option value="case_analysis">Case Analysis</option>
                  <option value="legal_drafting">Legal Drafting</option>
                  <option value="citation_checker">Citation Checker</option>
                </select>
              </div>
              
              <div>
                <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Tool Input
                </label>
                <input
                  bind:value={toolInput}
                  placeholder="Enter tool input..."
                  class="w-full rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
                />
              </div>
            </div>
            
            <button
              onclick={executeTool}
              disabled={isProcessing || !toolInput.trim()}
              class="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-slate-400 text-white rounded-lg transition-colors flex items-center space-x-2"
            >
              {#if isProcessing}
                <Loader2 class="w-4 h-4 animate-spin" />
              {:else}
                <Tool class="w-4 h-4" />
              {/if}
              <span>Execute Tool</span>
            </button>
            
            <!-- Tool Results -->
            {#if toolResults}
              <div class="p-4 border border-slate-200 dark:border-slate-600 rounded-lg">
                <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-3">Tool Results</h4>
                
                {#if toolResults.error}
                  <div class="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded">
                    <p class="text-red-700 dark:text-red-300">Error: {toolResults.error}</p>
                  </div>
                {:else}
                  <div class="space-y-3">
                    <div class="p-3 bg-slate-50 dark:bg-slate-700 rounded">
                      <p class="text-sm font-medium text-slate-900 dark:text-slate-100">Tool: {toolResults.tool}</p>
                      <p class="text-sm text-slate-600 dark:text-slate-400">Input: {toolResults.input}</p>
                      <p class="text-sm text-slate-600 dark:text-slate-400">Execution Time: {toolResults.executionTime}ms</p>
                    </div>
                    
                    <div class="p-3 bg-slate-50 dark:bg-slate-700 rounded">
                      <p class="text-sm font-medium text-slate-900 dark:text-slate-100 mb-2">Output:</p>
                      <pre class="text-xs text-slate-600 dark:text-slate-400 whitespace-pre-wrap max-h-64 overflow-y-auto">
                        {JSON.stringify(toolResults.output, null, 2)}
                      </pre>
                    </div>
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        {:else if selectedDemo === 'analysis'}
          <div class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div class="md:col-span-3">
                <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Legal Document Text
                </label>
                <textarea
                  bind:value={analysisText}
                  class="w-full h-64 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
                  placeholder="Paste legal document text here..."
                ></textarea>
              </div>
              
              <div>
                <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Analysis Type
                </label>
                <select
                  bind:value={analysisType}
                  class="w-full rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-slate-900 dark:text-slate-100"
                >
                  <option value="summary">Summary</option>
                  <option value="precedents">Precedents</option>
                  <option value="facts">Facts</option>
                  <option value="holding">Holding</option>
                  <option value="reasoning">Reasoning</option>
                </select>
                
                <button
                  onclick={analyzeDocument}
                  disabled={isProcessing || !analysisText.trim()}
                  class="w-full mt-4 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-400 text-white rounded-lg transition-colors flex items-center justify-center space-x-2"
                >
                  {#if isProcessing}
                    <Loader2 class="w-4 h-4 animate-spin" />
                  {:else}
                    <FileText class="w-4 h-4" />
                  {/if}
                  <span>Analyze</span>
                </button>
              </div>
            </div>
            
            <!-- Analysis Results -->
            {#if analysisResults}
              <div class="p-4 border border-slate-200 dark:border-slate-600 rounded-lg">
                <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-3">Analysis Results</h4>
                
                {#if analysisResults.error}
                  <div class="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded">
                    <p class="text-red-700 dark:text-red-300">Error: {analysisResults.error}</p>
                  </div>
                {:else}
                  <pre class="text-sm text-slate-600 dark:text-slate-400 whitespace-pre-wrap max-h-96 overflow-y-auto bg-slate-50 dark:bg-slate-700 p-3 rounded">
                    {JSON.stringify(analysisResults, null, 2)}
                  </pre>
                {/if}
              </div>
            {/if}
          </div>
        {:else if selectedDemo === 'streaming'}
          <div class="space-y-4">
            <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
              <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-2">Streaming Status</h4>
              <p class="text-sm text-slate-600 dark:text-slate-400">
                Active Streams: {serviceStatus.activeStreams}
              </p>
              <p class="text-sm text-slate-600 dark:text-slate-400">
                Current Session: {currentSession?.id || 'None'}
              </p>
            </div>
            
            <div class="text-center py-8">
              <Activity class="w-12 h-12 text-slate-400 mx-auto mb-4" />
              <p class="text-slate-600 dark:text-slate-400">
                Use the chat interface to test streaming responses
              </p>
              <p class="text-sm text-slate-500 dark:text-slate-500">
                Click the streaming button (⚡) next to the send button
              </p>
            </div>
          </div>
        {:else if selectedDemo === 'metrics'}
          <div class="space-y-4">
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <p class="text-sm text-slate-600 dark:text-slate-400">Total Executions</p>
                <p class="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {metrics.totalExecutions || 0}
                </p>
              </div>
              
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <p class="text-sm text-slate-600 dark:text-slate-400">Success Rate</p>
                <p class="text-2xl font-bold text-green-600">
                  {metrics.totalExecutions > 0 
                    ? Math.round((metrics.successfulExecutions / metrics.totalExecutions) * 100)
                    : 0}%
                </p>
              </div>
              
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <p class="text-sm text-slate-600 dark:text-slate-400">Avg Latency</p>
                <p class="text-2xl font-bold text-blue-600">
                  {Math.round(metrics.averageLatency || 0)}ms
                </p>
              </div>
              
              <div class="p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
                <p class="text-sm text-slate-600 dark:text-slate-400">Total Tokens</p>
                <p class="text-2xl font-bold text-purple-600">
                  {(metrics.totalTokens || 0).toLocaleString()}
                </p>
              </div>
            </div>
            
            <!-- Tool Usage -->
            {#if metrics.toolUsage && Object.keys(metrics.toolUsage).length > 0}
              <div class="p-4 border border-slate-200 dark:border-slate-600 rounded-lg">
                <h4 class="font-medium text-slate-900 dark:text-slate-100 mb-3">Tool Usage</h4>
                <div class="space-y-2">
                  {#each Object.entries(metrics.toolUsage) as [tool, count]}
                    <div class="flex justify-between items-center">
                      <span class="text-sm text-slate-600 dark:text-slate-400">{tool}</span>
                      <span class="font-medium text-slate-900 dark:text-slate-100">{count}</span>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>

<style>
  .langchain-demo {
    font-family: system-ui, -apple-system, sans-serif;
  }
</style>