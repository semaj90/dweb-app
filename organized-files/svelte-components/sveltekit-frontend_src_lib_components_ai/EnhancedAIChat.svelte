<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import { browser } from '$app/environment';
  
  import { ChatBubbleIcon, PaperPlaneIcon, MagnifyingGlassIcon, DocumentTextIcon } from '@radix-icons/svelte';
  import * as Dialog from '$lib/components/ui/dialog';
  import * as Card from '$lib/components/ui/card';
  import * as Button from '$lib/components/ui/button';
  import * as Badge from '$lib/components/ui/badge';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { Input } from '$lib/components/ui/input';
  import { Textarea } from '$lib/components/ui/textarea';
  import { chatStore, userHistoryStore, recommendationsStore } from '$lib/stores';
  import { createWebSocketConnection } from '$lib/services/websocket-client';
  import { createWebGPUAccelerator } from '$lib/services/webgpu-accelerator';
  import type { ChatMessage, MessageAnalysis, RAGContext, Recommendation } from '$lib/types/ai-chat';

  interface Props {
    caseId?: string;
    userId?: string;
    enableWebGPU?: boolean;
    enableAttentionTracking?: boolean;
    showAnalysisPanel?: boolean;
    maxMessages?: number;
  }

  let {
    caseId = '',
    userId = '',
    enableWebGPU = true,
    enableAttentionTracking = true,
    showAnalysisPanel = true,
    maxMessages = 100
  } = $props();

  // Component state
  let chatContainer: HTMLDivElement
  let messageInput: HTMLTextAreaElement
  let isConnected = $state(false);
  let isTyping = $state(false);
  let streamingResponse = $state('');
  let currentAnalysis = $state<MessageAnalysis | null>(null);
  let ragContext = $state<RAGContext | null>(null);
  let userAttention = $state({ focused: true, lastActivity: Date.now() });

  // Chat state
  let messages = $state<ChatMessage[]>([]);
  let sessionId = $state<string>('');
  let isLoading = $state(false);
  let error = $state<string | null>(null);

  // WebSocket connection
  let wsConnection: ReturnType<typeof createWebSocketConnection> | null = null;
  let webgpuAccelerator: ReturnType<typeof createWebGPUAccelerator> | null = null;

  // Enhanced AI features
  let recommendations = $state<Recommendation[]>([]);
  let didYouMean = $state<string[]>([]);
  let processingMetrics = $state({
    responseTime: 0,
    tokenCount: 0,
    confidenceScore: 0,
    somCluster: -1,
  });

  // UI state
  let inputText = $state('');
  let showRecommendations = $state(false);
  let showAnalysisDetails = $state(false);
  let selectedMessage = $state<ChatMessage | null>(null);

  // Derived values
  const lastUserMessage = $derived(messages.filter(m => m.role === 'user').slice(-1)[0] || null
  );

  const lastAIResponse = $derived(messages.filter(m => m.role === 'assistant').slice(-1)[0] || null
  );

  onMount(async () => {
    if (browser) {
      await initializeChat();
      setupAttentionTracking();
      
      if (enableWebGPU) {
        webgpuAccelerator = await createWebGPUAccelerator();
      }
    }
  });

  onDestroy(() => {
    if (wsConnection) {
      wsConnection.disconnect();
    }
    cleanupAttentionTracking();
  });

  async function initializeChat() {
    try {
      isLoading = true;
      
      // Create chat session
      const sessionResponse = await fetch('/api/chat/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, case_id: caseId }),
      });

      if (!sessionResponse.ok) {
        throw new Error('Failed to create chat session');
      }

      const session = await sessionResponse.json();
      sessionId = session.id;

      // Initialize WebSocket connection
      wsConnection = createWebSocketConnection(`/api/chat/ws/${session.id}`);
      
      wsConnection.onConnected(() => {
        isConnected = true;
        console.log('🔌 Chat WebSocket connected');
      });

      wsConnection.onDisconnected(() => {
        isConnected = false;
        console.log('🔌 Chat WebSocket disconnected');
      });

      wsConnection.onMessage((data) => {
        handleWebSocketMessage(data);
      });

      await wsConnection.connect();

      // Load chat history
      await loadChatHistory(session.id);

    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to initialize chat';
      console.error('❌ Chat initialization failed:', err);
    } finally {
      isLoading = false;
    }
  }

  async function loadChatHistory(sessionId: string) {
    try {
      const response = await fetch(`/api/chat/history/${sessionId}`);
      if (response.ok) {
        const history = await response.json();
        messages = history.messages || [];
      }
    } catch (err) {
      console.error('Failed to load chat history:', err);
    }
  }

  function handleWebSocketMessage(data: any) {
    switch (data.type) {
      case 'message_received':
        // User message was received by server
        break;

      case 'streaming_token':
        // Real-time AI response streaming
        streamingResponse += data.response.token;
        break;

      case 'streaming_complete':
        // AI response completed
        const aiMessage: ChatMessage = {
          id: data.response.message_id,
          session_id: sessionId,
          user_id: 'assistant',
          role: 'assistant',
          content: streamingResponse,
          timestamp: new Date(),
          token_count: estimateTokenCount(streamingResponse),
        };

        messages = [...messages, aiMessage];
        streamingResponse = '';
        isTyping = false;
        
        // Scroll to bottom
        scrollToBottom();
        break;

      case 'analysis_result':
        // Message analysis completed
        currentAnalysis = data.analysis;
        processingMetrics = {
          ...processingMetrics,
          confidenceScore: data.analysis.confidence,
          somCluster: data.analysis.som_cluster,
        };
        break;

      case 'rag_context':
        // RAG context updated
        ragContext = data.context;
        recommendations = data.context.recommendations || [];
        didYouMean = data.context.did_you_mean || [];
        break;

      case 'typing_indicator':
        isTyping = data.typing;
        break;

      case 'error':
        error = data.error;
        break;
    }
  }

  async function sendMessage() {
    if (!inputText.trim() || isLoading) return;

    const messageContent = inputText.trim();
    inputText = '';

    // Create user message
    const userMessage: ChatMessage = {
      id: generateId(),
      session_id: sessionId,
      user_id: userId,
      case_id: caseId,
      role: 'user',
      content: messageContent,
      timestamp: new Date(),
      token_count: estimateTokenCount(messageContent),
    };

    // Add to messages immediately for responsive UI
    messages = [...messages, userMessage];
    scrollToBottom();

    try {
      isLoading = true;
      isTyping = true;
      
      // Send via WebSocket for real-time processing
      if (wsConnection && isConnected) {
        wsConnection.send({
          type: 'chat_message',
          content: messageContent,
          user_id: userId,
          timestamp: new Date().toISOString(),
        });
      } else {
        // Fallback to HTTP API
        const response = await fetch('/api/chat/message', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            content: messageContent,
            user_id: userId,
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to send message');
        }

        const result = await response.json();
        
        // Add AI response
        if (result.ai_response) {
          messages = [...messages, result.ai_response];
          scrollToBottom();
        }

        // Update analysis and context
        if (result.user_message.analysis) {
          currentAnalysis = result.user_message.analysis;
        }

        if (result.user_message.context) {
          ragContext = result.user_message.context;
        }
      }

      // Track user attention
      trackUserActivity('message_sent', { content_length: messageContent.length });

    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to send message';
      console.error('❌ Failed to send message:', err);
    } finally {
      isLoading = false;
      if (!wsConnection || !isConnected) {
        isTyping = false;
      }
    }
  }

  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function handleInput() {
    // Send typing indicator
    if (wsConnection && $isConnected) {
      wsConnection.send({
        type: 'typing',
        user_id: userId,
        typing: inputText.length > 0,
      });
    }

    // Track typing for attention analysis
    if (enableAttentionTracking && inputText.length > 10) {
      trackUserActivity('typing', { query: inputText.slice(0, 50) });
    }
  }

  function scrollToBottom() {
    tick().then(() => {
      if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    });
  }

  function selectMessage(message: ChatMessage) {
    selectedMessage = message;
    showAnalysisDetails = true;
  }

  function applyRecommendation(recommendation: Recommendation) {
    switch (recommendation.action) {
      case 'refine_search':
        inputText = `Search for: ${recommendation.metadata?.suggested_query || ''}`;
        break;
      case 'use_template':
        inputText = `Generate using template: ${recommendation.title}`;
        break;
      case 'analyze_documents':
        inputText = 'Analyze the relevant documents and provide insights';
        break;
      case 'view_similar_cases':
        inputText = 'Show me similar cases and their outcomes';
        break;
    }
    
    messageInput?.focus();
  }

  function applyDidYouMean(suggestion: string) {
    inputText = suggestion;
    messageInput?.focus();
  }

  async function generateReport() {
    try {
      const response = await fetch('/api/chat/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          case_id: caseId,
          include_analysis: true,
          include_recommendations: true,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate report');
      }

      const report = await response.json();
      
      // Trigger download
      const blob = new Blob([report.content], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `chat-analysis-report-${new Date().toISOString().split('T')[0]}.md`;
      a.click();
      URL.revokeObjectURL(url);

    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to generate report';
    }
  }

  // Attention tracking setup
  let attentionListeners: Array<() => void> = [];

  function setupAttentionTracking() {
    if (!enableAttentionTracking || !browser) return;

    const trackEvent = (type: string, data?: any) => {
      userAttention = {
        ...userAttention,
        lastActivity: Date.now(),
      };

      if (wsConnection && isConnected) {
        wsConnection.send({
          type: 'attention',
          attention_type: type,
          data: { ...data, timestamp: Date.now() },
        });
      }
    };

    // Focus/blur tracking
    const focusHandler = () => {
      userAttention = { ...userAttention, focused: true };
      trackEvent('focus');
    };
    const blurHandler = () => {
      userAttention = { ...userAttention, focused: false };
      trackEvent('blur');
    };

    window.addEventListener('focus', focusHandler);
    window.addEventListener('blur', blurHandler);

    attentionListeners.push(
      () => window.removeEventListener('focus', focusHandler),
      () => window.removeEventListener('blur', blurHandler)
    );

    // Scroll tracking within chat container
    let scrollTimeout: number
    const scrollHandler = () => {
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        trackEvent('scroll', {
          scrollTop: chatContainer?.scrollTop,
          scrollHeight: chatContainer?.scrollHeight,
        });
      }, 100);
    };

    if (chatContainer) {
      chatContainer.addEventListener('scroll', scrollHandler);
      attentionListeners.push(() => {
        chatContainer?.removeEventListener('scroll', scrollHandler);
        clearTimeout(scrollTimeout);
      });
    }
  }

  function cleanupAttentionTracking() {
    attentionListeners.forEach(cleanup => cleanup());
    attentionListeners = [];
  }

  function trackUserActivity(type: string, data?: any) {
    if (!enableAttentionTracking) return;

    userHistoryStore.addActivity({
      type,
      timestamp: Date.now(),
      session_id: $sessionId,
      case_id: caseId,
      data,
    });
  }

  // Helper functions
  function generateId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  function estimateTokenCount(text: string): number {
    return Math.ceil(text.length / 4);
  }

  function formatTimestamp(timestamp: Date): string {
    return new Intl.DateTimeFormat('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    }).format(new Date(timestamp));
  }

  function getMessageTypeIcon(role: string) {
    switch (role) {
      case 'user':
        return ChatBubbleIcon;
      case 'assistant':
        return DocumentTextIcon;
      default:
        return ChatBubbleIcon;
    }
  }

  function getSentimentColor(sentiment: number): string {
    if (sentiment > 0.3) return 'text-green-600 dark:text-green-400';
    if (sentiment < -0.3) return 'text-red-600 dark:text-red-400';
    return 'text-gray-600 dark:text-gray-400';
  }

  function getConfidenceColor(confidence: number): string {
    if (confidence > 0.8) return 'bg-green-500';
    if (confidence > 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  }
</script>

<!-- Main Chat Interface -->
<div class="flex h-full w-full bg-background">
  <!-- Chat Messages Area -->
  <div class="flex flex-1 flex-col">
    <!-- Chat Header -->
    <div class="border-b border-border bg-card p-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="flex items-center gap-2">
            <div class="h-3 w-3 rounded-full {isConnected ? 'bg-green-500' : 'bg-red-500'}"></div>
            <h2 class="text-lg font-semibold">AI Legal Assistant</h2>
          </div>
          
          {#if isTyping}
            <Badge.Root variant="secondary" class="animate-pulse">
              <span class="flex items-center gap-1">
                <span class="h-1 w-1 rounded-full bg-current animate-bounce"></span>
                <span class="h-1 w-1 rounded-full bg-current animate-bounce [animation-delay:0.1s]"></span>
                <span class="h-1 w-1 rounded-full bg-current animate-bounce [animation-delay:0.2s]"></span>
                AI is typing...
              </span>
            </Badge.Root>
          {/if}
        </div>

        <div class="flex items-center gap-2">
          <!-- Analysis Toggle -->
          <Tooltip.Root>
            <Tooltip.Trigger asChild let:builder>
              <Button.Root
                builders={[builder]}
                variant="ghost"
                size="sm"
                class="h-8 w-8 p-0"
                onclick={() => showAnalysisPanel = !showAnalysisPanel}
              >
                <MagnifyingGlassIcon class="h-4 w-4" />
              </Button.Root>
            </Tooltip.Trigger>
            <Tooltip.Content>
              <p>{showAnalysisPanel ? 'Hide' : 'Show'} Analysis Panel</p>
            </Tooltip.Content>
          </Tooltip.Root>

          <!-- Generate Report -->
          <Button.Root variant="outline" size="sm" onclick={generateReport}>
            <DocumentTextIcon class="mr-2 h-4 w-4" />
            Report
          </Button.Root>
        </div>
      </div>

      <!-- Connection Status & Metrics -->
      {#if $processingMetrics.responseTime > 0}
        <div class="mt-2 flex gap-4 text-xs text-muted-foreground">
          <span>Response: {$processingMetrics.responseTime}ms</span>
          <span>Tokens: {$processingMetrics.tokenCount}</span>
          <span>Confidence: {Math.round($processingMetrics.confidenceScore * 100)}%</span>
          {#if $processingMetrics.somCluster >= 0}
            <span>Cluster: {$processingMetrics.somCluster}</span>
          {/if}
        </div>
      {/if}
    </div>

    <!-- Messages Container -->
    <div 
      bind:this={chatContainer}
      class="flex-1 overflow-y-auto p-4 space-y-4"
      onscroll={setupAttentionTracking}
    >
      {#each messages as message (message.id)}
        <div class="flex gap-3 {message.role === 'user' ? 'justify-end' : 'justify-start'}">
          <!-- Message Bubble -->
          <div 
            class="max-w-[80%] rounded-lg p-3 cursor-pointer transition-colors
              {message.role === 'user' 
                ? 'bg-primary text-primary-foreground ml-auto' 
                : 'bg-muted hover:bg-muted/80'}"
            onclick={() => selectMessage(message)}
            role="button"
            tabindex="0"
            onkeydown={(e) => e.key === 'Enter' && selectMessage(message)}
          >
            <!-- Message Content -->
            <div class="prose prose-sm dark:prose-invert max-w-none">
              {message.content}
            </div>

            <!-- Message Metadata -->
            <div class="mt-2 flex items-center justify-between text-xs opacity-70">
              <span>{formatTimestamp(message.timestamp)}</span>
              {#if message.token_count}
                <span>{message.token_count} tokens</span>
              {/if}
            </div>

            <!-- Analysis Preview -->
            {#if message.analysis && showAnalysisPanel}
              <div class="mt-2 pt-2 border-t border-border/50">
                <div class="flex flex-wrap gap-1">
                  {#each message.analysis.intent as intent}
                    <Badge.Root variant="outline" class="text-xs">
                      {intent}
                    </Badge.Root>
                  {/each}
                  
                  {#if message.analysis.sentiment !== undefined}
                    <Badge.Root 
                      variant="outline" 
                      class="text-xs {getSentimentColor(message.analysis.sentiment)}"
                    >
                      Sentiment: {message.analysis.sentiment > 0 ? '+' : ''}{message.analysis.sentiment.toFixed(2)}
                    </Badge.Root>
                  {/if}
                </div>
              </div>
            {/if}
          </div>
        </div>
      {/each}

      <!-- Streaming Response -->
      {#if $streamingResponse}
        <div class="flex gap-3 justify-start">
          <div class="max-w-[80%] rounded-lg p-3 bg-muted">
            <div class="prose prose-sm dark:prose-invert max-w-none">
              {$streamingResponse}<span class="animate-pulse">|</span>
            </div>
          </div>
        </div>
      {/if}

      <!-- Loading Indicator -->
      {#if $isLoading && !$streamingResponse}
        <div class="flex gap-3 justify-start">
          <div class="max-w-[80%] rounded-lg p-3 bg-muted">
            <div class="flex items-center gap-2 text-muted-foreground">
              <div class="flex gap-1">
                <div class="h-2 w-2 rounded-full bg-current animate-bounce"></div>
                <div class="h-2 w-2 rounded-full bg-current animate-bounce [animation-delay:0.1s]"></div>
                <div class="h-2 w-2 rounded-full bg-current animate-bounce [animation-delay:0.2s]"></div>
              </div>
              Thinking...
            </div>
          </div>
        </div>
      {/if}
    </div>

    <!-- Recommendations Bar -->
    {#if $recommendations.length > 0 || $didYouMean.length > 0}
      <div class="border-t border-border bg-card p-3">
        <!-- Recommendations -->
        {#if $recommendations.length > 0}
          <div class="mb-2">
            <p class="text-sm font-medium text-muted-foreground mb-2">Suggestions:</p>
            <div class="flex flex-wrap gap-2">
              {#each $recommendations.slice(0, 3) as rec}
                <Button.Root
                  variant="outline"
                  size="sm"
                  class="h-auto p-2 text-left"
                  onclick={() => applyRecommendation(rec)}
                >
                  <div>
                    <div class="font-medium text-xs">{rec.title}</div>
                    <div class="text-xs text-muted-foreground">{rec.description}</div>
                  </div>
                </Button.Root>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Did You Mean -->
        {#if $didYouMean.length > 0}
          <div>
            <p class="text-sm font-medium text-muted-foreground mb-2">Did you mean:</p>
            <div class="flex flex-wrap gap-2">
              {#each $didYouMean as suggestion}
                <Button.Root
                  variant="ghost"
                  size="sm"
                  onclick={() => applyDidYouMean(suggestion)}
                >
                  "{suggestion}"
                </Button.Root>
              {/each}
            </div>
          </div>
        {/if}
      </div>
    {/if}

    <!-- Message Input -->
    <div class="border-t border-border bg-card p-4">
      <div class="flex gap-2">
        <Textarea
          bind:this={messageInput}
          bind:value={inputText}
          placeholder="Ask your legal AI assistant..."
          class="flex-1 min-h-[40px] max-h-32 resize-none"
          onkeydown={handleKeyPress}
          oninput={handleInput}
          disabled={$isLoading}
        />
        
        <Button.Root
          size="sm"
          disabled={!inputText.trim() || $isLoading}
          onclick={sendMessage}
          class="h-10 w-10 p-0"
        >
          <PaperPlaneIcon class="h-4 w-4" />
        </Button.Root>
      </div>

      <!-- Input Helpers -->
      <div class="mt-2 flex justify-between text-xs text-muted-foreground">
        <span>Press Enter to send, Shift+Enter for new line</span>
        <span>{estimateTokenCount(inputText)} tokens</span>
      </div>
    </div>
  </div>

  <!-- Analysis Side Panel -->
  {#if showAnalysisPanel}
    <div class="w-80 border-l border-border bg-card">
      <div class="h-full flex flex-col">
        <!-- Panel Header -->
        <div class="border-b border-border p-4">
          <h3 class="font-semibold">Analysis</h3>
        </div>

        <!-- Analysis Content -->
        <div class="flex-1 overflow-y-auto p-4 space-y-4">
          <!-- Current Analysis -->
          {#if $currentAnalysis}
            <Card.Root>
              <Card.Header>
                <Card.Title class="text-sm">Message Analysis</Card.Title>
              </Card.Header>
              <Card.Content class="space-y-3">
                <!-- Intent -->
                {#if $currentAnalysis.intent.length > 0}
                  <div>
                    <p class="text-xs font-medium text-muted-foreground mb-1">Intent:</p>
                    <div class="flex flex-wrap gap-1">
                      {#each $currentAnalysis.intent as intent}
                        <Badge.Root variant="secondary" class="text-xs">{intent}</Badge.Root>
                      {/each}
                    </div>
                  </div>
                {/if}

                <!-- Entities -->
                {#if $currentAnalysis.entities.length > 0}
                  <div>
                    <p class="text-xs font-medium text-muted-foreground mb-1">Entities:</p>
                    <div class="space-y-1">
                      {#each $currentAnalysis.entities as entity}
                        <div class="flex justify-between items-center text-xs">
                          <span class="font-medium">{entity.text}</span>
                          <Badge.Root variant="outline" class="text-xs">{entity.type}</Badge.Root>
                        </div>
                      {/each}
                    </div>
                  </div>
                {/if}

                <!-- Metrics -->
                <div class="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <p class="text-muted-foreground">Sentiment:</p>
                    <p class="font-medium {getSentimentColor($currentAnalysis.sentiment)}">
                      {$currentAnalysis.sentiment.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p class="text-muted-foreground">Confidence:</p>
                    <div class="flex items-center gap-1">
                      <div class="w-8 h-1 bg-muted rounded-full overflow-hidden">
                        <div 
                          class="h-full {getConfidenceColor($currentAnalysis.confidence)}"
                          style="width: {$currentAnalysis.confidence * 100}%"
                        ></div>
                      </div>
                      <span class="font-medium">{Math.round($currentAnalysis.confidence * 100)}%</span>
                    </div>
                  </div>
                </div>
              </Card.Content>
            </Card.Root>
          {/if}

          <!-- RAG Context -->
          {#if $ragContext}
            <Card.Root>
              <Card.Header>
                <Card.Title class="text-sm">Context</Card.Title>
              </Card.Header>
              <Card.Content class="space-y-3">
                <!-- Relevant Documents -->
                {#if $ragContext.relevant_docs.length > 0}
                  <div>
                    <p class="text-xs font-medium text-muted-foreground mb-2">Relevant Documents:</p>
                    <div class="space-y-2">
                      {#each $ragContext.relevant_docs.slice(0, 3) as doc}
                        <div class="p-2 bg-muted rounded text-xs">
                          <p class="font-medium">{doc.title}</p>
                          <p class="text-muted-foreground mt-1">{doc.content.slice(0, 100)}...</p>
                          <div class="flex justify-between items-center mt-1">
                            <span class="text-muted-foreground">Relevance:</span>
                            <span class="font-medium">{Math.round(doc.relevance * 100)}%</span>
                          </div>
                        </div>
                      {/each}
                    </div>
                  </div>
                {/if}

                <!-- User History Summary -->
                {#if $ragContext.user_history.length > 0}
                  <div>
                    <p class="text-xs font-medium text-muted-foreground mb-1">Recent Activity:</p>
                    <p class="text-xs text-muted-foreground">
                      {$ragContext.user_history.length} recent messages considered
                    </p>
                  </div>
                {/if}
              </Card.Content>
            </Card.Root>
          {/if}

          <!-- Processing Metrics -->
          <Card.Root>
            <Card.Header>
              <Card.Title class="text-sm">Performance</Card.Title>
            </Card.Header>
            <Card.Content class="space-y-2 text-xs">
              <div class="flex justify-between">
                <span class="text-muted-foreground">Response Time:</span>
                <span class="font-medium">{$processingMetrics.responseTime}ms</span>
              </div>
              <div class="flex justify-between">
                <span class="text-muted-foreground">Tokens Processed:</span>
                <span class="font-medium">{$processingMetrics.tokenCount}</span>
              </div>
              {#if webgpuAccelerator}
                <div class="flex justify-between">
                  <span class="text-muted-foreground">WebGPU:</span>
                  <Badge.Root variant="secondary" class="text-xs">Enabled</Badge.Root>
                </div>
              {/if}
            </Card.Content>
          </Card.Root>
        </div>
      </div>
    </div>
  {/if}
</div>

<!-- Message Analysis Dialog -->
{#if selectedMessage}
  <Dialog.Root bind:open={showAnalysisDetails}>
    <Dialog.Content class="max-w-2xl">
      <Dialog.Header>
        <Dialog.Title>Message Analysis Details</Dialog.Title>
        <Dialog.Description>
          Comprehensive analysis of the selected message
        </Dialog.Description>
      </Dialog.Header>

      <div class="space-y-4">
        <!-- Message Content -->
        <div class="p-3 bg-muted rounded">
          <p class="text-sm font-medium mb-2">Message:</p>
          <p class="text-sm">{selectedMessage.content}</p>
          <p class="text-xs text-muted-foreground mt-2">
            {formatTimestamp(selectedMessage.timestamp)} • {selectedMessage.token_count} tokens
          </p>
        </div>

        <!-- Detailed Analysis -->
        {#if selectedMessage.analysis}
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Intent & Topics -->
            <div>
              <h4 class="text-sm font-medium mb-2">Intent & Topics</h4>
              <div class="space-y-2">
                <div>
                  <p class="text-xs text-muted-foreground">Intent:</p>
                  <div class="flex flex-wrap gap-1">
                    {#each selectedMessage.analysis.intent as intent}
                      <Badge.Root variant="secondary">{intent}</Badge.Root>
                    {/each}
                  </div>
                </div>
                
                {#if selectedMessage.analysis.topics.length > 0}
                  <div>
                    <p class="text-xs text-muted-foreground">Topics:</p>
                    <div class="flex flex-wrap gap-1">
                      {#each selectedMessage.analysis.topics as topic}
                        <Badge.Root variant="outline">{topic}</Badge.Root>
                      {/each}
                    </div>
                  </div>
                {/if}
              </div>
            </div>

            <!-- Metrics -->
            <div>
              <h4 class="text-sm font-medium mb-2">Analysis Metrics</h4>
              <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                  <span>Sentiment:</span>
                  <span class="{getSentimentColor(selectedMessage.analysis.sentiment)}">
                    {selectedMessage.analysis.sentiment.toFixed(3)}
                  </span>
                </div>
                <div class="flex justify-between">
                  <span>Confidence:</span>
                  <span>{Math.round(selectedMessage.analysis.confidence * 100)}%</span>
                </div>
                <div class="flex justify-between">
                  <span>SOM Cluster:</span>
                  <span>{selectedMessage.analysis.som_cluster}</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Entities -->
          {#if selectedMessage.analysis.entities.length > 0}
            <div>
              <h4 class="text-sm font-medium mb-2">Extracted Entities</h4>
              <div class="space-y-2">
                {#each selectedMessage.analysis.entities as entity}
                  <div class="flex justify-between items-center p-2 bg-muted rounded text-sm">
                    <div>
                      <span class="font-medium">{entity.text}</span>
                      <span class="text-muted-foreground ml-2">({entity.type})</span>
                    </div>
                    <span class="text-muted-foreground">{Math.round(entity.confidence * 100)}%</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}

          <!-- Legal Concepts -->
          {#if selectedMessage.analysis.legal_concepts.length > 0}
            <div>
              <h4 class="text-sm font-medium mb-2">Legal Concepts</h4>
              <div class="flex flex-wrap gap-2">
                {#each selectedMessage.analysis.legal_concepts as concept}
                  <Badge.Root variant="secondary">{concept}</Badge.Root>
                {/each}
              </div>
            </div>
          {/if}
        {/if}
      </div>

      <Dialog.Footer>
        <Button.Root variant="outline" onclick={() => showAnalysisDetails = false}>
          Close
        </Button.Root>
      </Dialog.Footer>
    </Dialog.Content>
  </Dialog.Root>
{/if}

<!-- Error Toast -->
{#if $error}
  <div class="fixed bottom-4 right-4 bg-destructive text-destructive-foreground p-3 rounded-lg shadow-lg">
    <p class="text-sm font-medium">Error</p>
    <p class="text-sm">{$error}</p>
    <Button.Root 
      variant="ghost" 
      size="sm" 
      class="mt-2"
      onclick={() => error = null}
    >
      Dismiss
    </Button.Root>
  </div>
{/if}

<style>
  .prose {
    @apply text-gray-900 dark:text-gray-100;
  }
  
  .prose h1, .prose h2, .prose h3, .prose h4, .prose h5, .prose h6 {
    @apply text-gray-900 dark:text-gray-100;
  }
  
  .prose code {
    @apply bg-muted px-1 py-0.5 rounded text-sm;
  }
  
  .prose pre {
    @apply bg-muted p-3 rounded-lg overflow-x-auto;
  }
  
  .prose blockquote {
    @apply border-l-4 border-border pl-4 italic;
  }
  
  .prose ul, .prose ol {
    @apply ml-4;
  }
  
  .prose a {
    @apply text-primary hover: underline
  }
</style>