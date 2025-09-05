<!-- ContextualBVectorChat.svelte - Enhanced AI Chat with BVector Store Integration -->
<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import { createEnhancedBVectorStore, type EnhancedBVectorStore, type SearchResult } from '$lib/services/enhanced-bvector-store';
  import { ContextualRLValidator, type ValidationResult } from '$lib/ai/contextual-rl-validator';
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/Card';
  import { Badge } from '$lib/components/ui/badge';
  import { ScrollArea } from '$lib/components/ui/scroll-area';
  import { Input } from '$lib/components/ui/input';
  import { 
    Brain, 
    Zap, 
    Database, 
    Cpu, 
    MessageSquare, 
    Loader2, 
    ThumbsUp, 
    ThumbsDown,
    Settings,
    TrendingUp,
    TestTube,
    CheckCircle2,
    AlertTriangle
  } from 'lucide-svelte';

  // Props with Svelte 5 patterns
  interface Props {
    userId: string;
    caseId?: string;
    userRole: 'prosecutor' | 'detective' | 'admin';
    className?: string;
  }

  let { 
    userId, 
    caseId = undefined, 
    userRole = 'admin', 
    className = '' 
  }: Props = $props();

  // Reactive state using Svelte 5 runes
  let message = $state('');
  let isLoading = $state(false);
  let showContextualResults = $state(true);
  let showMetrics = $state(false);
  let enableReinforcementLearning = $state(true);
  let useGPUAcceleration = $state(true);

  let chatHistory = $state<Array<{
    id: string;
    type: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    contextualResults?: SearchResult[];
    confidence?: number;
    processingTime?: number;
  }>>([]);

  let contextualResults = $state<SearchResult[]>([]);
  let bvectorStore: EnhancedBVectorStore | null = null;
  let storeMetrics = $state<any>(null);
  let lastUserFeedback = $state<{ messageId: string; rating: number } | null>(null);
  
  // Validation system state
  let validator: ContextualRLValidator | null = null;
  let showValidationPanel = $state(false);
  let isRunningValidation = $state(false);
  let validationResults = $state<ValidationResult[]>([]);
  let validationProgress = $state(0);

  const dispatch = createEventDispatcher();

  // Reactive computations
  const canSend = $derived(message.trim().length > 0 && !isLoading && bvectorStore !== null);
  const messageCount = $derived(chatHistory.length);
  const averageConfidence = $derived(
    chatHistory.length > 0 
      ? chatHistory.filter(m => m.confidence).reduce((sum, m) => sum + (m.confidence || 0), 0) / chatHistory.filter(m => m.confidence).length
      : 0
  );

  // Initialize BVector store
  onMount(async () => {
    try {
      bvectorStore = createEnhancedBVectorStore({
        reinforcementLearning: {
          enabled: enableReinforcementLearning,
          learningRate: 0.015,
          decayFactor: 0.95,
          feedbackThreshold: 3.5
        },
        gpuCache: {
          layers: 4,
          maxMemoryMB: 6144,
          batchSize: useGPUAcceleration ? 32 : 8,
          enableQuantization: true
        }
      });

      // Load user's conversation history for contextual training
      await loadUserHistory();

      // Start metrics polling
      setInterval(updateMetrics, 10000); // Every 10 seconds

      // Initialize validation system
      validator = new ContextualRLValidator(bvectorStore);

      addSystemMessage('🚀 Enhanced BVector Store initialized with GPU acceleration and reinforcement learning');
    } catch (error) {
      console.error('Failed to initialize BVector store:', error);
      addSystemMessage('❌ Failed to initialize enhanced AI system. Using fallback mode.');
    }
  });

  async function loadUserHistory(): Promise<void> {
    if (!bvectorStore) return;

    try {
      // Search for user's recent conversations to build context
      const recentContext = await bvectorStore.search('user conversation history', {
        userId,
        caseId,
        userRole,
        limit: 50,
        threshold: 0.3,
        useReinforcementLearning: false,
        enableGPUAcceleration: useGPUAcceleration
      });

      if (recentContext.length > 0) {
        addSystemMessage(`📚 Loaded ${recentContext.length} contextual conversations for personalized responses`);
      }
    } catch (error) {
      console.warn('Failed to load user history:', error);
    }
  }

  async function sendMessage(): Promise<void> {
    if (!canSend || !bvectorStore) return;

    const userMessage = message.trim();
    const messageId = Date.now().toString();
    const startTime = Date.now();

    // Add user message
    chatHistory.push({
      id: messageId,
      type: 'user',
      content: userMessage,
      timestamp: new Date()
    });

    message = '';
    isLoading = true;

    try {
      // Store user message in BVector store for future context
      await bvectorStore.store({
        id: messageId,
        content: userMessage,
        embedding: [], // Will be generated by store
        metadata: {
          userId,
          caseId,
          timestamp: Date.now(),
          conversationId: `conv_${Date.now()}`,
          userRole,
          intent: await classifyIntent(userMessage),
          confidence: 0.8
        }
      });

      // Search for contextual information
      const contextResults = await bvectorStore.search(userMessage, {
        userId,
        caseId,
        userRole,
        limit: 5,
        threshold: 0.7,
        useReinforcementLearning: enableReinforcementLearning,
        enableGPUAcceleration: useGPUAcceleration,
        contextualPrompting: true
      });

      contextualResults = contextResults;

      // Generate contextual prompt
      const contextualPrompt = buildContextualPrompt(userMessage, contextResults);

      // Send to AI service (Ollama with contextual enhancement)
      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: contextualPrompt,
          model: 'gemma3-legal',
          useRAG: true,
          caseId,
          userContext: {
            userId,
            userRole,
            conversationHistory: chatHistory.slice(-5).map(m => m.content),
            contextualResults: contextResults.map(r => r.content).slice(0, 3)
          }
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      const processingTime = Date.now() - startTime;

      // Add assistant response
      const assistantMessageId = Date.now().toString() + '_assistant';
      chatHistory.push({
        id: assistantMessageId,
        type: 'assistant',
        content: data.response,
        timestamp: new Date(),
        contextualResults: showContextualResults ? contextResults : undefined,
        confidence: calculateResponseConfidence(data, contextResults),
        processingTime
      });

      // Store assistant response for future context
      await bvectorStore.store({
        id: assistantMessageId,
        content: data.response,
        embedding: [],
        metadata: {
          userId,
          caseId,
          timestamp: Date.now(),
          conversationId: `conv_${Date.now()}`,
          userRole,
          confidence: calculateResponseConfidence(data, contextResults),
          rlMetrics: {
            responseQuality: 0.8,
            contextRelevance: contextResults.length > 0 ? 0.9 : 0.5
          }
        }
      });

      dispatch('message-sent', {
        userMessage,
        response: data.response,
        contextResults: contextResults.length,
        processingTime
      });

    } catch (error) {
      console.error('Chat error:', error);
      
      chatHistory.push({
        id: Date.now().toString() + '_error',
        type: 'system',
        content: `❌ Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
        timestamp: new Date()
      });
    } finally {
      isLoading = false;
    }
  }

  async function classifyIntent(message: string): Promise<string> {
    // Simple intent classification - in production, this could use ML
    const intents = {
      question: /\b(what|how|why|when|where|who)\b/i,
      request: /\b(please|can you|could you|would you)\b/i,
      search: /\b(find|search|look for|show me)\b/i,
      analysis: /\b(analyze|examine|review|assess)\b/i,
      legal: /\b(law|legal|court|case|statute|regulation)\b/i
    };

    for (const [intent, pattern] of Object.entries(intents)) {
      if (pattern.test(message)) {
        return intent;
      }
    }

    return 'general';
  }

  function buildContextualPrompt(userMessage: string, contextResults: SearchResult[]): string {
    if (contextResults.length === 0) {
      return userMessage;
    }

    const contextText = contextResults
      .slice(0, 3)
      .map(r => `- ${r.content} (relevance: ${(r.similarity * 100).toFixed(1)}%)`)
      .join('\n');

    return `Based on your previous conversations and legal knowledge:

Context from your history:
${contextText}

Current question: ${userMessage}

Please provide a contextual response that takes into account the relevant information from your conversation history while addressing the current question.`;
  }

  function calculateResponseConfidence(aiResponse: any, contextResults: SearchResult[]): number {
    let confidence = 0.7; // Base confidence

    // Boost if we had good contextual results
    if (contextResults.length > 0) {
      const avgSimilarity = contextResults.reduce((sum, r) => sum + r.similarity, 0) / contextResults.length;
      confidence += avgSimilarity * 0.3;
    }

    // Factor in AI model confidence if available
    if (aiResponse.confidence) {
      confidence = (confidence + aiResponse.confidence) / 2;
    }

    return Math.min(confidence, 1.0);
  }

  async function provideFeedback(messageId: string, rating: number): Promise<void> {
    lastUserFeedback = { messageId, rating };

    if (!bvectorStore) return;

    // Find the message and update RL metrics
    const messageIndex = chatHistory.findIndex(m => m.id === messageId);
    if (messageIndex === -1) return;

    const message = chatHistory[messageIndex];
    
    // Update reinforcement learning based on feedback
    if (message.type === 'assistant') {
      await bvectorStore.store({
        id: messageId + '_feedback',
        content: message.content,
        embedding: [],
        metadata: {
          userId,
          caseId,
          timestamp: Date.now(),
          userRole,
          confidence: message.confidence || 0.7,
          rlMetrics: {
            userSatisfaction: rating,
            responseQuality: rating / 5,
            followUpSuccess: rating > 3,
            contextRelevance: message.contextualResults ? 0.9 : 0.5
          }
        }
      });

      addSystemMessage(
        rating > 3 
          ? '👍 Positive feedback recorded. AI will learn from this interaction.'
          : '👎 Negative feedback recorded. AI will adjust future responses.'
      );
    }
  }

  function addSystemMessage(content: string): void {
    chatHistory.push({
      id: Date.now().toString() + '_system',
      type: 'system',
      content,
      timestamp: new Date()
    });
  }

  async function updateMetrics(): Promise<void> {
    if (!bvectorStore) return;

    try {
      storeMetrics = await bvectorStore.getMetrics();
    } catch (error) {
      console.warn('Failed to update metrics:', error);
    }
  }

  async function optimizeMemory(): Promise<void> {
    if (!bvectorStore) return;

    try {
      await bvectorStore.optimizeMemory();
      addSystemMessage('🧹 Memory optimization completed');
    } catch (error) {
      console.error('Memory optimization failed:', error);
    }
  }

  function handleKeyPress(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  // Validation system functions
  async function runContextualValidation(): Promise<void> {
    if (!validator || isRunningValidation) return;

    isRunningValidation = true;
    validationProgress = 0;
    
    try {
      addSystemMessage('🧪 Starting contextual prompting validation tests...');
      
      const contextualTests = ContextualRLValidator.getStandardContextualTests();
      const contextualResults = await validator.validateContextualPrompting(contextualTests);
      
      validationProgress = 50;
      addSystemMessage('🧠 Starting reinforcement learning validation tests...');
      
      const rlTests = ContextualRLValidator.getStandardRLTests();
      const rlResults = await validator.validateReinforcementLearning(rlTests);
      
      validationResults = [...contextualResults, ...rlResults];
      validationProgress = 100;
      
      const successRate = (validationResults.filter(r => r.success).length / validationResults.length) * 100;
      
      addSystemMessage(`✅ Validation completed: ${successRate.toFixed(1)}% success rate (${validationResults.filter(r => r.success).length}/${validationResults.length} tests passed)`);
      
      if (successRate >= 80) {
        addSystemMessage('🎉 Contextual AI system performing excellently!');
      } else if (successRate >= 60) {
        addSystemMessage('⚠️ Contextual AI system needs optimization');
      } else {
        addSystemMessage('❌ Contextual AI system requires immediate attention');
      }
      
    } catch (error) {
      addSystemMessage(`❌ Validation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      isRunningValidation = false;
    }
  }

  async function runQuickValidation(): Promise<void> {
    if (!validator || !bvectorStore || isRunningValidation) return;

    isRunningValidation = true;
    
    try {
      addSystemMessage('⚡ Running quick validation test...');
      
      // Quick contextual test
      const testQuery = 'contract liability terms';
      const baselineResults = await bvectorStore.search(testQuery, { topK: 5 });
      const contextualResults = await bvectorStore.contextualSearch(testQuery, {
        userContext: {
          userId,
          userRole,
          currentCase: caseId || 'test-case',
          workflowStage: 'evidence-review',
          recentQueries: chatHistory.filter(m => m.type === 'user').slice(-3).map(m => m.content)
        },
        topK: 5,
        enableRL: enableReinforcementLearning
      });
      
      const contextualBoost = contextualResults.reduce((sum, result) => sum + (result.contextualBoost || 0), 0) / contextualResults.length;
      const averageConfidenceImprovement = contextualResults.reduce((sum, result) => sum + result.metadata.confidence, 0) / contextualResults.length - 
                                         baselineResults.reduce((sum, result) => sum + result.metadata.confidence, 0) / baselineResults.length;
      
      addSystemMessage(`📊 Quick test results: ${contextualBoost.toFixed(3)} avg contextual boost, ${averageConfidenceImprovement.toFixed(3)} confidence improvement`);
      
    } catch (error) {
      addSystemMessage(`❌ Quick validation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      isRunningValidation = false;
    }
  }

  function getValidationStatusIcon(result: ValidationResult) {
    if (result.success) return CheckCircle2;
    return AlertTriangle;
  }

  function getValidationBadgeVariant(result: ValidationResult): 'default' | 'destructive' {
    return result.success ? 'default' : 'destructive';
  }
</script>

<!-- Enhanced Chat Interface -->
<div class="contextual-bvector-chat {className}">
  <!-- Header with Status and Controls -->
  <Card class="mb-4">
    <CardHeader class="pb-3">
      <div class="flex items-center justify-between">
        <CardTitle class="flex items-center gap-2">
          <Brain class="w-5 h-5 text-purple-500" />
          Enhanced Contextual AI
          {#if enableReinforcementLearning}
            <Badge variant="secondary" class="text-xs">
              <TrendingUp class="w-3 h-3 mr-1" />
              RL Enabled
            </Badge>
          {/if}
          {#if useGPUAcceleration}
            <Badge variant="secondary" class="text-xs">
              <Cpu class="w-3 h-3 mr-1" />
              GPU
            </Badge>
          {/if}
        </CardTitle>

        <div class="flex items-center gap-2">
          <!-- Validation Panel Toggle -->
          <Button
            variant="ghost"
            size="sm"
            on:click={() => (showValidationPanel = !showValidationPanel)}
            disabled={!validator}
          >
            <TestTube class="w-4 h-4" />
          </Button>

          <!-- Quick Validation -->
          <Button
            variant="ghost"
            size="sm"
            on:click={runQuickValidation}
            disabled={!validator || isRunningValidation}
          >
            ⚡
          </Button>

          <!-- Settings Toggle -->
          <Button
            variant="ghost"
            size="sm"
            on:click={() => (showMetrics = !showMetrics)}
          >
            <Settings class="w-4 h-4" />
          </Button>

          <!-- Memory Optimization -->
          <Button
            variant="ghost"
            size="sm"
            on:click={optimizeMemory}
            disabled={!bvectorStore}
          >
            🧹
          </Button>
        </div>
      </div>

      <!-- Performance Metrics -->
      {#if showMetrics && storeMetrics}
        <div class="mt-3 grid grid-cols-4 gap-4 text-xs">
          <div class="text-center">
            <div class="font-semibold">{storeMetrics.totalQueries}</div>
            <div class="text-muted-foreground">Queries</div>
          </div>
          <div class="text-center">
            <div class="font-semibold">{Math.round(storeMetrics.averageLatency)}ms</div>
            <div class="text-muted-foreground">Avg Latency</div>
          </div>
          <div class="text-center">
            <div class="font-semibold">{storeMetrics.gpuAccelerated}</div>
            <div class="text-muted-foreground">GPU Queries</div>
          </div>
          <div class="text-center">
            <div class="font-semibold">{(averageConfidence * 100).toFixed(1)}%</div>
            <div class="text-muted-foreground">Confidence</div>
          </div>
        </div>
      {/if}

      <!-- Validation Panel -->
      {#if showValidationPanel}
        <div class="mt-3 p-4 bg-gray-50 rounded-lg border">
          <div class="flex items-center justify-between mb-3">
            <h4 class="font-semibold text-sm flex items-center gap-2">
              <TestTube class="w-4 h-4" />
              Contextual AI Validation
            </h4>
            <div class="flex gap-2">
              <Button
                size="sm"
                on:click={runContextualValidation}
                disabled={isRunningValidation}
              >
                {#if isRunningValidation}
                  <Loader2 class="w-3 h-3 mr-1 animate-spin" />
                {:else}
                  <TestTube class="w-3 h-3 mr-1" />
                {/if}
                Full Test Suite
              </Button>
            </div>
          </div>

          {#if isRunningValidation}
            <div class="mb-3">
              <div class="flex items-center justify-between text-xs mb-1">
                <span>Running validation tests...</span>
                <span>{validationProgress}%</span>
              </div>
              <div class="w-full bg-gray-200 rounded-full h-1.5">
                <div 
                  class="bg-blue-600 h-1.5 rounded-full transition-all duration-300" 
                  style="width: {validationProgress}%"
                ></div>
              </div>
            </div>
          {/if}

          {#if validationResults.length > 0}
            <div class="space-y-2">
              <h5 class="font-medium text-sm">Test Results ({validationResults.filter(r => r.success).length}/{validationResults.length} passed)</h5>
              <div class="max-h-32 overflow-y-auto space-y-1">
                {#each validationResults as result}
                  <div class="flex items-center justify-between p-2 bg-white rounded border text-xs">
                    <div class="flex items-center gap-2">
                      <svelte:component 
                        this={getValidationStatusIcon(result)} 
                        class="w-3 h-3 {result.success ? 'text-green-600' : 'text-red-600'}"
                      />
                      <span class="font-medium">{result.testId.replace(/-/g, ' ')}</span>
                    </div>
                    <div class="flex items-center gap-2">
                      <Badge variant={getValidationBadgeVariant(result)} class="text-xs">
                        {(result.score * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      {/if}
    </CardHeader>
  </Card>

  <!-- Chat History -->
  <Card class="flex-1 mb-4">
    <ScrollArea class="h-96 p-4">
      {#if chatHistory.length === 0}
        <div class="text-center text-muted-foreground py-8">
          <MessageSquare class="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Start a conversation with enhanced contextual AI</p>
          <p class="text-sm mt-2">
            Your conversations are stored and used for better contextual responses
          </p>
        </div>
      {:else}
        {#each chatHistory as msg}
          <div class="mb-6 {msg.type === 'user' ? 'text-right' : 'text-left'}">
            <div
              class="inline-block max-w-[85%] {
                msg.type === 'user'
                  ? 'bg-blue-600 text-white rounded-l-lg rounded-br-lg'
                  : msg.type === 'system'
                  ? 'bg-yellow-100 text-yellow-800 rounded-lg border border-yellow-200'
                  : 'bg-gray-100 text-gray-900 rounded-r-lg rounded-bl-lg'
              } px-4 py-3"
            >
              <div class="whitespace-pre-wrap">{msg.content}</div>

              <!-- Message Metadata -->
              {#if msg.confidence !== undefined || msg.processingTime}
                <div class="text-xs opacity-70 mt-2 pt-2 border-t border-current/20">
                  {#if msg.confidence !== undefined}
                    Confidence: {(msg.confidence * 100).toFixed(1)}%
                  {/if}
                  {#if msg.processingTime}
                    • {msg.processingTime}ms
                  {/if}
                </div>
              {/if}

              <!-- Contextual Results Preview -->
              {#if msg.contextualResults && msg.contextualResults.length > 0}
                <details class="mt-3 text-xs">
                  <summary class="cursor-pointer opacity-70">
                    📚 {msg.contextualResults.length} contextual references
                  </summary>
                  <div class="mt-2 space-y-1">
                    {#each msg.contextualResults.slice(0, 3) as result}
                      <div class="p-2 bg-black/10 rounded text-xs">
                        <div class="font-medium">
                          Similarity: {(result.similarity * 100).toFixed(1)}%
                        </div>
                        <div class="mt-1 opacity-80">
                          {result.content.substring(0, 100)}...
                        </div>
                      </div>
                    {/each}
                  </div>
                </details>
              {/if}
            </div>

            <!-- Message Actions (for assistant messages) -->
            {#if msg.type === 'assistant'}
              <div class="mt-2 flex items-center gap-2 text-xs">
                <Button
                  variant="ghost"
                  size="sm"
                  class="h-6 px-2"
                  on:click={() => provideFeedback(msg.id, 5)}
                  disabled={lastUserFeedback?.messageId === msg.id}
                >
                  <ThumbsUp class="w-3 h-3" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  class="h-6 px-2"
                  on:click={() => provideFeedback(msg.id, 2)}
                  disabled={lastUserFeedback?.messageId === msg.id}
                >
                  <ThumbsDown class="w-3 h-3" />
                </Button>
                {#if lastUserFeedback?.messageId === msg.id}
                  <span class="text-green-600 text-xs">Feedback recorded</span>
                {/if}
              </div>
            {/if}

            <!-- Timestamp -->
            <div class="text-xs text-muted-foreground mt-1">
              {msg.timestamp.toLocaleTimeString()}
            </div>
          </div>
        {/each}
      {/if}

      <!-- Loading Indicator -->
      {#if isLoading}
        <div class="flex items-center gap-2 text-muted-foreground">
          <Loader2 class="w-4 h-4 animate-spin" />
          <span>AI is analyzing context and generating response...</span>
        </div>
      {/if}
    </ScrollArea>
  </Card>

  <!-- Contextual Results Sidebar -->
  {#if showContextualResults && contextualResults.length > 0}
    <Card class="mb-4">
      <CardHeader class="pb-2">
        <CardTitle class="text-sm flex items-center gap-2">
          <Database class="w-4 h-4" />
          Contextual References ({contextualResults.length})
        </CardTitle>
      </CardHeader>
      <CardContent class="space-y-2">
        {#each contextualResults.slice(0, 5) as result}
          <div class="p-3 bg-muted/50 rounded-lg text-sm">
            <div class="flex items-center justify-between mb-1">
              <Badge variant="outline" class="text-xs">
                {(result.similarity * 100).toFixed(1)}% match
              </Badge>
              {#if result.rlWeight > 0}
                <Badge variant="secondary" class="text-xs">
                  <Zap class="w-3 h-3 mr-1" />
                  RL: +{(result.rlWeight * 100).toFixed(0)}%
                </Badge>
              {/if}
            </div>
            <p class="text-xs opacity-80 line-clamp-3">
              {result.content.substring(0, 150)}...
            </p>
          </div>
        {/each}
      </CardContent>
    </Card>
  {/if}

  <!-- Input Area -->
  <div class="flex gap-2">
    <div class="flex-1">
      <Input
        bind:value={message}
        placeholder="Ask the enhanced contextual AI..."
        onkeypress={handleKeyPress}
        disabled={isLoading || !bvectorStore}
        class="pr-12"
      />
    </div>

    <Button
      on:click={sendMessage}
      disabled={!canSend}
      class="px-4"
    >
      {#if isLoading}
        <Loader2 class="w-4 h-4 animate-spin" />
      {:else}
        <Brain class="w-4 h-4" />
      {/if}
    </Button>
  </div>

  <!-- Stats Footer -->
  {#if messageCount > 0}
    <div class="mt-4 text-xs text-muted-foreground text-center">
      {messageCount} messages • {contextualResults.length} contextual refs
      • User: {userRole}
      {#if caseId}
        • Case: {caseId}
      {/if}
    </div>
  {/if}
</div>

<style>
  .contextual-bvector-chat {
    display: flex;
    flex-direction: column;
    height: 100%;
    max-width: 64rem;
    margin-left: auto;
    margin-right: auto;
  }

  .line-clamp-3 {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>