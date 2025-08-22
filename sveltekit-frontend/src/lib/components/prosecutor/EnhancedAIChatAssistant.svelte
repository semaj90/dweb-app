<!--
Enhanced AI Chat Assistant for Prosecutors
Features: Self-prompting, elemental awareness (YOLO), enhanced RAG, local LLM
-->
<script lang="ts">
  import { $props, $state } from 'svelte';

  import type { Props } from "$lib/types/global";
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Badge } from '$lib/components/ui/badge';
  import { webGPUProcessor } from '$lib/services/webgpu-vector-processor';
  import { 
    Bot, 
    User, 
    Send, 
    Brain, 
    Eye, 
    Zap, 
    Search,
    FileText,
    Scale,
    AlertTriangle
  } from 'lucide-svelte';
  import { onMount } from 'svelte';


  let {
    caseId,
    enableSelfPrompting = true,
    enableElementalAwareness = true,
    enableEnhancedRAG = true
  }: Props = $props();

  // Chat state
  let messages: any[] = $state([]);
  let currentMessage = $state('');
  let isTyping = $state(false);
  let hoveredElement: string | null = $state(null);
  let elementAnalysis: any = $state(null);

  // AI capabilities
  let ragSources: any[] = $state([]);
  let aiConfidence = $state(0);
  let selfPromptSuggestions: string[] = $state([]);

  // Initialize chat with prosecutor context
  onMount(() => {
    if (enableSelfPrompting) {
      generateSelfPromptSuggestions();
    }
    
    // Add welcome message
    messages = [{
      id: crypto.randomUUID(),
      role: 'assistant',
      content: `Hello! I'm your AI legal assistant powered by Gemma3Legal. I can help you with:

• Evidence analysis and correlation
• Legal precedent research  
• Case strategy recommendations
• Document synthesis and review
• Timeline reconstruction

${caseId ? `I'm ready to assist with Case ${caseId}.` : 'Select a case to get started with case-specific insights.'}`,
      timestamp: new Date(),
      metadata: {
        model: 'gemma3-legal:latest',
        confidence: 1.0,
        capabilities: ['evidence_analysis', 'legal_research', 'case_strategy']
      }
    }];
  });

  // Self-prompting system
  const generateSelfPromptSuggestions = async () => {
    if (!caseId) return;
    
    try {
      const response = await fetch('/api/ai/self-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          caseId,
          context: 'prosecutor_workflow',
          currentPhase: 'evidence_review'
        })
      });
      
      const result = await response.json();
      selfPromptSuggestions = result.suggestions || [
        "Analyze evidence strength for this case",
        "Find similar cases with comparable evidence",
        "Identify potential defense arguments",
        "Review timeline for inconsistencies"
      ];
    } catch (error) {
      console.error('Self-prompt generation failed:', error);
    }
  };

  // Elemental awareness (YOLO-style hover analysis)
  const handleElementHover = async (event: MouseEvent) => {
    if (!enableElementalAwareness) return;
    
    const target = event.target as HTMLElement;
    const elementType = target.tagName.toLowerCase();
    const elementText = target.textContent?.substring(0, 100) || '';
    
    if (elementText.length < 3) return;
    
    hoveredElement = elementType;
    
    // Analyze element with AI for legal relevance
    try {
      const response = await fetch('/api/ai/analyze-element', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          elementType,
          content: elementText,
          context: 'legal_analysis'
        })
      });
      
      const analysis = await response.json();
      elementAnalysis = analysis;
    } catch (error) {
      console.error('Element analysis failed:', error);
    }
  };

  // Enhanced RAG chat with vector search
  const sendMessage = async () => {
    if (!currentMessage.trim()) return;
    
    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: currentMessage,
      timestamp: new Date()
    };
    
    messages = [...messages, userMessage];
    const userQuery = currentMessage;
    currentMessage = '';
    isTyping = true;
    ragSources = [];

    try {
      // Enhanced RAG query with vector search
      if (enableEnhancedRAG) {
        const ragResponse = await fetch('/api/enhanced-rag/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: userQuery,
            caseId,
            includeEvidence: true,
            includePrecedents: true,
            vectorSearch: true
          })
        });
        
        const ragResult = await ragResponse.json();
        ragSources = ragResult.sources || [];
      }

      // Send to AI with context
      const aiResponse = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messages.slice(-5), // Last 5 messages for context
          query: userQuery,
          caseId,
          ragSources,
          enableSelfPrompting,
          context: {
            role: 'prosecutor',
            mode: 'evidence_analysis',
            capabilities: ['legal_research', 'evidence_correlation', 'strategy_planning']
          }
        })
      });

      const aiResult = await aiResponse.json();
      aiConfidence = aiResult.confidence || 0;

      const assistantMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: aiResult.content || aiResult.answer,
        timestamp: new Date(),
        sources: ragSources,
        metadata: {
          model: 'gemma3-legal:latest',
          confidence: aiConfidence,
          ragSources: ragSources.length,
          processingTime: aiResult.processingTime || 0
        }
      };
      
      messages = [...messages, assistantMessage];

      // Generate new self-prompt suggestions based on conversation
      if (enableSelfPrompting) {
        setTimeout(generateSelfPromptSuggestions, 1000);
      }
      
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
        error: true
      };
      messages = [...messages, errorMessage];
    } finally {
      isTyping = false;
    }
  };

  // Quick action for self-prompt suggestions
  const useSelfPrompt = (suggestion: string) => {
    currentMessage = suggestion;
    sendMessage();
  };

  // Keyboard handler
  const handleKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };
</script>

<svelte:window onmouseover={handleElementHover as any} />

<div class="flex flex-col h-full max-w-4xl mx-auto">
  <!-- Chat Header -->
  <Card class="mb-4">
    <CardHeader class="pb-3">
      <CardTitle class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <Bot class="w-5 h-5 text-blue-500" />
          Legal AI Assistant
          <Badge variant="secondary" class="ml-2">
            Gemma3Legal
          </Badge>
        </div>
        
        <div class="flex items-center gap-2 text-sm">
          {#if enableEnhancedRAG}
            <Badge variant="outline">
              <Search class="w-3 h-3 mr-1" />
              Enhanced RAG
            </Badge>
          {/if}
          {#if enableElementalAwareness}
            <Badge variant="outline">
              <Eye class="w-3 h-3 mr-1" />
              YOLO Aware
            </Badge>
          {/if}
          {#if enableSelfPrompting}
            <Badge variant="outline">
              <Brain class="w-3 h-3 mr-1" />
              Self-Prompting
            </Badge>
          {/if}
        </div>
      </CardTitle>
    </CardHeader>
  </Card>

  <!-- Self-Prompt Suggestions -->
  {#if enableSelfPrompting && selfPromptSuggestions.length > 0}
    <Card class="mb-4">
      <CardContent class="pt-4">
        <h4 class="text-sm font-medium mb-3 flex items-center gap-2">
          <Zap class="w-4 h-4" />
          Suggested Questions
        </h4>
        <div class="flex flex-wrap gap-2">
          {#each selfPromptSuggestions as suggestion}
            <Button 
              variant="outline" 
              size="sm"
              onclick={() => useSelfPrompt(suggestion)}
              disabled={isTyping}
            >
              {suggestion}
            </Button>
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Elemental Analysis Tooltip -->
  {#if enableElementalAwareness && elementAnalysis}
    <div class="fixed top-4 right-4 z-50 p-3 bg-black text-white rounded-lg shadow-lg max-w-xs">
      <div class="flex items-center gap-2 mb-2">
        <Eye class="w-4 h-4" />
        <span class="font-medium">Element Analysis</span>
      </div>
      <p class="text-sm">{elementAnalysis.relevance || 'Analyzing...'}</p>
      {#if elementAnalysis.legalContext}
        <Badge variant="secondary" class="mt-1 text-xs">
          {elementAnalysis.legalContext}
        </Badge>
      {/if}
    </div>
  {/if}

  <!-- Chat Messages -->
  <Card class="flex-1 flex flex-col">
    <CardContent class="flex-1 overflow-y-auto p-4 space-y-4">
      {#each messages as message}
        <div class="flex items-start gap-3 {message.role === 'user' ? 'justify-end' : ''}">
          {#if message.role === 'assistant'}
            <div class="flex-shrink-0">
              <Bot class="w-8 h-8 p-1.5 bg-blue-100 text-blue-600 rounded-full" />
            </div>
          {/if}
          
          <div class="flex-1 max-w-[80%] {message.role === 'user' ? 'order-1' : ''}">
            <div class="p-3 rounded-lg {message.role === 'user' ? 'bg-blue-500 text-white ml-auto' : 'bg-gray-100'}">
              <p class="text-sm whitespace-pre-wrap">{message.content}</p>
              
              {#if message.metadata?.confidence}
                <div class="mt-2 flex items-center gap-2">
                  <Badge variant="secondary" class="text-xs">
                    {Math.round(message.metadata.confidence * 100)}% confident
                  </Badge>
                  {#if message.metadata.ragSources > 0}
                    <Badge variant="outline" class="text-xs">
                      {message.metadata.ragSources} sources
                    </Badge>
                  {/if}
                </div>
              {/if}
              
              {#if message.sources?.length > 0}
                <div class="mt-3 space-y-2">
                  <h5 class="text-xs font-medium opacity-75">Sources:</h5>
                  {#each message.sources as source}
                    <div class="text-xs opacity-75 p-2 bg-black bg-opacity-10 rounded">
                      <div class="flex items-center gap-1">
                        <FileText class="w-3 h-3" />
                        {source.metadata?.fileName || source.id}
                        <Badge variant="outline" class="text-xs ml-auto">
                          {Math.round(source.score * 100)}% match
                        </Badge>
                      </div>
                      <p class="mt-1">{source.content.substring(0, 100)}...</p>
                    </div>
                  {/each}
                </div>
              {/if}
            </div>
            
            <p class="text-xs text-gray-500 mt-1 {message.role === 'user' ? 'text-right' : ''}">
              {new Date(message.timestamp).toLocaleTimeString()}
            </p>
          </div>
          
          {#if message.role === 'user'}
            <div class="flex-shrink-0">
              <User class="w-8 h-8 p-1.5 bg-gray-100 text-gray-600 rounded-full" />
            </div>
          {/if}
        </div>
      {/each}
      
      {#if isTyping}
        <div class="flex items-start gap-3">
          <Bot class="w-8 h-8 p-1.5 bg-blue-100 text-blue-600 rounded-full animate-pulse" />
          <div class="p-3 bg-gray-100 rounded-lg">
            <div class="flex space-x-1">
              <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
              <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
              <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
            </div>
          </div>
        </div>
      {/if}
    </CardContent>

    <!-- Chat Input -->
    <div class="border-t p-4">
      <div class="flex gap-2">
        <div class="flex-1">
          <Input
            bind:value={currentMessage}
            placeholder="Ask about evidence, legal precedents, case strategy..."
            onkeydown={handleKeyDown}
            disabled={isTyping}
          />
        </div>
        <Button 
          onclick={sendMessage}
          disabled={isTyping || !currentMessage.trim()}
        >
          <Send class="w-4 h-4" />
        </Button>
      </div>
      
      <!-- AI Status Indicators -->
      {#if caseId}
        <div class="flex items-center justify-between mt-2 text-xs text-gray-500">
          <div class="flex items-center gap-4">
            <span class="flex items-center gap-1">
              <Scale class="w-3 h-3" />
              Case {caseId}
            </span>
            {#if ragSources.length > 0}
              <span class="flex items-center gap-1">
                <FileText class="w-3 h-3" />
                {ragSources.length} sources loaded
              </span>
            {/if}
            {#if aiConfidence > 0}
              <span class="flex items-center gap-1">
                <Brain class="w-3 h-3" />
                {Math.round(aiConfidence * 100)}% confidence
              </span>
            {/if}
          </div>
          
          <div class="flex items-center gap-2">
            {#if enableEnhancedRAG}
              <Badge variant="outline" class="text-xs">RAG Active</Badge>
            {/if}
            {#if enableElementalAwareness}
              <Badge variant="outline" class="text-xs">YOLO Aware</Badge>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  </Card>
</div>

<style>
  /* Enhanced chat styling */
  :global(.chat-message) {
    animation: fadeIn 0.3s ease-in-out;
  }
  
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
  
  /* Elemental awareness hover effects */
  :global(*:hover) {
    position: relative;
  }
  
  /* Self-prompting suggestion animations */
  :global(.suggestion-button) {
    transition: all 0.2s ease;
  }
  
  :global(.suggestion-button:hover) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  }
</style>


