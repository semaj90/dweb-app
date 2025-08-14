<script lang="ts">
  import { Card, CardContent, CardHeader, CardTitle } from "$lib/components/ui/card/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import { Textarea } from "$lib/components/ui/textarea/index.js";
  import { Loader2, Bot, MessageSquare, FileText, Search, Sparkles, Zap } from "lucide-svelte";

  // Props
  let { 
    onAISearch = null,
    onAIChat = null, 
    onAISummarize = null,
    disabled = false,
    compact = false 
  } = $props();

  // State
  let aiSearchQuery = $state('');
  let aiChatMessage = $state('');
  let summarizeText = $state('');
  let isAISearching = $state(false);
  let isAIChatting = $state(false);
  let isSummarizing = $state(false);
  let aiSearchResults = $state([]);
  let aiChatResponse = $state('');
  let summaryResult = $state('');

  // Enhanced AI Search with LangChain.js and vector similarity
  async function performAISearch() {
    if (!aiSearchQuery.trim() || isAISearching) return;
    
    isAISearching = true;
    aiSearchResults = [];
    
    try {
      const response = await fetch('/api/ai/enhanced-legal-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: aiSearchQuery,
          jurisdiction: 'all',
          category: 'all',
          maxResults: 10,
          useAI: true,
          advancedOptions: {
            useVector: true,
            similarityThreshold: 0.7
          }
        })
      });

      const result = await response.json();
      
      if (result.success) {
        aiSearchResults = result.results || [];
        console.log(`🔍 Enhanced AI search found ${aiSearchResults.length} results in ${result.searchTime}`);
        console.log('Search analytics:', result.analytics);
        
        if (onAISearch) {
          onAISearch(result);
        }
      } else {
        console.error('Enhanced AI search failed:', result.error);
        // Fallback to basic search
        await performFallbackSearch();
      }
    } catch (error) {
      console.error('Enhanced AI search error:', error);
      // Fallback to basic search
      await performFallbackSearch();
    } finally {
      isAISearching = false;
    }
  }

  // Fallback search method
  async function performFallbackSearch() {
    try {
      const response = await fetch('/api/ai/legal-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: aiSearchQuery,
          jurisdiction: 'all',
          category: 'all',
          useAI: true
        })
      });

      const result = await response.json();
      
      if (result.success) {
        aiSearchResults = result.laws || [];
        if (onAISearch) {
          onAISearch(result);
        }
      }
    } catch (fallbackError) {
      console.error('Fallback search also failed:', fallbackError);
    }
  }

  // AI Chat
  async function performAIChat() {
    if (!aiChatMessage.trim() || isAIChatting) return;
    
    isAIChatting = true;
    aiChatResponse = '';
    
    try {
      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: aiChatMessage,
          temperature: 0.7
        })
      });

      const result = await response.json();
      
      if (result.response) {
        aiChatResponse = result.response;
        if (onAIChat) {
          onAIChat(result);
        }
      } else {
        console.error('AI chat failed:', result.error);
      }
    } catch (error) {
      console.error('AI chat error:', error);
    } finally {
      isAIChatting = false;
    }
  }

  // AI Summarization
  async function performAISummarization() {
    if (!summarizeText.trim() || isSummarizing) return;
    
    isSummarizing = true;
    summaryResult = '';
    
    try {
      const response = await fetch('/api/ai/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: summarizeText,
          type: 'legal',
          options: { max_tokens: 500 }
        })
      });

      const result = await response.json();
      
      if (result.success) {
        summaryResult = result.summary;
        if (onAISummarize) {
          onAISummarize(result);
        }
      } else {
        console.error('AI summarization failed:', result.error);
      }
    } catch (error) {
      console.error('AI summarization error:', error);
    } finally {
      isSummarizing = false;
    }
  }

  // Keyboard handlers
  function handleAISearchKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      performAISearch();
    }
  }

  function handleAIChatKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      performAIChat();
    }
  }

  function clearResults() {
    aiSearchResults = [];
    aiChatResponse = '';
    summaryResult = '';
  }
</script>

<div class="space-y-6">
  <div class="text-center">
    <h2 class="text-2xl font-bold flex items-center justify-center gap-2">
      <Sparkles class="h-6 w-6 text-primary" />
      AI Legal Assistant
    </h2>
    <p class="text-muted-foreground mt-2">
      Intelligent search, chat, and summarization powered by local AI
    </p>
  </div>

  <div class="grid grid-cols-1 {compact ? 'lg:grid-cols-1' : 'lg:grid-cols-3'} gap-6">
    
    <!-- AI Search -->
    <Card class="border-primary/20">
      <CardHeader class="pb-3">
        <CardTitle class="flex items-center gap-2 text-lg">
          <Bot class="h-5 w-5 text-primary" />
          AI Search
        </CardTitle>
      </CardHeader>
      <CardContent class="space-y-4">
        <div class="flex gap-2">
          <div class="relative flex-1">
            <Bot class="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-primary" />
            <Input
              placeholder="Ask AI to find laws..."
              bind:value={aiSearchQuery}
              onkeydown={handleAISearchKeydown}
              {disabled}
              class="pl-10"
            />
          </div>
          <Button 
            onclick={performAISearch} 
            disabled={disabled || isAISearching || !aiSearchQuery.trim()}
            size="sm"
          >
            {#if isAISearching}
              <Loader2 class="h-4 w-4 animate-spin" />
            {:else}
              <Search class="h-4 w-4" />
            {/if}
          </Button>
        </div>
        
        {#if aiSearchResults.length > 0}
          <div class="space-y-2 max-h-32 overflow-y-auto">
            {#each aiSearchResults.slice(0, 3) as result}
              <div class="p-2 bg-muted/50 rounded text-sm">
                <div class="font-medium truncate">{result.title}</div>
                <div class="text-xs text-muted-foreground">{result.jurisdiction}</div>
              </div>
            {/each}
            {#if aiSearchResults.length > 3}
              <Badge variant="outline" class="text-xs">
                +{aiSearchResults.length - 3} more results
              </Badge>
            {/if}
          </div>
        {/if}
      </CardContent>
    </Card>

    <!-- AI Chat -->
    <Card class="border-green-500/20">
      <CardHeader class="pb-3">
        <CardTitle class="flex items-center gap-2 text-lg">
          <MessageSquare class="h-5 w-5 text-green-600" />
          AI Chat
        </CardTitle>
      </CardHeader>
      <CardContent class="space-y-4">
        <div class="space-y-2">
          <Textarea
            placeholder="Ask a legal question..."
            bind:value={aiChatMessage}
            onkeydown={handleAIChatKeydown}
            {disabled}
            rows="2"
            class="resize-none"
          />
          <Button 
            onclick={performAIChat} 
            disabled={disabled || isAIChatting || !aiChatMessage.trim()}
            size="sm"
            class="w-full"
          >
            {#if isAIChatting}
              <Loader2 class="h-4 w-4 animate-spin mr-2" />
              Thinking...
            {:else}
              <MessageSquare class="h-4 w-4 mr-2" />
              Ask AI
            {/if}
          </Button>
        </div>
        
        {#if aiChatResponse}
          <div class="p-3 bg-green-50 dark:bg-green-950/30 rounded text-sm max-h-32 overflow-y-auto">
            <div class="prose prose-sm max-w-none">
              <p class="whitespace-pre-wrap">{aiChatResponse}</p>
            </div>
          </div>
        {/if}
      </CardContent>
    </Card>

    <!-- AI Summarization -->
    <Card class="border-blue-500/20">
      <CardHeader class="pb-3">
        <CardTitle class="flex items-center gap-2 text-lg">
          <FileText class="h-5 w-5 text-blue-600" />
          AI Summary
        </CardTitle>
      </CardHeader>
      <CardContent class="space-y-4">
        <div class="space-y-2">
          <Textarea
            placeholder="Paste legal text to summarize..."
            bind:value={summarizeText}
            {disabled}
            rows="2"
            class="resize-none"
          />
          <Button 
            onclick={performAISummarization} 
            disabled={disabled || isSummarizing || !summarizeText.trim()}
            size="sm"
            class="w-full"
          >
            {#if isSummarizing}
              <Loader2 class="h-4 w-4 animate-spin mr-2" />
              Summarizing...
            {:else}
              <Zap class="h-4 w-4 mr-2" />
              Summarize
            {/if}
          </Button>
        </div>
        
        {#if summaryResult}
          <div class="p-3 bg-blue-50 dark:bg-blue-950/30 rounded text-sm max-h-32 overflow-y-auto">
            <div class="prose prose-sm max-w-none">
              <p class="whitespace-pre-wrap">{summaryResult}</p>
            </div>
          </div>
        {/if}
      </CardContent>
    </Card>
  </div>

  <!-- Clear Results Button -->
  {#if aiSearchResults.length > 0 || aiChatResponse || summaryResult}
    <div class="text-center">
      <Button variant="outline" onclick={clearResults} size="sm">
        Clear All Results
      </Button>
    </div>
  {/if}

  <!-- Quick Actions -->
  <div class="flex flex-wrap gap-2 justify-center">
    <Button 
      variant="outline" 
      size="sm" 
      onclick={() => { aiSearchQuery = 'California murder laws'; performAISearch(); }}
      disabled={disabled || isAISearching}
    >
      <Bot class="h-3 w-3 mr-1" />
      Murder Laws
    </Button>
    <Button 
      variant="outline" 
      size="sm" 
      onclick={() => { aiChatMessage = 'What are the elements of a valid contract?'; performAIChat(); }}
      disabled={disabled || isAIChatting}
    >
      <MessageSquare class="h-3 w-3 mr-1" />
      Contract Elements
    </Button>
    <Button 
      variant="outline" 
      size="sm" 
      onclick={() => { aiSearchQuery = 'evidence admissibility rules'; performAISearch(); }}
      disabled={disabled || isAISearching}
    >
      <Search class="h-3 w-3 mr-1" />
      Evidence Rules
    </Button>
  </div>
</div>

<style>
  :global(.prose p) {
    @apply text-sm leading-relaxed mb-2 last:mb-0;
  }
</style>