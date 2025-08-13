<script lang="ts">
  import { onMount } from 'svelte';
  import * as Dialog from "$lib/components/ui/dialog/index.js";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { ScrollArea } from "$lib/components/ui/scroll-area/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import { MessageSquare, Send, Bot, User, Loader2, X, Copy, ThumbsUp, ThumbsDown } from "lucide-svelte";

  let { 
    open = $bindable(false),
    context = null,
    title = "AI Legal Assistant",
    placeholder = "Ask about legal matters...",
    caseId = null,
    documentId = null
  } = $props();

  let messages = $state([]);
  let currentMessage = $state('');
  let isLoading = $state(false);
  let chatContainer = $state(null);
  let inputElement = $state(null);

  // Auto-scroll to bottom when messages change
  $effect(() => {
    if (messages.length > 0 && chatContainer) {
      setTimeout(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }, 100);
    }
  });

  // Focus input when dialog opens
  $effect(() => {
    if (open && inputElement) {
      setTimeout(() => inputElement.focus(), 100);
    }
  });

  // Initialize with context message if provided
  $effect(() => {
    if (open && context && messages.length === 0) {
      addSystemMessage();
    }
  });

  function addSystemMessage() {
    if (context) {
      messages = [{
        id: Date.now(),
        role: 'system',
        content: `I have context about: ${context.title || 'Legal Document'}. How can I help you understand or analyze this?`,
        timestamp: new Date().toISOString(),
        type: 'context'
      }];
    }
  }

  async function sendMessage() {
    if (!currentMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: currentMessage.trim(),
      timestamp: new Date().toISOString()
    };

    messages = [...messages, userMessage];
    const messageToSend = currentMessage.trim();
    currentMessage = '';
    isLoading = true;

    try {
      // Prepare context for AI
      let contextText = '';
      if (context) {
        contextText = `Context: ${context.title}\n${context.description || ''}\n${context.fullText || ''}`;
      }

      const response = await fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: messageToSend,
          context: contextText ? [contextText] : undefined,
          caseId,
          documentId,
          temperature: 0.7
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get AI response');
      }

      const data = await response.json();
      
      const aiMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response || 'I apologize, but I could not generate a response.',
        timestamp: new Date().toISOString(),
        metadata: data.performance || {},
        suggestions: data.suggestions || []
      };

      messages = [...messages, aiMessage];

    } catch (error) {
      console.error('AI chat error:', error);
      
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'I apologize, but I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
        error: true
      };

      messages = [...messages, errorMessage];
    } finally {
      isLoading = false;
    }
  }

  function handleKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function clearChat() {
    messages = [];
    if (context) {
      addSystemMessage();
    }
  }

  async function copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      // Could add a toast notification here
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  }

  function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  }

  function handleSuggestionClick(suggestion) {
    currentMessage = suggestion;
    sendMessage();
  }

  async function provideFeedback(messageId, feedback) {
    try {
      await fetch('/api/ai/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messageId,
          feedback,
          caseId,
          documentId
        })
      });
    } catch (error) {
      console.error('Failed to provide feedback:', error);
    }
  }
</script>

<Dialog.Root bind:open>
  <Dialog.Content class="max-w-4xl max-h-[80vh] flex flex-col">
    <Dialog.Header class="flex-shrink-0">
      <Dialog.Title class="flex items-center gap-2">
        <Bot class="h-5 w-5" />
        {title}
      </Dialog.Title>
      <Dialog.Description>
        Ask questions about legal matters, get case analysis, and receive AI-powered assistance.
        {#if context}
          <br><strong>Context:</strong> {context.title}
        {/if}
      </Dialog.Description>
    </Dialog.Header>

    <!-- Chat Messages -->
    <div class="flex-1 overflow-hidden">
      <ScrollArea bind:element={chatContainer} class="h-[400px] w-full pr-4">
        <div class="space-y-4">
          {#each messages as message}
            <div class="flex gap-3 {message.role === 'user' ? 'justify-end' : 'justify-start'}">
              {#if message.role !== 'user'}
                <div class="flex-shrink-0">
                  {#if message.type === 'context'}
                    <div class="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                      <MessageSquare class="h-4 w-4 text-blue-600 dark:text-blue-400" />
                    </div>
                  {:else}
                    <div class="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900 flex items-center justify-center">
                      <Bot class="h-4 w-4 text-green-600 dark:text-green-400" />
                    </div>
                  {/if}
                </div>
              {/if}

              <div class="flex-1 max-w-[80%] {message.role === 'user' ? 'order-first' : ''}">
                <Card.Root class="{message.role === 'user' ? 'bg-primary text-primary-foreground' : ''} {message.error ? 'border-red-200 dark:border-red-800' : ''}">
                  <Card.Content class="p-3">
                    <div class="prose prose-sm max-w-none {message.role === 'user' ? 'prose-invert' : ''}">
                      <p class="whitespace-pre-wrap">{message.content}</p>
                    </div>
                    
                    <div class="flex items-center justify-between mt-3 pt-2 border-t border-border/50">
                      <div class="flex items-center gap-2">
                        <span class="text-xs opacity-70">
                          {formatTimestamp(message.timestamp)}
                        </span>
                        {#if message.metadata?.tokensPerSecond}
                          <Badge variant="outline" class="text-xs">
                            {Math.round(message.metadata.tokensPerSecond)} tok/s
                          </Badge>
                        {/if}
                      </div>
                      
                      {#if message.role === 'assistant' && !message.error}
                        <div class="flex items-center gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onclick={() => copyToClipboard(message.content)}
                          >
                            <Copy class="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onclick={() => provideFeedback(message.id, 'positive')}
                          >
                            <ThumbsUp class="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onclick={() => provideFeedback(message.id, 'negative')}
                          >
                            <ThumbsDown class="h-3 w-3" />
                          </Button>
                        </div>
                      {/if}
                    </div>
                  </Card.Content>
                </Card.Root>

                <!-- AI Suggestions -->
                {#if message.suggestions && message.suggestions.length > 0}
                  <div class="mt-2 space-y-1">
                    {#each message.suggestions as suggestion}
                      <Button
                        variant="outline"
                        size="sm"
                        class="text-xs h-auto py-1 px-2"
                        onclick={() => handleSuggestionClick(suggestion)}
                      >
                        {suggestion}
                      </Button>
                    {/each}
                  </div>
                {/if}
              </div>

              {#if message.role === 'user'}
                <div class="flex-shrink-0">
                  <div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                    <User class="h-4 w-4 text-primary-foreground" />
                  </div>
                </div>
              {/if}
            </div>
          {/each}

          {#if isLoading}
            <div class="flex gap-3 justify-start">
              <div class="flex-shrink-0">
                <div class="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900 flex items-center justify-center">
                  <Loader2 class="h-4 w-4 text-green-600 dark:text-green-400 animate-spin" />
                </div>
              </div>
              <div class="flex-1 max-w-[80%]">
                <Card.Root>
                  <Card.Content class="p-3">
                    <div class="flex items-center gap-2 text-muted-foreground">
                      <Loader2 class="h-4 w-4 animate-spin" />
                      <span>Thinking...</span>
                    </div>
                  </Card.Content>
                </Card.Root>
              </div>
            </div>
          {/if}
        </div>
      </ScrollArea>
    </div>

    <!-- Input Area -->
    <div class="flex-shrink-0 border-t pt-4">
      <div class="flex gap-2">
        <Input
          bind:element={inputElement}
          bind:value={currentMessage}
          {placeholder}
          onkeydown={handleKeydown}
          disabled={isLoading}
          class="flex-1"
        />
        <Button onclick={sendMessage} disabled={isLoading || !currentMessage.trim()}>
          {#if isLoading}
            <Loader2 class="h-4 w-4 animate-spin" />
          {:else}
            <Send class="h-4 w-4" />
          {/if}
        </Button>
        <Button variant="outline" onclick={clearChat}>
          <X class="h-4 w-4" />
        </Button>
      </div>
      
      {#if messages.length === 0 && context}
        <div class="mt-3 text-sm text-muted-foreground">
          <p>You can ask questions like:</p>
          <ul class="list-disc list-inside mt-1 space-y-1">
            <li>"Explain this law in simple terms"</li>
            <li>"What are the key elements to prove?"</li>
            <li>"How does this relate to other laws?"</li>
            <li>"What are common defenses or exceptions?"</li>
          </ul>
        </div>
      {/if}
    </div>
  </Dialog.Content>
</Dialog.Root>

<style>
  :global(.prose p) {
    @apply text-sm leading-relaxed mb-2 last:mb-0;
  }
</style>