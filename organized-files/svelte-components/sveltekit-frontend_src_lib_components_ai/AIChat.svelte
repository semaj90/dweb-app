<script lang="ts">
  import { useChatActor, chatActions } from '$lib/stores/chatStore';
  import { Card } from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { serviceStatus } from '$lib/stores/chatStore';
  import { onMount } from 'svelte';

  // Use the XState machine through the store
  const { state } = useChatActor();

  let userInput = '';
  let chatContainer: HTMLElement

  // Send message handler
  function handleSubmit() {
    if (!userInput.trim()) return;
    chatActions.sendMessage(userInput);
    userInput = '';
  }

  // Clear chat handler
  function handleClear() {
    chatActions.resetChat();
  }

  // Update scroll when messages change
  $effect(() => { if ($state.context.messages && chatContainer) {
    // Wait for DOM update
    setTimeout(() => {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 10);
  }

  // Check service status on mount
  onMount(() => {
    // Add any initialization here
  });
</script>

<Card class="flex flex-col h-[70vh] max-w-3xl mx-auto my-8">
  <div class="flex items-center justify-between border-b p-4">
    <div>
      <h2 class="text-xl font-semibold">Legal AI Assistant</h2>
      <p class="text-sm text-muted-foreground">
        {#if $serviceStatus.ollama === 'connected'}
          <span class="text-green-500">●</span> AI Connected
        {:else if $serviceStatus.ollama === 'error'}
          <span class="text-red-500">●</span> AI Service Error
        {:else}
          <span class="text-yellow-500">●</span> AI Status Unknown
        {/if}
      </p>
    </div>
    <Button variant="outline" size="sm" onclick={handleClear}>
      Clear Chat
    </Button>
  </div>

  <!-- Chat messages -->
  <div bind:this={chatContainer} class="flex-1 overflow-y-auto p-4 space-y-4">
    {#each $state.context.messages as message, i (i)}
      <div class="chat-message {message.role === 'user' ? 'user' : 'assistant'}">
        <div class="message-bubble">
          {@html message.content.replace(/\n/g, '<br>')}
          {#if $state.matches('loading') && i === $state.context.messages.length - 1}
            <span class="typing-indicator"></span>
          {/if}
        </div>
      </div>
    {/each}

    {#if $state.matches('error')}
      <div class="chat-message error">
        <div class="message-bubble error-bubble">
          <p>Error: {$state.context.error?.message || 'Unknown error'}</p>
          <p>Please try again.</p>
        </div>
      </div>
    {/if}
  </div>

  <!-- Input area -->
  <div class="border-t p-4">
    <form on:submit|preventDefault={handleSubmit} class="flex space-x-2">
      <Input
        type="text"
        placeholder="Ask about your legal case..."
        bind:value={userInput}
        disabled={$state.matches('loading')}
        class="flex-1"
      />
      <Button type="submit" disabled={$state.matches('loading') || !userInput.trim()}>
        {$state.matches('loading') ? 'Thinking...' : 'Send'}
      </Button>
    </form>
  </div>
</Card>

<style>
  .chat-message {
    display: flex
    margin-bottom: 1rem;
    max-width: 80%;
  }

  .chat-message.user {
    margin-left: auto
    justify-content: flex-end;
  }

  .chat-message.assistant {
    margin-right: auto
    justify-content: flex-start;
  }

  .chat-message.error {
    margin-right: auto
    justify-content: flex-start;
  }

  .message-bubble {
    padding: 0.75rem 1rem;
    border-radius: 1rem;
    background-color: #e9ecef;
    color: #212529;
  }

  .user .message-bubble {
    background-color: #3b82f6;
    color: white
    border-bottom-right-radius: 0.25rem;
  }

  .assistant .message-bubble {
    background-color: #f3f4f6;
    border-bottom-left-radius: 0.25rem;
  }

  .error-bubble {
    background-color: #fee2e2;
    color: #991b1b;
  }

  .typing-indicator {
    display: inline-block;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background-color: currentColor
    animation: typing 1s infinite steps(4, end);
    margin-left: 8px;
    vertical-align: middle
  }

  @keyframes typing {
    0%, 100% {
      transform: translateY(0);
    }
    50% {
      transform: translateY(-5px);
    }
  }
</style>
