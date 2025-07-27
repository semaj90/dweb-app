<script lang="ts">
  import { onMount } from 'svelte';
  import { useChatActor, chatActions, serviceStatus } from '$lib/stores/chatStore';
  import { Card, CardHeader, CardTitle, CardContent } from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Badge } from '$lib/components/ui/badge';
  import { ScrollArea } from '$lib/components/ui/scroll-area';
  import { AlertCircle, CheckCircle, Clock, Zap } from 'lucide-svelte';

  // Use the XState-compatible store
  const { state } = useChatActor();

  let userInput = '';
  let chatContainer: HTMLElement;
  let systemChecks = {
    ollama: { status: 'checking', message: 'Checking Ollama service...' },
    model: { status: 'checking', message: 'Verifying Gemma3 model...' },
    api: { status: 'checking', message: 'Testing API endpoints...' },
  };

  // Test message for demo
  let testMessages = [
    "What is contract law?",
    "Explain the difference between civil and criminal law",
    "What are the elements of a valid contract?",
    "How do I analyze a legal case?"
  ];

  onMount(async () => {
    await performSystemChecks();
  });

  async function performSystemChecks() {
    // Check Ollama service
    try {
      const ollamaResponse = await fetch('http://localhost:11434/api/version');
      if (ollamaResponse.ok) {
        const ollamaData = await ollamaResponse.json();
        systemChecks.ollama = {
          status: 'connected',
          message: `Ollama v${ollamaData.version} running`
        };
      } else {
        systemChecks.ollama = {
          status: 'error',
          message: 'Ollama service not responding'
        };
      }
    } catch (error) {
      systemChecks.ollama = {
        status: 'error',
        message: 'Cannot connect to Ollama on localhost:11434'
      };
    }

    // Check model availability
    try {
      const modelResponse = await fetch('http://localhost:11434/api/tags');
      if (modelResponse.ok) {
        const modelData = await modelResponse.json();
        const hasGemma3 = modelData.models?.some((model: any) =>
          model.name.includes('gemma3-legal')
        );

        if (hasGemma3) {
          systemChecks.model = {
            status: 'connected',
            message: 'Gemma3-legal model loaded'
          };
        } else {
          systemChecks.model = {
            status: 'error',
            message: 'Gemma3-legal model not found'
          };
        }
      }
    } catch (error) {
      systemChecks.model = {
        status: 'error',
        message: 'Cannot verify model status'
      };
    }

    // Check API endpoint
    try {
      const apiResponse = await fetch('/api/ai/chat');
      if (apiResponse.ok) {
        const apiData = await apiResponse.json();
        systemChecks.api = {
          status: 'connected',
          message: 'Chat API endpoint healthy'
        };
      } else {
        systemChecks.api = {
          status: 'error',
          message: 'Chat API endpoint error'
        };
      }
    } catch (error) {
      systemChecks.api = {
        status: 'error',
        message: 'Cannot reach chat API endpoint'
      };
    }

    // Trigger reactivity
    systemChecks = { ...systemChecks };
  }

  function handleSubmit() {
    if (!userInput.trim()) return;
    chatActions.sendMessage(userInput);
    userInput = '';
  }

  function sendTestMessage(message: string) {
    chatActions.sendMessage(message);
  }

  function handleClear() {
    chatActions.resetChat();
  }

  // Auto-scroll chat
  $: if ($state.context.messages && chatContainer) {
    setTimeout(() => {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 10);
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case 'connected': return CheckCircle;
      case 'error': return AlertCircle;
      case 'checking': return Clock;
      default: return Clock;
    }
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'connected': return 'text-green-500';
      case 'error': return 'text-red-500';
      case 'checking': return 'text-yellow-500';
      default: return 'text-gray-500';
    }
  }
</script>

<svelte:head>
  <title>Gemma3 Integration Test - Legal AI Chat</title>
</svelte:head>

<div class="container mx-auto p-6 space-y-6">
  <div class="text-center mb-8">
    <h1 class="text-3xl font-bold">Gemma3 Legal AI Integration Test</h1>
    <p class="text-muted-foreground mt-2">
      Complete system validation and chat interface
    </p>
  </div>

  <!-- System Status Cards -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
    {#each Object.entries(systemChecks) as [service, check]}
      <Card>
        <CardHeader class="pb-3">
          <CardTitle class="flex items-center gap-2 text-sm">
            <svelte:component
              this={getStatusIcon(check.status)}
              class="w-4 h-4 {getStatusColor(check.status)}"
            />
            {service.toUpperCase()}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Badge
            variant={check.status === 'connected' ? 'default' :
                    check.status === 'error' ? 'destructive' : 'secondary'}
            class="mb-2"
          >
            {check.status}
          </Badge>
          <p class="text-sm text-muted-foreground">{check.message}</p>
        </CardContent>
      </Card>
    {/each}
  </div>

  <!-- Quick Test Messages -->
  <Card>
    <CardHeader>
      <CardTitle class="flex items-center gap-2">
        <Zap class="w-5 h-5" />
        Quick Test Messages
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
        {#each testMessages as message}
          <Button
            variant="outline"
            size="sm"
            class="text-left justify-start"
            disabled={$state.matches('loading')}
            on:click={() => sendTestMessage(message)}
          >
            {message}
          </Button>
        {/each}
      </div>
    </CardContent>
  </Card>

  <!-- Chat Interface -->
  <Card class="flex flex-col h-[60vh]">
    <CardHeader class="border-b">
      <div class="flex items-center justify-between">
        <CardTitle class="flex items-center gap-2">
          Gemma3 Legal AI Chat
          {#if $state.matches('loading')}
            <Badge variant="secondary" class="animate-pulse">Thinking...</Badge>
          {:else if $state.matches('error')}
            <Badge variant="destructive">Error</Badge>
          {:else}
            <Badge variant="default">Ready</Badge>
          {/if}
        </CardTitle>
        <Button variant="outline" size="sm" on:click={handleClear}>
          Clear Chat
        </Button>
      </div>
    </CardHeader>

    <!-- Messages -->
    <ScrollArea class="flex-1 p-4">
      <div bind:this={chatContainer} class="space-y-4">
        {#each $state.context.messages as message, i (i)}
          <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
            <div class="max-w-[80%] rounded-lg p-3 {
              message.role === 'user'
                ? 'bg-primary text-primary-foreground ml-4'
                : 'bg-muted mr-4'
            }">
              <div class="text-sm mb-1 opacity-70">
                {message.role === 'user' ? 'You' : 'Gemma3 Legal AI'}
              </div>
              <div class="whitespace-pre-wrap">
                {@html message.content.replace(/\n/g, '<br>')}
              </div>
              {#if $state.matches('loading') && i === $state.context.messages.length - 1}
                <div class="flex items-center gap-1 mt-2 opacity-70">
                  <div class="w-2 h-2 bg-current rounded-full animate-bounce"></div>
                  <div class="w-2 h-2 bg-current rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                  <div class="w-2 h-2 bg-current rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                </div>
              {/if}
            </div>
          </div>
        {/each}

        {#if $state.context.messages.length === 0}
          <div class="text-center py-8 text-muted-foreground">
            <p class="text-lg mb-2">Welcome to Gemma3 Legal AI</p>
            <p class="text-sm">
              Ask questions about legal matters, contracts, case analysis, or use the quick test messages above.
            </p>
          </div>
        {/if}

        {#if $state.matches('error')}
          <div class="bg-destructive/10 border border-destructive/20 rounded-lg p-4 text-destructive">
            <div class="flex items-center gap-2 font-medium mb-1">
              <AlertCircle class="w-4 h-4" />
              Error
            </div>
            <p class="text-sm">
              {$state.context.error?.message || 'An unknown error occurred'}
            </p>
            <Button
              variant="outline"
              size="sm"
              class="mt-2"
              on:click={() => chatActions.clearError()}
            >
              Dismiss
            </Button>
          </div>
        {/if}
      </div>
    </ScrollArea>

    <!-- Input -->
    <div class="border-t p-4">
      <form on:submit|preventDefault={handleSubmit} class="flex gap-2">
        <Input
          type="text"
          placeholder="Ask about legal matters, case analysis, contracts..."
          bind:value={userInput}
          disabled={$state.matches('loading')}
          class="flex-1"
        />
        <Button
          type="submit"
          disabled={$state.matches('loading') || !userInput.trim()}
        >
          {$state.matches('loading') ? 'Sending...' : 'Send'}
        </Button>
      </form>
      <p class="text-xs text-muted-foreground mt-2">
        Powered by Gemma3 running locally via Ollama
      </p>
    </div>
  </Card>

  <!-- Debug Information -->
  <Card>
    <CardHeader>
      <CardTitle class="text-lg">Debug Information</CardTitle>
    </CardHeader>
    <CardContent>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div>
          <h4 class="font-medium mb-2">Chat State</h4>
          <ul class="space-y-1 text-muted-foreground">
            <li>Messages: {$state.context.messages.length}</li>
            <li>Loading: {$state.matches('loading')}</li>
            <li>Streaming: {$state.matches('streaming')}</li>
            <li>Error: {$state.matches('error')}</li>
            <li>Model: {$state.context.settings.model}</li>
          </ul>
        </div>
        <div>
          <h4 class="font-medium mb-2">System Status</h4>
          <ul class="space-y-1 text-muted-foreground">
            <li>Ollama: {systemChecks.ollama.status}</li>
            <li>Model: {systemChecks.model.status}</li>
            <li>API: {systemChecks.api.status}</li>
            <li>Temperature: {$state.context.settings.temperature}</li>
          </ul>
        </div>
      </div>

      <Button
        variant="outline"
        size="sm"
        class="mt-4"
        on:click={performSystemChecks}
      >
        Refresh System Status
      </Button>
    </CardContent>
  </Card>
</div>

<style>
  :global(.animate-bounce) {
    animation: bounce 1s infinite;
  }
</style>
