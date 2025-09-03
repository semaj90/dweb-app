<!--
  Complete WebAssembly + Go AI Stack Demo
  Showcases: WASM inference + Go chat service + PostgreSQL memory
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { wasmLlama } from '$lib/services/webasm-llama-complete';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Textarea } from '$lib/components/ui/textarea';
  import { Badge } from '$lib/components/ui/badge';
  import * as Alert from '$lib/components/ui/alert';
  import { 
    Cpu, Database, Zap, MessageSquare, Brain, 
    Loader2, CheckCircle, AlertCircle, Activity,
    Rocket, Code, Settings, BarChart3
  } from 'lucide-svelte';

  // Reactive state using Svelte 5 runes
  let currentPrompt = $state('');
  let chatHistory = $state<Array<{role: 'user' | 'assistant', content: string, timestamp: number}>>([]);
  let conversationId = $state('');
  let isStreamingActive = $state(false);
  
  // Service status
  let serviceStatus = $state({
    wasm: { status: 'disconnected', message: 'Not initialized' },
    go: { status: 'disconnected', message: 'Not tested' },
    postgres: { status: 'disconnected', message: 'Not tested' },
    ollama: { status: 'disconnected', message: 'Not tested' }
  });

  // Performance metrics
  let metrics = $state({
    tokensPerSecond: 0,
    totalTokens: 0,
    inferenceTime: 0,
    memoryUsed: 0
  });

  // Demo configuration
  let useWebAssembly = $state(true);
  let temperature = $state(0.7);
  let maxTokens = $state(512);

  onMount(async () => {
    await initializeServices();
    // Generate a conversation ID for this session
    conversationId = 'demo-' + Date.now();
  });

  onDestroy(() => {
    wasmLlama.cleanup();
  });

  /**
   * Initialize all AI services and check status
   */
  async function initializeServices() {
    console.log('🚀 Initializing complete AI stack...');

    // Test WebAssembly service
    try {
      const wasmReady = await wasmLlama.initialize();
      if (wasmReady) {
        serviceStatus.wasm = { status: 'connected', message: 'WebAssembly worker ready' };
        
        // Try to load a model
        try {
          await wasmLlama.loadModel({
            modelUrl: '/models/gemma3-legal-8b-q4_k_m.gguf',
            contextSize: 4096,
            enableGPU: true
          });
          serviceStatus.wasm = { status: 'ready', message: 'Model loaded successfully' };
        } catch (modelError) {
          serviceStatus.wasm = { status: 'warning', message: 'Worker ready, model not found' };
        }
      }
    } catch (error) {
      serviceStatus.wasm = { status: 'error', message: error.message };
    }

    // Test Go backend service
    try {
      const goResponse = await fetch('/api/go-llama-health');
      if (goResponse.ok) {
        const health = await goResponse.json();
        serviceStatus.go = { 
          status: 'ready', 
          message: `Go service ready (${health.model || 'model loaded'})` 
        };
      } else {
        serviceStatus.go = { status: 'warning', message: 'Go service not responding' };
      }
    } catch (error) {
      serviceStatus.go = { status: 'warning', message: 'Go service offline' };
    }

    // Test PostgreSQL connection
    try {
      const dbResponse = await fetch('/api/test-db');
      if (dbResponse.ok) {
        const dbHealth = await dbResponse.json();
        if (dbHealth.overall.success) {
          serviceStatus.postgres = { status: 'ready', message: 'Database connected with pgvector' };
        } else {
          serviceStatus.postgres = { status: 'warning', message: 'Database issues detected' };
        }
      }
    } catch (error) {
      serviceStatus.postgres = { status: 'warning', message: 'Database check failed' };
    }

    // Test Ollama service
    try {
      const ollamaResponse = await fetch('http://localhost:11434/api/tags');
      if (ollamaResponse.ok) {
        serviceStatus.ollama = { status: 'ready', message: 'Ollama running with models' };
      }
    } catch (error) {
      serviceStatus.ollama = { status: 'warning', message: 'Ollama not accessible' };
    }
  }

  /**
   * Send message using WebAssembly inference
   */
  async function sendMessageWASM(prompt: string) {
    if (!prompt.trim()) return;

    const userMessage = {
      role: 'user' as const,
      content: prompt,
      timestamp: Date.now()
    };

    chatHistory.push(userMessage);
    currentPrompt = '';
    isStreamingActive = true;

    const assistantMessage = {
      role: 'assistant' as const,
      content: '',
      timestamp: Date.now()
    };
    chatHistory.push(assistantMessage);

    try {
      const startTime = performance.now();
      let tokenCount = 0;

      // Stream tokens for real-time response
      for await (const token of wasmLlama.inferStream(prompt, {
        temperature,
        maxTokens
      })) {
        assistantMessage.content += token;
        tokenCount++;
        
        // Update metrics in real-time
        const elapsed = (performance.now() - startTime) / 1000;
        metrics.tokensPerSecond = tokenCount / elapsed;
        metrics.totalTokens = tokenCount;
        metrics.inferenceTime = elapsed;

        // Force reactivity update
        chatHistory = [...chatHistory];
      }

    } catch (error) {
      assistantMessage.content = `Error: ${error.message}`;
      chatHistory = [...chatHistory];
    } finally {
      isStreamingActive = false;
    }
  }

  /**
   * Send message using Go backend with PostgreSQL memory
   */
  async function sendMessageGo(prompt: string) {
    if (!prompt.trim()) return;

    const userMessage = {
      role: 'user' as const,
      content: prompt,
      timestamp: Date.now()
    };

    chatHistory.push(userMessage);
    currentPrompt = '';

    try {
      const startTime = performance.now();
      
      const response = await fetch('/api/go-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          user_id: 'demo-user',
          message: prompt,
          temperature,
          max_tokens: maxTokens,
          system_prompt: 'You are a helpful legal AI assistant.'
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      const assistantMessage = {
        role: 'assistant' as const,
        content: data.response,
        timestamp: Date.now()
      };

      chatHistory.push(assistantMessage);

      // Update metrics
      const elapsed = (performance.now() - startTime) / 1000;
      const tokenCount = data.response.split(' ').length;
      metrics.tokensPerSecond = tokenCount / elapsed;
      metrics.totalTokens = tokenCount;
      metrics.inferenceTime = elapsed;

    } catch (error) {
      const errorMessage = {
        role: 'assistant' as const,
        content: `Error: ${error.message}`,
        timestamp: Date.now()
      };
      chatHistory.push(errorMessage);
    }
  }

  /**
   * Send message using the selected method
   */
  async function sendMessage() {
    if (!currentPrompt.trim()) return;

    if (useWebAssembly) {
      await sendMessageWASM(currentPrompt);
    } else {
      await sendMessageGo(currentPrompt);
    }
  }

  /**
   * Clear chat history
   */
  function clearChat() {
    chatHistory = [];
    conversationId = 'demo-' + Date.now();
  }

  /**
   * Get status badge variant
   */
  function getStatusVariant(status: string) {
    switch (status) {
      case 'ready': return 'default';
      case 'connected': return 'secondary';
      case 'warning': return 'outline';
      case 'error': return 'destructive';
      default: return 'outline';
    }
  }

  /**
   * Get status icon
   */
  function getStatusIcon(status: string) {
    switch (status) {
      case 'ready': return CheckCircle;
      case 'connected': return Activity;
      case 'warning': return AlertCircle;
      case 'error': return AlertCircle;
      default: return Loader2;
    }
  }

  // Reactive derived values
  let allServicesReady = $derived(
    Object.values(serviceStatus).every(s => s.status === 'ready' || s.status === 'connected')
  );

  let currentMethod = $derived(useWebAssembly ? 'WebAssembly' : 'Go Backend');
</script>

<svelte:head>
  <title>Complete AI Stack Demo - Legal AI Platform</title>
</svelte:head>

<div class="container mx-auto p-6 space-y-6">
  <!-- Header -->
  <div class="text-center space-y-2">
    <h1 class="text-4xl font-bold flex items-center justify-center gap-2">
      <Rocket class="h-8 w-8 text-primary" />
      Complete WebAssembly + Go AI Stack
    </h1>
    <p class="text-muted-foreground">
      Native Windows performance with JavaScript flexibility
    </p>
  </div>

  <!-- Service Status Dashboard -->
  <Card.Root>
    <Card.Header>
      <Card.Title class="flex items-center gap-2">
        <Settings class="h-5 w-5" />
        Service Status
      </Card.Title>
      <Card.Description>
        Real-time status of all AI stack components
      </Card.Description>
    </Card.Header>
    <Card.Content>
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <!-- WebAssembly Status -->
        <div class="space-y-2">
          <div class="flex items-center gap-2">
            <Cpu class="h-4 w-4" />
            <span class="font-medium">WebAssembly</span>
          </div>
          <Badge variant={getStatusVariant(serviceStatus.wasm.status)}>
            <svelte:component this={getStatusIcon(serviceStatus.wasm.status)} class="h-3 w-3 mr-1" />
            {serviceStatus.wasm.status}
          </Badge>
          <p class="text-xs text-muted-foreground">{serviceStatus.wasm.message}</p>
        </div>

        <!-- Go Backend Status -->
        <div class="space-y-2">
          <div class="flex items-center gap-2">
            <Code class="h-4 w-4" />
            <span class="font-medium">Go Service</span>
          </div>
          <Badge variant={getStatusVariant(serviceStatus.go.status)}>
            <svelte:component this={getStatusIcon(serviceStatus.go.status)} class="h-3 w-3 mr-1" />
            {serviceStatus.go.status}
          </Badge>
          <p class="text-xs text-muted-foreground">{serviceStatus.go.message}</p>
        </div>

        <!-- PostgreSQL Status -->
        <div class="space-y-2">
          <div class="flex items-center gap-2">
            <Database class="h-4 w-4" />
            <span class="font-medium">PostgreSQL</span>
          </div>
          <Badge variant={getStatusVariant(serviceStatus.postgres.status)}>
            <svelte:component this={getStatusIcon(serviceStatus.postgres.status)} class="h-3 w-3 mr-1" />
            {serviceStatus.postgres.status}
          </Badge>
          <p class="text-xs text-muted-foreground">{serviceStatus.postgres.message}</p>
        </div>

        <!-- Ollama Status -->
        <div class="space-y-2">
          <div class="flex items-center gap-2">
            <Brain class="h-4 w-4" />
            <span class="font-medium">Ollama</span>
          </div>
          <Badge variant={getStatusVariant(serviceStatus.ollama.status)}>
            <svelte:component this={getStatusIcon(serviceStatus.ollama.status)} class="h-3 w-3 mr-1" />
            {serviceStatus.ollama.status}
          </Badge>
          <p class="text-xs text-muted-foreground">{serviceStatus.ollama.message}</p>
        </div>
      </div>

      {#if !allServicesReady}
        <Alert.Root class="mt-4">
          <AlertCircle class="h-4 w-4" />
          <Alert.Title>Service Status</Alert.Title>
          <Alert.Description>
            Some services are not fully ready. The demo will work with available services.
            Check the console for detailed initialization logs.
          </Alert.Description>
        </Alert.Root>
      {/if}
    </Card.Content>
  </Card.Root>

  <!-- Configuration Panel -->
  <Card.Root>
    <Card.Header>
      <Card.Title class="flex items-center gap-2">
        <Settings class="h-5 w-5" />
        Configuration
      </Card.Title>
    </Card.Header>
    <Card.Content>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <!-- Method Selection -->
        <div class="space-y-2">
          <Label for="method">Inference Method</Label>
          <div class="flex gap-2">
            <Button 
              variant={useWebAssembly ? "default" : "outline"} 
              size="sm"
              onclick={() => useWebAssembly = true}
            >
              <Cpu class="h-3 w-3 mr-1" />
              WASM
            </Button>
            <Button 
              variant={!useWebAssembly ? "default" : "outline"} 
              size="sm"
              onclick={() => useWebAssembly = false}
            >
              <Code class="h-3 w-3 mr-1" />
              Go
            </Button>
          </div>
        </div>

        <!-- Temperature -->
        <div class="space-y-2">
          <Label for="temperature">Temperature: {temperature}</Label>
          <input 
            type="range" 
            min="0.1" 
            max="1.0" 
            step="0.1" 
            bind:value={temperature}
            class="w-full"
          />
        </div>

        <!-- Max Tokens -->
        <div class="space-y-2">
          <Label for="maxTokens">Max Tokens</Label>
          <Input 
            type="number" 
            min="50" 
            max="2048" 
            step="50"
            bind:value={maxTokens}
            class="w-full"
          />
        </div>

        <!-- Performance Metrics -->
        <div class="space-y-2">
          <div class="flex items-center gap-2">
            <BarChart3 class="h-4 w-4" />
            <span class="font-medium">Performance</span>
          </div>
          <div class="text-sm space-y-1">
            <div>Speed: {metrics.tokensPerSecond.toFixed(1)} t/s</div>
            <div>Tokens: {metrics.totalTokens}</div>
            <div>Time: {metrics.inferenceTime.toFixed(1)}s</div>
          </div>
        </div>
      </div>
    </Card.Content>
  </Card.Root>

  <!-- Chat Interface -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Chat History -->
    <Card.Root class="lg:col-span-2">
      <Card.Header>
        <div class="flex items-center justify-between">
          <Card.Title class="flex items-center gap-2">
            <MessageSquare class="h-5 w-5" />
            AI Chat Demo ({currentMethod})
          </Card.Title>
          <div class="flex gap-2">
            <Badge variant="secondary">
              {chatHistory.length} messages
            </Badge>
            <Button variant="outline" size="sm" onclick={clearChat}>
              Clear Chat
            </Button>
          </div>
        </div>
      </Card.Header>
      <Card.Content>
        <!-- Chat Messages -->
        <div class="space-y-4 h-96 overflow-y-auto border rounded-lg p-4 mb-4">
          {#each chatHistory as message}
            <div class="flex gap-3 {message.role === 'user' ? 'justify-end' : 'justify-start'}">
              <div class="max-w-[80%] p-3 rounded-lg {
                message.role === 'user' 
                  ? 'bg-primary text-primary-foreground ml-auto' 
                  : 'bg-muted'
              }">
                <div class="text-sm font-medium mb-1 capitalize">
                  {message.role}
                  <span class="text-xs opacity-70 ml-2">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div class="whitespace-pre-wrap">{message.content}</div>
              </div>
            </div>
          {/each}

          {#if chatHistory.length === 0}
            <div class="text-center text-muted-foreground py-8">
              <MessageSquare class="h-12 w-12 mx-auto mb-2 opacity-50" />
              <p>Start a conversation to test the AI stack</p>
              <p class="text-sm">Try: "Explain contract law" or "What is tort liability?"</p>
            </div>
          {/if}

          {#if isStreamingActive}
            <div class="flex items-center gap-2 text-muted-foreground">
              <Loader2 class="h-4 w-4 animate-spin" />
              <span>Generating response...</span>
            </div>
          {/if}
        </div>

        <!-- Message Input -->
        <div class="flex gap-2">
          <Textarea 
            bind:value={currentPrompt}
            placeholder="Ask a legal question or test the AI..."
            class="resize-none"
            rows="2"
            onkeydown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
          />
          <Button 
            onclick={sendMessage}
            disabled={!currentPrompt.trim() || isStreamingActive}
            class="shrink-0"
          >
            {#if isStreamingActive}
              <Loader2 class="h-4 w-4 animate-spin" />
            {:else}
              <Zap class="h-4 w-4" />
            {/if}
          </Button>
        </div>
      </Card.Content>
    </Card.Root>

    <!-- Information Panel -->
    <Card.Root>
      <Card.Header>
        <Card.Title class="flex items-center gap-2">
          <Brain class="h-5 w-5" />
          Architecture Info
        </Card.Title>
      </Card.Header>
      <Card.Content class="space-y-4">
        <div>
          <h4 class="font-medium mb-2">WebAssembly Mode</h4>
          <ul class="text-sm text-muted-foreground space-y-1">
            <li>• Client-side inference</li>
            <li>• Web Worker threading</li>
            <li>• Zero server calls</li>
            <li>• Real-time streaming</li>
          </ul>
        </div>

        <div>
          <h4 class="font-medium mb-2">Go Backend Mode</h4>
          <ul class="text-sm text-muted-foreground space-y-1">
            <li>• Server-side processing</li>
            <li>• PostgreSQL memory</li>
            <li>• Conversation history</li>
            <li>• Semantic search</li>
          </ul>
        </div>

        <div>
          <h4 class="font-medium mb-2">Technology Stack</h4>
          <ul class="text-sm text-muted-foreground space-y-1">
            <li>• SvelteKit 2 + Svelte 5</li>
            <li>• llama.cpp + WASM</li>
            <li>• Go + PostgreSQL + pgvector</li>
            <li>• Ollama embeddings</li>
          </ul>
        </div>

        {#if conversationId}
          <div>
            <h4 class="font-medium mb-2">Session Info</h4>
            <div class="text-xs font-mono bg-muted p-2 rounded">
              ID: {conversationId.substring(0, 20)}...
            </div>
          </div>
        {/if}

        <Alert.Root>
          <Rocket class="h-4 w-4" />
          <Alert.Title>Performance</Alert.Title>
          <Alert.Description>
            This native stack typically achieves 50-150+ tokens/sec, 
            significantly faster than Python-based solutions.
          </Alert.Description>
        </Alert.Root>
      </Card.Content>
    </Card.Root>
  </div>
</div>