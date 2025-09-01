<!-- @migration-task Error while migrating Svelte code: `{@const}` must be the immediate child of `{#snippet}`, `{#if}`, `{:else if}`, `{:else}`, `{#each}`, `{:then}`, `{:catch}`, `<svelte:fragment>`, `<svelte:boundary` or `<Component>`
https://svelte.dev/e/const_tag_invalid_placement -->
<!-- AI Demo Page - SvelteKit 2.0 + Svelte 5 + Ollama Integration -->
<script lang="ts">
  import OllamaChatInterface from "$lib/components/OllamaChatInterface.svelte";
  import { Alert, AlertDescription } from "$lib/components/ui/alert";
  import { Badge } from "$lib/components/ui/badge";
  import { Button } from "$lib/components/ui/button";
  import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
  } from "$lib/components/ui/card";
  import {
    AlertTriangle,
    Brain,
    CheckCircle,
    Cpu,
    Database,
    MessageSquare,
    Settings,
    Zap,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  // Reactive state using Svelte 5 runes
  let ollamaStatus = $state<"checking" | "healthy" | "unhealthy">("checking");
  let availableModels = $state<string[]>([]);
  let selectedModel = $state("gemma3-legal");
  let systemStats = $state<{
    memory: string;
    containers: number;
    models: number;
  }>({ memory: "0MB", containers: 0, models: 0 });

  // Check Ollama health and available models
  async function checkOllamaHealth() {
    try {
      const response = await fetch("/api/ai/health");
      const data = await response.json();

      if (data.ollama?.healthy) {
        ollamaStatus = "healthy";
        availableModels = data.ollama.models || [];
      } else {
        ollamaStatus = "unhealthy";
      }

      // Update system stats
      systemStats = {
        memory: data.system?.memory || "0MB",
        containers: data.docker?.containers || 0,
        models: data.ollama?.models?.length || 0,
      };
    } catch (error) {
      console.error("Health check failed:", error);
      ollamaStatus = "unhealthy";
    }
  }

  // Test AI generation
  async function testGeneration() {
    const testPrompt =
      "What are the key elements of a criminal case prosecution?";

    try {
      const response = await fetch("/api/ai/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: testPrompt,
          model: selectedModel,
          useRAG: false,
        }),
      });

      const result = await response.json();
      console.log("Test generation result:", result);
    } catch (error) {
      console.error("Test generation failed:", error);
    }
  }

  onMount(() => {
    checkOllamaHealth();
    // Refresh health every 30 seconds
    const interval = setInterval(checkOllamaHealth, 30000);
    return () => clearInterval(interval);
  });

  // Computed values
  let healthColor = $derived(
    ollamaStatus === "healthy"
      ? "text-green-600"
      : ollamaStatus === "unhealthy"
        ? "text-red-600"
        : "text-yellow-600"
  );

  let healthIcon = $derived(
    ollamaStatus === "healthy" ? CheckCircle : AlertTriangle
  );
</script>

<svelte:head>
  <title>Legal AI Demo - Ollama Integration</title>
  <meta
    name="description"
    content="Legal AI system powered by Ollama and SvelteKit"
  />
</svelte:head>

<div class="container mx-auto p-6 space-y-6">
  <!-- Header -->
  <div class="text-center">
    <h1 class="text-4xl font-bold text-gray-900 dark:text-white mb-2">
      Legal AI System Demo
    </h1>
    <p class="text-xl text-gray-600 dark:text-gray-300">
      SvelteKit 2.0 + Svelte 5 + Ollama Integration
    </p>
  </div>

  <!-- System Status -->
  <Card>
    <CardHeader>
      <CardTitle class="flex items-center gap-2">
        <Cpu class="h-5 w-5" />
        System Status
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <!-- Ollama Status -->
        <div class="flex items-center gap-2">
          {#if healthIcon}
            {@const Icon = healthIcon}
            <Icon class={`h-5 w-5 ${healthColor}`} />
          {/if}
          <div>
            <p class="font-medium">Ollama Service</p>
            <p class="text-sm text-gray-500 capitalize">{ollamaStatus}</p>
          </div>
        </div>

        <!-- Available Models -->
        <div class="flex items-center gap-2">
          <Brain class="h-5 w-5 text-blue-600" />
          <div>
            <p class="font-medium">AI Models</p>
            <p class="text-sm text-gray-500">{systemStats.models} loaded</p>
          </div>
        </div>

        <!-- Docker Containers -->
        <div class="flex items-center gap-2">
          <Database class="h-5 w-5 text-purple-600" />
          <div>
            <p class="font-medium">Containers</p>
            <p class="text-sm text-gray-500">
              {systemStats.containers} running
            </p>
          </div>
        </div>

        <!-- Memory Usage -->
        <div class="flex items-center gap-2">
          <Zap class="h-5 w-5 text-yellow-600" />
          <div>
            <p class="font-medium">Memory</p>
            <p class="text-sm text-gray-500">{systemStats.memory}</p>
          </div>
        </div>
      </div>

      {#if ollamaStatus === "unhealthy"}
        <Alert class="mt-4">
          <AlertTriangle class="h-4 w-4" />
          <AlertDescription>
            Ollama service is not available. Please run:
            <code class="bg-gray-100 px-2 py-1 rounded ml-2"
              >npm run ollama:start</code
            >
          </AlertDescription>
        </Alert>
      {/if}
    </CardContent>
  </Card>

  <!-- Model Selection -->
  {#if ollamaStatus === "healthy" && availableModels.length > 0}
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center gap-2">
          <Settings class="h-5 w-5" />
          Model Configuration
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="flex flex-wrap gap-2">
          {#each availableModels as model}
            <Badge
              variant={selectedModel === model ? "default" : "outline"}
              class="cursor-pointer"
              on:on:click={() => (selectedModel = model)}
            >
              {model}
            </Badge>
          {/each}
        </div>

        <div class="mt-4 flex gap-2">
          <Button on:on:click={testGeneration} variant="outline">
            Test Generation
          </Button>
          <Button on:on:click={checkOllamaHealth} variant="outline">
            Refresh Status
          </Button>
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Chat Interface -->
  {#if ollamaStatus === "healthy"}
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center gap-2">
          <MessageSquare class="h-5 w-5" />
          Legal AI Chat Interface
        </CardTitle>
      </CardHeader>
      <CardContent>
        <OllamaChatInterface
          {selectedModel}
          useRAG={true}
          class="min-h-[400px]"
          data-testid="ollama-chat"
        />
      </CardContent>
    </Card>
  {/if}

  <!-- Quick Start Guide -->
  <Card>
    <CardHeader>
      <CardTitle>🚀 Quick Start Guide</CardTitle>
    </CardHeader>
    <CardContent>
      <div class="space-y-4">
        <div>
          <h3 class="font-semibold mb-2">1. Start Ollama Service</h3>
          <code class="bg-gray-100 dark:bg-gray-800 px-3 py-2 rounded block">
            npm run ollama:start
          </code>
        </div>

        <div>
          <h3 class="font-semibold mb-2">2. Setup Legal AI Models</h3>
          <code class="bg-gray-100 dark:bg-gray-800 px-3 py-2 rounded block">
            npm run ollama:setup
          </code>
        </div>

        <div>
          <h3 class="font-semibold mb-2">3. Start Full Development Stack</h3>
          <code class="bg-gray-100 dark:bg-gray-800 px-3 py-2 rounded block">
            npm run dev:full
          </code>
        </div>

        <div>
          <h3 class="font-semibold mb-2">4. GPU Acceleration (if available)</h3>
          <code class="bg-gray-100 dark:bg-gray-800 px-3 py-2 rounded block">
            npm run dev:gpu
          </code>
        </div>
      </div>
    </CardContent>
  </Card>
</div>

<style>
  :global(body) {
    font-family: "Inter", system-ui, sans-serif;
  }
</style>
