<!-- AI Chat Test Page - Showcasing Svelte 5 + bits-ui + Docker Ollama Integration -->
<script lang="ts">
  import { onMount } from "svelte";
  import Button from "$lib/components/ui/button/Button.svelte";
  import Card from "$lib/components/ui/Card.svelte";
  import Badge from "$lib/components/ui/Badge.svelte";
  import EnhancedAIChatTest from "$lib/components/ai/EnhancedAIChatTest.svelte";
  import {
    Bot,
    MessageCircle,
    Cpu,
    Database,
    Zap,
    CheckCircle,
    XCircle,
    Loader2,
    Server,
    HardDrive
  } from "lucide-svelte";

  // State using Svelte 5 runes
  let systemStatus = $state<{
    services?: {
      ollama: { status: string version?: string; error?: string };
      database: { status: string error?: string };
    };
    environment?: { ollamaUrl: string };
    // Legacy structure for compatibility
    ollama?: { status: string version?: string; error?: string };
    database?: { status: string error?: string };
  } | null>(null);

  let chatOpen = $state(false);
  let isLoading = $state(true);

  // Check system status on mount
  onMount(async () => {
    await checkSystemStatus();
  });

  async function checkSystemStatus() {
    try {
      isLoading = true;
      const response = await fetch("/api/system/check");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      
      // Normalize the response structure for compatibility
      systemStatus = {
        services: data.services,
        environment: data.environment,
        // Legacy compatibility
        ollama: data.services?.ollama || data.ollama,
        database: data.services?.database || data.database
      };
    } catch (error) {
      console.error("Failed to check system status:", error);
      systemStatus = {
        services: {
          ollama: { status: "error", error: "Failed to connect" },
          database: { status: "error", error: "Failed to connect" }
        },
        environment: { ollamaUrl: "unknown" },
        // Legacy compatibility
        ollama: { status: "error", error: "Failed to connect" },
        database: { status: "error", error: "Failed to connect" }
      };
    } finally {
      isLoading = false;
    }
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case "connected": return CheckCircle;
      case "error":
      case "disconnected":
        return XCircle;
      default: return Loader2;
    }
  }

  function getStatusColor(status: string) {
    switch (status) {
      case "connected": return "text-green-500";
      case "error":
      case "disconnected":
        return "text-red-500";
      default: return "text-yellow-500";
    }
  }

  function getStatusText(status: string) {
    switch (status) {
      case "connected": return "Connected";
      case "error": return "Error";
      case "disconnected": return "Disconnected";
      default: return "Checking...";
    }
  }
</script>

<svelte:head>
  <title>AI Chat Assistant Test - Deeds Legal AI</title>
  <meta name="description" content="Test page for the enhanced AI chat assistant with Docker Ollama integration" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
  <div class="container mx-auto px-4 py-8">
    <!-- Header -->
    <div class="text-center mb-8">
      <h1 class="text-4xl font-bold text-gray-900 mb-4">
        🤖 Enhanced AI Legal Assistant
      </h1>
      <p class="text-xl text-gray-600 max-w-3xl mx-auto">
        Powered by Gemma3 Legal Enhanced model running locally on RTX 3060 Ti GPU
        with Docker, Ollama, and Svelte 5 + bits-ui integration
      </p>
    </div>

    <!-- System Status Cards -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
      <!-- Ollama Service Status -->
      <Card class="p-6">
        <div class="flex items-center gap-3 mb-4">
          <div class="p-2 bg-blue-100 rounded-lg">
            <Bot class="h-6 w-6 text-blue-600" />
          </div>
          <div>
            <h3 class="text-lg font-semibold text-gray-900">Ollama Service</h3>
            <p class="text-sm text-gray-500">Local AI Model Server</p>
          </div>
        </div>

        {#if isLoading}
          <div class="flex items-center gap-2 text-yellow-600">
            <Loader2 class="h-4 w-4 animate-spin" />
            <span>Checking status...</span>
          </div>
        {:else if systemStatus}
          {@const StatusIcon = getStatusIcon(systemStatus.ollama.status)}
          <div class="flex items-center gap-2 {getStatusColor(systemStatus.ollama.status)}">
            <StatusIcon class="h-4 w-4" />
            <span>{getStatusText(systemStatus.ollama.status)}</span>
          </div>

          {#if systemStatus.ollama.version}
            <Badge variant="outline" class="mt-2">
              v{systemStatus.ollama.version}
            </Badge>
          {/if}

          {#if systemStatus.ollama.error}
            <p class="text-sm text-red-600 mt-2">
              {systemStatus.ollama.error}
            </p>
          {/if}
        {/if}
      </Card>

      <!-- Database Status -->
      <Card class="p-6">
        <div class="flex items-center gap-3 mb-4">
          <div class="p-2 bg-green-100 rounded-lg">
            <Database class="h-6 w-6 text-green-600" />
          </div>
          <div>
            <h3 class="text-lg font-semibold text-gray-900">Database</h3>
            <p class="text-sm text-gray-500">PostgreSQL with pgvector</p>
          </div>
        </div>

        {#if isLoading}
          <div class="flex items-center gap-2 text-yellow-600">
            <Loader2 class="h-4 w-4 animate-spin" />
            <span>Checking status...</span>
          </div>
        {:else if systemStatus}
          {@const StatusIcon = getStatusIcon(systemStatus.database.status)}
          <div class="flex items-center gap-2 {getStatusColor(systemStatus.database.status)}">
            <StatusIcon class="h-4 w-4" />
            <span>{getStatusText(systemStatus.database.status)}</span>
          </div>

          {#if systemStatus.database.error}
            <p class="text-sm text-red-600 mt-2">
              {systemStatus.database.error}
            </p>
          {/if}
        {/if}
      </Card>

      <!-- GPU/Hardware Status -->
      <Card class="p-6">
        <div class="flex items-center gap-3 mb-4">
          <div class="p-2 bg-purple-100 rounded-lg">
            <Cpu class="h-6 w-6 text-purple-600" />
          </div>
          <div>
            <h3 class="text-lg font-semibold text-gray-900">Hardware</h3>
            <p class="text-sm text-gray-500">GPU Acceleration</p>
          </div>
        </div>

        <div class="flex items-center gap-2 text-green-600">
          <CheckCircle class="h-4 w-4" />
          <span>RTX 3060 Ti Ready</span>
        </div>

        <div class="flex gap-2 mt-2">
          <Badge variant="outline">CUDA 12.9</Badge>
          <Badge variant="outline">8GB VRAM</Badge>
        </div>
      </Card>
    </div>

    <!-- Features Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <Card class="p-4 text-center">
        <MessageCircle class="h-8 w-8 text-blue-600 mx-auto mb-2" />
        <h4 class="font-semibold text-gray-900 mb-1">Real-time Chat</h4>
        <p class="text-sm text-gray-600">Streaming responses with GPU acceleration</p>
      </Card>

      <Card class="p-4 text-center">
        <HardDrive class="h-8 w-8 text-green-600 mx-auto mb-2" />
        <h4 class="font-semibold text-gray-900 mb-1">Local Processing</h4>
        <p class="text-sm text-gray-600">No data sent to external servers</p>
      </Card>

      <Card class="p-4 text-center">
        <Zap class="h-8 w-8 text-yellow-600 mx-auto mb-2" />
        <h4 class="font-semibold text-gray-900 mb-1">Legal Specialized</h4>
        <p class="text-sm text-gray-600">Fine-tuned for legal applications</p>
      </Card>

      <Card class="p-4 text-center">
        <Server class="h-8 w-8 text-purple-600 mx-auto mb-2" />
        <h4 class="font-semibold text-gray-900 mb-1">Docker Integrated</h4>
        <p class="text-sm text-gray-600">Containerized for easy deployment</p>
      </Card>
    </div>

    <!-- Chat Interface -->
    <div class="text-center mb-8">
      <div class="bg-white rounded-lg shadow-lg p-8 max-w-2xl mx-auto">
        <h2 class="text-2xl font-bold text-gray-900 mb-4">
          Test the AI Chat Assistant
        </h2>
        <p class="text-gray-600 mb-6">
          Click below to open the enhanced chat interface and start a conversation
          with your local AI legal assistant.
        </p>

        <EnhancedAIChatTest
          bind:open={chatOpen}
          caseId="TEST-CASE-001"
        />

        <div class="mt-4 text-sm text-gray-500">
          <p>💡 Try asking about legal concepts, case analysis, or document review</p>
        </div>
      </div>
    </div>

    <!-- Technical Details -->
    <Card class="p-6">
      <h3 class="text-xl font-bold text-gray-900 mb-4">Technical Implementation</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 class="font-semibold text-gray-900 mb-2">Frontend Stack</h4>
          <ul class="text-sm text-gray-600 space-y-1">
            <li>• Svelte 5 with runes and snippets</li>
            <li>• bits-ui headless components</li>
            <li>• shadcn-svelte UI components</li>
            <li>• TailwindCSS styling</li>
            <li>• TypeScript for type safety</li>
          </ul>
        </div>
        <div>
          <h4 class="font-semibold text-gray-900 mb-2">Backend Stack</h4>
          <ul class="text-sm text-gray-600 space-y-1">
            <li>• Docker containerization</li>
            <li>• Ollama for local LLM serving</li>
            <li>• Gemma3 legal-enhanced model</li>
            <li>• CUDA GPU acceleration</li>
            <li>• PostgreSQL with pgvector</li>
          </ul>
        </div>
      </div>

      {#if systemStatus}
        <div class="mt-4 p-4 bg-gray-50 rounded-lg">
          <h4 class="font-semibold text-gray-900 mb-2">Connection Details</h4>
          <p class="text-sm text-gray-600">
            Ollama URL: <code class="bg-white px-2 py-1 rounded">{systemStatus.environment.ollamaUrl}</code>
          </p>
        </div>
      {/if}
    </Card>

    <!-- Refresh Button -->
    <div class="text-center mt-8">
      <Button
        variant="outline"
        onclick={checkSystemStatus}
        disabled={isLoading}
        class="gap-2"
      >
        {#if isLoading}
          <Loader2 class="h-4 w-4 animate-spin" />
        {:else}
          <Server class="h-4 w-4" />
        {/if}
        Refresh System Status
      </Button>
    </div>
  </div>
</div>

<style>
  /* Custom styles for enhanced appearance */
  :global(body) {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }

  :global(code) {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  }
</style>
