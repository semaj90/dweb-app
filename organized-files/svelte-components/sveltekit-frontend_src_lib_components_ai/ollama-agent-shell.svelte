<!-- Ollama Agent Shell - Real-time Terminal Modal with Streaming Support -->
<script lang="ts">
  import { agentShellMachine } from "$lib/machines/agentShellMachine";
  import { cn } from "$lib/utils";
  import { useMachine } from "@xstate/svelte";
  import * as Dialog from "bits-ui/dialog";
  import { Bot, Check, Copy, Send, Terminal, User, X } from "lucide-svelte";
  import { onDestroy, onMount } from "svelte";

  // Props with Svelte 5 runes
  let {
    open = $bindable(false),
    docId = null,
    initialPrompt = "",
  }: {
    open?: boolean;
    docId?: string | null;
    initialPrompt?: string;
  } = $props();

  const { state, send } = useMachine(agentShellMachine);

  interface Message {
    role: "user" | "assistant" | "system";
    content: string
    timestamp: Date
    status?: "pending" | "streaming" | "complete" | "error";
    embeddings?: number[];
  }

  // State
  let messages: Message[] = $state([]);
  let input = $state("");
  let isLoading = $state(false);
  let copiedIndex = $state<number | null>(null);
  let terminalElement: HTMLDivElement
  let inputElement: HTMLTextAreaElement

  // WebSocket for real-time updates
  let ws: WebSocket | null = null;

  onMount(() => {
    // Initialize with system message
    messages.push({
      role: "system",
      content: `🚀 Ollama Agent Shell v1.0
Connected to: nomic-embed-text, gemma:3b
GPU: ${navigator.gpu ? "Enabled" : "Disabled"}
Type /help for commands`,
      timestamp: new Date(),
      status: "complete",
    });

    // Connect WebSocket if docId provided
    if (docId) {
      connectWebSocket();
    }

    // Process initial prompt
    if (initialPrompt) {
      input = initialPrompt;
      handleSubmit();
    }
  });

  onDestroy(() => {
    ws?.close();
  });

  function connectWebSocket() {
    if (!docId) return;

    ws = new WebSocket(`ws://localhost:8080/ws?docId=${docId}`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "status_update") {
        messages.push({
          role: "system",
          content: data.message,
          timestamp: new Date(),
          status: "complete",
        });
      }
    };

    ws.onerror = () => {
      messages.push({
        role: "system",
        content: "⚠️ WebSocket disconnected",
        timestamp: new Date(),
        status: "error",
      });
    };
  }

  async function handleSubmit() {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: "user",
      content: input,
      timestamp: new Date(),
      status: "complete",
    };
    messages.push(userMessage);

    // Handle commands
    if (input.startsWith("/")) {
      handleCommand(input);
      input = "";
      return;
    }

    // Send to XState machine
    send({ type: "PROMPT", input });

    input = "";
    isLoading = true;

    // Add placeholder for response
    const assistantMessage: Message = {
      role: "assistant",
      content: "",
      timestamp: new Date(),
      status: "pending",
    };
    messages.push(assistantMessage);

    try {
      // Get embeddings
      const embedResponse = await fetch("http://localhost:8081/batch-embed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          docId: docId || "shell-" + Date.now(),
          chunks: [userMessage.content],
          model: "nomic-embed-text",
        }),
      });

      if (embedResponse.ok) {
        const embedData = await embedResponse.json();
        messages[messages.length - 1].embeddings = embedData.embeddings[0];
      }

      // Stream response from Ollama
      const response = await fetch("http://localhost:11434/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemma:3b",
          prompt: userMessage.content,
          stream: true,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error("Failed to get response");
      }

      messages[messages.length - 1].status = "streaming";

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let content = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              if (data.response) {
                content += data.response;
                messages[messages.length - 1].content = content;
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }

      messages[messages.length - 1].status = "complete";
    } catch (error) {
      messages[messages.length - 1].content = `❌ Error: ${error}`;
      messages[messages.length - 1].status = "error";
    } finally {
      isLoading = false;
      scrollToBottom();
    }
  }

  function handleCommand(cmd: string) {
    const command = cmd.slice(1).toLowerCase();

    switch (command) {
      case "help":
        messages.push({
          role: "system",
          content: `Commands:
/help - Show this help
/clear - Clear chat
/embed - Show embeddings
/gpu - GPU status
/export - Export chat`,
          timestamp: new Date(),
          status: "complete",
        });
        break;

      case "clear":
        messages = [
          {
            role: "system",
            content: "🧹 Cleared",
            timestamp: new Date(),
            status: "complete",
          },
        ];
        break;

      case "embed":
        const lastEmbed = messages.findLast((m) => m.embeddings);
        if (lastEmbed?.embeddings) {
          messages.push({
            role: "system",
            content: `Embeddings (first 10): [${lastEmbed.embeddings
              .slice(0, 10)
              .map((e: number) => e.toFixed(3))
              .join(", ")}...]`,
            timestamp: new Date(),
            status: "complete",
          });
        }
        break;

      case "gpu":
        checkGPUStatus();
        break;

      case "export":
        exportChat();
        break;

      default:
        messages.push({
          role: "system",
          content: `Unknown command: ${cmd}`,
          timestamp: new Date(),
          status: "error",
        });
    }
  }

  async function checkGPUStatus() {
    if (navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      messages.push({
        role: "system",
        content: `🎮 GPU: ${adapter?.name || "Available"}`,
        timestamp: new Date(),
        status: "complete",
      });
    } else {
      messages.push({
        role: "system",
        content: "❌ WebGPU not available",
        timestamp: new Date(),
        status: "error",
      });
    }
  }

  function exportChat() {
    const data = JSON.stringify(messages, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat-${Date.now()}.json`;
    a.click();

    messages.push({
      role: "system",
      content: "✅ Exported",
      timestamp: new Date(),
      status: "complete",
    });
  }

  function copyMessage(content: string, index: number) {
    navigator.clipboard.writeText(content);
    copiedIndex = index;
    setTimeout(() => (copiedIndex = null), 2000);
  }

  function scrollToBottom() {
    terminalElement?.scrollTo(0, terminalElement.scrollHeight);
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  // Reactive state from XState
  let xstateResponse = $derived($state.context.response)
  $: if (
    xstateResponse &&
    messages.length > 0 &&
    messages[messages.length - 1].role === "assistant"
  ) {
    messages[messages.length - 1].content = xstateResponse;
  }
</script>

<Dialog.Root bind:open>
  <Dialog.Trigger
    class="fixed bottom-4 right-4 p-3 bg-primary text-primary-foreground rounded-full shadow-lg hover:scale-105 transition-transform"
  >
    <Terminal class="h-5 w-5" />
  </Dialog.Trigger>

  <Dialog.Portal>
    <Dialog.Overlay class="fixed inset-0 bg-black/50 backdrop-blur-sm" />
    <Dialog.Content
      class="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[90vw] max-w-4xl h-[80vh] bg-background border rounded-lg shadow-xl flex flex-col"
    >
      <div class="flex items-center justify-between p-4 border-b">
        <div class="flex items-center gap-2">
          <Terminal class="h-5 w-5" />
          <h2 class="text-lg font-semibold">Ollama Agent Shell</h2>
          {#if $state.matches("processing")}
            <span class="text-sm text-muted-foreground animate-pulse"
              >Processing...</span
            >
          {/if}
        </div>
        <Dialog.Close class="p-1 hover:bg-muted rounded">
          <X class="h-4 w-4" />
        </Dialog.Close>
      </div>

      <div
        bind:this={terminalElement}
        class="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {#each messages as message, i}
          <div class="flex items-start gap-3 group">
            {#if message.role === "assistant"}
              <div class="p-2 bg-primary bg-opacity-10 rounded-full shrink-0">
                <Bot class="h-4 w-4" />
              </div>
            {:else if message.role === "user"}
              <div class="p-2 bg-muted rounded-full shrink-0">
                <User class="h-4 w-4" />
              </div>
            {:else}
              <div class="p-2 bg-accent rounded-full shrink-0">
                <Terminal class="h-4 w-4" />
              </div>
            {/if}

            <div class="flex-1">
              <div class="flex items-center gap-2 mb-1">
                <span class="text-xs text-muted-foreground">
                  {message.timestamp.toLocaleTimeString()}
                </span>
                {#if message.status === "streaming"}
                  <span class="text-xs text-blue-500 animate-pulse">●</span>
                {:else if message.status === "error"}
                  <span class="text-xs text-red-500">Error</span>
                {/if}
              </div>

              <pre
                class="whitespace-pre-wrap font-mono text-sm">{message.content}</pre>
            </div>

            <button
              on:click={() => copyMessage(message.content, i)}
              class="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-muted rounded"
            >
              {#if copiedIndex === i}
                <Check class="h-4 w-4 text-green-500" />
              {:else}
                <Copy class="h-4 w-4" />
              {/if}
            </button>
          </div>
        {/each}
      </div>

      <div class="border-t p-4">
        <div class="flex items-end gap-2">
          <textarea
            bind:this={inputElement}
            bind:value={input}
            onkeydown={handleKeyDown}
            placeholder="Type a message or /help for commands..."
            class="flex-1 min-h-[60px] max-h-[120px] p-3 bg-muted rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-primary"
            disabled={isLoading}
          />
          <button
            on:click={handleSubmit}
            disabled={isLoading || !input.trim()}
            class={cn(
              "p-3 rounded-lg transition-colors",
              "bg-primary text-primary-foreground hover:bg-primary hover:bg-opacity-90",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
          >
            <Send class="h-4 w-4" />
          </button>
        </div>
      </div>
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>

<style>
  pre {
    font-family: "Cascadia Code", "SF Mono", Consolas, monospace;
  }
</style>
