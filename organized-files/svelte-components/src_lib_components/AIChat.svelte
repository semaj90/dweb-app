<script lang="ts">
  // @ts-nocheck
  import { onMount } from "svelte";
  import { writable, type Writable } from "svelte/store";
  const cn = (...classes: Array<string | false | null | undefined>) =>
    classes.filter(Boolean).join(" ");

  // Props
  export let caseId: string | null = null;
  export let sessionId: string | null = null;
  export let placeholder =
    "Ask about legal matters, case analysis, or evidence review...";
  export let maxHeight = "600px";
  export let enableRAG = true;
  export let maxContextChunks = 5;

  // Reactive stores
  /** @type {Writable<any[]>} */
  const messages = writable([] as any[]);
  const isLoading = writable(false);
  const isStreaming = writable(false);
  const currentInput = writable("");
  const systemStatus = writable({
    status: "checking",
    message: "Initializing AI system...",
  });
  /** @type {Writable<any[]>} */
  const contextSources = writable([] as any[]);

  // Component state
  let messageContainer: HTMLDivElement | undefined;
  let inputElement: HTMLInputElement | undefined;
  let currentSources: any[] = [];
  let chatSession = sessionId;
  let streamingMessage = "";
  let abortController: AbortController | null = null;

  // Configuration
  const API_BASE = "/api/chat";
  const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds

  onMount(async () => {
    await checkSystemHealth();
    // Set up periodic health checks
    const healthInterval = setInterval(
      checkSystemHealth,
      HEALTH_CHECK_INTERVAL
    );
    return () => clearInterval(healthInterval);
  });

  async function checkSystemHealth() {
    try {
      const response = await fetch(`${API_BASE}?action=health`);
      const data = await response.json();
      if (data.status === "healthy") {
        systemStatus.set({
          status: "ready",
          message: "AI system ready",
          details: data,
        });
      } else {
        systemStatus.set({
          status: "warning",
          message: "System partially available",
          details: data,
        });
      }
    } catch (error: any) {
      systemStatus.set({
        status: "error",
        message: "AI system unavailable",
        error: error.message,
      });
    }
  }

  async function sendMessage() {
    const input = $currentInput.trim();
    if (!input || $isLoading || $isStreaming) return;

    messages.update((msgs) => [
      ...msgs,
      {
        id: Date.now().toString(),
        role: "user",
        content: input,
        timestamp: new Date().toISOString(),
      },
    ]);

    currentInput.set("");
    isLoading.set(true);
    isStreaming.set(true);
    contextSources.set([]);
    streamingMessage = "";
    abortController = new AbortController();

    try {
      const response = await fetch(API_BASE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input,
          sessionId: chatSession,
          caseId,
          options: {
            stream: true,
            maxContextChunks,
            includeKnowledgeBase: enableRAG,
            includeDocuments: enableRAG,
          },
        }),
        signal: abortController.signal,
      });
      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body available");

      const assistantMessageId = Date.now().toString() + "_assistant";
      messages.update((msgs) => [
        ...msgs,
        {
          id: assistantMessageId,
          role: "assistant",
          content: "",
          timestamp: new Date().toISOString(),
          isStreaming: true,
        },
      ]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = new TextDecoder().decode(value);
        const lines = chunk.split("\n").filter((line) => line.trim());
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              await handleStreamingData(data, assistantMessageId);
            } catch (parseError) {
              console.error("Error parsing streaming data:", parseError);
            }
          }
        }
      }
    } catch (error: any) {
      if (error.name !== "AbortError") {
        console.error("Chat error:", error);
        messages.update((msgs) => [
          ...msgs,
          {
            id: Date.now().toString(),
            role: "system",
            content: `Error: ${error.message}. Please try again or check system status.`,
            timestamp: new Date().toISOString(),
            isError: true,
          },
        ]);
      }
    } finally {
      isLoading.set(false);
      isStreaming.set(false);
      abortController = null;
    }
    setTimeout(scrollToBottom, 100);
  }

  async function handleStreamingData(data: any, messageId: string) {
    switch (data.type) {
      case "status":
        systemStatus.update((status) => ({ ...status, message: data.message }));
        break;
      case "context":
        contextSources.set(data.sources || []);
        currentSources = data.sources || [];
        break;
      case "token":
        streamingMessage += data.content;
        messages.update((msgs) =>
          msgs.map((m) =>
            m.id === messageId ? { ...m, content: streamingMessage } : m
          )
        );
        break;
      case "complete":
        messages.update((msgs) =>
          msgs.map((m) =>
            m.id === messageId
              ? {
                  ...m,
                  isStreaming: false,
                  sources: data.sources,
                  responseTime: data.responseTime,
                  tokensUsed: data.tokensUsed,
                }
              : m
          )
        );
        if (data.sessionId && !chatSession) chatSession = data.sessionId;
        break;
      case "error":
        messages.update((msgs) =>
          msgs.map((m) =>
            m.id === messageId
              ? {
                  ...m,
                  content: `Error: ${data.error}`,
                  isStreaming: false,
                  isError: true,
                }
              : m
          )
        );
        break;
    }
  }

  function stopStreaming() {
    if (abortController) abortController.abort();
    isStreaming.set(false);
    isLoading.set(false);
  }

  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }
  function scrollToBottom() {
    if (messageContainer)
      messageContainer.scrollTop = messageContainer.scrollHeight;
  }
  function clearChat() {
    messages.set([]);
    contextSources.set([]);
    chatSession = null;
  }
  function formatTime(timestamp: string) {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  }
  function getStatusVariant(status: string) {
    switch (status) {
      case "ready":
        return "success";
      case "warning":
        return "warning";
      case "error":
        return "destructive";
      default:
        return "secondary";
    }
  }
</script>

<div class="chat" style={`max-height:${maxHeight}`}>
  <header class="chat__header">
    <h2>
      Legal AI Assistant {#if caseId}<span class="chat__case"
          >Case: {caseId}</span
        >{/if}
    </h2>
    <div class="chat__status">{$systemStatus.message}</div>
    {#if $messages.length > 0}
      <button class="chat__clear" onclick={clearChat}>Clear Chat</button>
    {/if}
  </header>

  <section class="chat__messages" bind:this={messageContainer}>
    {#each $messages as message (message.id)}
      <div
        class={cn(
          "msg",
          message.role === "user" ? "msg--user" : "msg--ai",
          message.isError && "msg--error"
        )}
      >
        <div class="msg__content">
          {#if message.role === "assistant" && message.isStreaming}
            {message.content} 
          {:else}
            {message.content}
          {/if}
        </div>
        {#if message.sources && message.sources.length > 0}
          <div class="msg__sources">
            <strong>Sources used:</strong>
            <ul>
              {#each message.sources as source}
                <li>
                  {source.type}: {source.title?.substring(0, 30)}... ({(
                    source.similarity * 100
                  ).toFixed(1)}%)
                </li>
              {/each}
            </ul>
          </div>
        {/if}
        {#if message.responseTime || message.tokensUsed}
          <div class="msg__meta">
            {#if message.responseTime}<span
                >Response: {message.responseTime}ms</span
              >{/if}
            {#if message.tokensUsed}<span>Tokens: {message.tokensUsed}</span
              >{/if}
          </div>
        {/if}
        <div class="msg__time">{formatTime(message.timestamp)}</div>
      </div>
    {/each}
  </section>

  {#if $isLoading && !$isStreaming}
    <div class="chat__loading">Processing your request...</div>
  {/if}

  {#if $contextSources.length > 0}
    <section class="chat__context">
      <h4>Using context from:</h4>
      <ul>
        {#each $contextSources as source}
          <li>
            {source.type}: {source.title?.substring(0, 40)}... ({(
              source.similarity * 100
            ).toFixed(1)}%)
          </li>
        {/each}
      </ul>
    </section>
  {/if}

  <footer class="chat__composer">
    <input
      bind:this={inputElement}
      bind:value={$currentInput}
      {placeholder}
      disabled={$isLoading || $systemStatus.status === "error"}
      on:keypress={handleKeyPress}
      class="composer__input"
    />
    {#if $isStreaming}
      <button class="composer__btn composer__btn--stop" onclick={stopStreaming}
        >Stop</button
      >
    {:else}
      <button
        class="composer__btn"
        onclick={sendMessage}
        disabled={!$currentInput.trim() ||
          $isLoading ||
          $systemStatus.status === "error"}>Send</button
      >
    {/if}
  </footer>

  {#if enableRAG}
    <div class="chat__rag">
      RAG-enhanced responses with legal context retrieval • Max context chunks: {maxContextChunks}
    </div>
  {/if}
</div>

<style>
  .chat {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 0.75rem;
    background: #0b0f19;
    color: #e5e7eb;
  }
  .chat__header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    justify-content: space-between;
  }
  .chat__case {
    font-weight: 500;
    margin-left: 0.5rem;
    color: #fbbf24;
  }
  .chat__status {
    font-size: 0.9rem;
    color: #9ca3af;
  }
  .chat__clear {
    background: transparent;
    color: #f87171;
    border: 1px solid #ef4444;
    border-radius: 6px;
    padding: 0.25rem 0.5rem;
  }
  .chat__messages {
    overflow: auto;
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .msg {
    max-width: 80%;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    border: 1px solid #374151;
  }
  .msg--user {
    background: #b45309;
    border-color: #92400e;
    color: #fff;
    margin-left: auto;
  }
  .msg--ai {
    background: #111827;
    color: #e5e7eb;
  }
  .msg--error {
    background: #7f1d1d;
    border-color: #991b1b;
    color: #fee2e2;
  }
  .msg__sources {
    margin-top: 0.25rem;
    font-size: 0.85rem;
    color: #9ca3af;
  }
  .msg__meta {
    margin-top: 0.25rem;
    display: flex;
    gap: 0.75rem;
    font-size: 0.8rem;
    color: #9ca3af;
  }
  .msg__time {
    margin-top: 0.25rem;
    font-size: 0.75rem;
    color: #9ca3af;
  }
  .chat__loading {
    text-align: center;
    color: #d1d5db;
  }
  .chat__context {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 0.5rem;
  }
  .chat__composer {
    display: flex;
    gap: 0.5rem;
  }
  .composer__input {
    flex: 1;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    border: 1px solid #374151;
    background: #0b0f19;
    color: #e5e7eb;
  }
  .composer__btn {
    padding: 0.5rem 0.75rem;
    background: #f59e0b;
    color: #111827;
    border: none;
    border-radius: 6px;
  }
  .composer__btn--stop {
    background: #ef4444;
    color: #fff;
  }
  .chat__rag {
    font-size: 0.85rem;
    color: #9ca3af;
  }
</style>
