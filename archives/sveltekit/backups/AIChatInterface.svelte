<!-- AI Chat Interface Component - SSR-safe with proper hydration -->
<script lang="ts">
  import { browser } from "$app/environment";
  import { onDestroy, onMount, tick } from "svelte";
  import { aiStore, conversation, status } from "../../../lib/stores/ai-store";
  import AIChatInput from "./AIChatInput.svelte";
  import AIChatMessage from "./AIChatMessage.svelte";
  import AIStatusIndicator from "./AIStatusIndicator.svelte";
  // Real-time evidence integration
  import { evidenceStore } from "../../stores/evidenceStore";

  // Props
  export let placeholder = "Ask a legal question...";
  export let maxHeight = "500px";
  export let showHistory = true;
  export let autoFocus = false;
  export let className = "";
  export let caseId: string | undefined = undefined;

  // Reactive state
  let chatContainer: HTMLElement;
  let isInitialized = false;
  let currentInput = "";
  let isProcessing = false;
  let currentEvidence: unknown[] = [];

  // Subscribe to real-time evidence for context
  let unsubscribeEvidence: (() => void) | undefined;

  // Initialize AI system on mount
  onMount(async () => {
    if (browser) {
      try {
        await aiStore.initialize();
        isInitialized = true;

        // Subscribe to real-time evidence updates for AI context
        if (caseId) {
          unsubscribeEvidence = evidenceStore.evidence.subscribe(
            (evidenceList) => {
              currentEvidence = evidenceList.filter((e) => e.caseId === caseId);
            }
          );
        }
      } catch (error) {
        console.error("Failed to initialize AI:", error);
      }
    }
  });

  // Auto-scroll to bottom when new messages arrive
  $: if (browser && $conversation.messages.length > 0) {
    tick().then(() => {
      if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    });
  }

  // Handle message sending
  async function handleSendMessage(content: string) {
    if (!content.trim() || isProcessing || !isInitialized) return;

    isProcessing = true;
    currentInput = "";

    try {
      const response = await aiStore.sendMessage(content, {
        includeHistory: showHistory,
        maxSources: 5,
        searchThreshold: 0.7,
        useCache: true,
        caseId,
      });

      if (!response) {
        throw new Error("Failed to get AI response");
      }
    } catch (error) {
      console.error("Failed to send message:", error);
      // Error is handled in the store and reflected in $status
    } finally {
      isProcessing = false;
    }
  }

  // Handle input changes
  function handleInputChange(event: CustomEvent<string>) {
    currentInput = event.detail;
  }

  // Handle clear conversation
  function handleClearConversation() {
    aiStore.clearConversation();
  }

  // Handle save conversation
  function handleSaveConversation() {
    aiStore.saveConversationToHistory();
  }

  // Cleanup on destroy
  onDestroy(() => {
    if (unsubscribeEvidence) {
      unsubscribeEvidence();
    }
  });
</script>

<div class="mx-auto px-4 max-w-7xl" style="max-height: {maxHeight}">
  <!-- Status Bar -->
  <div class="mx-auto px-4 max-w-7xl">
    <AIStatusIndicator
      isReady={$status.localModelAvailable || $status.cloudModelAvailable}
      isLoading={$status.isLoading || $status.isInitializing}
      provider={$status.currentProvider}
      model={$status.currentModel}
      error={$status.error}
    />

    {#if $conversation.messages.length > 0}
      <div class="mx-auto px-4 max-w-7xl">
        <button
          type="button"
          class="mx-auto px-4 max-w-7xl"
          onclick={() => handleSaveConversation()}
          title="Save conversation to history"
          aria-label="Save conversation to history"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
          </svg>
        </button>

        <button
          type="button"
          class="mx-auto px-4 max-w-7xl"
          onclick={() => handleClearConversation()}
          title="Clear conversation"
          aria-label="Clear conversation"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <path d="M3 6h18" />
            <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
            <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
          </svg>
        </button>
      </div>
    {/if}
  </div>

  <!-- Chat Messages -->
  <div
    class="mx-auto px-4 max-w-7xl"
    bind:this={chatContainer}
    role="log"
    aria-live="polite"
    aria-label="AI conversation"
  >
    {#if !isInitialized}
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl"></div>
        <p>Initializing AI system...</p>
      </div>
    {:else if $conversation.messages.length === 0}
      <div class="mx-auto px-4 max-w-7xl">
        <h3>Legal AI Assistant</h3>
        <p>Ask me anything about legal matters. I can help with:</p>
        <ul>
          <li>Contract analysis and interpretation</li>
          <li>Legal research and case law</li>
          <li>Document review and drafting</li>
          <li>Compliance and regulatory questions</li>
        </ul>
        <p class="mx-auto px-4 max-w-7xl">
          <strong>Note:</strong> This AI provides general information only and does
          not constitute legal advice.
        </p>
      </div>
    {:else}
      {#each $conversation.messages as message (message.id)}
        <AIChatMessage
          {message}
          showSources={message.role === "assistant" &&
            message.sources &&
            message.sources.length > 0}
          showMetadata={message.role === "assistant" && message.metadata}
        />
      {/each}
    {/if}

    {#if isProcessing}
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl"></div>
        <p>Processing your request...</p>
      </div>
    {/if}
  </div>

  <!-- Chat Input -->
  <div class="mx-auto px-4 max-w-7xl">
    <AIChatInput
      {placeholder}
      {autoFocus}
      disabled={!isInitialized || isProcessing}
      value={currentInput}
      on:send={(e) => handleSendMessage(e.detail)}
      oninput={handleInputChange}
    />
  </div>
</div>

<style>
  .ai-chat-interface {
    display: flex;
    flex-direction: column;
    background: var(--bg-primary, #ffffff);
    border: 1px solid var(--border-color, #e2e8f0);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  .status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: var(--bg-secondary, #f8fafc);
    border-bottom: 1px solid var(--border-color, #e2e8f0);
    font-size: 0.875rem;
  }

  .conversation-actions {
    display: flex;
    gap: 8px;
  }

  .btn-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: none;
    border: none;
    border-radius: 4px;
    color: var(--text-secondary, #64748b);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .btn-icon:hover {
    background: var(--bg-hover, #e2e8f0);
    color: var(--text-primary, #1e293b);
  }

  .chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    min-height: 200px;
    max-height: calc(100% - 120px);
  }

  .initialization-message,
  .processing-message {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: var(--bg-info, #eff6ff);
    border: 1px solid var(--border-info, #bfdbfe);
    border-radius: 6px;
    color: var(--text-info, #1e40af);
  }

  .welcome-message {
    padding: 24px;
    text-align: center;
    color: var(--text-secondary, #64748b);
  }

  .welcome-message h3 {
    margin: 0 0 16px 0;
    color: var(--text-primary, #1e293b);
    font-size: 1.25rem;
    font-weight: 600;
  }

  .welcome-message ul {
    text-align: left;
    max-width: 400px;
    margin: 16px auto;
  }

  .welcome-message li {
    margin: 8px 0;
  }

  .note {
    margin-top: 16px;
    padding: 12px;
    background: var(--bg-warning, #fef3c7);
    border: 1px solid var(--border-warning, #fbbf24);
    border-radius: 4px;
    font-size: 0.875rem;
    color: var(--text-warning, #92400e);
  }

  .chat-input-container {
    padding: 16px;
    background: var(--bg-primary, #ffffff);
    border-top: 1px solid var(--border-color, #e2e8f0);
  }

  .loading-spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border-color, #e2e8f0);
    border-top: 2px solid var(--accent-color, #3b82f6);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }

  /* Dark mode support */
  @media (prefers-color-scheme: dark) {
    .ai-chat-interface {
      background: var(--bg-primary, #0f172a);
      border-color: var(--border-color, #334155);
    }

    .status-bar {
      background: var(--bg-secondary, #1e293b);
      border-color: var(--border-color, #334155);
    }

    .welcome-message {
      color: var(--text-secondary, #94a3b8);
    }

    .welcome-message h3 {
      color: var(--text-primary, #f8fafc);
    }
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .chat-messages {
      padding: 12px;
    }

    .chat-input-container {
      padding: 12px;
    }

    .status-bar {
      padding: 8px 12px;
    }
  }
</style>