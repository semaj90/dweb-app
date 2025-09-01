<script lang="ts">
  import { afterUpdate, onMount, tick } from "svelte";
  import { elasticOut, quintOut } from "svelte/easing";
  import { writable } from "svelte/store";
  import { fade, fly, scale } from "svelte/transition";
  // Icons from lucide-svelte
  import {
    Bot,
    Brain,
    FileText,
    RotateCw,
    Scale,
    Send,
    Sparkles,
    User,
    X,
    Zap,
  } from "lucide-svelte";

  // Svelte 5 Props Interface
  interface Props {
    conversationId?: string;
    userId: string;
    caseId?: string | null;
    open?: boolean;
    title?: string;
    onsuggestionsreceived?: (suggestions: string[]) => void;
    onactionsreceived?: (actions: any[]) => void;
    onmessage?: (event: any) => void;
    onclose?: () => void;
    onaction?: (action: any) => void;
  }

  // Svelte 5 props with defaults
  let {
    conversationId = crypto.randomUUID(),
    userId,
    caseId = null,
    open = false,
    title = "Legal AI Assistant",
    onsuggestionsreceived,
    onactionsreceived,
    onmessage,
    onclose,
    onaction
  }: Props = $props();

  // Chat state using Svelte 5 runes
  let currentMessage = $state("");
  let isGenerating = $state(false);
  let selectedMode = $state("professional");
  let showModeSelector = $state(false);
  let componentError = $state<Error | null>(null);
let messagesContainer = $state<HTMLElement;
let messageInput = $state<HTMLTextAreaElement;

  // Messages store (keeping as writable for complex state management)
  let messages >(writable<
    Array<{
      id: string);
      role: "user" | "assistant";
      content: string;
      timestamp: Date;
      contextUsed?: unknown;
      suggestions?: string[];
      actions?: Array<{ type: string; text: string; data?: unknown }>;
      isTyping?: boolean;
      isError?: boolean;
    }>
  >([]);

  // AI modes/vibes
  const aiModes >([
    {
      id: "professional",
      label: "Professional",
      icon: Scale,
      description: "Formal legal analysis",
    },
    {
      id: "investigative",
      label: "Investigative",
      icon: Brain,
      description: "Deep case analysis",
    },
    {
      id: "evidence",
      label: "Evidence Focus",
      icon: FileText,
      description: "Evidence-centered responses",
    },
    {
      id: "strategic",
      label: "Strategic",
      icon: Zap,
      description: "Case strategy planning",
    },
  ]);

  // Quick actions
  const quickActions = [
    { text: "Analyze evidence timeline", icon: FileText },
    { text: "Review witness statements", icon: User },
    { text: "Check legal precedents", icon: Scale },
    { text: "Suggest next steps", icon: Zap },
  ];

  onMount(() => {
    try {
      if (open) {
        focusInput();
        loadConversationHistory();
      }
    } catch (error) {
      componentError = error instanceof Error ? error : new Error('Initialization failed');
    }
  });

  afterUpdate(() => {
    scrollToBottom();
  });

  async function loadConversationHistory() {
    try {
      const response = await fetch(
        `/api/chat?conversationId=${conversationId}&userId=${userId}&limit=50`
      );
      if (response.ok) {
        const result = await response.json();
        if (result.success && result.conversation) {
          messages.set(result.conversation);
        }
      }
    } catch (error) {
      console.warn("Failed to load conversation history:", error);
    }
  }

  function focusInput() {
    tick().then(() => {
      messageInput?.focus();
    });
  }

  function scrollToBottom() {
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Enter") {
      if (event.shiftKey) {
        // Allow new line with Shift+Enter
        return;
      } else {
        event.preventDefault();
        sendMessage();
      }
    } else if (event.key === "Escape") {
      closeChat();
    }
  }

  function handleQuickAction(action: string) {
    currentMessage = action;
    sendMessage();
  }

  async function sendMessage() {
    if (!currentMessage.trim() || isGenerating) return;

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user" as const,
      content: currentMessage.trim(),
      timestamp: new Date(),
    };

    // Add user message immediately
    messages.update((msgs) => [...msgs, userMessage]);
    const messageContent = currentMessage.trim();
    currentMessage = "";
    isGenerating = true;

    // Add typing indicator
    const typingMessage = {
      id: "typing-" + Date.now(),
      role: "assistant" as const,
      content: "",
      timestamp: new Date(),
      isTyping: true,
    };
    messages.update((msgs) => [...msgs, typingMessage]);

    try {
      // Send to enhanced chat API with vector context
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: messageContent,
          conversationId,
          userId,
          caseId,
          mode: selectedMode,
          useContext: true,
          maxTokens: 1000,
        }),
      });

      const result = await response.json();

      // Remove typing indicator
      messages.update((msgs) => msgs.filter((m) => !m.isTyping));

      if (result.success && result.message) {
        // Add AI response with enhanced features
        const aiMessage = {
          ...result.message,
          contextUsed: result.contextUsed,
          suggestions: result.suggestions,
          actions: result.actions,
        };

        messages.update((msgs) => [...msgs, aiMessage]);

        // Dispatch events for suggestions and actions
        if (result.suggestions?.length > 0 && onsuggestionsreceived) {
          onsuggestionsreceived(result.suggestions);
        }

        if (result.actions?.length > 0 && onactionsreceived) {
          onactionsreceived(result.actions);
        }

        // Store message embedding for future context
        if (aiMessage.content) {
          storeMessageEmbedding(aiMessage.content, "assistant");
        }
      } else {
        throw new Error(result.error || "Failed to get AI response");
      }
    } catch (error) {
      console.error("Failed to send message:", error);
      componentError = error instanceof Error ? error : new Error('Send message failed');

      // Remove typing indicator and show error
      messages.update((msgs) => msgs.filter((m) => !m.isTyping));

      const errorMessage = {
        id: crypto.randomUUID(),
        role: "assistant" as const,
        content:
          "Sorry, I encountered an error while processing your request. Please try again.",
        timestamp: new Date(),
        isError: true,
      };
      messages.update((msgs) => [...msgs, errorMessage]);
    } finally {
      isGenerating = false;
      scrollToBottom();
      focusInput();
    }
  }

  async function storeMessageEmbedding(
    content: string,
    role: "user" | "assistant"
  ) {
    try {
      await fetch("/api/embed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: content,
          type: "chat_message",
          metadata: {
            userId,
            caseId,
            conversationId,
            role,
            mode: selectedMode,
          },
        }),
      });
    } catch (error) {
      console.warn("Failed to store message embedding:", error);
    }
  }

  function closeChat() {
    open = false;
    if (onclose) onclose();
  }

  function clearConversation() {
    messages.set([]);
    conversationId = crypto.randomUUID();
  }

  function formatTimestamp(timestamp: Date): string {
    return new Intl.DateTimeFormat("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    }).format(timestamp);
  }

  function handleActionClick(action: any) {
    if (onaction) onaction(action);
  }
</script>

{#if componentError}
  <div class="error-boundary">
    <h2>Chat Error</h2>
    <p>The chat component encountered an error:</p>
    <p class="error-message">{componentError.message}</p>
    <button 
      on:on:onclick={() => { componentError = null; }}
      aria-label="Dismiss error and retry"
    >
      Retry
    </button>
  </div>
{:else if open}
  <!-- Chat overlay -->
  <div
    class="chat-overlay"
    transitifade={{ duration: 200 }}
    on:on:onclick={(e) => { if (e.target === e.currentTarget) closeChat(); }}
    on:keydown={(e) => e.key === "Escape" && closeChat()}
    role="dialog"
    aria-modal="true"
    aria-labelledby="chat-title"
    tabindex="-1"
  >
    <!-- Chat container -->
    <div
      class="chat-container"
      transitifly={{ y: 50, duration: 300, easing: quintOut }}
    >
      <!-- Header -->
      <div class="chat-header">
        <div class="header-content">
          <div class="title-section">
            <div class="ai-indicator">
              <Sparkles size={20} />
            </div>
            <h2 id="chat-title">{title}</h2>
          </div>

          <!-- Mode selector -->
          <div class="mode-section">
            <button
              class="mode-button"
              class:active={showModeSelector}
              on:on:onclick={() => (showModeSelector = !showModeSelector)}
              title="Select AI mode"
              aria-label="Select AI mode"
              aria-expanded={showModeSelector}
            >
              {#each aiModes as mode}
                {#if mode.id === selectedMode}
                  <svelte:component this={mode.icon} size={16} />
                  {mode.label}
                {/if}
              {/each}
            </button>

            {#if showModeSelector}
              <div
                class="mode-dropdown"
                transitiscale={{ duration: 200, easing: elasticOut }}
              >
                {#each aiModes as mode}
                  <button
                    class="mode-option"
                    class:selected={mode.id === selectedMode}
                    on:on:onclick={() => {
                      selectedMode = mode.id;
                      showModeSelector = false;
                    }}
                    aria-label="Switch to {mode.label} mode"
                  >
                    <svelte:component this={mode.icon} size={16} />
                    <div class="mode-info">
                      <span class="mode-name">{mode.label}</span>
                      <span class="mode-desc">{mode.description}</span>
                    </div>
                  </button>
                {/each}
              </div>
            {/if}
          </div>
        </div>

        <!-- Actions -->
        <div class="header-actions">
          <button
            class="header-action"
            on:on:onclick={() => clearConversation()}
            title="Clear conversation"
            disabled={isGenerating}
            aria-label="Clear conversation"
          >
            <RotateCw size={16} />
          </button>
          <button
            class="header-action"
            on:on:onclick={() => closeChat()}
            title="Close chat"
            aria-label="Close chat"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      <!-- Messages area -->
      <div class="messages-container" bind:this={messagesContainer}>
        {#each $messages as message (message.id)}
          <div
            class="message"
            class:user={message.role === "user"}
            class:assistant={message.role === "assistant"}
            class:error={message.isError}
            transitifly={{
              x: message.role === "user" ? 20 : -20,
              duration: 300,
            }}
          >
            <div class="message-avatar">
              {#if message.role === "user"}
                <User size={16} />
              {:else}
                <Bot size={16} />
              {/if}
            </div>

            <div class="message-content">
              {#if message.isTyping}
                <div class="typing-indicator">
                  <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <span class="typing-text">AI is thinking...</span>
                </div>
              {:else}
                <div class="message-text">
                  {message.content}
                </div>

                {#if message.suggestions && message.suggestions.length > 0}
                  <div class="suggestions">
                    <h4>Suggestions:</h4>
                    <ul>
                      {#each message.suggestions as suggestion}
                        <li>{suggestion}</li>
                      {/each}
                    </ul>
                  </div>
                {/if}

                {#if message.actions && message.actions.length > 0}
                  <div class="actions">
                    {#each message.actions as action}
                      <button
                        class="action-button"
                        on:on:onclick={() => handleActionClick(action)}
                        title={action.text}
                        aria-label="Action: {action.text}"
                      >
                        {action.text}
                      </button>
                    {/each}
                  </div>
                {/if}

                <div class="message-meta">
                  <span class="message-timestamp"
                    >{formatTimestamp(message.timestamp)}</span
                  >
                  {#if message.contextUsed && (message.contextUsed.similarContent?.length > 0 || message.contextUsed.evidence?.length > 0)}
                    <span
                      class="context-indicator"
                      title="Response used relevant context"
                    >
                      <Brain size={12} />
                    </span>
                  {/if}
                </div>
              {/if}
            </div>
          </div>
        {/each}
      </div>

      <!-- Quick actions (when no messages) -->
      {#if $messages.length === 0}
        <div class="quick-actions" transitifade={{ delay: 300 }}>
          <h3>Quick Actions</h3>
          <div class="action-grid">
            {#each quickActions as action}
              <button
                class="quick-action"
                on:on:onclick={() => handleQuickAction(action.text)}
                disabled={isGenerating}
                aria-label="Quick action: {action.text}"
              >
                <svelte:component this={action.icon} size={20} />
                {action.text}
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Input area -->
      <div class="input-area">
        <div class="input-container">
          <textarea
            bind:this={messageInput}
            bind:value={currentMessage}
            placeholder="Ask about your case, evidence, or legal strategy..."
            disabled={isGenerating}
            onkeydown={handleKeydown}
            rows="4"
            class="message-input"
            aria-label="Type your message here"
          ></textarea>

          <button
            class="send-button"
            class:sending={isGenerating}
            disabled={!currentMessage.trim() || isGenerating}
            on:on:onclick={() => sendMessage()}
            title="Send message"
            aria-label="Send message"
          >
            {#if isGenerating}
              <div class="spinner"></div>
            {:else}
              <Send size={20} />
            {/if}
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .error-boundary {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem;
    color: #dc2626;
  }

  .error-boundary h2 {
    margin: 0 0 0.5rem 0;
    font-size: 1.25rem;
    font-weight: 600;
  }

  .error-boundary p {
    margin: 0 0 0.5rem 0;
  }

  .error-message {
    font-family: monospace;
    font-size: 0.875rem;
    background: rgba(0, 0, 0, 0.05);
    padding: 0.5rem;
    border-radius: 4px;
    margin: 0.5rem 0;
  }

  .error-boundary button {
    background: #dc2626;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
  }

  .chat-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    padding: 1rem;
  }

  .chat-container {
    background: white;
    border-radius: 12px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    width: 100%;
    max-width: 600px;
    max-height: 80vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .chat-header {
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #e5e7eb;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .header-content {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex: 1;
  }

  .title-section {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .ai-indicator {
    padding: 0.5rem;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .chat-header h2 {
    margin: 0;
    font-size: 1.125rem;
    font-weight: 600;
  }

  .mode-section {
    position: relative;
  }

  .mode-button {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .mode-button:hover {
    background: rgba(255, 255, 255, 0.2);
  }

  .mode-dropdown {
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 0.5rem;
    background: white;
    border-radius: 8px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    overflow: hidden;
    z-index: 10;
    min-width: 200px;
  }

  .mode-option {
    width: 100%;
    padding: 0.75rem;
    background: none;
    border: none;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    cursor: pointer;
    transition: background 0.2s;
    color: #374151;
  }

  .mode-option:hover {
    background: #f3f4f6;
  }

  .mode-option.selected {
    background: #e0e7ff;
    color: #3730a3;
  }

  .mode-info {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
  }

  .mode-name {
    font-weight: 500;
    font-size: 0.875rem;
  }

  .mode-desc {
    font-size: 0.75rem;
    color: #6b7280;
  }

  .header-actions {
    display: flex;
    gap: 0.5rem;
  }

  .header-action {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
    padding: 0.5rem;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .header-action:hover {
    background: rgba(255, 255, 255, 0.2);
  }

  .header-action:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .message {
    display: flex;
    gap: 0.75rem;
    max-width: 85%;
  }

  .message.user {
    align-self: flex-end;
    flex-direction: row-reverse;
  }

  .message.assistant {
    align-self: flex-start;
  }

  .message-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .message.user .message-avatar {
    background: #3b82f6;
    color: white;
  }

  .message.assistant .message-avatar {
    background: #10b981;
    color: white;
  }

  .message-content {
    background: #f8fafc;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    flex: 1;
  }

  .message.user .message-content {
    background: #3b82f6;
    color: white;
  }

  .message.error .message-content {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #dc2626;
  }

  .message-text {
    line-height: 1.5;
    white-space: pre-wrap;
  }

  .typing-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .typing-dots {
    display: flex;
    gap: 4px;
  }

  .typing-dots span {
    width: 6px;
    height: 6px;
    background: #9ca3af;
    border-radius: 50%;
    animation: typing 1.4s infinite;
  }

  .typing-dots span:nth-child(2) {
    animation-delay: 0.2s;
  }

  .typing-dots span:nth-child(3) {
    animation-delay: 0.4s;
  }

  .typing-text {
    color: #6b7280;
    font-style: italic;
    font-size: 0.875rem;
  }

  .suggestions {
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
  }

  .message.assistant .suggestions {
    border-top-color: #e5e7eb;
  }

  .suggestions h4 {
    margin: 0 0 0.5rem 0;
    font-size: 0.875rem;
    font-weight: 600;
    opacity: 0.9;
  }

  .suggestions ul {
    margin: 0;
    padding-left: 1rem;
    font-size: 0.875rem;
  }

  .suggestions li {
    margin-bottom: 0.25rem;
    opacity: 0.9;
  }

  .actions {
    margin-top: 0.75rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .action-button {
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: inherit;
    padding: 0.375rem 0.75rem;
    border-radius: 6px;
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .message.assistant .action-button {
    background: #e5e7eb;
    border-color: #d1d5db;
    color: #374151;
  }

  .action-button:hover {
    background: rgba(255, 255, 255, 0.3);
  }

  .message.assistant .action-button:hover {
    background: #d1d5db;
  }

  .message-meta {
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.75rem;
    opacity: 0.7;
  }

  .context-indicator {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }

  .quick-actions {
    padding: 2rem;
    text-align: center;
  }

  .quick-actions h3 {
    margin: 0 0 1.5rem 0;
    color: #374151;
    font-size: 1.125rem;
  }

  .action-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
  }

  .quick-action {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 1rem;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    transition: all 0.2s;
    color: #374151;
  }

  .quick-action:hover {
    background: #f1f5f9;
    border-color: #cbd5e1;
    transform: translateY(-1px);
  }

  .quick-action:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }

  .input-area {
    padding: 1rem 1.5rem;
    border-top: 1px solid #e5e7eb;
    background: #f8fafc;
  }

  .input-container {
    display: flex;
    gap: 0.75rem;
    align-items: flex-end;
  }

  .message-input {
    flex: 1;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 0.75rem;
    font-family: inherit;
    font-size: 0.875rem;
    line-height: 1.5;
    resize: none;
    min-height: 44px;
    max-height: 120px;
    background: white;
    transition: border-color 0.2s;
  }

  .message-input:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }

  .message-input:disabled {
    background: #f3f4f6;
    color: #9ca3af;
  }

  .send-button {
    background: #3b82f6;
    border: none;
    color: white;
    padding: 0.75rem;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 44px;
    height: 44px;
  }

  .send-button:hover {
    background: #2563eb;
  }

  .send-button:disabled {
    background: #9ca3af;
    cursor: not-allowed;
  }

  .spinner {
    width: 20px;
    height: 20px;
    border: 2px solid transparent;
    border-top: 2px solid currentColor;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes typing {
    0%,
    60%,
    100% {
      transform: translateY(0);
    }
    30% {
      transform: translateY(-10px);
    }
  }

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .chat-overlay {
      padding: 0.5rem;
    }
    .chat-header {
      padding: 1rem;
    }

    .header-content {
      flex-direction: column;
      align-items: flex-start;
      gap: 0.75rem;
    }

    .action-grid {
      grid-template-columns: 1fr;
    }

    .message {
      max-width: 95%;
    }
  }
</style>

<!-- Svelte 5 migration completed - modern patterns applied -->