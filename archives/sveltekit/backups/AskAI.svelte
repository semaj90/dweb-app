<!-- Ask AI Component with Vector Search Integration -->
<script lang="ts">
  import { browser } from "$app/environment";
  import {
    AlertCircle,
    Brain,
    CheckCircle,
    Loader2,
    MessageCircle,
    Search,
  } from "lucide-svelte/icons";
  import { createEventDispatcher, onMount } from "svelte";

  export let caseId: string | undefined = undefined;
  export let evidenceIds: string[] = [];
  export let placeholder = "Ask AI about this case...";
  export let maxHeight = "400px";
  export let showReferences = true;
  export let enableVoiceInput = false;

  interface AIResponse {
    answer: string;
    references: Array<{
      id: string;
      type: string;
      title: string;
      relevanceScore: number;
      citation: string;
    }>;
    confidence: number;
    searchResults: number;
    model: string;
    processingTime: number;
}
  interface ConversationMessage {
    id: string;
    type: "user" | "ai";
    content: string;
    timestamp: number;
    references?: AIResponse["references"];
    confidence?: number;
    metadata?: Record<string, any>;
}
  // Component state
  let query = "";
  let isLoading = false;
  let error = "";
  let conversation: ConversationMessage[] = [];
  let textareaRef: HTMLTextAreaElement;
  let messagesContainer: HTMLDivElement;

  // Advanced options
  let showAdvancedOptions = false;
  let selectedModel: "openai" | "ollama" = "openai";
  let searchThreshold = 0.7;
  let maxResults = 10;
  let temperature = 0.7;

  // Voice input state
  let isListening = false;
  let recognition: SpeechRecognition | null = null;

  const dispatch = createEventDispatcher<{
    response: AIResponse;
    error: string;
    referenceClicked: { id: string; type: string };
  }>();

  // Simple IndexedDB wrapper for conversation storage
  const getIndexedDBService = () => ({
    async getSetting(key: string): Promise<any> {
      if (!browser) return null;
      try {
        const stored = localStorage.getItem(key);
        return stored ? JSON.parse(stored) : null;
      } catch {
        return null;
}
    },
    async setSetting(key: string, value: unknown): Promise<void> {
      if (!browser) return;
      try {
        localStorage.setItem(key, JSON.stringify(value));
      } catch (error) {
        console.warn("Storage failed:", error);
}
    },
  });

  // Simple user activity tracking
  async function trackUserActivity(activity: unknown): Promise<void> {
    if (!browser) return;
    try {
      console.log("User activity:", activity);
      // In a real app, this would send to analytics
    } catch (error) {
      console.warn("Activity tracking failed:", error);
}}
  onMount(() => {
    // Initialize speech recognition if supported and enabled
    if (enableVoiceInput && "webkitSpeechRecognition" in window) {
      recognition = new (window as any).webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = "en-US";

      recognition.onresult = (event: unknown) => {
        const transcript = event.results[0][0].transcript;
        query = transcript;
        textareaRef?.focus();
      };

      recognition.onerror = () => {
        isListening = false;
      };

      recognition.onend = () => {
        isListening = false;
      };
}
    // Load conversation history from IndexedDB
    loadConversationHistory();
  });

  async function loadConversationHistory() {
    try {
      const contextKey = caseId ? `case_${caseId}` : "general";
      const indexedDBService = getIndexedDBService();
      const history = await indexedDBService.getSetting(
        `ai_conversation_${contextKey}`
      );

      if (history && Array.isArray(history)) {
        conversation = history.slice(-10); // Load last 10 messages
}
    } catch (error) {
      console.warn("Failed to load conversation history:", error);
}}
  async function saveConversationHistory() {
    try {
      const contextKey = caseId ? `case_${caseId}` : "general";
      const indexedDBService = getIndexedDBService();
      await indexedDBService.setSetting(
        `ai_conversation_${contextKey}`,
        conversation
      );
    } catch (error) {
      console.warn("Failed to save conversation history:", error);
}}
  async function askAI() {
    if (!query.trim() || isLoading) return;

    const userMessage: ConversationMessage = {
      id: generateId(),
      type: "user",
      content: query.trim(),
      timestamp: Date.now(),
    };

    conversation = [...conversation, userMessage];

    const currentQuery = query;
    query = "";
    isLoading = true;
    error = "";

    // Auto-resize textarea
    if (textareaRef) {
      textareaRef.style.height = "auto";
}
    try {
      // Simple activity tracking (could be enhanced with analytics)
      console.log("User activity:", {
        type: "search",
        target: caseId ? "case" : "evidence",
        targetId: caseId || "general",
        query: currentQuery,
        timestamp: new Date().toISOString(),
      });

      // Prepare request
      const requestBody = {
        question: currentQuery,
        context: {
          caseId,
          evidenceIds,
          maxResults,
          searchThreshold,
        },
        options: {
          model: selectedModel,
          temperature,
          maxTokens: 1000,
          includeReferences: showReferences,
        },
      };

      // Call AI endpoint
      const response = await fetch("/api/ai/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to get AI response");
}
      const aiResponse: AIResponse = await response.json();

      // Add AI message to conversation
      const aiMessage: ConversationMessage = {
        id: generateId(),
        type: "ai",
        content: aiResponse.answer,
        timestamp: Date.now(),
        references: aiResponse.references,
        confidence: aiResponse.confidence,
        metadata: {
          model: aiResponse.model,
          processingTime: aiResponse.processingTime,
          searchResults: aiResponse.searchResults,
        },
      };

      conversation = [...conversation, aiMessage];

      // Scroll to bottom
      setTimeout(() => scrollToBottom(), 100);

      // Save conversation
      await saveConversationHistory();

      // Dispatch events
      dispatch("response", aiResponse);
    } catch (err) {
      error = err instanceof Error ? err.message : "An error occurred";
      console.error("AI request failed:", err);
      dispatch("error", error);
    } finally {
      isLoading = false;
}}
  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askAI();
}}
  function startVoiceInput() {
    if (recognition && !isListening) {
      isListening = true;
      recognition.start();
}}
  function stopVoiceInput() {
    if (recognition && isListening) {
      recognition.stop();
      isListening = false;
}}
  function handleReferenceClick(
    reference: NonNullable<ConversationMessage["references"]>[0]
  ) {
    dispatch("referenceClicked", {
      id: reference.id,
      type: reference.type,
    });
}
  function clearConversation() {
    conversation = [];
    saveConversationHistory();
}
  function scrollToBottom() {
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
}}
  function generateId(): string {
    return Math.random().toString(36).substr(2, 9);
}
  function formatTimestamp(timestamp: number): string {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
}
  function getConfidenceColor(confidence: number): string {
    if (confidence >= 0.8) return "text-green-600";
    if (confidence >= 0.6) return "text-yellow-600";
    return "text-red-600";
}
  function getConfidenceIcon(confidence: number) {
    if (confidence >= 0.8) return CheckCircle;
    if (confidence >= 0.6) return AlertCircle;
    return AlertCircle;
}
  // Auto-resize textarea
  function autoResize(event: Event) {
    const target = event.target as HTMLTextAreaElement;
    target.style.height = "auto";
    target.style.height = target.scrollHeight + "px";
}
</script>

<div class="mx-auto px-4 max-w-7xl">
  <!-- Header -->
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <Brain class="mx-auto px-4 max-w-7xl" />
        <h3 class="mx-auto px-4 max-w-7xl">Ask AI Assistant</h3>
        {#if caseId}
          <span class="mx-auto px-4 max-w-7xl">• Case Context</span>
        {/if}
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <button
          type="button"
          class="mx-auto px-4 max-w-7xl"
          onclick={() => (showAdvancedOptions = !showAdvancedOptions)}
        >
          Advanced
        </button>

        {#if conversation.length > 0}
          <button
            type="button"
            class="mx-auto px-4 max-w-7xl"
            onclick={() => clearConversation()}
          >
            Clear
          </button>
        {/if}
      </div>
    </div>

    <!-- Advanced Options -->
    {#if showAdvancedOptions}
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <div>
            <label
              class="mx-auto px-4 max-w-7xl"
              for="field-1"
            >
              Model
            </label>
            <select
              bind:value={selectedModel}
              class="mx-auto px-4 max-w-7xl"
              id="field-1"
            >
              <option value="openai">OpenAI GPT-3.5</option>
              <option value="ollama">Local LLM (Gemma)</option>
            </select>
          </div>

          <div>
            <label
              class="mx-auto px-4 max-w-7xl"
              for="field-2"
            >
              Search Threshold
            </label>
            <input
              type="range"
              min="0.5"
              max="0.9"
              step="0.1"
              bind:value={searchThreshold}
              class="mx-auto px-4 max-w-7xl"
              id="field-2"
            />
            <span class="mx-auto px-4 max-w-7xl">{searchThreshold}</span>
          </div>
        </div>

        <div class="mx-auto px-4 max-w-7xl">
          <div>
            <label
              class="mx-auto px-4 max-w-7xl"
              for="field-3"
            >
              Max Results
            </label>
            <input
              type="number"
              min="5"
              max="50"
              bind:value={maxResults}
              class="mx-auto px-4 max-w-7xl"
              id="field-3"
            />
          </div>

          <div>
            <label
              class="mx-auto px-4 max-w-7xl"
              for="field-4"
            >
              Temperature
            </label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              bind:value={temperature}
              class="mx-auto px-4 max-w-7xl"
              id="field-4"
            />
            <span class="mx-auto px-4 max-w-7xl">{temperature}</span>
          </div>
        </div>
      </div>
    {/if}
  </div>

  <!-- Conversation -->
  <div
    bind:this={messagesContainer}
    class="mx-auto px-4 max-w-7xl"
    style="max-height: {maxHeight};"
  >
    {#if conversation.length === 0}
      <div class="mx-auto px-4 max-w-7xl">
        <MessageCircle class="mx-auto px-4 max-w-7xl" />
        <p class="mx-auto px-4 max-w-7xl">Start a conversation with the AI assistant</p>
        <p class="mx-auto px-4 max-w-7xl">
          Ask questions about cases, evidence, or legal procedures
        </p>
      </div>
    {:else}
      {#each conversation as message (message.id)}
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">
              {#if message.type === "user"}
                <div
                  class="mx-auto px-4 max-w-7xl"
                >
                  <span class="mx-auto px-4 max-w-7xl">U</span>
                </div>
              {:else}
                <div
                  class="mx-auto px-4 max-w-7xl"
                >
                  <Brain class="mx-auto px-4 max-w-7xl" />
                </div>
              {/if}
            </div>

            <div class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">
                  {message.type === "user" ? "You" : "AI Assistant"}
                </span>
                <span class="mx-auto px-4 max-w-7xl">
                  {formatTimestamp(message.timestamp)}
                </span>

                {#if message.type === "ai" && message.confidence !== undefined}
                  <div class="mx-auto px-4 max-w-7xl">
                    <svelte:component
                      this={getConfidenceIcon(message.confidence)}
                      class="mx-auto px-4 max-w-7xl"
                    />
                    <span
                      class="mx-auto px-4 max-w-7xl"
                    >
                      {Math.round(message.confidence * 100)}%
                    </span>
                  </div>
                {/if}
              </div>

              <div class="mx-auto px-4 max-w-7xl">
                <p class="mx-auto px-4 max-w-7xl">{message.content}</p>
              </div>

              <!-- References -->
              {#if message.references && message.references.length > 0 && showReferences}
                <div class="mx-auto px-4 max-w-7xl">
                  <h4 class="mx-auto px-4 max-w-7xl">
                    References:
                  </h4>
                  <div class="mx-auto px-4 max-w-7xl">
                    {#each message.references as reference}
                      <button
                        type="button"
                        class="mx-auto px-4 max-w-7xl"
                        onclick={() => handleReferenceClick(reference)}
                      >
                        <span class="mx-auto px-4 max-w-7xl"
                          >{reference.type.toUpperCase()}:</span
                        >
                        {reference.title}
                        <span class="mx-auto px-4 max-w-7xl"
                          >({Math.round(reference.relevanceScore * 100)}%)</span
                        >
                      </button>
                    {/each}
                  </div>
                </div>
              {/if}

              <!-- Metadata -->
              {#if message.metadata}
                <div class="mx-auto px-4 max-w-7xl">
                  {#if message.metadata.model}
                    Model: {message.metadata.model}
                  {/if}
                  {#if message.metadata.processingTime}
                    • {message.metadata.processingTime}ms
                  {/if}
                  {#if message.metadata.searchResults}
                    • {message.metadata.searchResults} results
                  {/if}
                </div>
              {/if}
            </div>
          </div>
        </div>
      {/each}
    {/if}
  </div>

  <!-- Input Area -->
  <div class="mx-auto px-4 max-w-7xl">
    {#if error}
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <AlertCircle class="mx-auto px-4 max-w-7xl" />
          <span class="mx-auto px-4 max-w-7xl">{error}</span>
        </div>
      </div>
    {/if}

    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <textarea
          bind:this={textareaRef}
          bind:value={query}
          onkeypress={handleKeyPress}
          oninput={autoResize}
          {placeholder}
          disabled={isLoading}
          rows={${1"
          class="mx-auto px-4 max-w-7xl"
        ></textarea>

        {#if enableVoiceInput && recognition}
          <button
            type="button"
            class="mx-auto px-4 max-w-7xl"
            class:text-red-500={isListening}
            onclick={() => (isListening ? stopVoiceInput : startVoiceInput())}
            disabled={isLoading}
          >
            🎤
          </button>
        {/if}
      </div>

      <button
        type="button"
        onclick={() => askAI()}
        disabled={!query.trim() || isLoading}
        class="mx-auto px-4 max-w-7xl"
      >
        {#if isLoading}
          <Loader2 class="mx-auto px-4 max-w-7xl" />
          <span>Thinking...</span>
        {:else}
          <Search class="mx-auto px-4 max-w-7xl" />
          <span>Ask</span>
        {/if}
      </button>
    </div>

    <div class="mx-auto px-4 max-w-7xl">
      Press Enter to send, Shift+Enter for new line
      {#if caseId}
        • Context: Case {caseId.slice(0, 8)}
      {/if}
      {#if evidenceIds.length > 0}
        • {evidenceIds.length} evidence item(s)
      {/if}
      {#if selectedModel === "ollama"}
        • Using local LLM
      {/if}
    </div>
  </div>
</div>

<style>
  .ai-chat-component {
    font-family:
      system-ui,
      -apple-system,
      sans-serif;
}
  .message {
    animation: slideInFromBottom 0.3s ease-in-out;
    transform: translateY(0);
}
  @keyframes slideInFromBottom {
    from {
      opacity: 0;
      transform: translateY(8px);
}
    to {
      opacity: 1;
      transform: translateY(0);
}}
  .user-message {
    opacity: 0.9;
}
  .ai-message {
    background-color: rgb(249 250 251);
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin-left: -0.5rem;
    margin-right: -0.5rem;
}
  :global(.prose p) {
    margin-bottom: 0.5rem;
}
  :global(.prose p:last-child) {
    margin-bottom: 0;
}
  /* UnoCSS will handle the utility classes, this is for custom animations */
  .search-result:hover {
    background-color: rgb(239 246 255);
    border-color: rgb(147 197 253);
}
  .statute-reference {
    display: inline-block;
    font-weight: 500;
}
</style>