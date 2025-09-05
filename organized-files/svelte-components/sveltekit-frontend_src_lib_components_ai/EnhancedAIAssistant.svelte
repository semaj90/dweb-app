<script lang="ts">
  interface Props {
    caseId?: string;
    placeholder?: string;
    maxHeight?: string;
    showReferences?: boolean;
    onresponse?: (event?: any) => void;
    oncitation?: (event?: any) => void;
  }
  let {
    caseId = undefined,
    placeholder = "Ask AI about this case...",
    maxHeight = "400px",
    showReferences = true,
    onresponse,
    oncitation
  } = $props();



  import { Bot, Download, Loader2, MessageSquare, Quote, Search, Settings, User as UserIcon } from "lucide-svelte";
  
    export const evidenceIds: string[] = [];
      
  
  // State
  let query = $state("");
  let isLoading = $state(false);
  let messages = $state<any[]>([]);
  let showSettings = $state(false);
  let showCitationDialog = $state(false);
  let selectedCitation = $state("");

  // Settings
  let selectedModel = $state("gpt-4");
  let temperature = $state(0.7);
  let searchThreshold = $state(0.7);
  let maxResults = $state(5);

  async function handleSubmit() {
    if (!query.trim() || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      role: "user",
      content: query,
      timestamp: new Date(),
    };

    messages = [...messages, userMessage];
    const currentQuery = query;
    query = "";
    isLoading = true;

    try {
      const response = await fetch("/api/ai/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: currentQuery,
          caseId,
          evidenceIds,
          model: selectedModel,
          temperature,
          searchThreshold,
          maxResults,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get AI response");
}
      const data = await response.json();

      const assistantMessage = {
        id: Date.now().toString(),
        role: "assistant",
        content: data.answer,
        references: data.references || [],
        timestamp: new Date(),
      };

      messages = [...messages, assistantMessage];

      // Dispatch event for parent components
      onresponse?.();
    } catch (error) {
      console.error("AI chat error:", error);
      const errorMessage = {
        id: Date.now().toString(),
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again.",
        timestamp: new Date(),
      };
      messages = [...messages, errorMessage];
    } finally {
      isLoading = false;
}}
  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
}}
  function showCitation(citation: string) {
    selectedCitation = citation;
    showCitationDialog = true;
}
  function insertCitation() {
    oncitation?.();
    showCitationDialog = false;
}
  function clearChat() {
    messages = [];
}
  function downloadChat() {
    const chatData = {
      caseId,
      timestamp: new Date().toISOString(),
      messages,
    };

    const blob = new Blob([JSON.stringify(chatData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ai-chat-${caseId || "session"}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}
</script>

<div class="ai-assistant-container">
  <!-- Header -->
  <div class="ai-header">
    <div class="ai-title">
      <Bot size={20} />
      <span>AI Assistant</span>
      {#if caseId}
        <span class="case-id">Case: {caseId}</span>
      {/if}
    </div>
    <div class="ai-actions">
      <button
        class="action-btn"
        onclick={() => (showSettings = !showSettings)}
        title="Settings"
      >
        <Settings size={16} />
      </button>
      <button
        class="action-btn"
        onclick={() => downloadChat()}
        title="Download chat"
        disabled={messages.length === 0}
      >
        <Download size={16} />
      </button>
      <button
        class="action-btn"
        onclick={() => clearChat()}
        title="Clear chat"
        disabled={messages.length === 0}
      >
        <MessageSquare size={16} />
      </button>
    </div>
  </div>

  <!-- Settings Panel -->
  {#if showSettings}
    <div class="settings-panel">
      <div class="settings-row">
        <label for="model-select">Model:</label>
        <select id="model-select" bind:value={selectedModel}>
          <option value="gpt-4">GPT-4</option>
          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          <option value="claude-3-sonnet">Claude 3 Sonnet</option>
        </select>
      </div>

      <div class="settings-row">
        <label for="temperature-slider">Temperature:</label>
        <input
          id="temperature-slider"
          type="range"
          min="0"
          max="1"
          step="0.1"
          bind:value={temperature}
        />
        <span>{temperature}</span>
      </div>

      <div class="settings-row">
        <label for="threshold-slider">Search Threshold:</label>
        <input
          id="threshold-slider"
          type="range"
          min="0"
          max="1"
          step="0.1"
          bind:value={searchThreshold}
        />
        <span>{searchThreshold}</span>
      </div>

      <div class="settings-row">
        <label for="max-results-input">Max Results:</label>
        <input
          id="max-results-input"
          type="number"
          min="1"
          max="20"
          bind:value={maxResults}
        />
      </div>
    </div>
  {/if}

  <!-- Chat Messages -->
  <div class="chat-messages" style="max-height: {maxHeight}">
    {#each messages as message}
      <div class="message {message.role}">
        <div class="message-header">
          {#if message.role === "user"}
            <UserIcon size={16} />
          {:else}
            <Bot size={16} />
          {/if}
          <span>
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>
        <div class="message-content">
          {message.content}
        </div>
        {#if message.references && message.references.length > 0 && showReferences}
          <div class="message-references">
            <h4>References:</h4>
            <ul>
              {#each message.references as ref}
                <li>
                  <button
                    class="reference-link"
                    onclick={() => showCitation(ref.citation)}
                  >
                    {ref.title}
                  </button>
                </li>
              {/each}
            </ul>
          </div>
        {/if}
      </div>
    {/each}
    {#if isLoading}
      <div class="message assistant loading">
        <div class="message-header">
          <Bot size={16} />
          <span>Thinking...</span>
        </div>
        <div class="message-content">
          <Loader2 size={16} />
        </div>
      </div>
    {/if}
  </div>

  <!-- Input Area -->
  <div class="chat-input">
    <div class="input-wrapper">
      <textarea
        bind:value={query}
        onkeydown={handleKeyDown}
        {placeholder}
        rows="4"
        disabled={isLoading}
      ></textarea>
      <button
        class="submit-btn"
        onclick={() => handleSubmit()}
        disabled={!query.trim() || isLoading}
      >
        {#if isLoading}
          <Loader2 size={16} />
        {:else}
          <Search size={16} />
        {/if}
      </button>
    </div>
  </div>

  <!-- Citation Dialog -->
  {#if showCitationDialog}
    <div class="modal-overlay" tabindex="-1" aria-modal="true" role="dialog" aria-labelledby="citation-modal-title" onkeydown={(e) => { if (e.key === 'Escape') showCitationDialog = false; }}>
      <div class="modal" role="document">
        <div class="modal-header">
          <h2 id="citation-modal-title" class="visually-hidden">Legal Citation</h2>
          <Quote aria-hidden="true" />
          <span class="sr-only">Legal Citation</span>
        </div>
        <div class="modal-body">
          <div class="citation-box">
            <p>{selectedCitation}</p>
          </div>
          <div class="modal-actions">
            <button class="btn-primary" onclick={() => insertCitation()}>
              Insert Citation
            </button>
            <button class="btn-secondary" onclick={() => navigator.clipboard.writeText(selectedCitation)}>
              Copy
            </button>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn-close" onclick={() => (showCitationDialog = false)}>
            Close
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  .ai-assistant-container {
    display: flex
    flex-direction: column
    background: white
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    overflow: hidden
}
  .ai-header {
    display: flex
    justify-content: space-between;
    align-items: center
    padding: 1rem;
    background: #f9fafb;
    border-bottom: 1px solid #e5e7eb;
}
  .ai-title {
    display: flex
    align-items: center
    gap: 0.5rem;
    font-weight: 600;
}
  .case-id {
    font-size: 0.875rem;
    color: #6b7280;
    background: #e5e7eb;
    padding: 0.125rem 0.5rem;
    border-radius: 4px;
}
  .ai-actions {
    display: flex
    gap: 0.5rem;
}
  .action-btn {
    padding: 0.5rem;
    border: none
    background: transparent
    border-radius: 4px;
    cursor: pointer
    transition: background-color 0.2s;
}
  .action-btn:hover {
    background: #e5e7eb;
}
  .action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
  .settings-panel {
    padding: 1rem;
    background: #f9fafb;
    border-bottom: 1px solid #e5e7eb;
}
  .settings-row {
    display: flex
    align-items: center
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}
  .settings-row label {
    min-width: 120px;
    font-size: 0.875rem;
    font-weight: 500;
}
  .settings-row select,
  .settings-row input {
    flex: 1;
    padding: 0.25rem 0.5rem;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    font-size: 0.875rem;
}
  .chat-messages {
    flex: 1;
    overflow-y: auto
    padding: 1rem;
    display: flex
    flex-direction: column
    gap: 1rem;
}
  .message {
    display: flex
    flex-direction: column
    gap: 0.5rem;
}
  .message.user {
    align-items: flex-end;
}
  .message.assistant {
    align-items: flex-start;
}
  .message-header {
    display: flex
    align-items: center
    gap: 0.5rem;
    font-size: 0.875rem;
    color: #6b7280;
}
  .message-content {
    max-width: 80%;
    padding: 0.75rem;
    border-radius: 8px;
    font-size: 0.875rem;
    line-height: 1.5;
}
  .message.user .message-content {
    background: #3b82f6;
    color: white
}
  .message.assistant .message-content {
    background: #f3f4f6;
    color: #1f2937;
}
  .message.loading .message-content {
    display: flex
    align-items: center
    gap: 0.5rem;
}
  .message-references {
    max-width: 80%;
    margin-top: 0.5rem;
}
  .message-references h4 {
    font-size: 0.875rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
}
  .message-references ul {
    list-style: none
    padding: 0;
    margin: 0;
}
  .message-references li {
    margin-bottom: 0.25rem;
}
  .reference-link {
    color: #3b82f6;
    text-decoration: underline
    background: none
    border: none
    cursor: pointer
    font-size: 0.875rem;
}
  .reference-link:hover {
    color: #2563eb;
}
  .chat-input {
    padding: 1rem;
    border-top: 1px solid #e5e7eb;
}
  .input-wrapper {
    display: flex
    gap: 0.5rem;
}
  .input-wrapper textarea {
    flex: 1;
    padding: 0.75rem;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    resize: vertical
    min-height: 2.5rem;
    font-family: inherit
    font-size: 0.875rem;
}
  .input-wrapper textarea:focus {
    outline: none
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}
  .submit-btn {
    padding: 0.75rem;
    background: #3b82f6;
    color: white
    border: none
    border-radius: 6px;
    cursor: pointer
    transition: background-color 0.2s;
}
  .submit-btn:hover:not(:disabled) {
    background: #2563eb;
}
  .submit-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
  .modal-overlay {
    position: fixed
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex
    align-items: center
    justify-content: center
    z-index: 1000;
}
  .modal {
    background: white
    border-radius: 8px;
    max-width: 500px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
}
  .modal-header {
    display: flex
    align-items: center
    gap: 0.5rem;
    padding: 1rem;
    border-bottom: 1px solid #e5e7eb;
    font-weight: 600;
}
  .modal-body {
    padding: 1rem;
}
  .citation-box {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1rem;
}
  .citation-box p {
    margin: 0;
    font-family: monospace
    font-size: 0.875rem;
    line-height: 1.5;
}
  .modal-actions {
    display: flex
    gap: 0.5rem;
    margin-bottom: 1rem;
}
  .btn-primary {
    padding: 0.5rem 1rem;
    background: #3b82f6;
    color: white
    border: none
    border-radius: 6px;
    cursor: pointer
    font-size: 0.875rem;
    transition: background-color 0.2s;
}
  .btn-primary:hover {
    background: #2563eb;
}
  .btn-secondary {
    padding: 0.5rem 1rem;
    background: #f3f4f6;
    color: #374151;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    cursor: pointer
    font-size: 0.875rem;
    transition: background-color 0.2s;
}
  .btn-secondary:hover {
    background: #e5e7eb;
}
  .modal-footer {
    display: flex
    justify-content: flex-end;
    padding: 1rem;
    border-top: 1px solid #e5e7eb;
}
  .btn-close {
    padding: 0.5rem 1rem;
    background: #f3f4f6;
    color: #374151;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    cursor: pointer
    font-size: 0.875rem;
    transition: background-color 0.2s;
}
  .btn-close:hover {
    background: #e5e7eb;
}
</style>
</div>
