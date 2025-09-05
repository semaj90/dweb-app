<!-- Simplified Enhanced AI Assistant -->
<script lang="ts">
  interface Props {

  }
  let {
    caseId = undefined,
    placeholder = "Ask AI about this case...",
    maxHeight = "400px",
    showReferences = true
  } = $props();



  import {
    Brain,
    Loader2,
    Quote,
    Search,
    Settings,
    Trash2,
  } from "lucide-svelte";
  
    export const evidenceIds: string[] = [];
        export const enableVoiceInput = false;

  
  // State
  let query = "";
  let isLoading = false;
  let messages: any[] = [];
  let showSettings = false;
  let showCitationDialog = false;
  let selectedCitation = "";
  let selectedModel = "gpt-4";
  let searchThreshold = 0.7;
  let maxResults = 5;
  let temperature = 0.7;
  let enabledSources = ["cases", "statutes", "regulations", "secondary"];

  // Mock AI response
  async function handleSubmit() {
    if (!query.trim() || isLoading) return;

    isLoading = true;
    const userMessage = { role: "user", content: query };
    messages = [...messages, userMessage];

    // Clear input
    query = "";

    // Mock AI response
    setTimeout(() => {
      const aiResponse = {
        role: "assistant",
        content: `Based on the case information provided, here are my findings regarding "${userMessage.content}":

This appears to be a question about legal precedent and case law. The relevant statutes and regulations would need to be analyzed in the context of your specific jurisdiction.

Key considerations:
1. Applicable statutory framework
2. Relevant case precedents
3. Jurisdictional variations
4. Current regulatory environment

Would you like me to elaborate on any of these aspects?`,
        references: [
          {
            title: "Smith v. Jones",
            citation: "123 F.3d 456 (2023)",
            relevance: 0.9,
          },
          {
            title: "42 U.S.C. § 1983",
            citation: "Federal Civil Rights Statute",
            relevance: 0.8,
          },
        ],
      };

      messages = [...messages, aiResponse];
      isLoading = false;
    }, 1500);
}
  function handleReferenceClick(reference: any) {
    selectedCitation = `${reference.title} - ${reference.citation}`;
    showCitationDialog = true;
}
  function insertCitation() {
    dispatch("citation-inserted", selectedCitation);
    showCitationDialog = false;
}
  function clearMessages() {
    messages = [];
}
</script>

<div class="space-y-4">
  <!-- Main Chat Interface -->
  <div class="space-y-4" style="max-height: {maxHeight}">
    <!-- Header -->
    <div class="space-y-4">
      <div class="space-y-4">
        <Brain class="space-y-4" />
        <h3 class="space-y-4">Legal AI Assistant</h3>
        {#if caseId}
          <span class="space-y-4"
            >Case: {caseId}</span
          >
        {/if}
      </div>
      <div class="space-y-4">
        <button
          class="space-y-4"
          onclick={() => (showSettings = !showSettings)}
          title="Settings"
        >
          <Settings class="space-y-4" />
        </button>
        <button
          class="space-y-4"
          onclick={() => clearMessages()}
          title="Clear conversation"
        >
          <Trash2 class="space-y-4" />
        </button>
      </div>
    </div>

    <!-- Messages -->
    <div class="space-y-4">
      {#each messages as message}
        <div class="space-y-4">
          <div class="space-y-4">
            {message.content}
          </div>

          {#if message.references && showReferences}
            <div class="space-y-4">
              <h4 class="space-y-4">References:</h4>
              {#each message.references as reference}
                <button
                  class="space-y-4"
                  onclick={() => handleReferenceClick(reference)}
                >
                  <Quote class="space-y-4" />
                  <span class="space-y-4">{reference.title}</span>
                  <span class="space-y-4">{reference.citation}</span>
                </button>
              {/each}
            </div>
          {/if}
        </div>
      {/each}

      {#if isLoading}
        <div class="space-y-4">
          <div class="space-y-4">
            <Loader2 class="space-y-4" />
            Analyzing your query...
          </div>
        </div>
      {/if}
    </div>

    <!-- Input -->
    <form class="space-y-4" on:submit|preventDefault={handleSubmit}>
      <div class="space-y-4">
        <input
          type="text"
          bind:value={query}
          {placeholder}
          disabled={isLoading}
          class="space-y-4"
        />
        <button
          type="submit"
          disabled={!query.trim() || isLoading}
          class="space-y-4"
        >
          <Search class="space-y-4" />
        </button>
      </div>
    </form>
  </div>

  <!-- Settings Panel -->
  {#if showSettings}
    <div class="space-y-4">
      <div class="space-y-4">
        <h4 class="space-y-4">AI Assistant Settings</h4>
        <button class="space-y-4" onclick={() => (showSettings = false)}
          >×</button
        >
      </div>

      <div class="space-y-4">
        <div class="space-y-4">
          <label for="model-select">Model:</label>
          <select id="model-select" bind:value={selectedModel}>
            <option value="gpt-4">GPT-4</option>
            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
            <option value="claude-3">Claude 3</option>
          </select>
        </div>

        <div class="space-y-4">
          <label for="temperature-range">Temperature: {temperature}</label>
          <input
            id="temperature-range"
            type="range"
            min="0"
            max="1"
            step="0.1"
            bind:value={temperature}
          />
        </div>

        <div class="space-y-4">
          <label for="threshold-range"
            >Search Threshold: {searchThreshold}</label
          >
          <input
            id="threshold-range"
            type="range"
            min="0"
            max="1"
            step="0.1"
            bind:value={searchThreshold}
          />
        </div>

        <div class="space-y-4">
          <label for="max-results">Max Results:</label>
          <input
            id="max-results"
            type="number"
            min="1"
            max="20"
            bind:value={maxResults}
          />
        </div>
      </div>
    </div>
  {/if}

  <!-- Citation Dialog -->
  {#if showCitationDialog}
    <div
      class="space-y-4"
      onclick={() => (showCitationDialog = false)}
      onkeydown={(e) => {
        if (e.key === "Escape") {
          showCitationDialog = false;
        }
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="citation-dialog-title"
      tabindex={-1}
    >
      <div class="space-y-4" role="document">
        <div class="space-y-4">
          <h4 class="space-y-4" id="citation-dialog-title">
            <Quote class="space-y-4" />
            Legal Citation
          </h4>
        </div>

        <div class="space-y-4">
          <div class="space-y-4">
            <p>{selectedCitation}</p>
          </div>

          <div class="space-y-4">
            <button class="space-y-4" onclick={() => insertCitation()}>
              Insert Citation
            </button>
            <button
              class="space-y-4"
              onclick={() => navigator.clipboard.writeText(selectedCitation)}
            >
              Copy to Clipboard
            </button>
          </div>
        </div>

        <div class="space-y-4">
          <button
            class="space-y-4"
            onclick={() => (showCitationDialog = false)}
          >
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
    position: relative
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: white
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}
  .chat-container {
    display: flex
    flex-direction: column
    overflow: hidden
}
  .chat-header {
    display: flex
    justify-content: space-between;
    align-items: center
    padding: 12px 16px;
    border-bottom: 1px solid #e5e7eb;
    background: #f9fafb;
}
  .messages-container {
    flex: 1;
    overflow-y: auto
    padding: 16px;
    max-height: 300px;
}
  .message {
    margin-bottom: 16px;
    padding: 12px;
    border-radius: 8px;
}
  .message.user {
    background: #dbeafe;
    margin-left: 20%;
    text-align: right
}
  .message.assistant {
    background: #f3f4f6;
    margin-right: 20%;
}
  .message-content {
    white-space: pre-wrap;
    line-height: 1.5;
}
  .references {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #e5e7eb;
}
  .references-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 8px;
}
  .reference-item {
    display: flex
    align-items: center
    gap: 8px;
    padding: 8px 12px;
    margin-bottom: 4px;
    background: white
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    cursor: pointer
    transition: all 0.2s;
    width: 100%;
    text-align: left
}
  .reference-item:hover {
    background: #f9fafb;
    border-color: #d1d5db;
}
  .reference-title {
    font-weight: 500;
    color: #111827;
}
  .reference-citation {
    color: #6b7280;
    font-size: 0.875rem;
}
  .chat-input {
    padding: 16px;
    border-top: 1px solid #e5e7eb;
}
  .input-container {
    display: flex
    gap: 8px;
}
  .chat-input-field {
    flex: 1;
    padding: 12px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    outline: none
    transition: border-color 0.2s;
}
  .chat-input-field:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}
  .chat-submit-btn {
    padding: 12px;
    background: #3b82f6;
    color: white
    border: none
    border-radius: 6px;
    cursor: pointer
    transition: background 0.2s;
}
  .chat-submit-btn:hover:not(:disabled) {
    background: #2563eb;
}
  .chat-submit-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
  .btn-icon {
    padding: 8px;
    background: none
    border: none
    border-radius: 4px;
    cursor: pointer
    color: #6b7280;
    transition: all 0.2s;
}
  .btn-icon:hover {
    background: #f3f4f6;
    color: #374151;
}
  .settings-panel {
    position: absolute
    top: 0;
    right: 0;
    width: 300px;
    height: 100%;
    background: white
    border-left: 1px solid #e5e7eb;
    box-shadow: -2px 0 10px rgba(0, 0, 0, 0.1);
    z-index: 10;
}
  .settings-header {
    display: flex
    justify-content: space-between;
    align-items: center
    padding: 16px;
    border-bottom: 1px solid #e5e7eb;
}
  .settings-title {
    font-weight: 600;
    color: #111827;
}
  .btn-close {
    background: none
    border: none
    font-size: 20px;
    cursor: pointer
    color: #6b7280;
}
  .settings-content {
    padding: 16px;
}
  .setting-group {
    margin-bottom: 16px;
}
  .setting-group label {
    display: block
    margin-bottom: 4px;
    font-size: 0.875rem;
    font-weight: 500;
    color: #374151;
}
  .setting-group select,
  .setting-group input[type="number"] {
    width: 100%;
    padding: 8px;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    outline: none
}
  .setting-group input[type="range"] {
    width: 100%;
}
  .dialog-overlay {
    position: fixed
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: flex
    align-items: center
    justify-content: center
    z-index: 1000;
}
  .dialog-content {
    background: white
    border-radius: 8px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
    max-width: 500px;
    width: 90%;
    max-height: 90vh;
    overflow-y: auto
}
  .dialog-header {
    padding: 16px;
    border-bottom: 1px solid #e5e7eb;
}
  .dialog-title {
    display: flex
    align-items: center
    gap: 8px;
    font-weight: 600;
    color: #111827;
}
  .dialog-body {
    padding: 16px;
}
  .citation-display {
    padding: 16px;
    background: #f9fafb;
    border-radius: 6px;
    margin-bottom: 16px;
    border-left: 4px solid #3b82f6;
}
  .dialog-actions {
    display: flex
    gap: 8px;
    margin-bottom: 16px;
}
  .btn-primary {
    padding: 8px 16px;
    background: #3b82f6;
    color: white
    border: none
    border-radius: 6px;
    cursor: pointer
    font-weight: 500;
}
  .btn-primary:hover {
    background: #2563eb;
}
  .btn-secondary {
    padding: 8px 16px;
    background: #6b7280;
    color: white
    border: none
    border-radius: 6px;
    cursor: pointer
    font-weight: 500;
}
  .btn-secondary:hover {
    background: #4b5563;
}
  .dialog-footer {
    padding: 16px;
    border-top: 1px solid #e5e7eb;
    text-align: right
}
  .dialog-footer .btn-close {
    padding: 8px 16px;
    background: #f3f4f6;
    color: #374151;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    cursor: pointer
}
  .dialog-footer .btn-close:hover {
    background: #e5e7eb;
}
</style>
