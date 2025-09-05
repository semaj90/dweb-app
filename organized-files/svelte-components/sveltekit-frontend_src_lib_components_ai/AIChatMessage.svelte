<script context="module">
  interface Props {
    message: {
    showSources?: any;
    showMetadata?: any;
  }
  let {
    message,
    showSources = false,
    showMetadata = false
  } = $props();



  import { slide } from "svelte/transition";
</script>

<!-- AI Chat Message Component -->
<script lang="ts">
  import { browser } from "$app/environment";
  import { onMount } from "svelte";

      id: string
    role: "user" | "assistant" | "system";
    content: string
    timestamp: Date
    sources?: Array<{
      id: string
      title: string
      content: string
      score: number
      type: string
    }>;
    metadata?: {
      provider: "local" | "cloud" | "hybrid";
      model: string
      confidence: number
      executionTime: number
      fromCache: boolean
    };
  };

  let { showSources = $bindable() } = $props(); // false;
  let { showMetadata = $bindable() } = $props(); // false;

  let formattedTime = "";
  let isSourcesExpanded = false;
  let isMetadataExpanded = false;

  // Format timestamp
  onMount(() => {
    if (browser) {
      formattedTime = new Date(message.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
}
  });

  // Copy message content to clipboard
  async function copyToClipboard() {
    if (!browser) return;

    try {
      await navigator.clipboard.writeText(message.content);
      // TODO: Show toast notification
    } catch (error) {
      console.error("Failed to copy:", error);
}}
  // Format confidence as percentage
  function formatConfidence(confidence: number): string {
    return Math.round(confidence * 100) + "%";
}
  // Format execution time
  function formatExecutionTime(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
}
</script>

<div
  class="space-y-4"
  role="article"
  aria-label="{message.role} message"
>
  <div class="space-y-4">
    <div class="space-y-4">
      {#if message.role === "user"}
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
          <circle cx="12" cy="7" r="4" />
        </svg>
        You
      {:else if message.role === "assistant"}
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <rect width="18" height="11" x="3" y="11" rx="2" ry="2" />
          <circle cx="12" cy="5" r="2" />
          <path d="M12 7v4" />
          <line x1="8" y1="16" x2="8" y2="16" />
          <line x1="16" y1="16" x2="16" y2="16" />
        </svg>
        AI Assistant
        {#if message.metadata?.provider}
          <span
            class="space-y-4"
            class:local={message.metadata.provider === "local"}
          >
            {message.metadata.provider}
          </span>
        {/if}
      {/if}
    </div>

    <div class="space-y-4">
      <span
        class="space-y-4"
        title="Message sent at {new Date(message.timestamp).toLocaleString()}"
      >
        {formattedTime}
      </span>

      <button
        type="button"
        class="space-y-4"
        onclick={() => copyToClipboard()}
        title="Copy message"
        aria-label="Copy message to clipboard"
      >
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
          <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
        </svg>
      </button>
    </div>
  </div>

  <div class="space-y-4">
    <div class="space-y-4">
      {message.content}
    </div>

    {#if showSources && message.sources && message.sources.length > 0}
      <div class="space-y-4">
        <button
          type="button"
          class="space-y-4"
          onclick={() => (isSourcesExpanded = !isSourcesExpanded)}
          aria-expanded={isSourcesExpanded}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            class:rotated={isSourcesExpanded}
          >
            <polyline points="6,9 12,15 18,9" />
          </svg>
          Sources ({message.sources.length})
        </button>

        {#if isSourcesExpanded}
          <div class="space-y-4" transitionslide={{ duration: 200 }}>
            {#each message.sources as source (source.id)}
              <div class="space-y-4">
                <div class="space-y-4">
                  <span class="space-y-4">{source.title}</span>
                  <span class="space-y-4"
                    >{Math.round(source.score * 100)}%</span
                  >
                  <span class="space-y-4">{source.type}</span>
                </div>
                <div class="space-y-4">
                  {source.content}
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}

    {#if showMetadata && message.metadata}
      <div class="space-y-4">
        <button
          type="button"
          class="space-y-4"
          onclick={() => (isMetadataExpanded = !isMetadataExpanded)}
          aria-expanded={isMetadataExpanded}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            class:rotated={isMetadataExpanded}
          >
            <polyline points="6,9 12,15 18,9" />
          </svg>
          Details
        </button>

        {#if isMetadataExpanded}
          <div class="space-y-4" transitionslide={{ duration: 200 }}>
            <div class="space-y-4">
              <span class="space-y-4">Model:</span>
              <span class="space-y-4">{message.metadata.model}</span>
            </div>
            <div class="space-y-4">
              <span class="space-y-4">Provider:</span>
              <span class="space-y-4">{message.metadata.provider}</span>
            </div>
            <div class="space-y-4">
              <span class="space-y-4">Confidence:</span>
              <span class="space-y-4"
                >{formatConfidence(message.metadata.confidence)}</span
              >
            </div>
            <div class="space-y-4">
              <span class="space-y-4">Response Time:</span>
              <span class="space-y-4"
                >{formatExecutionTime(message.metadata.executionTime)}</span
              >
            </div>
            {#if message.metadata.fromCache}
              <div class="space-y-4">
                <span class="space-y-4">Source:</span>
                <span class="space-y-4">Cached</span>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  /* @unocss-include */
  .chat-message {
    margin: 16px 0;
    padding: 16px;
    border-radius: 8px;
    background: var(--bg-primary, #ffffff);
    border: 1px solid var(--border-color, #e2e8f0);
}
  .chat-message.user {
    margin-left: 20%;
    background: var(--bg-user, #3b82f6);
    color: white
    border-color: var(--border-user, #2563eb);
}
  .chat-message.assistant {
    margin-right: 20%;
    background: var(--bg-assistant, #f8fafc);
    border-color: var(--border-assistant, #e2e8f0);
}
  .message-header {
    display: flex
    justify-content: space-between;
    align-items: center
    margin-bottom: 12px;
    font-size: 0.875rem;
}
  .message-role {
    display: flex
    align-items: center
    gap: 8px;
    font-weight: 600;
    color: var(--text-primary, #1e293b);
}
  .chat-message.user .message-role {
    color: white
}
  .provider-badge {
    font-size: 0.75rem;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--bg-secondary, #e2e8f0);
    color: var(--text-secondary, #64748b);
    font-weight: normal
}
  .provider-badge.local {
    background: var(--bg-success, #dcfce7);
    color: var(--text-success, #166534);
}
  .message-actions {
    display: flex
    align-items: center
    gap: 8px;
}
  .timestamp {
    color: var(--text-muted, #94a3b8);
    font-size: 0.75rem;
}
  .chat-message.user .timestamp {
    color: rgba(255, 255, 255, 0.8);
}
  .action-btn {
    display: flex
    align-items: center
    justify-content: center
    width: 24px;
    height: 24px;
    background: none
    border: none
    border-radius: 4px;
    color: var(--text-muted, #94a3b8);
    cursor: pointer
    transition: all 0.2s ease;
}
  .action-btn:hover {
    background: var(--bg-hover, rgba(0, 0, 0, 0.05));
    color: var(--text-primary, #1e293b);
}
  .chat-message.user .action-btn {
    color: rgba(255, 255, 255, 0.8);
}
  .chat-message.user .action-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: white
}
  .message-content {
    line-height: 1.6;
}
  .content-text {
    margin-bottom: 12px;
    white-space: pre-wrap;
    word-wrap: break-word;
}
  .sources-section,
  .metadata-section {
    margin-top: 16px;
    border-top: 1px solid var(--border-color, #e2e8f0);
    padding-top: 12px;
}
  .sources-toggle,
  .metadata-toggle {
    display: flex
    align-items: center
    gap: 6px;
    background: none
    border: none
    padding: 4px 0;
    font-size: 0.875rem;
    color: var(--text-secondary, #64748b);
    cursor: pointer
    transition: color 0.2s ease;
}
  .sources-toggle:hover,
  .metadata-toggle:hover {
    color: var(--text-primary, #1e293b);
}
  .sources-toggle svg,
  .metadata-toggle svg {
    transition: transform 0.2s ease;
}
  .sources-toggle svg.rotated,
  .metadata-toggle svg.rotated {
    transform: rotate(180deg);
}
  .sources-list {
    margin-top: 8px;
    border-left: 2px solid var(--border-accent, #3b82f6);
    padding-left: 12px;
}
  .source-item {
    margin: 8px 0;
    padding: 8px;
    background: var(--bg-secondary, #f8fafc);
    border-radius: 4px;
    font-size: 0.875rem;
}
  .source-header {
    display: flex
    justify-content: space-between;
    align-items: center
    margin-bottom: 4px;
    font-weight: 500;
}
  .source-title {
    color: var(--text-primary, #1e293b);
}
  .source-score {
    color: var(--text-accent, #3b82f6);
    font-weight: 600;
}
  .source-type {
    font-size: 0.75rem;
    padding: 2px 6px;
    background: var(--bg-muted, #e2e8f0);
    color: var(--text-muted, #64748b);
    border-radius: 2px;
}
  .source-content {
    color: var(--text-secondary, #64748b);
    font-size: 0.8125rem;
    line-height: 1.4;
}
  .metadata-content {
    margin-top: 8px;
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 8px;
    font-size: 0.875rem;
}
  .metadata-item {
    display: flex
    justify-content: space-between;
    align-items: center
    padding: 4px 8px;
    background: var(--bg-secondary, #f8fafc);
    border-radius: 4px;
}
  .metadata-item .label {
    color: var(--text-secondary, #64748b);
    font-weight: 500;
}
  .metadata-item .value {
    color: var(--text-primary, #1e293b);
    font-weight: 600;
}
  .metadata-item .value.cache {
    color: var(--text-info, #0369a1);
}
  /* Dark mode support */
  @media (prefers-color-scheme: dark) {
    .chat-message {
      background: var(--bg-primary, #1e293b);
      border-color: var(--border-color, #475569);
}
    .chat-message.assistant {
      background: var(--bg-assistant, #0f172a);
      border-color: var(--border-assistant, #334155);
}
    .source-item,
    .metadata-item {
      background: var(--bg-secondary, #334155);
}}
  /* Responsive design */
  @media (max-width: 768px) {
    .chat-message.user {
      margin-left: 10%;
}
    .chat-message.assistant {
      margin-right: 10%;
}
    .metadata-content {
      grid-template-columns: 1fr;
}}
</style>
