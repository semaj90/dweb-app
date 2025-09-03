<!-- AI Status Indicator Component -->
<script lang="ts">
  interface Props {
    isReady?: boolean;
    isLoading?: boolean;
    provider?: "local" | "cloud" | "hybrid" | null;
    model?: string | null;
    error?: string | null;
  }

  let {
    isReady = false,
    isLoading = false,
    provider = null,
    model = null,
    error = null
  }: Props = $props();

  // Status computation
  let currentStatus = $derived(error
    ? "error"
    : isLoading
      ? "loading"
      : isReady
        ? "ready"
        : "unavailable");

  let statusText = $derived({
    ready: "AI Ready",
    loading: "Loading...",
    error: "AI Error",
    unavailable: "AI Unavailable",
  }[currentStatus]);

  let statusColor = $derived({
    ready: "var(--status-success, #10b981)",
    loading: "var(--status-warning, #f59e0b)",
    error: "var(--status-error, #ef4444)",
    unavailable: "var(--status-muted, #94a3b8)",
  }[currentStatus]);

  // Provider details
  let providerText = $derived(provider === "local"
      ? "Local AI"
      : provider === "cloud"
        ? "Cloud AI"
        : provider === "hybrid"
          ? "Hybrid AI"
          : "No Provider");

  let isErrorState = $derived(currentStatus === "error");
  let isLoadingState = $derived(currentStatus === "loading");
  let isReadyState = $derived(currentStatus === "ready");
  let modelText = $derived(model || "No Model");
</script>

<div
  class="mx-auto px-4 max-w-7xl"
  class:error={isErrorState}
  class:loading={isLoadingState}
  class:ready={isReadyState}
>
  <!-- Status Icon -->
  <div class="mx-auto px-4 max-w-7xl" style="color: {statusColor}">
    {#if currentStatus === "loading"}
      <div class="mx-auto px-4 max-w-7xl"></div>
    {:else if currentStatus === "ready"}
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <path d="M9 12l2 2 4-4" />
        <circle cx="12" cy="12" r="9" />
      </svg>
    {:else if currentStatus === "error"}
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <circle cx="12" cy="12" r="9" />
        <line x1="15" y1="9" x2="9" y2="15" />
        <line x1="9" y1="9" x2="15" y2="15" />
      </svg>
    {:else}
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <circle cx="12" cy="12" r="9" />
        <line x1="9" y1="9" x2="15" y2="15" />
        <line x1="15" y1="9" x2="9" y2="15" />
      </svg>
    {/if}
  </div>

  <!-- Status Text -->
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl" style="color: {statusColor}">
      {statusText}
    </div>

    {#if isReady && provider && model}
      <div class="mx-auto px-4 max-w-7xl">
        <span class="mx-auto px-4 max-w-7xl" class:local={provider === "local"}>
          {providerText}
        </span>
        <span class="mx-auto px-4 max-w-7xl">•</span>
        <span class="mx-auto px-4 max-w-7xl" title="Current AI model: {model}">
          {modelText}
        </span>
      </div>
    {:else if error}
      <div class="mx-auto px-4 max-w-7xl" title={error}>
        {error.length > 50 ? error.substring(0, 50) + "..." : error}
      </div>
    {/if}
  </div>

  <!-- Detailed Tooltip -->
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <strong>Status:</strong>
        {statusText}
      </div>

      {#if provider && model}
        <div class="mx-auto px-4 max-w-7xl">
          <strong>Provider:</strong>
          {providerText}
        </div>
        <div class="mx-auto px-4 max-w-7xl">
          <strong>Model:</strong>
          {model}
        </div>
      {/if}

      {#if error}
        <div class="mx-auto px-4 max-w-7xl">
          <strong>Error:</strong>
          {error}
        </div>
      {/if}

      <div class="mx-auto px-4 max-w-7xl">
        <small>
          {#if currentStatus === "ready"}
            AI system is ready to process requests
          {:else if currentStatus === "loading"}
            Initializing AI system...
          {:else if currentStatus === "error"}
            AI system encountered an error
          {:else}
            AI system is not available
          {/if}
        </small>
      </div>
    </div>
  </div>
</div>

<style>
  .ai-status-indicator {
    position: relative;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 0.875rem;
    transition: all 0.2s ease;
    cursor: help;
  }

  .ai-status-indicator:hover {
    background: var(--bg-hover, rgba(0, 0, 0, 0.05));
  }

  .ai-status-indicator:hover .status-tooltip {
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
  }

  .status-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
@keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }

  .status-details {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0; /* Allow text truncation */
  }

  .status-text {
    font-weight: 600;
    line-height: 1.2;
  }

  .provider-info {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.75rem;
    color: var(--text-secondary, #64748b);
    line-height: 1.2;
  }

  .provider {
    font-weight: 500;
  }

  .provider.local {
    color: var(--text-success, #059669);
  }

  .separator {
    color: var(--text-muted, #94a3b8);
  }

  .model {
    font-family: monospace;
    background: var(--bg-muted, #f1f5f9);
    padding: 1px 4px;
    border-radius: 2px;
    color: var(--text-primary, #1e293b);
  }

  .error-text {
    font-size: 0.75rem;
    color: var(--status-error, #ef4444);
    line-height: 1.2;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .status-tooltip {
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%) translateY(-8px);
    background: var(--bg-tooltip, #1e293b);
    color: white;
    padding: 12px;
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    opacity: 0;
    visibility: hidden;
    transition: all 0.2s ease;
    z-index: 1000;
    white-space: nowrap;
    font-size: 0.75rem;
    min-width: 200px;
  }

  .status-tooltip::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: var(--bg-tooltip, #1e293b);
  }

  .tooltip-content {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .tooltip-section {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }

  .tooltip-section.error {
    color: var(--status-error, #fca5a5);
  }

  .tooltip-section strong {
    color: var(--text-primary-inverse, #f8fafc);
  }

  .tooltip-section small {
    font-style: italic;
    opacity: 0.8;
    white-space: normal;
    max-width: 180px;
  }

  /* Dark mode support */
  @media (prefers-color-scheme: dark) {
    .ai-status-indicator:hover {
      background: var(--bg-hover, rgba(255, 255, 255, 0.05));
    }

    .model {
      background: var(--bg-muted, #334155);
      color: var(--text-primary, #f8fafc);
    }

    .status-tooltip {
      background: var(--bg-tooltip, #0f172a);
      border: 1px solid var(--border-color, #334155);
    }

    .status-tooltip::after {
      border-top-color: var(--bg-tooltip, #0f172a);
    }
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .ai-status-indicator {
      padding: 2px 6px;
      font-size: 0.8125rem;
    }

    .status-icon {
      width: 14px;
      height: 14px;
    }

    .status-icon svg,
.provider-info {
      font-size: 0.6875rem;
    }

    .status-tooltip {
      min-width: 180px;
      font-size: 0.6875rem;
    }

    .tooltip-section small {
      max-width: 160px;
    }
  }
</style>

