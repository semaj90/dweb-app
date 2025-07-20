<script lang="ts">
  import type { User } from '$lib/types';
  import Button from "$lib/components/ui/button";
  import {
    errorHandler,
    type UserFriendlyError,
  } from "$lib/stores/error-handler";
  import { notifications } from "$lib/stores/notification";
  import {
    AlertCircle,
    AlertTriangle,
    Bug,
    ChevronDown,
    ChevronUp,
    Copy,
    Info,
    RefreshCw,
    X,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  export let showInline = false; // Show as inline alert vs modal
  export let autoHide = true; // Auto-hide non-critical errors
  export let maxWidth = "max-w-lg"; // Maximum width class

  let currentError: UserFriendlyError | null = null;
  let showDetails = false;
  let retryInProgress = false;

  onMount(() => {
    const unsubscribe = errorHandler.subscribe((error) => {
      currentError = error;

      // Auto-hide info level errors
      if (error && autoHide && error.severity === "info") {
        setTimeout(() => {
          if (currentError === error) {
            clearError();
}
        }, 5000);
}
    });

    return unsubscribe;
  });

  function clearError() {
    errorHandler.clear();
    currentError = null;
    showDetails = false;
    retryInProgress = false;
}
  async function retryAction() {
    if (!currentError?.canRetry) return;

    retryInProgress = true;
    try {
      // The retry function should be attached to the error
      // This is a placeholder - actual retry would depend on the error context
      await new Promise((resolve) => setTimeout(resolve, 1000));
      clearError();
      notifications.add({
        type: "success",
        title: "Retry Successful",
        message: "The operation completed successfully.",
      });
    } catch (error) {
      // If retry fails, show a new error
      errorHandler.handle(error, { context: "retry_failed" });
    } finally {
      retryInProgress = false;
}}
  function copyErrorDetails() {
    if (!currentError) return;

    const errorText = `Error: ${currentError.title}
Message: ${currentError.message}
Suggestion: ${currentError.suggestion || "None"}
Severity: ${currentError.severity}
Timestamp: ${new Date().toISOString()}`;

    navigator.clipboard
      .writeText(errorText)
      .then(() => {
        notifications.add({
          type: "success",
          title: "Copied",
          message: "Error details copied to clipboard.",
        });
      })
      .catch(() => {
        // Fallback for older browsers
        const textarea = document.createElement("textarea");
        textarea.value = errorText;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);

        notifications.add({
          type: "success",
          title: "Copied",
          message: "Error details copied to clipboard.",
        });
      });
}
  function getIcon(severity: string) {
    switch (severity) {
      case "critical":
      case "error":
        return AlertCircle;
      case "warning":
        return AlertTriangle;
      case "info":
      default:
        return Info;
}}
  function getAlertClass(severity: string) {
    switch (severity) {
      case "critical":
        return "alert-error border-error/20 bg-error/10";
      case "error":
        return "alert-error border-error/20 bg-error/5";
      case "warning":
        return "alert-warning border-warning/20 bg-warning/5";
      case "info":
      default:
        return "alert-info border-info/20 bg-info/5";
}}
  function getButtonClass(severity: string) {
    switch (severity) {
      case "critical":
      case "error":
        return "btn-error";
      case "warning":
        return "btn-warning";
      case "info":
      default:
        return "btn-info";
}}
  // Report error to support (placeholder)
  function reportError() {
    if (!currentError) return;

    // This would integrate with your error reporting service
    console.log("Reporting error:", currentError);

    notifications.add({
      type: "success",
      title: "Error Reported",
      message: "Thank you for reporting this issue. Our team will investigate.",
    });
}
</script>

{#if currentError}
  {#if showInline}
    <!-- Inline Alert -->
    <div
      class="container mx-auto px-4"
      role="alert"
    >
      {#if currentError.severity === "critical" || currentError.severity === "error"}
        <AlertCircle class="container mx-auto px-4" />
      {:else if currentError.severity === "warning"}
        <AlertTriangle class="container mx-auto px-4" />
      {:else}
        <Info class="container mx-auto px-4" />
      {/if}

      <div class="container mx-auto px-4">
        <h3 class="container mx-auto px-4">{currentError.title}</h3>
        <p class="container mx-auto px-4">{currentError.message}</p>

        {#if currentError.suggestion}
          <p class="container mx-auto px-4">
            <strong>Suggestion:</strong>
            {currentError.suggestion}
          </p>
        {/if}

        {#if showDetails && currentError.showDetails}
          <div class="container mx-auto px-4">
            <div class="container mx-auto px-4">
              <span class="container mx-auto px-4">Technical Details</span>
              <Button
                variant="ghost"
                size="sm"
                on:click={() => copyErrorDetails()}
                class="container mx-auto px-4"
                aria-label="Copy error details"
              >
                <Copy class="container mx-auto px-4" />
              </Button>
            </div>
            <div class="container mx-auto px-4">
              <div>Severity: {currentError.severity}</div>
              <div>Time: {new Date().toLocaleString()}</div>
            </div>
          </div>
        {/if}
      </div>

      <div class="container mx-auto px-4">
        {#if currentError.canRetry}
          <Button
            size="sm"
            variant="outline"
            class={getButtonClass(currentError.severity)}
            on:click={() => retryAction()}
            disabled={retryInProgress}
            aria-label="Retry action"
          >
            {#if retryInProgress}
              <div class="container mx-auto px-4"></div>
            {:else}
              <RefreshCw class="container mx-auto px-4" />
            {/if}
            Retry
          </Button>
        {/if}

        {#if currentError.showDetails}
          <Button
            size="sm"
            variant="ghost"
            on:click={() => (showDetails = !showDetails)}
            aria-label="Toggle error details"
          >
            {#if showDetails}
              <ChevronUp class="container mx-auto px-4" />
            {:else}
              <ChevronDown class="container mx-auto px-4" />
            {/if}
          </Button>
        {/if}

        <Button
          size="sm"
          variant="ghost"
          on:click={() => clearError()}
          aria-label="Dismiss error"
        >
          <X class="container mx-auto px-4" />
        </Button>
      </div>
    </div>
  {:else}
    <!-- Modal Error -->
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          {#if currentError.severity === "critical" || currentError.severity === "error"}
            <AlertCircle class="container mx-auto px-4" />
          {:else if currentError.severity === "warning"}
            <AlertTriangle class="container mx-auto px-4" />
          {:else}
            <Info class="container mx-auto px-4" />
          {/if}

          <div class="container mx-auto px-4">
            <h3 class="container mx-auto px-4">{currentError.title}</h3>
            <p class="container mx-auto px-4">
              {currentError.message}
            </p>

            {#if currentError.suggestion}
              <div class="container mx-auto px-4">
                <p class="container mx-auto px-4">
                  <strong>💡 Suggestion:</strong>
                  {currentError.suggestion}
                </p>
              </div>
            {/if}

            {#if showDetails && currentError.showDetails}
              <div class="container mx-auto px-4">
                <div class="container mx-auto px-4">
                  <h4 class="container mx-auto px-4">Technical Details</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    on:click={() => copyErrorDetails()}
                    class="container mx-auto px-4"
                    aria-label="Copy error details"
                  >
                    <Copy class="container mx-auto px-4" />
                    Copy
                  </Button>
                </div>
                <div class="container mx-auto px-4">
                  <div>Severity: {currentError.severity}</div>
                  <div>Time: {new Date().toLocaleString()}</div>
                  <div>Browser: {navigator.userAgent}</div>
                </div>
              </div>
            {/if}
          </div>
        </div>

        <div class="container mx-auto px-4">
          {#if currentError.severity === "critical" || currentError.severity === "error"}
            <Button
              variant="outline"
              size="sm"
              on:click={() => reportError()}
              class="container mx-auto px-4"
            >
              <Bug class="container mx-auto px-4" />
              Report Issue
            </Button>
          {/if}

          {#if currentError.showDetails}
            <Button
              variant="outline"
              size="sm"
              on:click={() => (showDetails = !showDetails)}
              class="container mx-auto px-4"
            >
              {#if showDetails}
                <ChevronUp class="container mx-auto px-4" />
                Hide Details
              {:else}
                <ChevronDown class="container mx-auto px-4" />
                Show Details
              {/if}
            </Button>
          {/if}

          {#if currentError.canRetry}
            <Button
              class={`gap-2 ${getButtonClass(currentError.severity)}`}
              on:click={() => retryAction()}
              disabled={retryInProgress}
            >
              {#if retryInProgress}
                <div class="container mx-auto px-4"></div>
                Retrying...
              {:else}
                <RefreshCw class="container mx-auto px-4" />
                Retry
              {/if}
            </Button>
          {/if}

          <Button
            variant={currentError.canRetry ? "outline" : "default"}
            on:click={() => clearError()}
          >
            {currentError.canRetry ? "Cancel" : "Close"}
          </Button>
        </div>
      </div>
    </div>
  {/if}
{/if}
