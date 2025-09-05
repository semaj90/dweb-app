<script lang="ts">
  interface Props {
    showInline?: any;
    autoHide?: any;
    maxWidth?: any;
  }
  let {
    showInline = false,
    autoHide = true,
    maxWidth = "max-w-lg"
  } = $props();



  import type { User } from '$lib/types';
  import { Button } from "$lib/components/ui/button/index.js";
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

   // false; // Show as inline alert vs modal
  let { autoHide = $bindable() } = $props(); // true; // Auto-hide non-critical errors
  let { maxWidth = $bindable() } = $props(); // "max-w-lg"; // Maximum width class

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
      class={`alert ${getAlertClass(currentError.severity)} ${maxWidth} mx-auto`}
      role="alert"
    >
      {#if currentError.severity === "critical" || currentError.severity === "error"}
        <AlertCircle class="h-5 w-5 flex-shrink-0" />
      {:else if currentError.severity === "warning"}
        <AlertTriangle class="h-5 w-5 flex-shrink-0" />
      {:else}
        <Info class="h-5 w-5 flex-shrink-0" />
      {/if}

      <div class="flex-1">
        <h3 class="font-semibold">{currentError.title}</h3>
        <p class="text-sm mt-1">{currentError.message}</p>

        {#if currentError.suggestion}
          <p class="text-sm mt-2">
            <strong>Suggestion:</strong>
            {currentError.suggestion}
          </p>
        {/if}

        {#if showDetails && currentError.showDetails}
          <div class="mt-3 p-3 bg-base-200 rounded-lg">
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm font-medium">Technical Details</span>
              <Button
                variant="outline"
                size="sm"
                onclick={() => copyErrorDetails()}
                class="h-8 w-8 p-0"
                aria-label="Copy error details"
              >
                <Copy class="h-4 w-4" />
              </Button>
            </div>
            <div class="text-xs space-y-1">
              <div>Severity: {currentError.severity}</div>
              <div>Time: {new Date().toLocaleString()}</div>
            </div>
          </div>
        {/if}
      </div>

      <div class="flex items-start gap-1">
        {#if currentError.canRetry}
          <Button
            size="sm"
            variant="outline"
            class={getButtonClass(currentError.severity)}
            onclick={() => retryAction()}
            disabled={retryInProgress}
            aria-label="Retry action"
          >
            {#if retryInProgress}
              <div class="loading loading-spinner loading-xs"></div>
            {:else}
              <RefreshCw class="h-4 w-4" />
            {/if}
            Retry
          </Button>
        {/if}

        {#if currentError.showDetails}
          <Button
            size="sm"
            variant="outline"
            onclick={() => (showDetails = !showDetails)}
            aria-label="Toggle error details"
          >
            {#if showDetails}
              <ChevronUp class="h-4 w-4" />
            {:else}
              <ChevronDown class="h-4 w-4" />
            {/if}
          </Button>
        {/if}

        <Button
          size="sm"
          variant="outline"
          onclick={() => clearError()}
          aria-label="Dismiss error"
        >
          <X class="h-4 w-4" />
        </Button>
      </div>
    </div>
  {:else}
    <!-- Modal Error -->
    <div class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-50">
      <div class={`modal-box ${maxWidth} max-h-[90vh] overflow-y-auto`}>
        <div class="flex items-start gap-4">
          {#if currentError.severity === "critical" || currentError.severity === "error"}
            <AlertCircle class="h-6 w-6 text-error flex-shrink-0" />
          {:else if currentError.severity === "warning"}
            <AlertTriangle class="h-6 w-6 text-warning flex-shrink-0" />
          {:else}
            <Info class="h-6 w-6 text-info flex-shrink-0" />
          {/if}

          <div class="flex-1">
            <h3 class="text-lg font-semibold mb-2">{currentError.title}</h3>
            <p class="text-base-content/80">
              {currentError.message}
            </p>

            {#if currentError.suggestion}
              <div class="mt-4 p-3 bg-base-200 rounded-lg">
                <p class="text-sm">
                  <strong>💡 Suggestion:</strong>
                  {currentError.suggestion}
                </p>
              </div>
            {/if}

            {#if showDetails && currentError.showDetails}
              <div class="mt-4 p-4 bg-base-200 rounded-lg">
                <div class="flex items-center justify-between mb-3">
                  <h4 class="font-medium">Technical Details</h4>
                  <Button
                    variant="outline"
                    size="sm"
                    onclick={() => copyErrorDetails()}
                    class="gap-2"
                    aria-label="Copy error details"
                  >
                    <Copy class="h-4 w-4" />
                    Copy
                  </Button>
                </div>
                <div class="text-sm space-y-1 font-mono">
                  <div>Severity: {currentError.severity}</div>
                  <div>Time: {new Date().toLocaleString()}</div>
                  <div>Browser: {navigator.userAgent}</div>
                </div>
              </div>
            {/if}
          </div>
        </div>

        <div class="modal-action">
          {#if currentError.severity === "critical" || currentError.severity === "error"}
            <Button
              variant="outline"
              size="sm"
              onclick={() => reportError()}
              class="gap-2"
            >
              <Bug class="h-4 w-4" />
              Report Issue
            </Button>
          {/if}

          {#if currentError.showDetails}
            <Button
              variant="outline"
              size="sm"
              onclick={() => (showDetails = !showDetails)}
              class="gap-2"
            >
              {#if showDetails}
                <ChevronUp class="h-4 w-4" />
                Hide Details
              {:else}
                <ChevronDown class="h-4 w-4" />
                Show Details
              {/if}
            </Button>
          {/if}

          {#if currentError.canRetry}
            <Button
              class={`gap-2 ${getButtonClass(currentError.severity)}`}
              onclick={() => retryAction()}
              disabled={retryInProgress}
            >
              {#if retryInProgress}
                <div class="loading loading-spinner loading-xs"></div>
                Retrying...
              {:else}
                <RefreshCw class="h-4 w-4" />
                Retry
              {/if}
            </Button>
          {/if}

          <Button
            variant={currentError.canRetry ? "outline" : "primary"}
            onclick={() => clearError()}
          >
            {currentError.canRetry ? "Cancel" : "Close"}
          </Button>
        </div>
      </div>
    </div>
  {/if}
{/if}
