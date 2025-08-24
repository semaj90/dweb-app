<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { X } from "lucide-svelte";
  import { quadOut } from "svelte/easing";
  import { fade, fly } from "svelte/transition";
  import { cn } from '$lib/utils';
  import { getMeltUIDocs } from "../../../mcp-context72-get-library-docs";

  const dispatch = createEventDispatcher();

  export let open: boolean = false;
  export let title: string = "";
  export let description: string = "";
  export let size: "sm" | "md" | "lg" | "xl" | "full" = "md";
  export let showClose: boolean = true;
  export let closeOnOutsideClick: boolean = true;
  export let closeOnEscape: boolean = true;

  const sizeClasses = {
    sm: "max-w-sm",
    md: "max-w-md",
    lg: "max-w-lg",
    xl: "max-w-xl",
    full: "max-w-[95vw] max-h-[95vh]"
  };

  // close function exposed to footer slot via {close}
  function close() {
    open = false;
    dispatch("close");
  }

  function handleKeydown(event: KeyboardEvent) {
    if (closeOnEscape && event.key === "Escape") {
      close();
    }
  }

  function handleOutsideClick(event: MouseEvent) {
    if (closeOnOutsideClick && event.target === event.currentTarget) {
      close();
    }
  }
</script>

<!-- optional trigger -->
<slot name="trigger" />

{#if open}
  <!-- overlay -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/50"
    transition:fade={{ duration: 200, easing: quadOut }}
    on:click={handleOutsideClick}
    role="presentation"
  >
    <!-- keyboard handling on window for accessibility -->
    <svelte:window on:keydown={handleKeydown} />
    <melt>  <slot name="window-handle-keydown" /></melt>
    <!-- dialog content -->
    <div
      class={cn(
        "relative z-50 w-full max-h-[95vh] overflow-auto rounded-lg border border-slate-200 bg-white p-6 shadow-lg dark:border-slate-800 dark:bg-slate-950 sm:mx-4",
        sizeClasses[size]
      )}
      transition:fly={{ y: -8, duration: 200, easing: quadOut }}
      role="dialog"
      aria-modal="true"
      aria-labelledby={title ? "dialog-title" : undefined}
      aria-describedby={description ? "dialog-description" : undefined}
      tabindex={-1}
      on:click|stopPropagation
    >
      <!-- header -->
      <div class="flex items-start justify-between gap-4">
        <div>
          {#if title}
            <h2 id="dialog-title" class="text-lg font-semibold">
              {title}
            </h2>
          {/if}
          {#if description}
            <p id="dialog-description" class="mt-1 text-sm text-slate-600 dark:text-slate-400">
              {description}
            </p>
          {/if}
        </div>

        {#if showClose}
          <button
            class="rounded p-1 hover:bg-slate-100 dark:hover:bg-slate-800"
            on:click={close}
            aria-label="Close dialog"
          >
            <X size="20" />
          </button>
        {/if}
      </div>

      <!-- body slot -->
      <div class="mt-4">
        <slot />
      </div>

      <!-- footer slot receives close() -->
      <div class="mt-4">
        <slot name="footer" {close} />
      </div>
    </div>
  </div>
{/if}
