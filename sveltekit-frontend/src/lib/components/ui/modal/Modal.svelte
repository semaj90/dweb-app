<script lang="ts">
  import { X } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";
  import { fade, fly } from "svelte/transition";

  export let open = false;
  export let onClose: () => void = () => {};
  export let title = "";
  export let size: "sm" | "md" | "lg" | "xl" = "md";
  export let closable = true;

  const dispatch = createEventDispatcher();

  function close() {
    open = false;
    onClose();
    dispatch("close");}
  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Escape" && closable) {
      close();
}}
  function handleBackdropClick(event: MouseEvent) {
    if (event.target === event.currentTarget && closable) {
      close();
}}
  function getSizeClass(size: string) {
    switch (size) {
      case "sm":
        return "max-w-sm";
      case "md":
        return "max-w-md";
      case "lg":
        return "max-w-lg";
      case "xl":
        return "max-w-xl";
      default:
        return "max-w-md";
}}
  $: sizeClass = getSizeClass(size);
</script>

<svelte:window on:keydown={handleKeydown} />

{#if open}
  <!-- Backdrop -->
  <div
    class="container mx-auto px-4"
    on:click={handleBackdropClick}
    on:keydown={(e) => { if (e.key === 'Escape') close(); }}
    role="presentation"
    aria-hidden="true"
    transition:fade={{ duration: 200 }}
  >
    <!-- Modal -->
    <div
      class="container mx-auto px-4"
      transition:fly={{ y: 50, duration: 300 }}
      on:click|stopPropagation
      on:keydown|stopPropagation
      role="dialog"
      aria-modal="true"
      aria-labelledby={title ? 'modal-title' : undefined}
    >
      <!-- Header -->
      {#if title || closable}
        <div
          class="container mx-auto px-4"
        >
          {#if title}
            <h2 id="modal-title" class="container mx-auto px-4">{title}</h2>
          {:else}
            <div></div>
          {/if}

          {#if closable}
            <button
              on:click={() => close()}
              class="container mx-auto px-4"
            >
              <X class="container mx-auto px-4" />
            </button>
          {/if}
        </div>
      {/if}

      <!-- Content -->
      <div class="container mx-auto px-4">
        <slot />
      </div>
    </div>
  </div>
{/if}

<style>
  /* @unocss-include */
  .modal-overlay {
    backdrop-filter: blur(2px);}
</style>

