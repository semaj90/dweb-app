<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { X } from "lucide-svelte";

  let { open = $bindable() } = $props(); // false;
  let { title = $bindable() } = $props(); // "";
  let { description = $bindable() } = $props(); // "";
  let { side = $bindable() } = $props(); // "left" | "right" | "top" | "bottom" = "right";
  let { size = $bindable() } = $props(); // "sm" | "md" | "lg" | "xl" = "md";

  const dispatch = createEventDispatcher();

  function handleClose() {
    open = false;
    dispatch("close");
  }

  function handleBackdropClick(e: MouseEvent) {
    if (e.target === e.currentTarget) {
      handleClose();
    }
  }
</script>

{#if open}
  <div
    class="drawer-overlay"
    role="dialog"
    aria-modal="true"
    aria-label={title ? title : "Drawer"}
    on:onclick={handleBackdropClick}
  >
    <div class="drawer drawer-{size} drawer-{side}" click|stopPropagation>
      <div class="drawer-header">
        <div>
          {#if title}
            <h2 class="drawer-title">{title}</h2>
          {/if}
          {#if description}
            <p class="drawer-description">{description}</p>
          {/if}
        </div>
        <button
          class="drawer-close"
          aria-label="Close drawer"
          on:click|stopPropagation={handleClose}
        >
          <X size="24" />
        </button>
      </div>

      <div class="drawer-body">
        {@render children}
      </div>
    </div>
  </div>
{/if}

<style>
  /* @unocss-include */
  .drawer-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .drawer {
    background: white;
    border-radius: 8px;
    padding: 20px;
    max-width: 90vw;
    max-height: 90vh;
    overflow-y: auto;
  }
  .drawer-sm {
    width: 300px;
  }
  .drawer-md {
    width: 500px;
  }
  .drawer-lg {
    width: 700px;
  }
  .drawer-xl {
    width: 900px;
  }
  .drawer-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
  }
  .drawer-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0;
  }
  .drawer-description {
    color: #666;
    margin: 4px 0 0 0;
  }
  .drawer-close {
    background: none;
    border: none;
    padding: 4px;
    cursor: pointer;
    border-radius: 4px;
  }
  .drawer-close:hover {
    background: #f5f5f5;
  }
</style>
