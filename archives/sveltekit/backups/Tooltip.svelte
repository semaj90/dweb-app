<script lang="ts">
  export let content: string = "";
  export let placement: "top" | "bottom" | "left" | "right" = "top";
  export let disabled: boolean = false;

  let showTooltip = false;
  let timeoutId: ReturnType<typeof setTimeout>;

  function handleMouseEnter() {
    if (disabled) return;
    timeoutId = setTimeout(() => {
      showTooltip = true;
    }, 500);
  }

  function handleMouseLeave() {
    clearTimeout(timeoutId);
    showTooltip = false;
  }
</script>

<div
  class="mx-auto px-4 max-w-7xl"
  role="tooltip"
  onmouseenter={handleMouseEnter}
  onmouseleave={handleMouseLeave}
>
  <slot />
  {#if showTooltip && content}
    <div class="mx-auto px-4 max-w-7xl" role="tooltip">
      {content}
    </div>
  {/if}
</div>

<style>
  .tooltip-wrapper {
    position: relative;
    display: inline-block;
  }

  .tooltip {
    position: absolute;
    z-index: 9999;
    background: #1f2937;
    color: white;
    padding: 0.5rem 0.75rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    line-height: 1.25rem;
    white-space: nowrap;
    max-width: 200px;
    word-wrap: break-word;
    white-space: normal;
    pointer-events: none;
  }

  .tooltip-top {
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    margin-bottom: 0.5rem;
  }

  .tooltip-bottom {
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    margin-top: 0.5rem;
  }

  .tooltip-left {
    right: 100%;
    top: 50%;
    transform: translateY(-50%);
    margin-right: 0.5rem;
  }

  .tooltip-right {
    left: 100%;
    top: 50%;
    transform: translateY(-50%);
    margin-left: 0.5rem;
  }
</style>