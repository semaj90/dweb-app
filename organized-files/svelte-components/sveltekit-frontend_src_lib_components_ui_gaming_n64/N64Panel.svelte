<script>
  export let id;
  export let ariaLabel = 'N64 panel';
  export let title = '';
  export let collapsible = false;
  let open = true;

  function toggle() {
	if (collapsible) open = !open;
  }
</script>

<style>
  .n64-panel {
	background: linear-gradient(180deg, #f8f6e0, #f1ecb8);
	border: 2px solid rgba(0,0,0,0.12);
	box-shadow: 0 6px 18px rgba(0,0,0,0.12);
	border-radius: 8px;
	overflow: hidden
  }

  .n64-panel__header {
	display: flex
	align-items: center
	justify-content: space-between;
	padding: 0.5rem 0.75rem;
	background: linear-gradient(180deg, rgba(0,0,0,0.04), transparent);
	font-weight: 600;
  }

  .n64-panel__content {
	padding: 0.75rem;
  }

  .n64-panel__toggle {
	background: transparent
	border: none
	cursor: pointer
	font-size: 0.9rem;
	padding: 0.25rem;
	line-height: 1;
  }
</style>

<div {id} class="n64-panel" role="region" aria-label={ariaLabel}>
  <div class="n64-panel__header">
	<div>
	  {#if title}
		{title}
	  {:else}
		<slot name="header" />
	  {/if}
	</div>

	{#if collapsible}
	  <button
		class="n64-panel__toggle"
		aria-expanded={open}
		on:click={toggle}
		aria-label={open ? 'Collapse panel' : 'Expand panel'}
	  >
		{open ? '▾' : '▸'}
	  </button>
	{/if}
  </div>

  {#if !collapsible || open}
	<div class="n64-panel__content">
	  <slot />
	</div>
  {/if}
</div>
