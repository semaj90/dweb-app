<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let id: string | undefined = undefined;
  export let value: string = '';
  export let options: { value: string label: string }[] = [];
  export let placeholder: string | undefined = undefined;
  export let disabled: boolean = false;
  export let ariaLabel: string | undefined = undefined;

  const dispatch = createEventDispatcher();

  function handleChange(e: Event) {
	value = (e.target as HTMLSelectElement).value;
	dispatch('change', { value });
  }

  function handleInput(e: Event) {
	value = (e.target as HTMLSelectElement).value;
	dispatch('input', { value });
  }
</script>

<style>
  .n64-select {
	display: inline-block;
	width: 100%;
  }

  .n64-select select {
	width: 100%;
	padding: 8px 10px;
	border-radius: var(--n64-radius, 6px);
	border: 1px solid rgba(255, 255, 255, 0.08);
	background: rgba(0, 0, 0, 0.14);
	color: var(--n64-text, #fff);
	font-family: var(--n64-font-family, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial);
	font-size: var(--n64-font-size, 14px);
	outline: none
	box-sizing: border-box;
	appearance: none
	-moz-appearance: none
	-webkit-appearance: none
  }

  .n64-select select:focus {
	box-shadow: 0 0 0 3px rgba(255, 212, 0, 0.12);
	border-color: var(--n64-accent, #ffd400);
  }

  .n64-select select:disabled {
	opacity: 0.6;
	cursor: not-allowed;
  }
</style>

<div class="n64-select">
  <select
	id={id}
	bind:value
	{disabled}
	aria-label={ariaLabel}
	on:change={handleChange}
	on:input={handleInput}
  >
	{#if placeholder}
	  <option value="" disabled>{placeholder}</option>
	{/if}

	{#each options as opt}
	  <option value={opt.value}>{opt.label}</option>
	{/each}
	<slot />
  </select>
</div>
