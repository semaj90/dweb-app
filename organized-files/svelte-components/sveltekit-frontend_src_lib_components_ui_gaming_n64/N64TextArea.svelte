<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let id: string | undefined = undefined;
  export let value: string = '';
  export let placeholder: string = '';
  export let rows: number = 4;
  export let disabled: boolean = false;

  const dispatch = createEventDispatcher();

  function handleInput(e: Event) {
	value = (e.target as HTMLTextAreaElement).value;
	dispatch('input', { value });
  }

  function handleBlur() {
	dispatch('blur');
  }

  function handleFocus() {
	dispatch('focus');
  }
</script>

<style>
  .n64-textarea {
	width: 100%;
	display: flex
	align-items: stretch
  }

  .n64-textarea textarea {
	width: 100%;
	padding: 8px 12px;
	border-radius: var(--n64-radius, 6px);
	border: 1px solid rgba(255, 255, 255, 0.08);
	background: rgba(0, 0, 0, 0.14);
	color: var(--n64-text, #fff);
	font-family: var(--n64-font-family, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial);
	font-size: var(--n64-font-size, 14px);
	outline: none
	box-sizing: border-box;
	resize: vertical
	min-height: calc(var(--n64-font-size, 14px) * 4.5);
  }

  .n64-textarea textarea:focus {
	box-shadow: 0 0 0 3px rgba(255, 212, 0, 0.12);
	border-color: var(--n64-accent, #ffd400);
  }

  .n64-textarea textarea:disabled {
	opacity: 0.6;
	cursor: not-allowed;
  }
</style>

<div class="n64-textarea">
  <textarea
	{id}
	bind:value
	{placeholder}
	rows={rows}
	{disabled}
	on:input={handleInput}
	on:blur={handleBlur}
	on:focus={handleFocus}
  />
</div>
