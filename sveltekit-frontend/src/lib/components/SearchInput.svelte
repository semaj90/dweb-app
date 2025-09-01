<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Search, X } from 'lucide-svelte';

  // Convert to Svelte 5 runes pattern
  let {
    placeholder = 'Search...',
    value = $bindable(''),
    debounceTime = 300,
    onsearch = undefined
  } = $props<{
    placeholder?: string;
    value?: string;
    debounceTime?: number;
    onsearch?: ((payload?: unknown) => void) | undefined;
  }>();

  const dispatch = createEventDispatcher();
  let debounceTimer = $state<number | undefined>(undefined);
  let inputElement = $state<HTMLInputElement | null>(null);
  let isFocused = $state(false);

  function triggerSearch() {
	dispatch('search', { query: value });
	onsearch?.({ query: value });
  }

  function handleInput() {
	if (debounceTimer) clearTimeout(debounceTimer);
	debounceTimer = window.setTimeout(() => {
	  triggerSearch();
	}, debounceTime);
  }

  function handleKeydown(event: KeyboardEvent) {
	if (event.key === 'Enter') {
	  if (debounceTimer) clearTimeout(debounceTimer);
	  triggerSearch();
	} else if (event.key === 'Escape') {
	  clearValue();
	  inputElement?.blur();
	}
  }

  function handleFocus() {
	isFocused = true;
  }
  function handleBlur() {
	isFocused = false;
  }
  function clearValue() {
	value = '';
	triggerSearch();
	inputElement?.focus();
  }
</script>

<div class="search-input-container" class:focused={isFocused}>
  <div class="search-icon" aria-hidden="true">
	<Search size={18} />
  </div>

  <input
	bind:this={inputElement}
	bind:value
	{placeholder}
	class="search-input"
	type="text"
	input={handleInput}
	keydown={handleKeydown}
	onfocus={handleFocus}
	onblur={handleBlur}
	aria-label="Search"
  />

  {#if value}
	<button
	  class="clear-button"
	  on:onclick={clearValue}
	  aria-label="Clear search"
	  type="button"
	>
	  <X size={16} />
	</button>
  {/if}
</div>

<style>
  /* @unocss-include */
  .search-input-container {
	position: relative;
	display: flex;
	align-items: center;
	background: var(--bg-primary);
	border: 1px solid var(--border-light);
	border-radius: 8px;
	transition: all 0.2s ease;
	min-height: 40px;
  }
  .search-input-container:hover {
	border-color: var(--harvard-crimson);
  }
  .search-input-container.focused {
	border-color: var(--harvard-crimson);
	box-shadow: 0 0 0 2px var(--bg-secondary);
  }
  .search-icon {
	display: flex;
	align-items: center;
	justify-content: center;
	padding: 0 12px;
	color: var(--text-muted);
	pointer-events: none;
  }
  .search-input {
	flex: 1;
	padding: 8px 0;
	background: transparent;
	border: none;
	outline: none;
	color: var(--text-primary);
	font-size: 0.875rem;
  }
  .search-input::placeholder {
	color: var(--text-muted);
  }
  .clear-button {
	display: flex;
	align-items: center;
	justify-content: center;
	padding: 8px 12px;
	background: transparent;
	border: none;
	cursor: pointer;
	color: var(--text-muted);
	border-radius: 4px;
	transition: all 0.2s ease;
  }
  .clear-button:hover {
	color: var(--text-primary);
	background: var(--bg-tertiary);
  }
  .clear-button:active {
	transform: scale(0.95);
  }
</style>
