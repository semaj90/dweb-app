<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { Search, X } from 'lucide-svelte';

	export let placeholder = 'Search...';
	export let value = '';
	export let debounceTime = 300;

	const dispatch = createEventDispatcher();

	let debounceTimer: NodeJS.Timeout;
	let inputElement: HTMLInputElement;
	let isFocused = false;

	function handleInput() {
		clearTimeout(debounceTimer);
		debounceTimer = setTimeout(() => {
			dispatch('search', { query: value });
		}, debounceTime);
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter') {
			clearTimeout(debounceTimer);
			dispatch('search', { query: value });
		} else if (event.key === 'Escape') {
			clearValue();
			inputElement.blur();
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
		dispatch('search', { query: '' });
		inputElement.focus();
	}
</script>

<div class="mx-auto px-4 max-w-7xl" class:focused={isFocused}>
	<div class="mx-auto px-4 max-w-7xl">
		<Search size={18} />
	</div>
	
	<input
		bind:this={inputElement}
		bind:value
		{placeholder}
		class="mx-auto px-4 max-w-7xl"
		type="text"
		oninput={handleInput}
		onkeydown={handleKeydown}
		onfocus={handleFocus}
		onblur={handleBlur}
		aria-label="Search"
	/>
	
	{#if value}
		<button
			class="mx-auto px-4 max-w-7xl"
			onclick={() => clearValue()}
			aria-label="Clear search"
			type="button"
		>
			<X size={16} />
		</button>
	{/if}
</div>

<style>
	.search-input-container {
		position: relative;
		display: flex;
		align-items: center;
		background: var(--pico-background-color);
		border: 1px solid var(--pico-muted-border-color);
		border-radius: 8px;
		transition: all 0.2s ease;
		min-height: 40px;
	}

	.search-input-container:hover {
		border-color: var(--pico-primary);
	}

	.search-input-container.focused {
		border-color: var(--pico-primary);
		box-shadow: 0 0 0 2px var(--pico-primary-background);
	}

	.search-icon {
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 0 12px;
		color: var(--pico-muted-color);
		pointer-events: none;
	}

	.search-input {
		flex: 1;
		padding: 8px 0;
		background: transparent;
		border: none;
		outline: none;
		color: var(--pico-color);
		font-size: 0.875rem;
	}

	.search-input::placeholder {
		color: var(--pico-muted-color);
	}

	.clear-button {
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 8px 12px;
		background: transparent;
		border: none;
		cursor: pointer;
		color: var(--pico-muted-color);
		border-radius: 4px;
		transition: all 0.2s ease;
	}

	.clear-button:hover {
		color: var(--pico-color);
		background: var(--pico-secondary-background);
	}

	.clear-button:active {
		transform: scale(0.95);
	}
</style>