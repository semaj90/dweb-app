<script lang="ts">
  interface Props {
    onsearch?: (event?: any) => void;
  }
  let {
    placeholder = 'Search...',
    value = '',
    debounceTime = 300
  } = $props();



		import { Search, X } from 'lucide-svelte';

			
	
	let debounceTimer: NodeJS.Timeout;
	let inputElement: HTMLInputElement
	let isFocused = false;

	function handleInput() {
		clearTimeout(debounceTimer);
		debounceTimer = setTimeout(() => {
			onsearch?.();
		}, debounceTime);
}
	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter') {
			clearTimeout(debounceTimer);
			onsearch?.();
		} else if (event.key === 'Escape') {
			clearValue();
			inputElement.blur();
}}
	function handleFocus() {
		isFocused = true;
}
	function handleBlur() {
		isFocused = false;
}
	function clearValue() {
		value = '';
		onsearch?.();
		inputElement.focus();
}
</script>

<div class="space-y-4" class:focused={isFocused}>
	<div class="space-y-4">
		<Search size={18} />
	</div>
	
	<input
		bind:this={inputElement}
		bind:value
		{placeholder}
		class="space-y-4"
		type="text"
		oninput={handleInput}
		onkeydown={handleKeydown}
		onfocus={handleFocus}
		onblur={handleBlur}
		aria-label="Search"
	/>
	
	{#if value}
		<button
			class="space-y-4"
			onclick={() => clearValue()}
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
		position: relative
		display: flex
		align-items: center
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
		display: flex
		align-items: center
		justify-content: center
		padding: 0 12px;
		color: var(--text-muted);
		pointer-events: none
}
	.search-input {
		flex: 1;
		padding: 8px 0;
		background: transparent
		border: none
		outline: none
		color: var(--text-primary);
		font-size: 0.875rem;
}
	.search-input::placeholder {
		color: var(--text-muted);
}
	.clear-button {
		display: flex
		align-items: center
		justify-content: center
		padding: 8px 12px;
		background: transparent
		border: none
		cursor: pointer
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
