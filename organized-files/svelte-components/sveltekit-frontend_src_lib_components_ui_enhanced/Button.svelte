<script lang="ts">
	import { type ButtonVariants, buttonVariants } from './button-variants';
	import type { HTMLButtonAttributes } from 'svelte/elements';

	interface Props extends Omit<HTMLButtonAttributes, 'class'> {
		variant?: ButtonVariants['variant'];
		size?: ButtonVariants['size'];
		loading?: boolean;
		children?: import('svelte').Snippet;
		class?: string;
	}

	let {
		variant = 'default',
		size = 'default',
		loading = false,
		children,
		class: className,
		disabled,
		...props
	} = $props();

	const isDisabled = $derived(disabled || loading)
</script>

<button
	class="yorha-button {buttonVariants({ variant, size })} {className || ''}"
	disabled={isDisabled}
	{...props}
>
	{#if loading}
		<div class="i-lucide-loader-2 animate-spin mr-2 h-4 w-4" aria-hidden="true"></div>
	{/if}
	{@render children?.()}
</button>

<style>
	/* Scoped UnoCSS integration for enhanced performance */
	.yorha-button {
		/* Base styles handled by UnoCSS shortcuts */
		transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
	}
	
	.yorha-button:hover {
		transform: translateY(-1px);
	}
	
	.yorha-button:active {
		transform: translateY(0);
	}
	
	.yorha-button:disabled {
		transform: none
		cursor: not-allowed;
	}
</style>