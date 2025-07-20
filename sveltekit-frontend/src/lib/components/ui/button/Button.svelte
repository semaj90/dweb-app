<script>
	let { variant = 'default', class: className, children, ...restProps } = $props();
</script>

<button class="btn btn-{variant} {className}" {...restProps}>
	{#if children}
		{@render children()}
	{:else}
		<slot />
	{/if}
</button>

<style>
	.btn {
		padding: 0.5rem 1rem;
		border: none;
		border-radius: 0.25rem;
		cursor: pointer;
	}
	.btn-default {
		background: #f3f4f6;
		color: #1f2937;
	}
	.btn-primary {
		background: #3b82f6;
		color: white;
	}
</style>
