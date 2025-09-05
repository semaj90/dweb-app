<script lang="ts">
	import type { HTMLInputAttributes } from 'svelte/elements';

	interface Props extends Omit<HTMLInputAttributes, 'class' | 'value'> {
		label?: string;
		error?: string;
		hint?: string;
		icon?: string;
		loading?: boolean;
		class?: string;
		value?: string;
	}

	let {
		label,
		error,
		hint,
		icon,
		loading = false,
		class: className,
		id = crypto.randomUUID(),
		value = $bindable(''),
		...props
	} = $props();

	const inputClasses = $derived([
		'yorha-input bits-input',
		error && 'border-destructive focus:border-destructive',
		icon && 'pl-10',
		loading && 'pr-10',
		className
	].filter(Boolean).join(' '));
</script>

<div class="space-y-2">
	{#if label}
		<label for={id} class="bits-label">
			{label}
		</label>
	{/if}
	
	<div class="relative">
		{#if icon}
			<div class="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
				<div class="i-lucide-{icon} h-4 w-4"></div>
			</div>
		{/if}
		
		<input
			{id}
			bind:value
			class={inputClasses}
			aria-invalid={error ? 'true' : undefined}
			aria-describedby={error ? `${id}-error` : hint ? `${id}-hint` : undefined}
			{...props}
		/>
		
		{#if loading}
			<div class="absolute right-3 top-1/2 -translate-y-1/2">
				<div class="i-lucide-loader-2 h-4 w-4 animate-spin text-muted-foreground"></div>
			</div>
		{/if}
	</div>
	
	{#if error}
		<p id="{id}-error" class="text-sm text-destructive">
			{error}
		</p>
	{:else if hint}
		<p id="{id}-hint" class="text-sm text-muted-foreground">
			{hint}
		</p>
	{/if}
</div>

<style>
	/* Enhanced Input with NieR styling */
	.yorha-input {
		/* Base styles from UnoCSS shortcuts */
		transition: all 0.2s ease;
	}
	
	.yorha-input:focus {
		box-shadow: 0 0 0 1px var(--color-nier-border-primary);
	}
	
	.yorha-input[aria-invalid="true"] {
		animation: shake 0.3s ease-in-out;
	}
	
	@keyframes shake {
		0%, 100% { transform: translateX(0); }
		25% { transform: translateX(-2px); }
		75% { transform: translateX(2px); }
	}
</style>