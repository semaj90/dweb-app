<script lang="ts">
	import type {     Snippet     } from 'svelte';
	import { cva, type VariantProps } from 'class-variance-authority';
	import { cn } from '$lib/utils';

	const cardVariants = cva(
		'rounded-lg border bg-card text-card-foreground shadow-sm',
		{
			variants: {
				variant: {
					default: 'border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-950',
					outline: 'border-gray-200 bg-transparent dark:border-gray-800',
					ghost: 'border-transparent bg-transparent shadow-none',
					legal: 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950',
					evidence: 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950',
					case: 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950',
					yorha: 'border-yellow-400 bg-black/90 text-yellow-400'
				},
				size: {
					default: 'p-6',
					sm: 'p-4',
					lg: 'p-8',
					xl: 'p-10'
				}
			},
			defaultVariants: {
				variant: 'default',
				size: 'default'
			}
		}
	);

	interface Props {
		variant?: VariantProps<typeof cardVariants>['variant'];
		size?: VariantProps<typeof cardVariants>['size'];
		class?: string;
		
		// Content props
		title?: string;
		description?: string;
		
		// Snippets
		children?: Snippet;
		header?: Snippet;
		content?: Snippet;
		footer?: Snippet;
		
		// Accessibility
		role?: string;
		'data-testid'?: string;
		
		// Event handlers
		onclick?: (event: MouseEvent) => void;
	}
	
	let {
		variant = 'default',
		size = 'default',
		class: className = '',
		title,
		description,
		children,
		header,
		content,
		footer,
		role = 'region',
		'data-testid': testId,
		onclick
	}: Props = $props();
	
	let cardClass = $derived(cn(cardVariants({ variant, size }), class));
	
	type $$Props = Props;
</script>

<div
	class={cardClass}
	{role}
	data-testid={testId || "melt-card"}
	{onclick}
	tabindex={onclick ? 0 : undefined}
	keydown={onclick ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onclick?.(e as any); } } : undefined}
>
	<!-- Header -->
	{#if header}
		<div class="flex flex-col space-y-1.5 pb-4">
			{@render header()}
		</div>
	{:else if title || description}
		<div class="flex flex-col space-y-1.5 pb-4">
			{#if title}
				<h3 class="text-lg font-semibold leading-none tracking-tight">{title}</h3>
			{/if}
			{#if description}
				<p class="text-sm text-gray-500 dark:text-gray-400">{description}</p>
			{/if}
		</div>
	{/if}
	
	<!-- Content -->
	{#if content}
		<div class="pt-0">
			{@render content()}
		</div>
	{:else if children}
		{@render children()}
	{/if}
	
	<!-- Footer -->
	{#if footer}
		<div class="flex items-center pt-4">
			{@render footer()}
		</div>
	{/if}
</div>