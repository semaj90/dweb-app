<script lang="ts">
	import { cva, type VariantProps } from 'class-variance-authority';
	import { cn } from '$lib/utils';

	const badgeVariants = cva(
		'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 dark:focus:ring-gray-800',
		{
			variants: {
				variant: {
					default: 'border-transparent bg-gray-900 text-gray-50 hover:bg-gray-900/80 dark:bg-gray-50 dark:text-gray-900 dark:hover:bg-gray-50/80',
					secondary: 'border-transparent bg-gray-100 text-gray-900 hover:bg-gray-100/80 dark:bg-gray-800 dark:text-gray-50 dark:hover:bg-gray-800/80',
					destructive: 'border-transparent bg-red-500 text-gray-50 hover:bg-red-500/80 dark:bg-red-900 dark:text-red-50 dark:hover:bg-red-900/80',
					success: 'border-transparent bg-green-500 text-gray-50 hover:bg-green-500/80 dark:bg-green-900 dark:text-green-50 dark:hover:bg-green-900/80',
					warning: 'border-transparent bg-yellow-500 text-gray-50 hover:bg-yellow-500/80 dark:bg-yellow-900 dark:text-yellow-50 dark:hover:bg-yellow-900/80',
					outline: 'text-gray-900 dark:text-gray-50',
					legal: 'border-transparent bg-blue-500 text-gray-50 hover:bg-blue-500/80',
					evidence: 'border-transparent bg-red-600 text-gray-50 hover:bg-red-600/80',
					case: 'border-transparent bg-green-600 text-gray-50 hover:bg-green-600/80',
					yorha: 'border-yellow-400 bg-black/90 text-yellow-400 hover:bg-black'
				},
				size: {
					default: 'px-2.5 py-0.5 text-xs',
					sm: 'px-2 py-0.5 text-xs',
					lg: 'px-3 py-1 text-sm',
					xl: 'px-4 py-1.5 text-base'
				}
			},
			defaultVariants: {
				variant: 'default',
				size: 'default'
			}
		}
	);

	import type { BadgeProps } from '$lib/types/component-props.js';

	interface Props extends BadgeProps {
		// Content
		text?: string;
		
		// Event handlers
		onclick?: (event: MouseEvent) => void;
		
		// Accessibility
		role?: string;
		'aria-label'?: string;
	}

	let {
		variant = 'default',
		size = 'default',
		class: className = '',
		text,
		onclick,
		role,
		'aria-label': ariaLabel,
		'data-testid': testId,
		children
	}: Props = $props();

	let badgeClass = $derived(cn(badgeVariants({ variant, size }), class));
	let isClickable = $derived(!!onclick);

	type $$Props = Props;
</script>

{#if isClickable}
	<button
		class={badgeClass}
		{onclick}
		{role}
		aria-label={ariaLabel}
		data-testid={testId || "melt-badge"}
		type="button"
		tabindex={0}
		keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onclick?.(e as any); } }}
	>
		{#if children}
			{@render children()}
		{:else if text}
			{text}
		{/if}
	</button>
{:else}
	<span
		class={badgeClass}
		{role}
		aria-label={ariaLabel}
		data-testid={testId || "melt-badge"}
	>
		{#if children}
			{@render children()}
		{:else if text}
			{text}
		{/if}
	</span>
{/if}