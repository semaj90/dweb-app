<script lang="ts">
	import type { ComponentProps } from 'svelte';
	import { cva, type VariantProps } from 'class-variance-authority';
	import { cn } from '$lib/utils';
	// import { Button as ButtonPrimitive } from 'bits-ui';
	import { createEventDispatcher, onMount } from 'svelte';
	import { browser } from '$app/environment';
	
	// XState integration
	import { useMachine } from '@xstate/svelte';
	
	// User analytics and tracking
	import { userAnalyticsStore } from '$lib/stores/analytics';
	import { lokiButtonCache } from '$lib/services/loki-cache';
	import { searchableButtonIndex } from '$lib/services/fuse-search';
	
	// JSON SSR rendering support
	import type { UIJsonSSRConfig, ButtonAnalyticsEvent } from '$lib/types/ui-json-ssr';
	
	const buttonVariants = cva(
		'inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none',
		{
			variants: {
				variant: {
					default: 'bg-primary text-primary-foreground hover:bg-primary/90',
					destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
					outline: 'border border-input bg-background hover:bg-accent hover:text-accent-foreground',
					secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
					ghost: 'hover:bg-accent hover:text-accent-foreground',
					link: 'text-primary underline-offset-4 hover:underline',
					legal: 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500',
					evidence: 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500',
					case: 'bg-purple-600 text-white hover:bg-purple-700 focus:ring-purple-500'
				},
				size: {
					default: 'h-10 px-4 py-2',
					sm: 'h-9 rounded-md px-3',
					lg: 'h-11 rounded-md px-8',
					icon: 'h-10 w-10',
					xs: 'h-8 rounded px-2 text-xs'
				}
			},
			defaultVariants: {
				variant: 'default',
				size: 'default'
			}
		}
	);

	interface Props {
		variant?: VariantProps<typeof buttonVariants>['variant'];
		size?: VariantProps<typeof buttonVariants>['size'];
		disabled?: boolean;
		type?: 'button' | 'submit' | 'reset';
		href?: string;
		target?: string;
		loading?: boolean;
		loadingText?: string;
		class?: string;
		
		// Enhanced modular properties
		id?: string;
		analyticsCategory?: string;
		analyticsAction?: string;
		analyticsLabel?: string;
		xstateContext?: any;
		uiJsonConfig?: UIJsonSSRConfig;
		searchKeywords?: string[];
		cacheKey?: string;
		role?: string;
		'data-testid'?: string;
	}
	
	let {
		variant = 'default',
		size = 'default',
		disabled = false,
		type = 'button',
		href,
		target,
		loading = false,
		loadingText = 'Loading...',
		class: className = '',
		
		// Enhanced modular properties
		id = crypto.randomUUID(),
		analyticsCategory = 'ui',
		analyticsAction = 'click',
		analyticsLabel = '',
		xstateContext,
		uiJsonConfig,
		searchKeywords = [],
		cacheKey,
		role = 'button',
		'data-testid': testId,
		...restProps
	}: Props = $props();
	
	let isDisabled = $derived(disabled || loading);
	let buttonClass = $derived(cn(buttonVariants({ variant, size }), className));
	
	// Basic button component props
	type $$Props = Props;
	
	// Event dispatcher for component communication
	const dispatch = createEventDispatcher<{
		click: ButtonAnalyticsEvent;
		analytics: ButtonAnalyticsEvent;
		cache: { key: string; action: string };
	}>();
	
	// Enhanced click handler with analytics and XState integration
	function handleClick(event: MouseEvent) {
		if (isDisabled || loading) return;
		
		// Analytics tracking
		const analyticsEvent: ButtonAnalyticsEvent = {
			id,
			category: analyticsCategory,
			action: analyticsAction,
			label: analyticsLabel || (event.target as HTMLElement)?.textContent || '',
			timestamp: Date.now(),
			context: xstateContext,
			variant,
			size
		};
		
		// Store analytics
		if (browser) {
			userAnalyticsStore.trackButtonClick(analyticsEvent);
			dispatch('analytics', analyticsEvent);
		}
		
		// Cache interaction if cacheKey provided
		if (cacheKey && browser) {
			lokiButtonCache.recordInteraction(cacheKey, analyticsEvent);
			dispatch('cache', { key: cacheKey, action: 'click' });
		}
		
		dispatch('click', analyticsEvent);
	}
	
	// Register with searchable index on mount
	onMount(() => {
		if (browser && searchKeywords.length > 0) {
			searchableButtonIndex.addButton({
				id,
				keywords: searchKeywords,
				variant,
				size,
				label: analyticsLabel,
				element: document.getElementById(id)
			});
		}
	});
</script>

{#if href}
	<a 
		{href} 
		{target}
		class={buttonClass}
		role="button"
		tabindex="0"
		aria-disabled={isDisabled}
		data-testid="button"
		{...restProps}
	>
		{#if loading}
			<svg 
				class="mr-2 h-4 w-4 animate-spin" 
				xmlns="http://www.w3.org/2000/svg" 
				fill="none" 
				viewBox="0 0 24 24"
				aria-hidden="true"
			>
				<circle 
					class="opacity-25" 
					cx="12" 
					cy="12" 
					r="10" 
					stroke="currentColor" 
					stroke-width="4"
				/>
				<path 
					class="opacity-75" 
					fill="currentColor" 
					d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
				/>
			</svg>
			{loadingText}
		{:else}
			<slot />
		{/if}
	</a>
{:else}
	<button
		{type}
		disabled={isDisabled}
		class={buttonClass}
		data-testid="button"
		on:click={handleClick}
		{...restProps}
	>
		{#if loading}
			<svg 
				class="mr-2 h-4 w-4 animate-spin" 
				xmlns="http://www.w3.org/2000/svg" 
				fill="none" 
				viewBox="0 0 24 24"
				aria-hidden="true"
			>
				<circle 
					class="opacity-25" 
					cx="12" 
					cy="12" 
					r="10" 
					stroke="currentColor" 
					stroke-width="4"
				/>
				<path 
					class="opacity-75" 
					fill="currentColor" 
					d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
				/>
			</svg>
			{loadingText}
		{:else}
			<slot />
		{/if}
	</button>
{/if}

