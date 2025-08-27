<script lang="ts">
	import type { ComponentProps, Snippet } from 'svelte';
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
		'inline-flex items-center justify-center font-medium transition-all duration-200 focus-visible:outline-none nes-focus disabled:opacity-50 disabled:pointer-events-none',
		{
			variants: {
				variant: {
					default: 'btn-nes-primary',
					destructive: 'btn-nes-danger',
					outline: 'border-2 border-gray-400 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800',
					secondary: 'btn-nes-secondary',
					ghost: 'hover:bg-gray-100 dark:hover:bg-gray-800 rounded-none',
					link: 'text-blue-500 underline-offset-4 hover:underline hover:text-blue-600',
					legal: 'nes-legal-priority-medium yorha-3d-button',
					evidence: 'nes-legal-priority-critical yorha-3d-button',
					case: 'nes-legal-priority-high yorha-3d-button',
					success: 'btn-nes-success',
					yorha: 'yorha-3d-button bg-black/80 text-yellow-400 border-2 border-yellow-400',
					neural: 'neural-sprite-active bg-gradient-to-r from-purple-600 to-blue-600 text-white border-2 border-purple-400'
				},
				size: {
					default: 'h-10 px-4 py-2',
					sm: 'h-9 rounded-md px-3',
					lg: 'h-11 rounded-md px-8',
					icon: 'h-8 w-8',
				icon_sm: 'h-6 w-6',
				icon_lg: 'h-12 w-12',
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
		children?: Snippet;
		onclick?: (event: MouseEvent) => void;
		
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
		children,
		onclick,
		
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
		
		// Call the onclick prop if provided
		if (onclick) {
			onclick(event);
		}
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
			{@render children?.()}
		{/if}
	</a>
{:else}
	<button
		{type}
		disabled={isDisabled}
		class={buttonClass}
		data-testid="button"
		onclick={handleClick}
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
			{@render children?.()}
		{/if}
	</button>
{/if}

