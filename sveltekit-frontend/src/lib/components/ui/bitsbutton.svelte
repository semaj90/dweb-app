<script lang="ts">
	import { cva, type VariantProps } from 'class-variance-authority';
	import { cn } from '$lib/utils';
	import { createEventDispatcher } from 'svelte';
	import { browser } from '$app/environment';
	// NES.css now imported globally in +layout.svelte to avoid duplicate CSS injection

	import { userAnalyticsStore } from '$lib/stores/analytics';
	import { lokiButtonCache } from '$lib/services/loki-cache';
	import { searchableButtonIndex } from '$lib/services/fuse-search';

	import type { UIJsonSSRConfig, ButtonAnalyticsEvent } from '$lib/types/ui-json-ssr';

	// (Optional) If you want to leverage Bits UI's Button component, import it.
	// Here we keep native <button>/<a> to preserve existing structure.
	// import { Button as BitsButton } from 'bits-ui';

	const buttonVariants = cva(
		'inline-flex items-center justify-center font-medium transition-all duration-200 focus-visible:outline-none nes-focus disabled:opacity-50 disabled:pointer-events-none cursor-pointer',
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

	// Props
	export let variant: VariantProps<typeof buttonVariants>['variant'] = 'default';
	export let size: VariantProps<typeof buttonVariants>['size'] = 'default';
	export let disabled = false;
	export let type: 'button' | 'submit' | 'reset' = 'button';
	export let href: string | undefined = undefined;
	export let target: string | undefined = undefined;
	export let loading = false;
	export let loadingText = 'Loading...';
	export let className = '';
	export let onclick: ((event: MouseEvent) => void) | undefined = undefined;
	export let id: string = (typeof globalThis !== 'undefined' && (globalThis.crypto as any)?.randomUUID)
		? (globalThis.crypto as any).randomUUID()
		: `bits-btn-${Math.random().toString(36).slice(2, 9)}`;
	export let analyticsCategory = 'ui';
	export let analyticsAction = 'click';
	export let analyticsLabel = '';
	export let xstateContext: any = undefined; // eslint-disable-line @typescript-eslint/no-explicit-any
	export let uiJsonConfig: UIJsonSSRConfig | undefined = undefined;
	export let searchKeywords: string[] = [];
	export let cacheKey: string | undefined = undefined;
	export let role: string = 'button';
	export let dataTestid: string | undefined = undefined;

	const dispatch = createEventDispatcher<{
		click: ButtonAnalyticsEvent;
		analytics: ButtonAnalyticsEvent;
		cache: { key: string; action: string };
	}>();

	$: isDisabled = disabled || loading;
	$: buttonClass = cn(buttonVariants({ variant, size }), className);

	let isDisabled: boolean;
	let buttonClass: string;

	function handleClick(event: MouseEvent) {
		if (isDisabled) {
			event.preventDefault();
			return;
		}
		const analyticsEvent: ButtonAnalyticsEvent = {
			id,
			category: analyticsCategory,
			action: analyticsAction,
			label: analyticsLabel || (event.currentTarget as HTMLElement)?.textContent || '',
			timestamp: Date.now(),
			context: xstateContext,
			variant,
			size
		};
		if (browser) {
			userAnalyticsStore.trackButtonClick(analyticsEvent);
			dispatch('analytics', analyticsEvent);
			if (cacheKey) {
				lokiButtonCache.recordInteraction(cacheKey, analyticsEvent);
				dispatch('cache', { key: cacheKey, action: 'click' });
			}
			if (searchKeywords.length > 0) {
				searchableButtonIndex.addButton({ id, keywords: searchKeywords });
			}
		}
		dispatch('click', analyticsEvent);
		if (onclick) onclick(event);
	}
</script>

{#if href}
	<a
		{href}
		{target}
		id={id}
		class={buttonClass}
		role={role}
		tabindex="0"
		aria-disabled={isDisabled}
		data-testid={dataTestid || 'bits-button'}
		onclick={handleClick}
		{...$$restProps}
	>
		{#if loading}
			<svg class="mr-2 h-4 w-4 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" aria-hidden="true">
				<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
				<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
			</svg>
			{loadingText}
		{:else}
			<slot />
		{/if}
	</a>
{:else}
	<button
		id={id}
		type={type}
		disabled={isDisabled}
		class={buttonClass}
		data-testid={dataTestid || 'bits-button'}
		onclick={handleClick}
		{...$$restProps}
	>
		{#if loading}
			<svg class="mr-2 h-4 w-4 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" aria-hidden="true">
				<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
				<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
			</svg>
			{loadingText}
		{:else}
			<slot />
		{/if}
	</button>
{/if}
