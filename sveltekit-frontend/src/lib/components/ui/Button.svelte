<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

	// Using standard button element instead of melt-ui Button
	import { cva, type VariantProps } from 'class-variance-authority';
	import { cn } from '$lib/utils/cn';
	import { createButton } from '@melt-ui/svelte';
	
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
	
	type $$Props = VariantProps<typeof buttonVariants> & {
		class?: string;
		disabled?: boolean;
		type?: 'button' | 'submit' | 'reset';
		href?: string;
		target?: string;
		loading?: boolean;
		loadingText?: string;
		onclick?: (e: MouseEvent) => void;
		'data-testid'?: string;
	};

	interface $$RestProps {
		[key: string]: any;
	}

	let $$restProps: $$RestProps = {};
	$: $$restProps = Object.fromEntries(
		Object.entries($$props).filter(([key]) => ![
			'variant', 'size', 'disabled', 'type', 'href', 'target', 
			'loading', 'loadingText', 'onclick', 'class'
		].includes(key))
	);
	
	export let variant: $$Props['variant'] = 'default';
	export let size: $$Props['size'] = 'default';
	export let disabled = false;
	export let type: $$Props['type'] = 'button';
	export let href: $$Props['href'] = undefined;
	export let target: $$Props['target'] = undefined;
	export let loading = false;
	export let loadingText = 'Loading...';
	
	let className: $$Props['class'] = '';
	export { className as class };
	
	$: isDisabled = disabled || loading;
	$: buttonClass = cn(buttonVariants({ variant, size }), className);

	// Create melt-ui button
	const {
		elements: { root: MeltButton },
		options: { disabled: disabledStore }
	} = createButton({
		disabled: isDisabled
	});

	$: disabledStore.set(isDisabled);
</script>

{#if href}
	<a 
		{href} 
		{target}
		class={buttonClass}
		role="button"
		tabindex="0"
		aria-disabled={isDisabled}
		data-testid={$$restProps['data-testid']}
		onclick
		onkeydown={(e) => {
			if (e.key === 'Enter' || e.key === ' ') {
				e.preventDefault();
				e.currentTarget.click();
			}
		}}
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
		use:MeltButton
		{type}
		disabled={isDisabled}
		class={buttonClass}
		data-testid={$$restProps['data-testid']}
		onclick
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

<!-- TODO: migrate export lets to $props(); CommonProps assumed. -->
