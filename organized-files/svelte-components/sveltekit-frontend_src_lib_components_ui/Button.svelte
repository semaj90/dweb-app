<script lang="ts">
	import { Button as BitsButton } from 'bits-ui';
	import { cva, type VariantProps } from 'class-variance-authority';
	import { cn } from '$lib/utils/cn';

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

	type Props = VariantProps<typeof buttonVariants> & {
		class?: string;
		disabled?: boolean;
		type?: 'button' | 'submit' | 'reset';
		href?: string;
		target?: string;
		loading?: boolean;
		loadingText?: string;
		onclick?: (e: MouseEvent) => void;
		children?: any;
		'data-testid'?: string;
	};

	let { 
		class: className = '',
		variant = 'default', 
		size = 'default',
		disabled = false,
		type = 'button',
		href = undefined,
		target = undefined,
		loading = false,
		loadingText = 'Loading...',
		onclick,
		children,
		'data-testid': testId,
		...rest 
	} = $props();

	let isDisabled = $derived(disabled || loading)
	let buttonClass = $derived(cn(buttonVariants({ variant, size }), className));
</script>

{#if href}
	<a 
		{href} 
		{target}
		class={buttonClass}
		role="button"
		tabindex="0"
		aria-disabled={isDisabled}
		data-testid={testId}
		{onclick}
		onkeydown={(e) => {
			if (e.key === 'Enter' || e.key === ' ') {
				e.preventDefault();
				onclick?.(e as any);
			}
		}}
		{...rest}
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
		{:else if children}
			{@render children()}
		{:else}
			<slot />
		{/if}
	</a>
{:else}
	<BitsButton.Root
		{type}
		disabled={isDisabled}
		class={buttonClass}
		data-testid={testId}
		{onclick}
		{...rest}
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
		{:else if children}
			{@render children()}
		{:else}
			<slot />
		{/if}
	</BitsButton.Root>
{/if}