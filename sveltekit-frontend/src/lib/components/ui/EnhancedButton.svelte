<script lang="ts">
	import type {     ComponentProps     } from 'svelte';
	import { cva, type VariantProps } from 'class-variance-authority';
	import { cn } from '$lib/utils';
	// import { createButton, melt } from 'melt'; // Removed melt dependency
	import { Button as BitsButton } from 'bits-ui';
	import type { Button as BitsButtonType } from 'bits-ui';
	
	const buttonVariants = cva(
		'inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background',
		{
			variants: {
				variant: {
					default: 'bg-primary text-primary-foreground hover:bg-primary/90',
					destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
					outline: 'border border-input bg-background hover:bg-accent hover:text-accent-foreground',
					secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
					ghost: 'hover:bg-accent hover:text-accent-foreground',
					link: 'text-primary underline-offset-4 hover:underline',
					legal: 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500 shadow-md',
					evidence: 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500 shadow-md',
					case: 'bg-purple-600 text-white hover:bg-purple-700 focus:ring-purple-500 shadow-md',
					yorha: 'bg-gradient-to-r from-yellow-400 to-yellow-600 text-black hover:from-yellow-500 hover:to-yellow-700 font-bold uppercase tracking-wider border-2 border-yellow-700 shadow-lg'
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

	interface Props extends ComponentProps<BitsButtonType.Root> {
		variant?: VariantProps<typeof buttonVariants>['variant'];
		size?: VariantProps<typeof buttonVariants>['size'];
		disabled?: boolean;
		type?: 'button' | 'submit' | 'reset';
		href?: string;
		target?: string;
		loading?: boolean;
		loadingText?: string;
		class?: string;
		useMelt?: boolean; // Option to use melt-ui enhancements
		useBits?: boolean; // Option to use bits-ui
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
		useMelt = false, // Disabled melt functionality
		useBits = false,
		...restProps
	}: Props = $props();
	
	let isDisabled = $derived(disabled || loading);
	let buttonClass = $derived(cn(buttonVariants({ variant, size }), class));
	
	// Create melt-ui button for enhanced accessibility and interactions - conditionally
	// const meltButtonBuilder = useMelt ? createButton({
	// 	disabled: isDisabled
	// }) : null;
	
	// const meltButton = meltButtonBuilder?.elements.root;
	// const pressed = meltButtonBuilder?.states.pressed;
	
	// Loading spinner SVG
	const LoadingSpinner = () => (
		`<svg class="mr-2 h-4 w-4 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" aria-hidden="true">
			<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
			<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
		</svg>`
	);
</script>

{#if useBits && !href}
	<!-- Use Bits-UI Button -->
	<BitsButton.Root
		{type}
		disabled={isDisabled}
		class={buttonClass}
		data-testid="enhanced-button"
		data-variant={variant}
		data-pressed={undefined}
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
			{@render children}
		{/if}
	</BitsButton.Root>
{:else if href}
	<!-- Link variant -->
	<a 
		{href} 
		{target}
		class={buttonClass}
		role="button"
		tabindex="0"
		aria-disabled={isDisabled}
		data-testid="enhanced-button"
		data-variant={variant}
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
			{@render children}
		{/if}
	</a>
{:else}
	{#if useMelt}
		<!-- Melt-ui enhanced button -->
		<button
			<!-- <!-- <!-- use:melt={$meltButton} -->
			{type}
			disabled={isDisabled}
			class={buttonClass}
			data-testid="enhanced-button"
			data-variant={variant}
			data-pressed={undefined}
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
				{@render children}
			{/if}
		</button>
	{:else}
		<!-- Standard button without melt-ui -->
		<button
			{type}
			disabled={isDisabled}
			class={buttonClass}
			data-testid="enhanced-button"
			data-variant={variant}
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
				{@render children}
			{/if}
		</button>
	{/if}
{/if}

<style>
	/* YoRHa terminal-style button animations */
	:global([data-variant="yorha"]) {
		position: relative;
		overflow: hidden;
	}
	
	:global([data-variant="yorha"]:before) {
		content: '';
		position: absolute;
		top: 0;
		left: -100%;
		width: 100%;
		height: 100%;
		background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
		transition: left 0.5s;
	}
	
	:global([data-variant="yorha"]:hover:before) {
		left: 100%;
	}
	
	/* Enhanced focus states for accessibility */
	:global([data-testid="enhanced-button"]:focus-visible) {
		outline: 2px solid currentColor;
		outline-offset: 2px;
	}
</style>