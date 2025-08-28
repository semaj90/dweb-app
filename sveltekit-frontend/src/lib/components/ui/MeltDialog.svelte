<script lang="ts">
	import type { ComponentProps, Snippet } from 'svelte';
	import { createDialog, melt } from '@melt-ui/svelte';
	import { createEventDispatcher } from 'svelte';
	import { cn } from '$lib/utils';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
		
		// Dialog configuration
		preventScroll?: boolean;
		closeOnOutsideClick?: boolean;
		closeOnEscape?: boolean;
		role?: 'dialog' | 'alertdialog';
		
		// Content configuration
		title?: string;
		description?: string;
		
		// Styling
		class?: string;
		overlayClass?: string;
		contentClass?: string;
		
		// Snippets
		trigger?: Snippet;
		children?: Snippet;
		footer?: Snippet;
		
		// Event handlers
		onClose?: () => void;
	}
	
	let {
		open = false,
		onOpenChange,
		preventScroll = true,
		closeOnOutsideClick = true,
		closeOnEscape = true,
		role = 'dialog',
		title,
		description,
		class: className = '',
		overlayClass = '',
		contentClass = '',
		trigger,
		children,
		footer,
		onClose
	}: Props = $props();
	
	// Create the dialog with configuration
	const {
		elements: { 
			trigger: triggerElement, 
			portalled, 
			overlay, 
			content, 
			title: titleElement, 
			description: descriptionElement, 
			close 
		},
		states: { open: dialogOpen }
	} = createDialog({
		preventScroll,
		closeOnOutsideClick,
		escapeBehavior: closeOnEscape ? 'close' : 'ignore',
		role,
		onOpenChange: ({ curr, next }) => {
			if (onOpenChange) {
				onOpenChange(next);
			}
			return next;
		}
	});
	
	const dispatch = createEventDispatcher<{
		close: void;
		open: void;
		'open-change': { open: boolean };
	}>();
	
	// Sync external open prop with internal state
	$effect(() => {
		if (open !== $dialogOpen) {
			dialogOpen.set(open);
		}
	});
	
	// Watch for state changes and dispatch events
	$effect(() => {
		if ($dialogOpen !== open) {
			dispatch('open-change', { open: $dialogOpen });
			if ($dialogOpen) {
				dispatch('open');
			} else {
				dispatch('close');
				if (onClose) {
					onClose();
				}
			}
		}
	});
	
	// Default overlay styles
	const defaultOverlayClass = 'fixed inset-0 z-50 bg-black/80 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0';
	
	// Default content styles
	const defaultContentClass = 'fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border border-gray-200 bg-white p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] sm:rounded-lg md:w-full dark:border-gray-800 dark:bg-gray-950';
	
	type $$Props = Props;
</script>

<!-- Trigger -->
{#if trigger}
	<button 
		use:melt={$triggerElement}
		class="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 dark:ring-offset-gray-950 dark:focus-visible:ring-gray-800"
		type="button"
	>
		{@render trigger()}
	</button>
{/if}

<!-- Dialog -->
{#if $dialogOpen}
	<div use:melt={$portalled}>
		<!-- Overlay -->
		<div 
			use:melt={$overlay} 
			class={cn(defaultOverlayClass, overlayClass)}
		></div>
		
		<!-- Content -->
		<div 
			use:melt={$content}
			class={cn(defaultContentClass, contentClass, className)}
		>
			<!-- Close button -->
			<button
				use:melt={$close}
				class="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-white transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 disabled:pointer-events-none dark:ring-offset-gray-950 dark:focus:ring-gray-800"
				type="button"
			>
				<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
				</svg>
				<span class="sr-only">Close</span>
			</button>
			
			<!-- Title -->
			{#if title}
				<div class="flex flex-col space-y-1.5 text-center sm:text-left">
					<h2 
						use:melt={$titleElement}
						class="text-lg font-semibold leading-none tracking-tight"
					>
						{title}
					</h2>
				</div>
			{/if}
			
			<!-- Description -->
			{#if description}
				<p 
					use:melt={$descriptionElement}
					class="text-sm text-gray-500 dark:text-gray-400"
				>
					{description}
				</p>
			{/if}
			
			<!-- Main content -->
			{#if children}
				<div class="py-4">
					{@render children()}
				</div>
			{/if}
			
			<!-- Footer -->
			{#if footer}
				<div class="flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2">
					{@render footer()}
				</div>
			{/if}
		</div>
	</div>
{/if}