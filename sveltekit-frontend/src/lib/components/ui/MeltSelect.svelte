<script lang="ts">
	import type { Snippet } from 'svelte';
	import { createSelect, melt } from '@melt-ui/svelte';
	import { createEventDispatcher } from 'svelte';
	import { cn } from '$lib/utils';

	interface SelectOption {
		value: string;
		label: string;
		disabled?: boolean;
	}

	interface Props {
		options: SelectOption[];
		value?: string | string[];
		placeholder?: string;
		disabled?: boolean;
		multiple?: boolean;
		
		// Select configuration
		defaultOpen?: boolean;
		loop?: boolean;
		preventScroll?: boolean;
		closeOnEscape?: boolean;
		closeOnOutsideClick?: boolean;
		portal?: boolean;
		sameWidth?: boolean;
		positioning?: 'top' | 'bottom' | 'left' | 'right';
		
		// Styling
		class?: string;
		triggerClass?: string;
		menuClass?: string;
		itemClass?: string;
		
		// Event handlers
		onSelectedChange?: (selected: string | undefined) => void;
		onMultipleSelectedChange?: (selected: string[]) => void;
		onOpenChange?: (open: boolean) => void;
		
		// Snippets
		trigger?: Snippet<[{ selected: any; open: boolean }]>;
		option?: Snippet<[{ option: SelectOption; isSelected: boolean }]>;
		
		// Accessibility
		name?: string;
		required?: boolean;
		'aria-label'?: string;
		'aria-labelledby'?: string;
		'data-testid'?: string;
	}
	
	let {
		options,
		value = multiple ? [] : undefined,
		placeholder = 'Select an option...',
		disabled = false,
		multiple = false,
		defaultOpen = false,
		loop = false,
		preventScroll = true,
		closeOnEscape = true,
		closeOnOutsideClick = true,
		portal = true,
		sameWidth = true,
		positioning = 'bottom',
		class: className = '',
		triggerClass = '',
		menuClass = '',
		itemClass = '',
		onSelectedChange,
		onMultipleSelectedChange,
		onOpenChange,
		trigger,
		option,
		name,
		required = false,
		'aria-label': ariaLabel,
		'aria-labelledby': ariaLabelledBy,
		'data-testid': testId
	}: Props = $props();
	
	// Create the select with configuration
	const {
		elements: { 
			trigger: triggerElement,
			menu,
			item,
			label,
			arrow
		},
		states: { 
			selected, 
			selectedLabel,
			open 
		},
		helpers: { 
			isSelected 
		}
	} = createSelect({
		multiple,
		defaultOpen,
		loop,
		preventScroll,
		closeOnEscape,
		closeOnOutsideClick,
		portal,
		sameWidth,
		positioning,
		onSelectedChange: ({ curr, next }) => {
			if (multiple) {
				if (onMultipleSelectedChange) {
					onMultipleSelectedChange(next as string[]);
				}
			} else {
				if (onSelectedChange) {
					onSelectedChange(next as string | undefined);
				}
			}
			return next;
		},
		onOpenChange: ({ curr, next }) => {
			if (onOpenChange) {
				onOpenChange(next);
			}
			return next;
		}
	});
	
	const dispatch = createEventDispatcher<{
		'selected-change': { selected: string | undefined };
		'multiple-selected-change': { selected: string[] };
		'open-change': { open: boolean };
	}>();
	
	// Sync external value with internal state
	$effect(() => {
		if (multiple) {
			if (Array.isArray(value) && JSON.stringify(value) !== JSON.stringify($selected)) {
				selected.set(value);
			}
		} else {
			if (value !== $selected) {
				selected.set(value);
			}
		}
	});
	
	// Watch for state changes and dispatch events
	$effect(() => {
		if (multiple) {
			dispatch('multiple-selected-change', { selected: $selected as string[] });
		} else {
			dispatch('selected-change', { selected: $selected as string | undefined });
		}
	});
	
	$effect(() => {
		dispatch('open-change', { open: $open });
	});
	
	// Default styles
	const defaultTriggerClass = 'flex h-10 items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm ring-offset-white placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-950 dark:ring-offset-gray-950 dark:placeholder:text-gray-400 dark:focus:ring-gray-800';
	
	const defaultMenuClass = 'z-50 min-w-[8rem] overflow-hidden rounded-md border border-gray-200 bg-white p-1 text-gray-950 shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 dark:border-gray-800 dark:bg-gray-950 dark:text-gray-50';
	
	const defaultItemClass = 'relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-gray-100 focus:text-gray-900 data-[disabled]:pointer-events-none data-[disabled]:opacity-50 data-[highlighted]:bg-gray-100 data-[highlighted]:text-gray-900 dark:focus:bg-gray-800 dark:focus:text-gray-50 dark:data-[highlighted]:bg-gray-800 dark:data-[highlighted]:text-gray-50';
	
	// Get display text for trigger
	let displayText = $derived(() => {
		if (multiple) {
			const selectedArray = $selected as string[];
			if (selectedArray && selectedArray.length > 0) {
				if (selectedArray.length === 1) {
					const option = options.find(opt => opt.value === selectedArray[0]);
					return option?.label || selectedArray[0];
				}
				return `${selectedArray.length} items selected`;
			}
			return placeholder;
		} else {
			if ($selected) {
				const option = options.find(opt => opt.value === $selected);
				return option?.label || $selected;
			}
			return placeholder;
		}
	});
	
	type $$Props = Props;
</script>

<!-- Hidden input for form submission -->
{#if name}
	{#if multiple}
		{#each ($selected as string[]) || [] as selectedValue}
			<input type="hidden" {name} value={selectedValue} />
		{/each}
	{:else}
		<input type="hidden" {name} value={$selected || ''} />
	{/if}
{/if}

<!-- Trigger -->
<button
	use:melt={$triggerElement}
	class={cn(defaultTriggerClass, triggerClass, className)}
	{disabled}
	aria-label={ariaLabel}
	aria-labelledby={ariaLabelledBy}
	data-testid={testId || "melt-select-trigger"}
	type="button"
>
	{#if trigger}
		{@render trigger({ selected: $selected, open: $open })}
	{:else}
		<span class="truncate">{displayText}</span>
		
		<!-- Arrow -->
		<div use:melt={$arrow} class="ml-2 h-4 w-4 opacity-50">
			<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" class="h-4 w-4">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
			</svg>
		</div>
	{/if}
</button>

<!-- Menu -->
{#if $open}
	<div
		use:melt={$menu}
		class={cn(defaultMenuClass, menuClass)}
	>
		{#each options as optionItem (optionItem.value)}
			{@const selected = isSelected(optionItem.value)}
			<div
				use:melt={$item({ value: optionItem.value, disabled: optionItem.disabled })}
				class={cn(defaultItemClass, itemClass)}
			>
				{#if option}
					{@render option({ option: optionItem, isSelected: selected })}
				{:else}
					<!-- Selected indicator -->
					{#if selected}
						<span class="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
							<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
							</svg>
						</span>
					{/if}
					
					<span class="truncate">{optionItem.label}</span>
				{/if}
			</div>
		{/each}
	</div>
{/if}