<script lang="ts">
	import type {    Snippet    } from 'svelte';
	import { Select as BitsSelect } from 'bits-ui';
	import { createEventDispatcher } from 'svelte';
	import { cn } from '$lib/utils';

	interface SelectOption {
		value: string;
		label: string;
		disabled?: boolean;
	}

	interface Props {
		options: SelectOption[];
		value?: string;
		placeholder?: string;
		disabled?: boolean;
		
		// Select configuration
		open?: boolean;
		
		// Styling
		class?: string;
		triggerClass?: string;
		contentClass?: string;
		itemClass?: string;
		
		// Event handlers
		onValueChange?: (value: string | undefined) => void;
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
		value = undefined,
		placeholder = 'Select an option...',
		disabled = false,
		open = false,
		class: className = '',
		triggerClass = '',
		contentClass = '',
		itemClass = '',
		onValueChange,
		onOpenChange,
		trigger,
		option,
		name,
		required = false,
		'aria-label': ariaLabel,
		'aria-labelledby': ariaLabelledBy,
		'data-testid': testId
	}: Props = $props();
	
	const dispatch = createEventDispatcher<{
		'value-change': { value: string | undefined };
		'open-change': { open: boolean };
	}>();

	function handleValueChange(newValue: string | undefined) {
		if (onValueChange) {
			onValueChange(newValue);
		}
		dispatch('value-change', { value: newValue });
	}

	function handleOpenChange(newOpen: boolean) {
		if (onOpenChange) {
			onOpenChange(newOpen);
		}
		dispatch('open-change', { open: newOpen });
	}
	
	// Default styles
	const defaultTriggerClass = 'flex h-10 items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm ring-offset-white placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-950 dark:ring-offset-gray-950 dark:placeholder:text-gray-400 dark:focus:ring-gray-800';
	
	const defaultContentClass = 'z-50 min-w-[8rem] overflow-hidden rounded-md border border-gray-200 bg-white p-1 text-gray-950 shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 dark:border-gray-800 dark:bg-gray-950 dark:text-gray-50';
	
	const defaultItemClass = 'relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-gray-100 focus:text-gray-900 data-[disabled]:pointer-events-none data-[disabled]:opacity-50 data-[highlighted]:bg-gray-100 data-[highlighted]:text-gray-900 dark:focus:bg-gray-800 dark:focus:text-gray-50 dark:data-[highlighted]:bg-gray-800 dark:data-[highlighted]:text-gray-50';
	
	// Get display text for trigger
	let displayText = $derived(() => {
		if (value) {
			const option = options.find(opt => opt.value === value);
			return option?.label || value;
		}
		return placeholder;
	});
	
	type $$Props = Props;
</script>

<!-- Hidden input for form submission -->
{#if name}
	<input type="hidden" {name} value={value || ''} />
{/if}

<BitsSelect.Root {value} {open} {disabled} onValueChange={handleValueChange} onOpenChange={handleOpenChange}>
	<!-- Trigger -->
	<BitsSelect.Trigger
		class={cn(defaultTriggerClass, triggerClass, className)}
		aria-label={ariaLabel}
		aria-labelledby={ariaLabelledBy}
		data-testid={testId || "bits-select-trigger"}
	>
		{#if trigger}
			{@render trigger({ selected: value, open: open })}
		{:else}
			<BitsSelect.Value placeholder={placeholder}>
				{displayText}
			</BitsSelect.Value>
			
			<!-- Arrow -->
			<BitsSelect.Icon class="ml-2 h-4 w-4 opacity-50">
				<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" class="h-4 w-4">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
				</svg>
			</BitsSelect.Icon>
		{/if}
	</BitsSelect.Trigger>

	<!-- Content -->
	<BitsSelect.Portal>
		<BitsSelect.Content
			class={cn(defaultContentClass, contentClass)}
		>
			{#each options as optionItem (optionItem.value)}
				{@const isSelected = value === optionItem.value}
				<BitsSelect.Item
					value={optionItem.value}
					disabled={optionItem.disabled}
					class={cn(defaultItemClass, itemClass)}
				>
					{#if option}
						{@render option({ option: optionItem, isSelected })}
					{:else}
						<!-- Selected indicator -->
						<BitsSelect.ItemIndicator class="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
							<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
							</svg>
						</BitsSelect.ItemIndicator>
						
						<BitsSelect.ItemText class="truncate">
							{optionItem.label}
						</BitsSelect.ItemText>
					{/if}
				</BitsSelect.Item>
			{/each}
		</BitsSelect.Content>
	</BitsSelect.Portal>
</BitsSelect.Root>