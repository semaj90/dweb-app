<script lang="ts">
	import type { Snippet } from 'svelte';
	import { createCheckbox, melt } from '@melt-ui/svelte';
	import { createEventDispatcher } from 'svelte';
	import { cn } from '$lib/utils';

	type CheckedState = boolean | 'indeterminate';

	interface Props {
		checked?: CheckedState;
		disabled?: boolean;
		required?: boolean;
		
		// Form integration
		name?: string;
		id?: string;
		value?: string;
		
		// Styling
		class?: string;
		
		// Label and description
		label?: string;
		description?: string;
		
		// Snippets
		children?: Snippet;
		
		// Accessibility
		'aria-label'?: string;
		'aria-labelledby'?: string;
		'aria-describedby'?: string;
		'data-testid'?: string;
		
		// Event handlers
		onCheckedChange?: (checked: CheckedState) => void;
	}
	
	let {
		checked = $bindable(false),
		disabled = false,
		required = false,
		name,
		id,
		value,
		class: className = '',
		label,
		description,
		children,
		'aria-label': ariaLabel,
		'aria-labelledby': ariaLabelledBy,
		'aria-describedby': ariaDescribedBy,
		'data-testid': testId,
		onCheckedChange
	}: Props = $props();
	
	// Create the checkbox with melt-ui
	const {
		elements: { root, input },
		states: { checked: checkedState, isChecked, isIndeterminate },
		options
	} = createCheckbox({
		checked,
		disabled,
		required,
		onCheckedChange: ({ curr, next }) => {
			if (onCheckedChange) {
				onCheckedChange(next);
			}
			return next;
		}
	});
	
	const dispatch = createEventDispatcher<{
		'checked-change': { checked: CheckedState };
	}>();
	
	// Sync external checked prop with internal state
	$effect(() => {
		if (checked !== $checkedState) {
			checkedState.set(checked);
		}
	});
	
	// Update external binding when internal state changes
	$effect(() => {
		if (checked !== $checkedState) {
			checked = $checkedState;
			dispatch('checked-change', { checked: $checkedState });
		}
	});
	
	// Sync disabled state
	$effect(() => {
		options.update(prev => ({ ...prev, disabled }));
	});
	
	// Default checkbox styles
	const checkboxClass = cn(
		'peer h-4 w-4 shrink-0 rounded-sm border border-gray-300 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-gray-900 data-[state=checked]:text-gray-50 data-[state=indeterminate]:bg-gray-900 data-[state=indeterminate]:text-gray-50 dark:border-gray-600 dark:ring-offset-gray-950 dark:focus-visible:ring-gray-800 dark:data-[state=checked]:bg-gray-50 dark:data-[state=checked]:text-gray-900 dark:data-[state=indeterminate]:bg-gray-50 dark:data-[state=indeterminate]:text-gray-900',
		className
	);
	
	type $$Props = Props;
</script>

<!-- Hidden input for form submission -->
{#if name}
	<input
		type="hidden"
		{name}
		{value}
		bind:checked={$isChecked}
	/>
{/if}

<div class="flex items-center space-x-2">
	<!-- Checkbox -->
	<button
		use:melt={$root}
		{id}
		class={checkboxClass}
		aria-label={ariaLabel}
		aria-labelledby={ariaLabelledBy}
		aria-describedby={ariaDescribedBy}
		data-testid={testId || "melt-checkbox"}
		type="button"
	>
		<!-- Hidden native input -->
		<input use:melt={$input} />
		
		<!-- Checked state indicator -->
		{#if $isChecked}
			<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
			</svg>
		{:else if $isIndeterminate}
			<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"></path>
			</svg>
		{/if}
	</button>
	
	<!-- Label and description -->
	{#if children}
		<div class="grid gap-1.5 leading-none">
			{@render children()}
		</div>
	{:else if label || description}
		<div class="grid gap-1.5 leading-none">
			{#if label}
				<label 
					for={id}
					class="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
				>
					{label}
					{#if required}
						<span class="text-red-500 ml-1">*</span>
					{/if}
				</label>
			{/if}
			{#if description}
				<p class="text-xs text-gray-500 dark:text-gray-400">{description}</p>
			{/if}
		</div>
	{/if}
</div>