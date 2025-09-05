<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { HTMLInputAttributes } from 'svelte/elements';

	interface $$Props extends Omit<HTMLInputAttributes, 'size' | 'disabled' | 'required' | 'readonly'> {
		label?: string;
		error?: string | null;
		hint?: string;
		variant?: 'default' | 'filled' | 'underlined';
		size?: 'sm' | 'md' | 'lg';
		icon?: string;
		iconPosition?: 'left' | 'right';
		clearable?: boolean;
		loading?: boolean;
		success?: boolean;
		disabled?: boolean | null | undefined;
		required?: boolean | null | undefined;
		readonly?: boolean | null | undefined;
	}

	export let label: $$Props['label'] = undefined;
	export let error: $$Props['error'] = undefined;
	export let hint: $$Props['hint'] = undefined;
	export let variant: NonNullable<$$Props['variant']> = 'default';
	export let size: NonNullable<$$Props['size']> = 'md';
	export let icon: $$Props['icon'] = undefined;
	export let iconPosition: NonNullable<$$Props['iconPosition']> = 'left';
	export let clearable: NonNullable<$$Props['clearable']> = false;
	export let loading: NonNullable<$$Props['loading']> = false;
	export let success: NonNullable<$$Props['success']> = false;
	export let value: $$Props['value'] = '';
	export let disabled: $$Props['disabled'] = false;
	export let required: $$Props['required'] = false;
	export let readonly: $$Props['readonly'] = false;

	const dispatch = createEventDispatcher<{
		input: Event;
		change: Event;
		focus: FocusEvent;
		blur: FocusEvent;
		clear: void;
	}>();

	let inputElement: HTMLInputElement;
	let isFocused = false;

	$: hasValue = value !== '' && value !== null && value !== undefined;
	$: showClearButton = clearable && hasValue && !disabled && !readonly;
	$: hasError = !!error;

	// Generate unique ID for accessibility
	const inputId = `input-${Math.random().toString(36).substr(2, 9)}`;
	const errorId = `${inputId}-error`;
	const hintId = `${inputId}-hint`;

	function handleInput(event: Event) {
		const target = event.target as HTMLInputElement;
		value = target.value;
		dispatch('input', event);
	}

	function handleChange(event: Event) {
		dispatch('change', event);
	}

	function handleFocus(event: FocusEvent) {
		isFocused = true;
		dispatch('focus', event);
	}

	function handleBlur(event: FocusEvent) {
		isFocused = false;
		dispatch('blur', event);
	}

	function handleClear() {
		value = '';
		dispatch('clear');
		inputElement?.focus();
	}

	// Dynamic classes
	$: containerClasses = 'relative flex flex-col gap-1';

	$: labelClasses = [
		'text-sm font-medium transition-colors',
		hasError ? 'text-red-600 dark:text-red-400' : 'text-gray-700 dark:text-gray-300',
		required ? "after:content-['*'] after:ml-1 after:text-red-500" : ''
	].filter(Boolean).join(' ');

	$: inputContainerClasses = 'relative flex items-center';

	$: inputClasses = [
		'block w-full rounded-md border transition-all duration-200',
		'focus:outline-none focus:ring-2 focus:ring-offset-1',
		'disabled:opacity-50 disabled:cursor-not-allowed',
		'placeholder:text-gray-400 dark:placeholder:text-gray-500',
		
		// Size classes
		size === 'sm' ? 'px-3 py-1.5 text-sm' : '',
		size === 'md' ? 'px-3 py-2 text-base' : '',
		size === 'lg' ? 'px-4 py-3 text-lg' : '',

		// Icon padding
		icon && iconPosition === 'left' ? (size === 'sm' ? 'pl-10' : size === 'lg' ? 'pl-12' : 'pl-11') : '',
		icon && iconPosition === 'right' ? (size === 'sm' ? 'pr-10' : size === 'lg' ? 'pr-12' : 'pr-11') : '',
		showClearButton ? (size === 'sm' ? 'pr-10' : size === 'lg' ? 'pr-12' : 'pr-11') : '',

		// Variant classes
		variant === 'default' ? 'bg-white dark:bg-gray-900' : '',
		variant === 'filled' ? 'bg-gray-50 dark:bg-gray-800 border-transparent' : '',
		variant === 'underlined' ? 'bg-transparent border-0 border-b-2 rounded-none px-0' : '',

		// State classes
		hasError ? 'border-red-300 text-red-900 focus:border-red-500 focus:ring-red-500 dark:border-red-600 dark:text-red-100' :
		success ? 'border-green-300 text-green-900 focus:border-green-500 focus:ring-green-500 dark:border-green-600 dark:text-green-100' :
		'border-gray-300 text-gray-900 focus:border-blue-500 focus:ring-blue-500 dark:border-gray-600 dark:text-gray-100'
	].filter(Boolean).join(' ');

	$: iconClasses = [
		'absolute flex items-center justify-center pointer-events-none',
		size === 'sm' ? 'w-8 h-8' : size === 'lg' ? 'w-12 h-12' : 'w-10 h-10',
		iconPosition === 'left' ? 'left-0' : 'right-0',
		hasError ? 'text-red-400' : success ? 'text-green-400' : 'text-gray-400'
	].join(' ');

	$: hintClasses = [
		'text-xs transition-colors',
		hasError ? 'text-red-600 dark:text-red-400' : 'text-gray-500 dark:text-gray-400'
	].join(' ');
</script>

<div class={containerClasses}>
	<!-- Label -->
	{#if label}
		<label for={inputId} class={labelClasses}>
			{label}
		</label>
	{/if}

	<!-- Input Container -->
	<div class={inputContainerClasses}>
		<!-- Left Icon -->
		{#if icon && iconPosition === 'left'}
			<div class={iconClasses}>
				<iconify-icon data-icon="${1}" class="mx-auto px-4 max-w-7xl"></iconify-icon>
			</div>
		{/if}

		<!-- Input -->
		<input
			bind:this={inputElement}
			id={inputId}
			class={inputClasses}
			{value}
			{disabled}
			{readonly}
			{required}
			aria-invalid={hasError}
			aria-describedby={(hasError ? errorId : '') + (hint ? ' ' + hintId : '')}
			oninput={handleInput}
			onchange={handleChange}
			onfocus={handleFocus}
			onblur={handleBlur}
			{...$$restProps}
		/>

		<!-- Right Icon or Status -->
		{#if loading}
			<div class="mx-auto px-4 max-w-7xl">
				<div class="mx-auto px-4 max-w-7xl"></div>
			</div>
		{:else if success}
			<div class="mx-auto px-4 max-w-7xl">
				<iconify-icon data-icon="${1}" class="mx-auto px-4 max-w-7xl"></iconify-icon>
			</div>
		{:else if hasError}
			<div class="mx-auto px-4 max-w-7xl">
				<iconify-icon data-icon="${1}" class="mx-auto px-4 max-w-7xl"></iconify-icon>
			</div>
		{:else if icon && iconPosition === 'right'}
			<div class={iconClasses}>
				<iconify-icon data-icon="${1}" class="mx-auto px-4 max-w-7xl"></iconify-icon>
			</div>
		{/if}

		<!-- Clear Button -->
		{#if showClearButton}
			<button
				type="button"
				class="mx-auto px-4 max-w-7xl"
				onclick={() => handleClear()}
				tabindex={-1}
				aria-label="Clear input"
			>
				<iconify-icon data-icon="${1}" class="mx-auto px-4 max-w-7xl"></iconify-icon>
			</button>
		{/if}
	</div>

	<!-- Error Message -->
	{#if hasError}
		<p id={errorId} class={hintClasses} role="alert">
			{error}
		</p>
	{/if}

	<!-- Hint Text -->
	{#if hint && !hasError}
		<p id={hintId} class={hintClasses}>
			{hint}
		</p>
	{/if}
</div>