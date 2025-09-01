<script lang="ts">
	import { cva, type VariantProps } from 'class-variance-authority';
	import { cn } from '$lib/utils';

	const inputVariants = cva(
		'flex w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm ring-offset-white file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-gray-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-950 dark:ring-offset-gray-950 dark:placeholder:text-gray-400 dark:focus-visible:ring-gray-800',
		{
			variants: {
				variant: {
					default: '',
					destructive: 'border-red-500 focus-visible:ring-red-500',
					success: 'border-green-500 focus-visible:ring-green-500',
					warning: 'border-yellow-500 focus-visible:ring-yellow-500'
				},
				size: {
					default: 'h-10 px-3 py-2',
					sm: 'h-9 px-3',
					lg: 'h-11 px-4',
					xl: 'h-12 px-4'
				}
			},
			defaultVariants: {
				variant: 'default',
				size: 'default'
			}
		}
	);

	interface Props {
		value?: string | number;
		type?: 'text' | 'email' | 'password' | 'number' | 'tel' | 'url' | 'search';
		placeholder?: string;
		disabled?: boolean;
		readonly?: boolean;
		required?: boolean;
		
		// Styling
		variant?: VariantProps<typeof inputVariants>['variant'];
		size?: VariantProps<typeof inputVariants>['size'];
		class?: string;
		
		// Validation
		pattern?: string;
		minlength?: number;
		maxlength?: number;
		min?: number;
		max?: number;
		step?: number;
		
		// Form integration
		name?: string;
		id?: string;
		form?: string;
		
		// Accessibility
		'aria-label'?: string;
		'aria-labelledby'?: string;
		'aria-describedby'?: string;
		'data-testid'?: string;
		
		// Event handlers
		oninput?: (event: Event & { currentTarget: HTMLInputElement }) => void;
		onchange?: (event: Event & { currentTarget: HTMLInputElement }) => void;
		onfocus?: (event: FocusEvent & { currentTarget: HTMLInputElement }) => void;
		onblur?: (event: FocusEvent & { currentTarget: HTMLInputElement }) => void;
		onkeydown?: (event: KeyboardEvent & { currentTarget: HTMLInputElement }) => void;
		onkeyup?: (event: KeyboardEvent & { currentTarget: HTMLInputElement }) => void;
	}
	
	let {
		value = $bindable(''),
		type = 'text',
		placeholder,
		disabled = false,
		readonly = false,
		required = false,
		variant = 'default',
		size = 'default',
		class: className = '',
		pattern,
		minlength,
		maxlength,
		min,
		max,
		step,
		name,
		id,
		form,
		'aria-label': ariaLabel,
		'aria-labelledby': ariaLabelledBy,
		'aria-describedby': ariaDescribedBy,
		'data-testid': testId,
		oninput,
		onchange,
		onfocus,
		onblur,
		onkeydown,
		onkeyup
	}: Props = $props();
	
	let inputClass = $derived(cn(inputVariants({ variant, size }), class));
	
	type $$Props = Props;
</script>

<input
	bind:value
	{type}
	{placeholder}
	{disabled}
	{readonly}
	{required}
	{pattern}
	{minlength}
	{maxlength}
	{min}
	{max}
	{step}
	{name}
	{id}
	{form}
	class={inputClass}
	aria-label={ariaLabel}
	aria-labelledby={ariaLabelledBy}
	aria-describedby={ariaDescribedBy}
	data-testid={testId || "melt-input"}
	{oninput}
	{onchange}
	{onfocus}
	{onblur}
	{onkeydown}
	{onkeyup}
/>