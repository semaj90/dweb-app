<script lang="ts">
	import { cva, type VariantProps } from 'class-variance-authority';
	import { cn } from '$lib/utils';

	const textareaVariants = cva(
		'flex min-h-[80px] w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm ring-offset-white placeholder:text-gray-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-950 dark:ring-offset-gray-950 dark:placeholder:text-gray-400 dark:focus-visible:ring-gray-800',
		{
			variants: {
				variant: {
					default: '',
					destructive: 'border-red-500 focus-visible:ring-red-500',
					success: 'border-green-500 focus-visible:ring-green-500',
					warning: 'border-yellow-500 focus-visible:ring-yellow-500'
				},
				size: {
					default: 'min-h-[80px]',
					sm: 'min-h-[60px]',
					lg: 'min-h-[120px]',
					xl: 'min-h-[160px]'
				}
			},
			defaultVariants: {
				variant: 'default',
				size: 'default'
			}
		}
	);

	interface Props {
		value?: string;
		placeholder?: string;
		disabled?: boolean;
		readonly?: boolean;
		required?: boolean;
		
		// Styling
		variant?: VariantProps<typeof textareaVariants>['variant'];
		size?: VariantProps<typeof textareaVariants>['size'];
		class?: string;
		
		// Validation
		minlength?: number;
		maxlength?: number;
		rows?: number;
		cols?: number;
		
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
		oninput?: (event: Event & { currentTarget: HTMLTextAreaElement }) => void;
		onchange?: (event: Event & { currentTarget: HTMLTextAreaElement }) => void;
		onfocus?: (event: FocusEvent & { currentTarget: HTMLTextAreaElement }) => void;
		onblur?: (event: FocusEvent & { currentTarget: HTMLTextAreaElement }) => void;
		onkeydown?: (event: KeyboardEvent & { currentTarget: HTMLTextAreaElement }) => void;
		onkeyup?: (event: KeyboardEvent & { currentTarget: HTMLTextAreaElement }) => void;
	}
	
	let {
		value = $bindable(''),
		placeholder,
		disabled = false,
		readonly = false,
		required = false,
		variant = 'default',
		size = 'default',
		class: className = '',
		minlength,
		maxlength,
		rows,
		cols,
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
	
	let textareaClass = $derived(cn(textareaVariants({ variant, size }), class));
	
	type $$Props = Props;
</script>

<textarea
	bind:value
	{placeholder}
	{disabled}
	{readonly}
	{required}
	{minlength}
	{maxlength}
	{rows}
	{cols}
	{name}
	{id}
	{form}
	class={textareaClass}
	aria-label={ariaLabel}
	aria-labelledby={ariaLabelledBy}
	aria-describedby={ariaDescribedBy}
	data-testid={testId || "melt-textarea"}
	{oninput}
	{onchange}
	{onfocus}
	{onblur}
	{onkeydown}
	{onkeyup}
></textarea>