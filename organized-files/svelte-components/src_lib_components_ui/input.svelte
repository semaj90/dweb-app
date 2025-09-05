<script>
  interface Props {
    value?: any;
    type?: string;
    placeholder?: string;
    disabled?: boolean;
    readonly?: boolean;
    className?: string;
    id?: string;
    name?: string;
    required?: boolean;
    min?: any;
    max?: any;
    step?: any;
    pattern?: any;
    autocomplete?: any;
  }
  
  let {
    value = $bindable(''),
    type = 'text',
    placeholder = '',
    disabled = false,
    readonly = false,
    className = '',
    id = '',
    name = '',
    required = false,
    min = undefined,
    max = undefined,
    step = undefined,
    pattern = undefined,
    autocomplete = undefined,
    ...restProps
  }: Props = $props();
</script>

<input
  {type}
  {id}
  {name}
  {placeholder}
  {disabled}
  {readonly}
  {required}
  {min}
  {max}
  {step}
  {pattern}
  {autocomplete}
  bind:value
  {...restProps}
  onchange
  onfocus
  onblur
  onkeydown
  onkeyup
  onkeypress
  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed dark:bg-gray-800 dark:border-gray-600 dark:text-white {className}"
/>

<style>
  input[type="number"]::-webkit-inner-spin-button,
  input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }

  input[type="number"] {
  /* Standard */
  appearance: textfield;
  /* Firefox */
  -moz-appearance: textfield;
  }
</style>