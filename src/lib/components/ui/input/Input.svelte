<script lang="ts">
  import type { HTMLAttributes } from 'svelte/elements';
  
  interface Props extends HTMLAttributes<HTMLInputElement> {
    type?: string;
    value?: string;
    placeholder?: string;
    disabled?: boolean;
    readonly?: boolean;
    className?: string;
    id?: string;
    name?: string;
    required?: boolean;
    min?: unknown;
    max?: unknown;
    step?: unknown;
    pattern?: unknown;
    autocomplete?: unknown;
  }
  
  export let type: string = 'text';
  export let value: string = '';
  export let placeholder: string = '';
  export let disabled: boolean = false;
  export let readonly: boolean = false;
  export let className: string = '';
  export let id: string = '';
  export let name: string = '';
  export let required: boolean = false;
  export let min: unknown = undefined;
  export let max: unknown = undefined;
  export let step: unknown = undefined;
  export let pattern: unknown = undefined;
  export let autocomplete: unknown = undefined;
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