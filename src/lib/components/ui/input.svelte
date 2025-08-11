<script>
  export let value = '';
  export let type = 'text';
  export let placeholder = '';
  export let disabled = false;
  export let readonly = false;
  export let className = '';
  export let id = '';
  export let name = '';
  export let required = false;
  export let min = undefined;
  export let max = undefined;
  export let step = undefined;
  export let pattern = undefined;
  export let autocomplete = undefined;
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
  {...$$restProps}
  on:change
  on:focus
  on:blur
  on:keydown
  on:keyup
  on:keypress
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