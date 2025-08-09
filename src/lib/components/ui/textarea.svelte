<script>
  export let value = '';
  export let placeholder = '';
  export let disabled = false;
  export let readonly = false;
  export let className = '';
  export let id = '';
  export let name = '';
  export let required = false;
  export let rows = 4;
  export let cols = undefined;
  export let maxlength = undefined;
  export let minlength = undefined;
  export let wrap = undefined;
  export let resize = true;
  
  function handleInput(event) {
    value = event.target.value;
  }
  
  $: resizeClass = resize ? '' : 'resize-none';
</script>

<textarea
  {id}
  {name}
  {placeholder}
  {disabled}
  {readonly}
  {required}
  {rows}
  {cols}
  {maxlength}
  {minlength}
  {wrap}
  {value}
  on:input={handleInput}
  on:change
  on:focus
  on:blur
  on:keydown
  on:keyup
  on:keypress
  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed dark:bg-gray-800 dark:border-gray-600 dark:text-white {resizeClass} {className}"
></textarea>