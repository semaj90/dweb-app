<script>
  // Basic Textarea component
  let {
    value = $bindable(''),
    placeholder = '',
    disabled = false,
    rows = 4,
    class: className = ''
  } = $props();
</script>

<textarea
  bind:value
  {placeholder}
  {disabled}
  {rows}
  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 {className}"
></textarea>