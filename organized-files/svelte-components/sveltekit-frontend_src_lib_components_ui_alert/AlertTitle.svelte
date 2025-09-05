<script lang="ts">
  let {
    class: className = '',
    children,
    ...restProps
  } = $props();
</script>

<h5 class="mb-1 font-medium leading-none tracking-tight {className}" {...restProps}>
  <slot></slot>
</h5>
