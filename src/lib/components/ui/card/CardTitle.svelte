<script lang="ts">
  import type { HTMLAttributes } from 'svelte/elements';
  
  interface Props extends HTMLAttributes<HTMLHeadingElement> {
    className?: string;
  }
  
  export let className: string = '';
</script>

<h3
  class="text-2xl font-semibold leading-none tracking-tight {className}"
>
  <slot />
</h3>