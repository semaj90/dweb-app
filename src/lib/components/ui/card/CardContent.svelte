<script lang="ts">
  import type { HTMLAttributes } from 'svelte/elements';
  
  interface Props extends HTMLAttributes<HTMLDivElement> {
    className?: string;
  }
  
  export let className: string = '';
</script>

<div
  class="p-6 pt-0 {className}"
>
  <slot />
</div>