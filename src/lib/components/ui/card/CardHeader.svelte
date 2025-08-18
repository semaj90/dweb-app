<script lang="ts">
  import type { HTMLAttributes } from 'svelte/elements';
  
  interface Props extends HTMLAttributes<HTMLDivElement> {
    className?: string;
  }
  
  export let className: string = '';
</script>

<div
  class="flex flex-col space-y-1.5 p-6 {className}"
>
  <slot />
</div>