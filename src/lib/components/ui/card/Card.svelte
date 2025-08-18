<script lang="ts">
  import type { HTMLAttributes } from 'svelte/elements';
  
  interface Props extends HTMLAttributes<HTMLDivElement> {
    className?: string;
  }
  
  export let className: string = '';
</script>

<div
  class="rounded-lg border bg-card text-card-foreground shadow-sm {className}"
>
  <slot />
</div>