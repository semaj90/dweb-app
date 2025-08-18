<script lang="ts">
  import type { HTMLAttributes } from 'svelte/elements';
  
  interface Props extends HTMLAttributes<HTMLParagraphElement> {
    className?: string;
  }
  
  export let className: string = '';
</script>

<p
  class="text-sm text-muted-foreground {className}"
>
  <slot />
</p>