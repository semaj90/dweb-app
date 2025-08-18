<script lang="ts">
  import type { HTMLAttributes } from 'svelte/elements';
  
  interface Props extends HTMLAttributes<HTMLLabelElement> {
    className?: string;
    for?: string;
  }
  
  export let className: string = '';
  export let htmlFor: string | undefined = undefined;
</script>

<label
  for={htmlFor}
  class="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 {className}"
>
  <slot />
</label>