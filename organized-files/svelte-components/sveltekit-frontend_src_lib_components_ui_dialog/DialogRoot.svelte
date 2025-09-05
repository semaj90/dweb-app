<script lang="ts">
  interface Props {
    open?: boolean;
    onOpenChange?: ((open: boolean) => void) | undefined;
  }
  let {
    open = false,
    onOpenChange = undefined,
    children
  }: Props & { children?: any } = $props();



  
  import { writable } from 'svelte/store';


  const openWritable = writable(open);

  // Keep the writable in sync with the prop
  $effect(() => { openWritable.set(open); });

  const {
    elements: { trigger, overlay, content, title, description, close },
    states: { open: openState }
  } = createDialog({
    open: openWritable,
    onOpenChange: ({ next }) => {
      open = next;
      onOpenChange?.(next);
      return next;
}
  });

</script>

{#if children}
  {@render children({ trigger, overlay, content, title, description, close, openState })}
{/if}
