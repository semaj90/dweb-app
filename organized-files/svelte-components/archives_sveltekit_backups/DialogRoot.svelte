<script lang="ts">
  import { createDialog } from '@melt-ui/svelte';
  import { writable } from 'svelte/store';
  
  export let open: boolean = false;
  export let onOpenChange: ((open: boolean) => void) | undefined = undefined;
  
  const openWritable = writable(open);
  
  // Keep the writable in sync with the prop
  $: openWritable.set(open);
  
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
  
  export { trigger, overlay, content, title, description, close, openState };
</script>

<slot {trigger} {overlay} {content} {title} {description} {close} {openState} />
