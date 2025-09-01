<!-- @migration-task Error while migrating Svelte code: Unexpected token
https://svelte.dev/e/js_parse_error -->
<script lang="ts">
  import { $props, $effect } from 'svelte';

  interface Props {
    open?: boolean;
    onOpenChange?: ((open: boolean) => void) | undefined;
  }
  let {
    open = false,
    onOpenChange = undefined,
    children
  }: Props & { children?: unknown } = $props();



  // import { createDialog } from 'melt'; // Removed melt dependency
  import { writable } from 'svelte/store';


  const openWritable = writable(open);

  // Keep the writable in sync with the prop
  $effect(() => {
    openWritable.set(open);
  });

  // const {
  //   elements: { trigger, overlay, content, title, description, close },
  //   states: { open: openState }
  // } = createDialog({
  //   open: openWritable,
  //   onOpenChange: ({ next }) => {
  //     open = next;
  //     onOpenChange?.(next);
  //     return next;
  //   }
  // });
  
  // Mock objects for now - replace with bits-ui or plain HTML
  const trigger = {};
  const overlay = {};
  const content = {};
  const title = {};
  const description = {};
  const close = {};
  const openState = false;

</script>

{#if children}
  {@render children({ trigger, overlay, content, title, description, close, openState })}
{/if}
