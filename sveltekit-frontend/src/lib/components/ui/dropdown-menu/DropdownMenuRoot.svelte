<script lang="ts">

  import { createDropdownMenu } from 'melt';
  import { $props } from 'svelte';
  interface Props {
    children?: import('svelte').Snippet<[any]>;
  }

  let { children }: Props = $props();
  
  const { elements, states } = createDropdownMenu();
  const { menu, trigger } = elements;
  
  export { menu, trigger, states };
</script>

{@render children?.({ menu, trigger, states, })}
