<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { createContextMenu } from '@melt-ui/svelte';
  interface Props extends CommonProps {
    children?: import('svelte').Snippet<[any]>;
  }

  let { children }: Props = $props();
  
  const { elements, states } = createContextMenu();
  const { menu, trigger } = elements;
  
  export { menu, trigger, states };
</script>

{@render children?.({ menu, trigger, states, })}
