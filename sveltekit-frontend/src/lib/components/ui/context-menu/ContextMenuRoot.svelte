<script lang="ts">
  import { createContextMenu } from '@melt-ui/svelte';
  
  const { elements, states } = createContextMenu();
  const { menu, trigger } = elements;
  
  export { menu, trigger, states };
</script>

<slot {menu} {trigger} {states} />
