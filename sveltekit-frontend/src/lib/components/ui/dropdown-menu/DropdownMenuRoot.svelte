<script lang="ts">
  import { createDropdownMenu } from '@melt-ui/svelte';
  
  const { elements, states } = createDropdownMenu();
  const { menu, trigger } = elements;
  
  export { menu, trigger, states };
</script>

<slot {menu} {trigger} {states} />
