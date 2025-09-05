<script lang="ts">
  
  
  const { elements, states } = createContextMenu();
  const { menu, trigger } = elements;
  
  export { menu, trigger, states };
</script>

<slot {menu} {trigger} {states} />
