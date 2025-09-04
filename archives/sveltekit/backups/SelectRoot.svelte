<script lang="ts">
  import { createSelect, melt } from '@melt-ui/svelte';
  import { setContext } from 'svelte';
  
  export let selected = undefined;
  
  const {
    elements: { trigger, menu, option },
    states: { selectedLabel, open },
    helpers: { isSelected }
  } = createSelect({
    defaultSelected: selected
  });
  
  // Provide context for child components
  setContext('select', {
    trigger,
    menu,
    option,
    selectedLabel,
    open,
    isSelected,
    select: (value: unknown) => {
      selected = value;
    }
  });
  
  export { trigger, menu, option, selectedLabel, open, isSelected };
</script>

<slot {trigger} {menu} {option} {selectedLabel} {open} {isSelected} />
