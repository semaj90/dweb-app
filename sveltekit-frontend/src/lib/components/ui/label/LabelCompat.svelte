<script lang="ts">
  import Label from '../Label.svelte';
  // Accept legacy props and map to base Label's for_/class_
  export let for: string | undefined;
  export let class: string | undefined;
  // Collect rest props after extracting known ones
  const { for: _f, class: _c, ...rest } = $props();
</script>

<Label for_={for} class_={class} {...rest}>
  <slot />
</Label>
