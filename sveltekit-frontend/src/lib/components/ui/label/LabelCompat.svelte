<script lang="ts">
  import Label from '../Label.svelte';
  import type { HTMLAttributes } from 'svelte/elements';
  
  // Svelte 5 props pattern - Accept legacy props and map to base Label
  interface Props extends HTMLAttributes<HTMLLabelElement> {
    for?: string;
    class?: string;
  }
  
  let { 'for': forProp = '', class: classProp = '', ...rest }: Props = $props();
</script>

<Label for_={forProp} class_={classProp} {...rest}>
  <slot />
</Label>
