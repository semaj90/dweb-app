<script lang="ts">
  import { melt } from 'melt';

  interface Props {
    item?: any;
    children?: import('svelte').Snippet;
    class?: string;
    href?: string;
    disabled?: boolean;
  }
  
  let { 
    children, 
    item,
    class: className = '',
    href,
    disabled = false
  }: Props = $props();
</script>

{#if href}
  <a 
    {href}
    <!-- use:melt={item}
    class="relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 {className}"
    data-disabled={disabled ? '' : undefined}
  >
    {@render children?.()}
  </a>
{:else}
  <button 
    <!-- use:melt={item}
    class="relative flex w-full cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 {className}"
    {disabled}
  >
    {@render children?.()}
  </button>
{/if}
