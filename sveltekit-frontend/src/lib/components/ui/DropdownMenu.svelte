<!-- DropdownMenu component combining dropdown-menu parts -->
<script lang="ts">
  import DropdownMenuRoot from './dropdown-menu/DropdownMenuRoot.svelte';
  import DropdownMenuTrigger from './dropdown-menu/DropdownMenuTrigger.svelte';
  import DropdownMenuContent from './dropdown-menu/DropdownMenuContent.svelte';
  import DropdownMenuItem from './dropdown-menu/DropdownMenuItem.svelte';
  import DropdownMenuSeparator from './dropdown-menu/DropdownMenuSeparator.svelte';
  
  // Component props
  let { 
    items = [],
    trigger = 'Menu',
    ...props 
  } = $props();
</script>

<DropdownMenuRoot {...props}>
  <DropdownMenuTrigger>{trigger}</DropdownMenuTrigger>
  
  <DropdownMenuContent>
    {#each items as item, index}
      {#if item.separator}
        <DropdownMenuSeparator />
      {:else}
        <DropdownMenuItem 
          value={item.value}
          disabled={item.disabled || false}
          on:on:click={() => item.onClick?.(item.value)}
        >
          {item.label}
        </DropdownMenuItem>
      {/if}
    {/each}
  </DropdownMenuContent>
</DropdownMenuRoot>