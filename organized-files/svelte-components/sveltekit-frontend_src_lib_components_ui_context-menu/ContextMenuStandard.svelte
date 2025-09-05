<script lang="ts">
  import { ContextMenu } from "bits-ui";
  import type { Snippet } from "svelte";
  import type { WithoutChild } from "bits-ui";

  interface Props extends ContextMenu.RootProps {
    trigger: Snippet
    items: Array<{
      type: 'item' | 'checkbox' | 'radio' | 'separator' | 'sub';
      label?: string;
      value?: string;
      checked?: boolean;
      disabled?: boolean;
      onSelect?: () => void;
      items?: Props['items']; // For sub-menus
    }>;
    contentProps?: WithoutChild<ContextMenu.ContentProps>;
  }

  let {
    open = $bindable(false),
    trigger,
    items,
    contentProps,
    children,
    ...restProps
  } = $props();
</script>

<ContextMenu.Root bind:open {...restProps}>
  <ContextMenu.Trigger>
    {@render trigger()}
  </ContextMenu.Trigger>
  
  <ContextMenu.Portal>
    <ContextMenu.Content {...contentProps}>
      {#each items as item}
        {#if item.type === 'separator'}
          <ContextMenu.Separator />
        {:else if item.type === 'checkbox'}
          <ContextMenu.CheckboxItem 
            value={item.value} 
            disabled={item.disabled}
            onSelect={item.onSelect}
          >
            {#snippet children({ checked })}
              {#if checked}✓{/if}
              {item.label}
            {/snippet}
          </ContextMenu.CheckboxItem>
        {:else if item.type === 'radio'}
          <ContextMenu.RadioItem 
            value={item.value} 
            disabled={item.disabled}
            onSelect={item.onSelect}
          >
            {#snippet children({ checked })}
              {#if checked}●{/if}
              {item.label}
            {/snippet}
          </ContextMenu.RadioItem>
        {:else if item.type === 'sub' && item.items}
          <ContextMenu.Sub>
            <ContextMenu.SubTrigger>{item.label}</ContextMenu.SubTrigger>
            <ContextMenu.Portal>
              <ContextMenu.SubContent>
                {#each item.items as subItem}
                  <ContextMenu.Item 
                    textValue={subItem.label}
                    disabled={subItem.disabled}
                    onSelect={subItem.onSelect}
                  >
                    {subItem.label}
                  </ContextMenu.Item>
                {/each}
              </ContextMenu.SubContent>
            </ContextMenu.Portal>
          </ContextMenu.Sub>
        {:else}
          <ContextMenu.Item 
            textValue={item.label}
            disabled={item.disabled}
            onSelect={item.onSelect}
          >
            {item.label}
          </ContextMenu.Item>
        {/if}
      {/each}
      {@render children?.()}
    </ContextMenu.Content>
  </ContextMenu.Portal>
</ContextMenu.Root>