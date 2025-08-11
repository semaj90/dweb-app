<script lang="ts">
  import { Dialog } from "bits-ui";
  import { X } from "lucide-svelte";
  import { fade, fly } from "svelte/transition";
  
  interface Props {
    open?: boolean;
    title?: string;
    description?: string;
    size?: 'sm' | 'md' | 'lg' | 'xl';
    closeOnOutsideClick?: boolean;
    closeOnEscape?: boolean;
    showCloseButton?: boolean;
    class?: string;
    onopen?: () => void;
    onclose?: () => void;
    children?: any;
    footer?: any;
  }
  
  let {
    open = $bindable(false),
    title = "",
    description = "",
    size = "md",
    closeOnOutsideClick = true,
    closeOnEscape = true,
    showCloseButton = true,
    class: className = "",
    onopen,
    onclose,
    children,
    footer
  }: Props = $props();
  
  // Size mappings
  const sizeClasses = {
    sm: "max-w-sm",
    md: "max-w-md", 
    lg: "max-w-lg",
    xl: "max-w-xl"
  };
  
  function handleOpenChange(isOpen: boolean) {
    if (isOpen && !open) {
      onopen?.();
    } else if (!isOpen && open) {
      onclose?.();
    }
    open = isOpen;
  }
</script>

<Dialog.Root bind:open onOpenChange={handleOpenChange} {closeOnEscape} {closeOnOutsideClick}>
  <Dialog.Trigger>
    <slot name="trigger" />
  </Dialog.Trigger>
  
  <Dialog.Portal>
    <Dialog.Overlay 
      class="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm"
      transition={fade}
      transitionConfig={{ duration: 150 }}
    />
    <Dialog.Content 
      class="fixed left-1/2 top-1/2 z-50 w-full {sizeClasses[size]} -translate-x-1/2 -translate-y-1/2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-6 shadow-xl {className}"
      transition={fly}
      transitionConfig={{ duration: 200, y: -10 }}
    >
      {#if title || showCloseButton}
        <Dialog.Header class="flex items-center justify-between pb-4 border-b border-gray-200 dark:border-gray-700">
          <div>
            {#if title}
              <Dialog.Title class="text-lg font-semibold text-gray-900 dark:text-white">
                {title}
              </Dialog.Title>
            {/if}
            {#if description}
              <Dialog.Description class="text-sm text-gray-600 dark:text-gray-400 mt-1">
                {description}
              </Dialog.Description>
            {/if}
          </div>
          
          {#if showCloseButton}
            <Dialog.Close class="rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none">
              <X class="h-4 w-4" />
              <span class="sr-only">Close</span>
            </Dialog.Close>
          {/if}
        </Dialog.Header>
      {/if}
      
      <div class="py-4">
        {#if children}
          {@render children()}
        {:else}
          <slot />
        {/if}
      </div>
      
      {#if footer}
        <Dialog.Footer class="flex items-center justify-end gap-2 pt-4 border-t border-gray-200 dark:border-gray-700">
          {@render footer()}
        </Dialog.Footer>
      {:else}
        <slot name="footer" />
      {/if}
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>