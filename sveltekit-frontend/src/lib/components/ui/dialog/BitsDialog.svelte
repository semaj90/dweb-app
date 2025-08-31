<!-- Modern Bits-UI Dialog Component for Legal AI App -->
<script lang="ts">
  import { Dialog } from 'bits-ui';
  import { X } from 'lucide-svelte';
  import { cn } from '$lib/utils';

  interface Props {
    open?: boolean;
    title?: string;
    description?: string;
    size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
    showClose?: boolean;
    onOpenChange?: (open: boolean) => void;
    class?: string;
    children?: any;
    trigger?: any;
    footer?: any;
  }

  let {
    open = $bindable(false),
    title = '',
    description = '',
    size = 'md',
    showClose = true,
    onOpenChange,
    class: className = '',
    children,
    trigger,
    footer
  }: Props = $props();

  const sizeClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
    full: 'max-w-[95vw] max-h-[95vh]'
  };

  function handleOpenChange(newOpen: boolean) {
    open = newOpen;
    onOpenChange?.(newOpen);
  }
</script>

<Dialog.Root bind:open onOpenChange={handleOpenChange}>
  <!-- Trigger (optional) -->
  {#if trigger}
    <Dialog.Trigger class="dialog-trigger">
      {@render trigger()}
    </Dialog.Trigger>
  {/if}

  <!-- Dialog Portal and Overlay -->
  <Dialog.Portal>
    <Dialog.Overlay 
      class="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0"
    />
    
    <!-- Dialog Content -->
    <Dialog.Content
      className={cn(
        'fixed left-1/2 top-1/2 z-50 grid w-full translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] sm:rounded-lg',
        sizeClasses[size],
        'legal-dialog', // Legal AI app specific styling
        className
      )}
    >
      <!-- Header -->
      {#if title || showClose}
        <div class="flex flex-col space-y-1.5 text-center sm:text-left">
          {#if title}
            <Dialog.Title class="text-lg font-semibold leading-none tracking-tight legal-dialog-title">
              {title}
            </Dialog.Title>
          {/if}
          
          {#if description}
            <Dialog.Description class="text-sm text-muted-foreground legal-dialog-description">
              {description}
            </Dialog.Description>
          {/if}
        </div>
      {/if}

      <!-- Close Button -->
      {#if showClose}
        <Dialog.Close class="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground">
          <X class="h-4 w-4" />
          <span class="sr-only">Close</span>
        </Dialog.Close>
      {/if}

      <!-- Main Content -->
      <div class="legal-dialog-content">
        {@render children?.()}
      </div>

      <!-- Footer -->
      {#if footer}
        <div class="flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2 legal-dialog-footer">
          {@render footer()}
        </div>
      {/if}
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>

<style>
  /* Legal AI App Specific Styling */
  :global(.legal-dialog) {
    @apply border-yorha-border bg-yorha-bg-secondary;
  }

  :global(.legal-dialog-title) {
    @apply text-yorha-text-primary font-mono;
  }

  :global(.legal-dialog-description) {
    @apply text-yorha-text-secondary font-mono text-xs;
  }

  :global(.legal-dialog-content) {
    @apply text-yorha-text-primary;
  }

  :global(.legal-dialog-footer) {
    @apply border-t border-yorha-border pt-4;
  }
</style>