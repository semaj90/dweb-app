<script lang="ts">
  import { Dialog as BitsDialog } from 'bits-ui';
  import { fade, fly } from 'svelte/transition';
  import { cn } from '$lib/utils/cn';

  interface DialogProps {
    /** Whether the dialog is open */
    open?: boolean;
    /** Callback when open state changes */
    onOpenChange?: (open: boolean) => void;
    /** Dialog size */
    size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
    /** Legal context styling */
    legal?: boolean;
    /** Evidence analysis specific styling */
    evidenceAnalysis?: boolean;
    /** Case management styling */
    caseManagement?: boolean;
    /** Modal vs non-modal behavior */
    modal?: boolean;
    /** Custom overlay class */
    overlayClass?: string;
    /** Custom content class */
    contentClass?: string;
    /** Children content */
    children?: import('svelte').Snippet;
    /** Content snippet for dialog content */
    content?: import('svelte').Snippet;
  }

  let {
    open = $bindable(false),
    onOpenChange,
    size = 'md',
    legal = false,
    evidenceAnalysis = false,
    caseManagement = false,
    modal = true,
    overlayClass = '',
    contentClass = '',
    children,
    content
  } = $props();

  // Reactive size classes using $derived
  const sizeClasses = $derived({
    'max-w-md': size === 'sm',
    'max-w-lg': size === 'md',
    'max-w-2xl': size === 'lg',
    'max-w-4xl': size === 'xl',
    'max-w-[95vw] max-h-[95vh]': size === 'full'
  }[size] ? {
    'max-w-md': size === 'sm',
    'max-w-lg': size === 'md',
    'max-w-2xl': size === 'lg',
    'max-w-4xl': size === 'xl',
    'max-w-[95vw] max-h-[95vh]': size === 'full'
  } : 'max-w-lg')

  // Reactive content classes using $derived
  const dialogContentClasses = $derived(cn(
    'bits-dialog-content',
    {
      'max-w-md': size === 'sm',
      'max-w-lg': size === 'md',
      'max-w-2xl': size === 'lg',
      'max-w-4xl': size === 'xl',
      'max-w-[95vw] max-h-[95vh]': size === 'full',
      'nier-bits-dialog': legal,
      'yorha-panel border-2 border-nier-border-primary': evidenceAnalysis,
      'yorha-card-elevated shadow-2xl': caseManagement,
      'font-gothic': legal
    },
    contentClass));

  // Reactive overlay classes using $derived
  const overlayClasses = $derived(cn(
    'bits-dialog-overlay',
    {
      'backdrop-blur-md': legal,
      'bg-nier-overlay': evidenceAnalysis || caseManagement
    },
    overlayClass));

  // Handle open change
  function handleOpenChange(newOpen: boolean) {
    open = newOpen;
    onOpenChange?.(newOpen);
  }
</script>

<BitsDialog.Root {open} onOpenChange={handleOpenChange}>
  {@render children?.()}

  <!-- Portal rendering for dialog content -->
  <BitsDialog.Portal>
    <BitsDialog.Overlay
      class={overlayClasses}
    />
    <BitsDialog.Content
      class={dialogContentClasses}
    >
      {@render content?.()}
    </BitsDialog.Content>
  </BitsDialog.Portal>
</BitsDialog.Root>

<!-- Export sub-components for easy use -->
<script lang="ts" module>
  export { BitsDialog as Dialog };

  // Re-export commonly used sub-components
  export const DialogTrigger = BitsDialog.Trigger;
  export const DialogPortal = BitsDialog.Portal;
  export const DialogOverlay = BitsDialog.Overlay;
  export const DialogContent = BitsDialog.Content;
  export const DialogTitle = BitsDialog.Title;
  export const DialogDescription = BitsDialog.Description;
  export const DialogClose = BitsDialog.Close;

  // Create custom header and footer components since they don't exist in newer Bits UI
  export const DialogHeader = 'div';
  export const DialogFooter = 'div';
</script>

<style>
  /* @unocss-include */
  /* Enhanced dialog animations for legal AI context */
  :global(.bits-dialog-overlay) {
    animation: overlay-show 200ms cubic-bezier(0.16, 1, 0.3, 1);
  }

  :global(.bits-dialog-content) {
    animation: content-show 300ms cubic-bezier(0.16, 1, 0.3, 1);
  }

  @keyframes overlay-show {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  @keyframes content-show {
    from {
      opacity: 0;
      transform: translate(-50%, -48%) scale(0.96);
    }
    to {
      opacity: 1;
      transform: translate(-50%, -50%) scale(1);
    }
  }

  /* Legal AI specific styling */
  :global(.nier-bits-dialog) {
    background: linear-gradient(
      135deg,
      var(--color-nier-bg-primary) 0%,
      var(--color-nier-bg-secondary) 100%
    );
    border: 2px solid var(--color-nier-border-primary);
  }

  :global(.nier-bits-dialog::before) {
    content: '';
    position: absolute
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(
      90deg,
      var(--color-nier-accent-warm),
      var(--color-nier-accent-cool),
      var(--color-nier-accent-warm)
    );
  }

  /* Evidence analysis specific styling */
  :global([data-evidence-analysis] .bits-dialog-content) {
    background-image:
      linear-gradient(45deg, transparent 25%, rgba(0,0,0,0.02) 25%),
      linear-gradient(-45deg, transparent 25%, rgba(0,0,0,0.02) 25%),
      linear-gradient(45deg, rgba(0,0,0,0.02) 75%, transparent 75%),
      linear-gradient(-45deg, rgba(0,0,0,0.02) 75%, transparent 75%);
    background-size: 20px 20px;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
  }

  /* Case management specific styling */
  :global([data-case-management] .bits-dialog-content) {
    box-shadow:
      0 20px 25px -5px rgba(0, 0, 0, 0.1),
      0 10px 10px -5px rgba(0, 0, 0, 0.04),
      inset 0 1px 0 rgba(255, 255, 255, 0.1);
  }

  /* Enhanced focus and accessibility */
  :global(.bits-dialog-content:focus) {
    outline: 2px solid var(--color-nier-border-primary);
    outline-offset: 2px;
  }

  /* Responsive adjustments */
  @media (max-width: 640px) {
    :global(.bits-dialog-content) {
      margin: 1rem;
      max-width: calc(100vw - 2rem);
      max-height: calc(100vh - 2rem);
    }
  }
</style>