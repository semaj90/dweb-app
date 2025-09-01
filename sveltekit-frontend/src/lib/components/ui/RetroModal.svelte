<!--
  RetroModal Component - bits-ui + nes.css integration
  Demonstrates the perfect marriage of headless functionality with retro styling
-->
<script lang="ts">
  import { Dialog } from "bits-ui";
  
  interface Props {
    open?: boolean;
    title?: string;
    onClose?: () => void;
  }
  
  let { 
    open = false, 
    title = "Dialog", 
    onClose 
  }: Props = $props();
  
  function handleOpenChange(isOpen: boolean) {
    if (!isOpen && onClose) {
      onClose();
    }
  }
</script>

<!-- bits-ui provides the functionality, nes.css provides the styling -->
<Dialog.Root {open} onOpenChange={handleOpenChange}>
  
  <!-- Trigger button with retro styling -->
  <Dialog.Trigger class="nes-btn is-primary">
    <slot name="trigger">Open Modal</slot>
  </Dialog.Trigger>
  
  <!-- Portal for proper z-index layering -->
  <Dialog.Portal>
    
    <!-- Overlay with retro dark background -->
    <Dialog.Overlay class="fixed inset-0 bg-black/50 z-50" />
    
    <!-- Modal content with full NES.css styling -->
    <Dialog.Content class="fixed left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50">
      <div class="nes-dialog is-rounded" style="min-width: 400px; max-width: 600px;">
        
        <!-- Header with close button -->
        <div class="flex justify-between items-center mb-4">
          <Dialog.Title class="nes-text is-primary font-bold text-lg">
            {title}
          </Dialog.Title>
          <Dialog.Close class="nes-btn is-error" style="padding: 4px 8px;">
            ×
          </Dialog.Close>
        </div>
        
        <!-- Modal body content -->
        <div class="modal-content">
          <slot>
            <p class="nes-text">Modal content goes here...</p>
          </slot>
        </div>
        
        <!-- Optional footer -->
        {#if $$slots.footer}
          <div class="modal-footer mt-4 pt-4 border-t-2 border-gray-300">
            {@render footer?.()}
          </div>
        {/if}
        
      </div>
    </Dialog.Content>
    
  </Dialog.Portal>
</Dialog.Root>

<style>
  /* Additional custom styling to enhance nes.css */
  :global(.nes-dialog) {
    animation: modalSlideIn 0.3s ease-out;
  }
  
  @keyframes modalSlideIn {
    from {
      opacity: 0;
      transform: translateY(-20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  /* Custom modal content styling */
  .modal-content {
    max-height: 400px;
    overflow-y: auto;
  }
  
  .modal-content :global(.nes-field) {
    margin-bottom: 1rem;
  }
</style>