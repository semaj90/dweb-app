<!-- Dialog Wrapper: Svelte 5, Bits UI, UnoCSS, analytics logging -->
<script lang="ts">
  

  interface Props {
    open?: boolean;
    title?: string;
    description?: string;
    analyticsLog?: (event: any) => void;
    onClose?: () => void;
  }

  let {
    open = $bindable(false),
    title = '',
    description = '',
    analyticsLog = () => {},
    onClose = () => {}
  } = $props();

  const {
    elements: { trigger, overlay, content, title: titleEl, description: descEl, close },
    states: { open: dialogOpen }
  } = createDialog({ open, onOpenChange: (v) => { open = v; if (!v) onClose(); analyticsLog({ event: 'dialog_closed', timestamp: Date.now() }); } });

  $effect(() => {
    if (open) analyticsLog({ event: 'dialog_opened', title, timestamp: Date.now() });
  });
</script>

{#if $dialogOpen}
  <div use:overlay class="modal-overlay" />
  <div use:content class="modal-content">
    {#if title}
      <h2 use:titleEl class="modal-title">{title}</h2>
    {/if}
    {#if description}
      <p use:descEl class="modal-description">{description}</p>
    {/if}
    <div class="modal-body">
      <slot></slot>
    </div>
    <button use:close class="modal-close">×</button>
  </div>
{/if}

<style>
  .modal-overlay {
    @apply fixed inset-0 bg-black/60 z-40;
  }
  .modal-content {
    @apply fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2;
    @apply bg-nier-surface border border-nier-border rounded-lg p-6 shadow-2xl;
  }
  .modal-title {
    @apply text-xl font-bold text-nier-accent mb-2;
  }
  .modal-description {
    @apply text-nier-text-muted mb-4;
  }
  .modal-close {
    @apply absolute top-4 right-4 w-8 h-8 rounded-full bg-nier-surface-light;
    @apply hover:bg-nier-surface-lighter transition-colors;
    @apply flex items-center justify-center text-nier-text-muted hover:text-nier-white;
  }
</style>
