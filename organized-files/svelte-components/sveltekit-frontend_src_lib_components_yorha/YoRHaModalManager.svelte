<!-- YoRHa Modal Manager Component -->
<script lang="ts">
  import YoRHaModal from './YoRHaModal.svelte';
  import { modalStore, type Modal } from '$lib/stores/dialogs';

  // Subscribe to modal store
  let modals = $state<Modal[]>([]);
  
  $effect(() => {
    const unsubscribe = modalStore.subscribe((value) => {
      modals = value;
    });
    
    return unsubscribe;
  });

  function handleModalClose(modal: Modal) {
    modalStore.remove(modal.id);
  }

  function handleModalConfirm(modal: Modal, event?: CustomEvent) {
    const result = event?.detail || true;
    modalStore.remove(modal.id, result);
  }

  function handleModalCancel(modal: Modal) {
    modalStore.reject(modal.id, 'cancelled');
  }
</script>

<!-- Render active modals -->
{#each modals as modal (modal.id)}
  <YoRHaModal
    open={true}
    size={modal.size}
    type={modal.type}
    persistent={modal.persistent}
    showHeader={true}
    showFooter={modal.type === 'confirm' || modal.type === 'alert'}
    onclose={() => handleModalClose(modal)}
    onconfirm={(event) => handleModalConfirm(modal, event)}
    oncancel={() => handleModalCancel(modal)}
  >
    {#snippet children()}
      {#if modal.component}
        <!-- Note: Dynamic components deprecated in Svelte 5 - replace with explicit components -->
        <div class="modal-component-placeholder">
          <p>Dynamic component: {modal.component?.name || 'Unknown'}</p>
          {#if modal.props}
            <pre>{JSON.stringify(modal.props, null, 2)}</pre>
          {/if}
        </div>
      {/if}
    {/snippet}
  </YoRHaModal>
{/each}