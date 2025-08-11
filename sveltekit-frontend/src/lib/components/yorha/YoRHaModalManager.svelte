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
    on:close={() => handleModalClose(modal)}
    on:confirm={(event) => handleModalConfirm(modal, event)}
    on:cancel={() => handleModalCancel(modal)}
  >
    {#if modal.component}
      <svelte:component this={modal.component} {...modal.props} />
    {/if}
  </YoRHaModal>
{/each}