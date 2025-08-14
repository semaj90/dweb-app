<!-- YoRHa Dialog Manager Component -->
<script lang="ts">
  import YoRHaDialog from './YoRHaDialog.svelte';
  import { dialogStore, type Dialog } from '$lib/stores/dialogs';

  // Subscribe to dialog store
  let dialogs = $state<Dialog[]>([]);
  
  $effect(() => {
    const unsubscribe = dialogStore.subscribe((value) => {
      dialogs = value;
    });
    
    return unsubscribe;
  });

  function handleDialogClose(dialog: Dialog) {
    dialogStore.remove(dialog.id);
  }

  function handleDialogConfirm(dialog: Dialog, event?: CustomEvent) {
    const result = event?.detail || true;
    dialogStore.remove(dialog.id, result);
  }

  function handleDialogCancel(dialog: Dialog) {
    dialogStore.reject(dialog.id, 'cancelled');
  }
</script>

<!-- Render active dialogs -->
{#each dialogs as dialog (dialog.id)}
  <YoRHaDialog
    open={true}
    type={dialog.type}
    title={dialog.title}
    message={dialog.message}
    position={dialog.position}
    persistent={dialog.persistent}
    value={dialog.value}
    onclose={() => handleDialogClose(dialog)}
    onconfirm={(event) => handleDialogConfirm(dialog, event)}
    oncancel={() => handleDialogCancel(dialog)}
  />
{/each}