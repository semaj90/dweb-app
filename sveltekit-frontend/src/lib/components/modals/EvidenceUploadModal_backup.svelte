<!-- @migration-task Error while migrating Svelte code: Unexpected token
https://svelte.dev/e/js_parse_error -->
<script lang="ts">
  import { $derived } from 'svelte';

  import { uploadActions, uploadModal } from "$lib/stores/evidence-store";

  let isOpen = $derived($uploadModal.isOpen);

  function closeModal() {
    uploadActions.closeModal();
}
</script>

{#if isOpen}
  <div
    class="space-y-4"
  >
    <div class="space-y-4">
      <h2 class="space-y-4">Upload Evidence</h2>
      <p class="space-y-4">Modal is working!</p>
      <button
        class="space-y-4"
        on:onclick={() => closeModal()}
      >
        Close
      </button>
    </div>
  </div>
{/if}


