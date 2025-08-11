<script lang="ts">
  import { uploadActions, uploadModal } from "$lib/stores/evidence-store";

  $: isOpen = $uploadModal.isOpen;

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
        on:click={() => closeModal()}
      >
        Close
      </button>
    </div>
  </div>
{/if}
