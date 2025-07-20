<script lang="ts">
  import { uploadActions, uploadModal } from "$lib/stores/evidence-store";

  $: isOpen = $uploadModal.isOpen;

  function closeModal() {
    uploadActions.closeModal();}
</script>

{#if isOpen}
  <div
    class="container mx-auto px-4"
  >
    <div class="container mx-auto px-4">
      <h2 class="container mx-auto px-4">Upload Evidence</h2>
      <p class="container mx-auto px-4">Modal is working!</p>
      <button
        class="container mx-auto px-4"
        on:click={() => closeModal()}
      >
        Close
      </button>
    </div>
  </div>
{/if}
