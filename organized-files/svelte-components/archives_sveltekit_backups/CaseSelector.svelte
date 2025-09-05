<script>
  import { user } from "$lib/stores/user";
  import { Button } from "$lib/components/ui/button";
  import Modal from "$lib/components/ui/Modal.svelte";

  let cases = [
    { id: "case-1", name: "State v. John Doe" },
    { id: "case-2", name: "People v. Jane Smith" },
  ];

  let showModal = false;

  function selectCase(caseId) {
    user.selectCase(caseId);
    showModal = false;
  }
</script>

<Button onclick={() => showModal = true}>Select Case</Button>

<Modal bind:open={showModal} title="Select a Case">
  <div slot="description">
    Choose a case to view its details and evidence.
  </div>
  
  <div class="mx-auto px-4 max-w-7xl">
    {#each cases as caseItem}
      <Button onclick={() => selectCase(caseItem.id)} variant="secondary">
        {caseItem.name}
      </Button>
    {/each}
  </div>
</Modal>
