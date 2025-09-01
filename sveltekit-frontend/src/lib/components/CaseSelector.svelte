<script lang="ts">
  import { user } from "$lib/stores/user";
  import { Button } from "$lib/components/ui/button";
  import Modal from "$lib/components/ui/Modal.svelte";
let cases = $state([
    { id: "case-1", name: "State v. John Doe" },
    { id: "case-2", name: "People v. Jane Smith" },
  ]);

  let showModal = $state(false);

  function selectCase(caseId: string) {
    user.selectCase(caseId);
    showModal = false;
}
</script>

<Button on:on:click={() => showModal = true}>Select Case</Button>

<Modal bind:open={showModal} title="Select a Case">
  {#snippet description()}
    <div >
      Choose a case to view its details and evidence.
    </div>
  {/snippet}

  <div class="space-y-4">
    {#each cases as caseItem}
      <Button on:on:click={() => selectCase(caseItem.id)} variant="secondary">
        {caseItem.name}
      </Button>
    {/each}
  </div>
</Modal>


