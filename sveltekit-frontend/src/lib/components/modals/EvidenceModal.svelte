<script lang="ts">
  import { Dialog, Button, Input, Tag } from 'bits-ui';
  import { superValidate } from 'sveltekit-superforms/client';
  import { evidenceSchema } from '$lib/server/schemas';
  import { createMachine, assign } from 'xstate';
  export let item;
  export let open = false;
  let form = superValidate(evidenceSchema, { initialValues: item });
  // Helper to handle tags as a comma separated string
  if (!form.values.jsonData.tagsString) {
    form.values.jsonData.tagsString = (form.values.jsonData.tags ?? []).join(', ');
  }

  // XState machine for tag/type grouping
  const evidenceMachine = createMachine({
    id: 'evidence',
    initial: 'view',
    context: { item },
    states: {
      view: { on: { EDIT: 'edit' } },
      edit: { on: { SAVE: 'view', CANCEL: 'view' } }
    }
  function handleSave() {
    // Convert tagsString to array before saving
    form.values.jsonData.tags = form.values.jsonData.tagsString
      ? form.values.jsonData.tagsString.split(',').map(t => t.trim()).filter(Boolean)
      : [];
    state = evidenceMachine.transition(state, 'SAVE');
    open = false;
  }
  let state = evidenceMachine.initialState;

  function handleEdit() { state = evidenceMachine.transition(state, 'EDIT'); }
  function handleSave() { state = evidenceMachine.transition(state, 'SAVE'); open = false; }
  function handleCancel() { state = evidenceMachine.transition(state, 'CANCEL'); open = false; }
</script>

<Dialog.Root bind:open>
  <Dialog.Content class="uno-p-4 uno-bg-white uno-shadow">
    <Dialog.Header>
      <Dialog.Title>Evidence Details</Dialog.Title>
    </Dialog.Header>
    {#if state.value === 'view'}
      <div class="mb-2">
        <div class="font-bold">{item.jsonData.title}</div>
        <div class="text-sm text-gray-600">{item.jsonData.description}</div>
      <form use:form class="flex flex-col gap-2" on:submit|preventDefault={handleSave}>
        <Input name="jsonData.title" bind:value={form.values.jsonData.title} placeholder="Title" />
        <Input name="jsonData.description" bind:value={form.values.jsonData.description} placeholder="Description" />
        <Input name="jsonData.tags" bind:value={form.values.jsonData.tagsString} placeholder="Tags (comma separated)" />
        <Input name="jsonData.type" bind:value={form.values.jsonData.type} placeholder="Type" />
        <div class="flex gap-2 mt-2">
          <Button type="submit" class="uno-bg-green-600 uno-text-white uno-px-3 uno-py-1 uno-rounded">Save</Button>
          <Button variant="outline" on:click={handleCancel}>Cancel</Button>
        </div>
      </form>
    {:else}
      <form use:form class="flex flex-col gap-2">
        <Input name="jsonData.title" bind:value={form.values.jsonData.title} placeholder="Title" />
        <Input name="jsonData.description" bind:value={form.values.jsonData.description} placeholder="Description" />
        <Input name="jsonData.tags" bind:value={form.values.jsonData.tags} placeholder="Tags (comma separated)" />
        <Input name="jsonData.type" bind:value={form.values.jsonData.type} placeholder="Type" />
        <div class="flex gap-2 mt-2">
          <Button type="submit" on:click={handleSave} class="uno-bg-green-600 uno-text-white uno-px-3 uno-py-1 uno-rounded">Save</Button>
          <Button variant="outline" on:click={handleCancel}>Cancel</Button>
        </div>
      </form>
    {/if}
    <Dialog.Footer>
      <Button on:click={() => (open = false)} variant="ghost">Close</Button>
    </Dialog.Footer>
  </Dialog.Content>
</Dialog.Root>

<style>
  /* @unocss-include */
  .uno-shadow {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
</style>