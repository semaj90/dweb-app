<script lang="ts">
  interface Props {
    evidence: Evidence | null ;
    data: any
  }
  let {
    evidence = null,
    data = null
  } = $props();



  import { invalidateAll } from "$app/navigation";
  import { createEventDispatcher } from "svelte";
  import { superForm } from "sveltekit-superforms";
// Corrected UI component import paths
  import Textarea from "$lib/components/ui/Textarea.svelte";
  import Button from "$lib/components/ui/button";
  import { Input } from "$lib/components/ui/input";
  import { Label } from "$lib/components/ui/label";
  import type { Evidence } from "$lib/types/api";
  
  
   // Evidence | null = null;
  let { data = $bindable() } = $props(); // any = null;

  const dispatch = createEventDispatcher();

  // Melt UI builder for the Select component with proper typing
  const {
    elements: { trigger, menu, option },
    states: { selectedLabel },
  } = createSelect({
    defaultValue: evidence?.type || undefined,
  });

  const { form, enhance, errors, submitting } = superForm(
    evidence || data.form,
    {
      onUpdated: async ({ form }) => {
        if (form.valid) {
          await invalidateAll();
          dispatch("success");
}
      },
}
  );
</script>

<form method="POST" use:enhance class="space-y-4">
  {#if evidence}
    <input type="hidden" name="id" bind:value={$form.id} />
  {/if}

  {#if data?.form?.message}
    <div class="space-y-4">{data.form.message}</div>
  {/if}

  <div>
    <Label for_="title">Title</Label>
    <Input id="title" name="title" bind:value={$form.title} required />
    {#if $errors.title}<span class="space-y-4">{$errors.title}</span
      >{/if}
  </div>

  <div>
    <Label for_="description">Description</Label>
    <Textarea id="description" name="description" bind:value={$form.description}></Textarea>
  </div>

  <div>
    <Label>Type</Label>
    <button
      use:melt={$trigger}
      aria-label="Select evidence type"
      class="space-y-4"
    >
      <span>{$selectedLabel || "Select a type"}</span>
    </button>
    <div use:melt={$menu} class="space-y-4">
      <div use:melt={$option({ value: "document", label: "Document" })}>
        Document
      </div>
      <div use:melt={$option({ value: "image", label: "Image" })}>Image</div>
      <div use:melt={$option({ value: "video", label: "Video" })}>Video</div>
      <div use:melt={$option({ value: "audio", label: "Audio" })}>Audio</div>
      <div use:melt={$option({ value: "other", label: "Other" })}>Other</div>
    </div>
    <input type="hidden" name="type" bind:value={$form.type} />
    {#if $errors.type}<span class="space-y-4">{$errors.type}</span
      >{/if}
  </div>

  <div>
    <Label for_="url">URL</Label>
    <Input
      id="url"
      name="url"
      bind:value={$form.url}
      placeholder="https://example.com/evidence"
    />
    {#if $errors.url}<span class="space-y-4">{$errors.url}</span>{/if}
  </div>

  <div>
    <Label for_="tags">Tags (comma-separated)</Label>
    <Input id="tags" name="tags" bind:value={$form.tags} placeholder="tag1, tag2, tag3" />
    {#if $errors.tags}<span class="space-y-4">{$errors.tags}</span
      >{/if}
  </div>

  <div class="space-y-4">
    <Button type="button" variant="ghost" onclick={() => dispatch("cancel")}
      >Cancel</Button
    >
    <Button type="submit" disabled={$submitting}>
      {#if $submitting}
        Saving...
      {:else}
        {evidence ? "Save Changes" : "Create Evidence"}
      {/if}
    </Button>
  </div>
</form>

<style>
  /* @unocss-include */
  form {
    max-width: 500px;
    margin: 0 auto;
}
  .select-trigger {
    display: inline-flex;
    align-items: center
    padding: 0.5rem 1rem;
    border: 1px solid #ccc;
    border-radius: 6px;
    background: #f9fafb;
    cursor: pointer
    font-size: 1rem;
    min-width: 160px;
    transition: box-shadow 0.2s;
}
  .select-trigger:focus {
    outline: none
    box-shadow: 0 0 0 2px #6366f1;
}
  .select-menu {
    position: absolute
    z-index: 50;
    background: white
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
    margin-top: 0.5rem;
    min-width: 160px;
    padding: 0.5rem 0;
}
  .select-menu > div {
    padding: 0.5rem 1rem;
    cursor: pointer
    transition: background 0.2s;
}
  .select-menu > div:hover {
    background: #f3f4f6;
}
</style>
