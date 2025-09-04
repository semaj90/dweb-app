<script lang="ts">
  import { createDialog, melt } from "@melt-ui/svelte";
  import {
    Bookmark,
    BookmarkCheck,
    Calendar,
    Edit3,
    Eye,
    Tag,
    User,
    X,
  } from "lucide-svelte";
  import { marked } from "marked";
  import { writable } from "svelte/store";
  import { fade, fly } from "svelte/transition";
  import {
    removeSavedNote,
    saveNoteForLater,
  } from "../../../lib/stores/saved-notes";
  import RichTextEditor from "./RichTextEditor.svelte";

  export let noteId: string;
  export let title: string = "";
  export let content: string = "";
  export let markdown: string = "";
  export let html: string = "";
  export let contentJson: unknown = null;
  export let noteType: string = "general";
  export let tags: string[] = [];
  export let userId: string = "";
  export let caseId: string | undefined = undefined;
  export let createdAt: Date = new Date();
  export let isOpen = false;
  export let mode: "view" | "edit" = "view";
  export let canEdit = true;
  export let onSave: ((data: unknown) => void) | undefined = undefined;

  let isSaved = false;
  let editedContent = content;
  let editedTitle = title;
  let editedTags = [...tags];
  let newTag = "";

  const {
    elements: { trigger, overlay, content: dialogContent, close },
    states: { open },
  } = createDialog({
    open: writable(isOpen),
    onOpenChange: ({ next }) => {
      isOpen = next;
      return next;
    },
  });

  $: if (isOpen !== $open) {
    open.set(isOpen);
}
  // Parse markdown to HTML for display
  $: displayHtml = html || (markdown ? marked.parse(markdown) : "");

  async function handleSaveForLater() {
    try {
      await saveNoteForLater({
        id: noteId,
        title,
        content,
        markdown,
        html,
        contentJson,
        noteType,
        tags,
        userId,
        caseId,
      });
      isSaved = true;
      setTimeout(() => (isSaved = false), 2000);
    } catch (error) {
      console.error("Failed to save note:", error);
}}
  async function handleRemoveFromSaved() {
    try {
      await removeSavedNote(noteId);
      isSaved = false;
    } catch (error) {
      console.error("Failed to remove note:", error);
}}
  function addTag() {
    if (newTag.trim() && !editedTags.includes(newTag.trim())) {
      editedTags = [...editedTags, newTag.trim()];
      newTag = "";
}}
  function removeTag(tag: string) {
    editedTags = editedTags.filter((t) => t !== tag);
}
  function handleEditorSave(event: CustomEvent) {
    const {
      html: newHtml,
      markdown: newMarkdown,
      json: newJson,
    } = event.detail;

    const updatedNote = {
      id: noteId,
      title: editedTitle,
      content: newMarkdown || newHtml,
      markdown: newMarkdown,
      html: newHtml,
      contentJson: newJson,
      noteType,
      tags: editedTags,
      userId,
      caseId,
    };

    onSave?.(updatedNote);
    mode = "view";

    // Update local data
    title = editedTitle;
    content = newMarkdown || newHtml;
    markdown = newMarkdown;
    html = newHtml;
    contentJson = newJson;
    tags = [...editedTags];
}
  function startEdit() {
    mode = "edit";
    editedContent = content;
    editedTitle = title;
    editedTags = [...tags];
}
  function cancelEdit() {
    mode = "view";
    editedContent = content;
    editedTitle = title;
    editedTags = [...tags];
}
</script>

{#if isOpen}
  <div
    use:melt={$overlay}
    class="mx-auto px-4 max-w-7xl"
    transition:fade={{ duration: 150 "
  >
    <div
      use:melt={$dialogContent}
      class="mx-auto px-4 max-w-7xl"
      transition:fly={{ y: -20, duration: 200 "
    >
      <!-- Header -->
      <div
        class="mx-auto px-4 max-w-7xl"
      >
        <div class="mx-auto px-4 max-w-7xl">
          {#if mode === "edit"}
            <input
              bind:value={editedTitle}
              class="mx-auto px-4 max-w-7xl"
              placeholder="Note title..."
            />
          {:else}
            <h2
              class="mx-auto px-4 max-w-7xl"
            >
              {title || "Untitled Note"}
            </h2>
          {/if}

          <div
            class="mx-auto px-4 max-w-7xl"
          >
            <Calendar class="mx-auto px-4 max-w-7xl" />
            {createdAt.toLocaleDateString()}

            {#if userId}
              <User class="mx-auto px-4 max-w-7xl" />
              <span class="mx-auto px-4 max-w-7xl">{userId}</span>
            {/if}

            <span
              class="mx-auto px-4 max-w-7xl"
            >
              {noteType}
            </span>
          </div>
        </div>

        <div class="mx-auto px-4 max-w-7xl">
          {#if canEdit}
            {#if mode === "view"}
              <button
                type="button"
                class="mx-auto px-4 max-w-7xl"
                onclick={() => startEdit()}
                title="Edit Note"
              >
                <Edit3 class="mx-auto px-4 max-w-7xl" />
              </button>
            {:else}
              <button
                type="button"
                class="mx-auto px-4 max-w-7xl"
                onclick={() => cancelEdit()}
              >
                Cancel
              </button>
            {/if}
          {/if}

          <button
            type="button"
            class="mx-auto px-4 max-w-7xl"
            onclick={() => isSaved ? handleRemoveFromSaved : handleSaveForLater()}
            title={isSaved ? "Remove from saved" : "Save for later"}
          >
            {#if isSaved}
              <BookmarkCheck class="mx-auto px-4 max-w-7xl" />
            {:else}
              <Bookmark class="mx-auto px-4 max-w-7xl" />
            {/if}
          </button>

          <button
            type="button"
            use:melt={$close}
            class="mx-auto px-4 max-w-7xl"
          >
            <X class="mx-auto px-4 max-w-7xl" />
          </button>
        </div>
      </div>

      <!-- Tags Section -->
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <Tag class="mx-auto px-4 max-w-7xl" />

          {#if mode === "edit"}
            {#each editedTags as tag}
              <span
                class="mx-auto px-4 max-w-7xl"
              >
                {tag}
                <button
                  type="button"
                  onclick={() => removeTag(tag)}
                  class="mx-auto px-4 max-w-7xl"
                >
                  <X class="mx-auto px-4 max-w-7xl" />
                </button>
              </span>
            {/each}

            <input
              bind:value={newTag}
              onkeydown={(e) => e.key === "Enter" && addTag()}
              class="mx-auto px-4 max-w-7xl"
              placeholder="Add tag..."
            />
          {:else}
            {#each tags as tag}
              <span
                class="mx-auto px-4 max-w-7xl"
              >
                {tag}
              </span>
            {/each}
          {/if}
        </div>
      </div>

      <!-- Content -->
      <div class="mx-auto px-4 max-w-7xl">
        {#if mode === "edit"}
          <RichTextEditor
            content={editedContent}
            placeholder="Edit your note..."
            on:save={handleEditorSave}
            autoSave={false}
          />
        {:else if displayHtml}
          <div class="mx-auto px-4 max-w-7xl">
            {@html displayHtml}
          </div>
        {:else if content}
          <div class="mx-auto px-4 max-w-7xl">
            {content}
          </div>
        {:else}
          <div class="mx-auto px-4 max-w-7xl">
            No content available
          </div>
        {/if}
      </div>

      <!-- Footer -->
      {#if mode === "view"}
        <div
          class="mx-auto px-4 max-w-7xl"
        >
          <div class="mx-auto px-4 max-w-7xl">
            {#if caseId}
              <span>Associated with case: {caseId}</span>
            {:else}
              <span>General note</span>
            {/if}
          </div>

          <div class="mx-auto px-4 max-w-7xl">
            <Eye class="mx-auto px-4 max-w-7xl" />
            <span class="mx-auto px-4 max-w-7xl"
              >Read-only</span
            >
          </div>
        </div>
      {/if}
    </div>
  </div>
{/if}
