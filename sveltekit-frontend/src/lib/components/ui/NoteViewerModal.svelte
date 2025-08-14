<!-- @migration-task Error while migrating Svelte code: Unexpected token
https://svelte.dev/e/js_parse_error -->
<script lang="ts">
  interface Props {
    noteId: string;
    title: string ;
    content: string ;
    markdown: string ;
    html: string ;
    contentJson: any ;
    noteType: string ;
    tags: string[] ;
    userId: string ;
    caseId: string | undefined ;
    createdAt: Date ;
    isOpen?: any;
    mode: "view" | "edit" ;
    canEdit?: any;
    onSave: (data: any) => void;
  }
  let {
    noteId,
    title = "",
    content = "",
    markdown = "",
    html = "",
    contentJson = null,
    noteType = "general",
    tags = [],
    userId = "",
    caseId = undefined,
    createdAt = new Date(),
    isOpen = false,
    mode = "view",
    canEdit = true,
    onSave = undefined
  }: Props = $props();



  import { createDialog, melt } from "@melt-ui/svelte";
  import { Bookmark, BookmarkCheck, Calendar, Edit3, Eye, Tag, User as UserIcon, X } from "lucide-svelte";
  import { marked } from "marked";
  import { writable } from "svelte/store";
  import { fade, fly } from "svelte/transition";
  import {
    removeSavedNote,
    saveNoteForLater,
  } from '$lib/stores/saved-notes';
  import RichTextEditor from "./RichTextEditor.svelte";

                              
  let isSaved = $state(false);
  let editedContent = $state(content);
  let editedTitle = $state(title);
  let editedTags = $state([...tags]);
  let newTag = $state("");

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

  $effect(() => { 
    if (isOpen !== $open) {
      open.set(isOpen);
    }
  });
  // Parse markdown to HTML for display
  let displayHtml = $derived(html || (markdown ? marked.parse(markdown) : ""));

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
    class="space-y-4"
    transitionfade={{ duration: 150 }}
  >
    <div
      use:melt={$dialogContent}
      class="space-y-4"
      transitionfly={{ y: -20, duration: 200 }}
    >
      <!-- Header -->
      <div
        class="space-y-4"
      >
        <div class="space-y-4">
          {#if mode === "edit"}
            <input
              bind:value={editedTitle}
              class="space-y-4"
              placeholder="Note title..."
            />
          {:else}
            <h2
              class="space-y-4"
            >
              {title || "Untitled Note"}
            </h2>
          {/if}

          <div
            class="space-y-4"
          >
            <Calendar class="space-y-4" />
            {createdAt.toLocaleDateString()}

            {#if userId}
              <UserIcon class="space-y-4" />
              <span class="space-y-4">{userId}</span>
            {/if}

            <span
              class="space-y-4"
            >
              {noteType}
            </span>
          </div>
        </div>

        <div class="space-y-4">
          {#if canEdit}
            {#if mode === "view"}
              <button
                type="button"
                class="space-y-4"
                onclick={() => startEdit()}
                title="Edit Note"
              >
                <Edit3 class="space-y-4" />
              </button>
            {:else}
              <button
                type="button"
                class="space-y-4"
                onclick={() => cancelEdit()}
              >
                Cancel
              </button>
            {/if}
          {/if}

          <button
            type="button"
            class="space-y-4"
            onclick={() => isSaved ? handleRemoveFromSaved : handleSaveForLater()}
            title={isSaved ? "Remove from saved" : "Save for later"}
          >
            {#if isSaved}
              <BookmarkCheck class="space-y-4" />
            {:else}
              <Bookmark class="space-y-4" />
            {/if}
          </button>

          <button
            type="button"
            use:melt={$close}
            class="space-y-4"
          >
            <X class="space-y-4" />
          </button>
        </div>
      </div>

      <!-- Tags Section -->
      <div class="space-y-4">
        <div class="space-y-4">
          <Tag class="space-y-4" />

          {#if mode === "edit"}
            {#each editedTags as tag}
              <span
                class="space-y-4"
              >
                {tag}
                <button
                  type="button"
                  onclick={() => removeTag(tag)}
                  class="space-y-4"
                >
                  <X class="space-y-4" />
                </button>
              </span>
            {/each}

            <input
              bind:value={newTag}
              onkeydown={(e) => e.key === "Enter" && addTag()}
              class="space-y-4"
              placeholder="Add tag..."
            />
          {:else}
            {#each tags as tag}
              <span
                class="space-y-4"
              >
                {tag}
              </span>
            {/each}
          {/if}
        </div>
      </div>

      <!-- Content -->
      <div class="space-y-4">
        {#if mode === "edit"}
          <RichTextEditor
            content={editedContent}
            placeholder="Edit your note..."
            onsave={handleEditorSave}
            autoSave={false}
          />
        {:else if displayHtml}
          <div class="space-y-4">
            {@html displayHtml}
          </div>
        {:else if content}
          <div class="space-y-4">
            {content}
          </div>
        {:else}
          <div class="space-y-4">
            No content available
          </div>
        {/if}
      </div>

      <!-- Footer -->
      {#if mode === "view"}
        <div
          class="space-y-4"
        >
          <div class="space-y-4">
            {#if caseId}
              <span>Associated with case: {caseId}</span>
            {:else}
              <span>General note</span>
            {/if}
          </div>

          <div class="space-y-4">
            <Eye class="space-y-4" />
            <span class="space-y-4"
              >Read-only</span
            >
          </div>
        </div>
      {/if}
    </div>
  </div>
{/if}
