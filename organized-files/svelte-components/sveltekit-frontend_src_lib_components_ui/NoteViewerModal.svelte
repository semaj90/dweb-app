<script lang="ts">
  import { Bookmark, BookmarkCheck, Calendar, Edit3, Eye, Tag, User as UserIcon, X } from "lucide-svelte";
  import { marked } from "marked";
  import { fade, fly } from "svelte/transition";
  import {
    removeSavedNote,
    saveNoteForLater,
  } from '$lib/stores/saved-notes';
  import RichTextEditor from "./RichTextEditor.svelte";

  // Props
  export let noteId: string = "";
  export let title: string = "";
  export let content: string = "";
  export let markdown: string = "";
  export let html: string = "";
  export let contentJson: any = null;
  export let noteType: string = "general";
  export let tags: string[] = [];
  export let userId: string = "";
  export let caseId: string | undefined = undefined;
  export let createdAt: Date = new Date();
  export let isOpen: boolean = false;
  export let mode: "view" | "edit" = "view";
  export let canEdit: boolean = true;
  export let onSave: ((data: any) => void) | undefined = undefined;

  let isSaved = false;
  let editedContent = content;
  let editedTitle = title;
  let editedTags: string[] = [...tags];
  let newTag = "";

  // Reactive display HTML from html or markdown
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
    }
  }

  async function handleRemoveFromSaved() {
    try {
      await removeSavedNote(noteId);
      isSaved = false;
    } catch (error) {
      console.error("Failed to remove note:", error);
    }
  }

  function addTag() {
    const trimmed = newTag.trim();
    if (trimmed && !editedTags.includes(trimmed)) {
      editedTags = [...editedTags, trimmed];
      newTag = "";
    }
  }

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

  function closeModal() {
    isOpen = false;
  }
</script>

{#if isOpen}
  <div class="space-y-4" transition:fade={{ duration: 150 }}>
    <div class="space-y-4" transition:fly={{ y: -20, duration: 200 }}>
      <!-- Header -->
      <div class="space-y-4">
        <div class="space-y-4">
          {#if mode === "edit"}
            <input
              bind:value={editedTitle}
              class="space-y-4"
              placeholder="Note title..."
            />
          {:else}
            <h2 class="space-y-4">
              {title || "Untitled Note"}
            </h2>
          {/if}

          <div class="space-y-4">
            <Calendar class="space-y-4" />
            {createdAt instanceof Date ? createdAt.toLocaleDateString() : new Date(createdAt).toLocaleDateString()}

            {#if userId}
              <UserIcon class="space-y-4" />
              <span class="space-y-4">{userId}</span>
            {/if}

            <span class="space-y-4">{noteType}</span>
          </div>
        </div>

        <div class="space-y-4">
          {#if canEdit}
            {#if mode === "view"}
              <button
                type="button"
                class="space-y-4"
                on:click={startEdit}
                title="Edit Note"
              >
                <Edit3 class="space-y-4" />
              </button>
            {:else}
              <button
                type="button"
                class="space-y-4"
                on:click={cancelEdit}
              >
                Cancel
              </button>
            {/if}
          {/if}

          <button
            type="button"
            class="space-y-4"
            on:click={() => (isSaved ? handleRemoveFromSaved() : handleSaveForLater())}
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
            class="space-y-4"
            on:click={closeModal}
            title="Close"
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
              <span class="space-y-4">
                {tag}
                <button
                  type="button"
                  on:click={() => removeTag(tag)}
                  class="space-y-4"
                >
                  <X class="space-y-4" />
                </button>
              </span>
            {/each}

            <input
              bind:value={newTag}
              on:keydown={(e) => e.key === "Enter" && addTag()}
              class="space-y-4"
              placeholder="Add tag..."
            />
          {:else}
            {#each tags as tag}
              <span class="space-y-4">{tag}</span>
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
            on:save={handleEditorSave}
            autoSave={false}
          />
        {:else if displayHtml}
          <div class="space-y-4">
            {@html displayHtml}
          </div>
        {:else if content}
          <div class="space-y-4">{content}</div>
        {:else}
          <div class="space-y-4">No content available</div>
        {/if}
      </div>

      <!-- Footer -->
      {#if mode === "view"}
        <div class="space-y-4">
          <div class="space-y-4">
            {#if caseId}
              <span>Associated with case: {caseId}</span>
            {:else}
              <span>General note</span>
            {/if}
          </div>

          <div class="space-y-4">
            <Eye class="space-y-4" />
            <span class="space-y-4">Read-only</span>
          </div>
        </div>
      {/if}
    </div>
  </div>
{/if}
