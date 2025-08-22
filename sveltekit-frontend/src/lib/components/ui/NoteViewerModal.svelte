<script lang="ts">
  import { createDialog, melt } from "@melt-ui/svelte";
  import { $props, $state, $derived, $effect } from 'svelte';
  import { Bookmark, BookmarkCheck, Calendar, Edit3, Eye, Tag, User as UserIcon, X } from "lucide-svelte";
  import { marked } from "marked";
  import { writable } from "svelte/store";
  import { fade, fly } from "svelte/transition";
  import {
    removeSavedNote,
    saveNoteForLater,
  } from '$lib/stores/saved-notes';
  import RichTextEditor from "./RichTextEditor.svelte";

  interface Props {
    noteId: string;
    title?: string;
    content?: string;
    markdown?: string;
    html?: string;
    contentJson?: unknown;
    noteType?: string;
    tags?: string[];
    userId?: string;
    caseId?: string;
    createdAt?: Date;
    isOpen?: boolean;
    mode?: "view" | "edit";
    canEdit?: boolean;
    onSave?: (data: any) => void;
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
    isOpen = $bindable(false),
    mode = $bindable<"view" | "edit">("view"),
    canEdit = true,
    onSave = undefined
  }: Props = $props();

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
    class="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4"
    transition:fade={{ duration: 150 }}
  >
    <div
      use:melt={$dialogContent}
      class="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
      transition:fly={{ y: -20, duration: 200 }}
    >
      <!-- Header -->
      <div class="border-b border-gray-200 dark:border-gray-700 p-6">
        <div class="flex items-start justify-between">
          <div class="flex-1">
            {#if mode === "edit"}
              <input
                bind:value={editedTitle}
                class="text-2xl font-bold bg-transparent border-none outline-none w-full"
                placeholder="Note title..."
              />
            {:else}
              <h2 class="text-2xl font-bold text-gray-900 dark:text-white">
                {title || "Untitled Note"}
              </h2>
            {/if}

            <div class="flex items-center gap-4 mt-2 text-sm text-gray-500 dark:text-gray-400">
              <div class="flex items-center gap-1">
                <Calendar class="w-4 h-4" />
                {createdAt.toLocaleDateString()}
              </div>

              {#if userId}
                <div class="flex items-center gap-1">
                  <UserIcon class="w-4 h-4" />
                  <span>{userId}</span>
                </div>
              {/if}

              <span class="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">
                {noteType}
              </span>
            </div>
          </div>

          <div class="flex items-center gap-2 ml-4">
            {#if canEdit}
              {#if mode === "view"}
                <button
                  type="button"
                  class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded"
                  onclick={() => startEdit()}
                  title="Edit Note"
                >
                  <Edit3 class="w-5 h-5" />
                </button>
              {:else}
                <button
                  type="button"
                  class="px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
                  onclick={() => cancelEdit()}
                >
                  Cancel
                </button>
              {/if}
            {/if}

            <button
              type="button"
              class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded"
              onclick={() => isSaved ? handleRemoveFromSaved() : handleSaveForLater()}
              title={isSaved ? "Remove from saved" : "Save for later"}
            >
              {#if isSaved}
                <BookmarkCheck class="w-5 h-5 text-blue-500" />
              {:else}
                <Bookmark class="w-5 h-5" />
              {/if}
            </button>

            <button
              type="button"
              use:melt={$close}
              class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded"
            >
              <X class="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      <!-- Tags Section -->
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div class="flex items-center gap-2 flex-wrap">
          <Tag class="w-4 h-4 text-gray-400" />

          {#if mode === "edit"}
            {#each editedTags as tag}
              <span class="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded text-sm">
                {tag}
                <button
                  type="button"
                  onclick={() => removeTag(tag)}
                  class="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200"
                >
                  <X class="w-3 h-3" />
                </button>
              </span>
            {/each}

            <input
              bind:value={newTag}
              onkeydown={(e) => e.key === "Enter" && addTag()}
              class="px-2 py-1 border border-gray-300 dark:border-gray-600 rounded text-sm min-w-20"
              placeholder="Add tag..."
            />
          {:else}
            {#each tags as tag}
              <span class="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded text-sm">
                {tag}
              </span>
            {/each}
          {/if}
        </div>
      </div>

      <!-- Content -->
      <div class="p-6">
        {#if mode === "edit"}
          <RichTextEditor
            content={editedContent}
            placeholder="Edit your note..."
            onsave={handleEditorSave}
            autoSave={false}
          />
        {:else if displayHtml}
          <div class="prose dark:prose-invert max-w-none">
            {@html displayHtml}
          </div>
        {:else if content}
          <div class="whitespace-pre-wrap text-gray-700 dark:text-gray-300">
            {content}
          </div>
        {:else}
          <div class="text-gray-500 dark:text-gray-400 italic">
            No content available
          </div>
        {/if}
      </div>

      <!-- Footer -->
      {#if mode === "view"}
        <div class="border-t border-gray-200 dark:border-gray-700 px-6 py-4">
          <div class="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
            <div>
              {#if caseId}
                <span>Associated with case: {caseId}</span>
              {:else}
                <span>General note</span>
              {/if}
            </div>

            <div class="flex items-center gap-1">
              <Eye class="w-4 h-4" />
              <span>Read-only</span>
            </div>
          </div>
        </div>
      {/if}
    </div>
  </div>
{/if}

