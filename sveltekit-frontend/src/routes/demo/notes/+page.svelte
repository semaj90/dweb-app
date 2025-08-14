<script lang="ts">
  import { page } from '$app/stores';
  import { BookOpen, Plus, Search, Tag } from 'lucide-svelte';
  import { onMount } from 'svelte';
  import DragDropZone from '$lib/components/ui/DragDropZone.svelte';
  import MarkdownRenderer from '$lib/components/ui/MarkdownRenderer.svelte';
  import NoteViewerModal from '$lib/components/ui/NoteViewerModal.svelte';
  import RichTextEditor from '$lib/components/ui/RichTextEditor.svelte';
  import { filteredNotes, notesManager, setNoteFilter } from '$lib/stores/saved-notes';

  let searchQuery = '';
  let selectedNoteType = '';
  let showNoteModal = false;
  let selectedNote: any = null;
  let isCreatingNote = false;

  // Demo data
  let demoMarkdown = `# Legal Case Analysis

## Case Summary
This is a **comprehensive analysis** of the evidence collected in case #2024-001.

### Key Findings
1. Witness testimony corroborates timeline
2. Physical evidence supports suspect identification
3. *Digital forensics* reveal additional context

### Evidence List
- Security footage from 14:30-15:00
- Fingerprint analysis results
- Mobile phone records

> **Important Note**: All evidence has been properly catalogued and chain of custody maintained.

\`\`\`
Case ID: 2024-001
Status: Active Investigation
Priority: High
\`\`\`

![Evidence Photo](https://via.placeholder.com/400x200?text=Evidence+Photo)

---

**Next Steps:**
- Schedule additional witness interviews
- Request additional forensic analysis
- Coordinate with district attorney's office`;

  let editorContent = '';
  let currentNote = {
    id: '',
    title: 'New Case Note',
    content: '',
    markdown: '',
    html: '',
    contentJson: null,
    noteType: 'general',
    tags: ['demo', 'test'],
    userId: 'demo-user',
    caseId: undefined as string | undefined
  };

  onMount(async () => {
    // Load saved notes from IndexedDB
    await notesManager.loadSavedNotes();
  });

  function handleEditorSave(event: CustomEvent) {
    const { html, markdown, json } = event.detail;
    console.log('Editor saved:', { html, markdown, json });

    // Save the note
    saveCurrentNote(html, markdown, json);
}
  function handleEditorChange(event: CustomEvent) {
    const { html, markdown, json } = event.detail;
    editorContent = markdown || html;

    // Update current note
    currentNote = {
      ...currentNote,
      content: markdown || html,
      markdown,
      html,
      contentJson: json
    };
}
  async function saveCurrentNote(html: string, markdown: string, json: any) {
    if (!currentNote.title.trim() && !markdown.trim()) return;

    const noteToSave = {
      ...currentNote,
      id: currentNote.id || generateNoteId(),
      content: markdown || html,
      markdown,
      html,
      contentJson: json,
      savedAt: new Date()
    };

    try {
      // Save to API
      await fetch(`/api/notes/${noteToSave.id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(noteToSave)
      });

      // Save to local store
      await notesManager.saveNote(noteToSave);

      console.log('Note saved successfully');
    } catch (error) {
      console.error('Failed to save note:', error);
}}
  function generateNoteId(): string {
    return `note-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}
  function createNewNote() {
    currentNote = {
      id: generateNoteId(),
      title: 'New Case Note',
      content: '',
      markdown: '',
      html: '',
      contentJson: null,
      noteType: 'general',
      tags: [],
      userId: 'demo-user',
      caseId: $page.url.searchParams.get('caseId') || undefined
    };
    isCreatingNote = true;
}
  function viewNote(note: any) {
    selectedNote = note;
    showNoteModal = true;
}
  function handleFilesDropped(event: CustomEvent) {
    const files = event.detail as File[];
    console.log('Files dropped:', files);
    // Handle file upload logic here
}
  function updateSearch() {
    setNoteFilter({ search: searchQuery, noteType: selectedNoteType });
}
  $effect(() => {
    updateSearch();
  });
</script>

<svelte:head>
  <title>Rich Text Editor Demo - Warden-Net</title>
</svelte:head>

<div class="space-y-4">
  <div class="space-y-4">
    <!-- Header -->
    <div class="space-y-4">
      <h1 class="space-y-4">
        Rich Text Editor Demo
      </h1>
      <p class="space-y-4">
        Test the WYSIWYG editor, markdown rendering, and note management system
      </p>
    </div>

    <div class="space-y-4">
      <!-- Main Editor Column -->
      <div class="space-y-4">
        <!-- Rich Text Editor -->
        <div class="space-y-4">
          <div class="space-y-4">
            <h2 class="space-y-4">
              <BookOpen class="space-y-4" />
              Rich Text Editor
            </h2>
            <p class="space-y-4">
              Create and edit notes with full WYSIWYG support
            </p>
          </div>

          <div class="space-y-4">
            <input
              bind:value={currentNote.title}
              class="space-y-4"
              placeholder="Note title..."
            />

            <div class="space-y-4">
              <select bind:value={currentNote.noteType} class="space-y-4">
                <option value="general">General</option>
                <option value="evidence">Evidence</option>
                <option value="poi">Person of Interest</option>
                <option value="case_summary">Case Summary</option>
              </select>

              <input
                type="text"
                placeholder="Add tags (comma separated)"
                class="space-y-4"
                onblur={(e) => {
                  const tags = (e.target as HTMLInputElement).value.split(',').map(t => t.trim()).filter(t => t);
                  currentNote.tags = tags;
                }}
              />
            </div>

            <RichTextEditor
              content={currentNote.content}
              placeholder="Start writing your note..."
              onsave={handleEditorSave}
              onchange={handleEditorChange}
              autoSave={true}
              autoSaveDelay={3000}
            />
          </div>
        </div>

        <!-- File Upload -->
        <div class="space-y-4">
          <div class="space-y-4">
            <h2 class="space-y-4">
              File Upload
            </h2>
          </div>

          <div class="space-y-4">
            <DragDropZone
              accept="image/*,.pdf,.doc,.docx,.txt"
              maxSize={10485760}
              on:filesDropped={handleFilesDropped}
            />
          </div>
        </div>

        <!-- Markdown Renderer Demo -->
        <div class="space-y-4">
          <div class="space-y-4">
            <h2 class="space-y-4">
              Markdown Renderer (LLM Output Demo)
            </h2>
          </div>

          <div class="space-y-4">
            <MarkdownRenderer markdown={demoMarkdown} class="prose-sm" />
          </div>
        </div>
      </div>

      <!-- Sidebar - Saved Notes -->
      <div class="space-y-4">
        <!-- Search and Filters -->
        <div class="space-y-4">
          <div class="space-y-4">
            <h2 class="space-y-4">
              <Search class="space-y-4" />
              Saved Notes
            </h2>
          </div>

          <div class="space-y-4">
            <div class="space-y-4">
              <Search class="space-y-4" />
              <input
                bind:value={searchQuery}
                type="text"
                placeholder="Search notes..."
                class="space-y-4"
              />
            </div>

            <select bind:value={selectedNoteType} class="space-y-4">
              <option value="">All Types</option>
              <option value="general">General</option>
              <option value="evidence">Evidence</option>
              <option value="poi">Person of Interest</option>
              <option value="case_summary">Case Summary</option>
            </select>

            <button
              type="button"
              onclick={() => createNewNote()}
              class="space-y-4"
            >
              <Plus class="space-y-4" />
              New Note
            </button>
          </div>
        </div>

        <!-- Notes List -->
        <div class="space-y-4">
          <div class="space-y-4">
            <h3 class="space-y-4">
              Recent Notes ({$filteredNotes.length})
            </h3>
          </div>

          <div class="space-y-4">
            {#each $filteredNotes as note (note.id)}
              <button
                type="button"
                onclick={() => viewNote(note)}
                class="space-y-4"
              >
                <div class="space-y-4">
                  {note.title}
                </div>
                <div class="space-y-4">
                  {note.content.slice(0, 100)}...
                </div>
                <div class="space-y-4">
                  <span class="space-y-4">
                    {note.noteType}
                  </span>
                  <span class="space-y-4">
                    {new Date(note.savedAt).toLocaleDateString()}
                  </span>
                </div>
                {#if note.tags.length > 0}
                  <div class="space-y-4">
                    {#each note.tags.slice(0, 3) as tag}
                      <span class="space-y-4">
                        <Tag class="space-y-4" />
                        {tag}
                      </span>
                    {/each}
                    {#if note.tags.length > 3}
                      <span class="space-y-4">
                        +{note.tags.length - 3} more
                      </span>
                    {/if}
                  </div>
                {/if}
              </button>
            {:else}
              <div class="space-y-4">
                <BookOpen class="space-y-4" />
                <p>No notes found</p>
                <p class="space-y-4">Create your first note to get started</p>
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Note Viewer Modal -->
{#if selectedNote}
  <NoteViewerModal
    bind:isOpen={showNoteModal}
    noteId={selectedNote.id}
    title={selectedNote.title}
    content={selectedNote.content}
    markdown={selectedNote.markdown}
    html={selectedNote.html}
    contentJson={selectedNote.contentJson}
    noteType={selectedNote.noteType}
    tags={selectedNote.tags}
    userId={selectedNote.userId}
    caseId={selectedNote.caseId}
    createdAt={new Date(selectedNote.savedAt)}
    canEdit={true}
    onsave={(event) => {
      console.log('Note updated:', event.detail);
      // Refresh the note in the list
      notesManager.saveNote(event.detail);
    }}
  />
{/if}
