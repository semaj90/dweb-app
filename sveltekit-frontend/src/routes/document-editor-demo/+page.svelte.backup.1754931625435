<!-- Demo page for the Legal Document Editor -->
<script lang="ts">
  import LegalDocumentEditor from "$lib/components/editor/LegalDocumentEditor.svelte";

  let documentId = "doc-1"; // Use sample document ID
  let caseId = "case-123";
  let selectedDocumentType: "brief" | "contract" | "motion" | "evidence" =
    "brief";
  let editorTitle = "Criminal Case Brief Demo";
  let isReadonly = false;

  function handleSave(event: CustomEvent) {
    console.log("Document saved:", event.detail);
}
  function handleAIRequest(event: CustomEvent) {
    console.log("AI request:", event.detail);
}
  function handleCitationAdded(event: CustomEvent) {
    console.log("Citation added:", event.detail);
}
  function createNewDocument() {
    documentId = `doc-${Date.now()}`;
    console.log("Created new document:", documentId);
}
  function switchDocumentType(type: typeof selectedDocumentType) {
    selectedDocumentType = type;
    editorTitle = `${type.charAt(0).toUpperCase() + type.slice(1)} Demo`;
}
</script>

<svelte:head>
  <title>Legal Document Editor Demo</title>
  <meta
    name="description"
    content="Demo of the Legal Document Editor component"
  />
</svelte:head>

<div class="container mx-auto px-4">
  <!-- Demo Controls -->
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <h1 class="container mx-auto px-4">
          Legal Document Editor Demo
        </h1>

        <div class="container mx-auto px-4">
          <!-- Document Type Selector -->
          <div class="container mx-auto px-4">
            <label
              for="document-type-selector"
              class="container mx-auto px-4">Type:</label
            >
            <select
              id="document-type-selector"
              bind:value={selectedDocumentType}
              on:change={() => switchDocumentType(selectedDocumentType)}
              class="container mx-auto px-4"
            >
              <option value="brief">Brief</option>
              <option value="motion">Motion</option>
              <option value="contract">Contract</option>
              <option value="evidence">Evidence</option>
            </select>
          </div>

          <!-- New Document Button -->
          <button
            on:click={() => createNewDocument()}
            class="container mx-auto px-4"
          >
            New Document
          </button>

          <!-- Readonly Toggle -->
          <label class="container mx-auto px-4">
            <input
              type="checkbox"
              bind:checked={isReadonly}
              class="container mx-auto px-4"
            />
            <span class="container mx-auto px-4">Read-only</span>
          </label>
        </div>
      </div>
    </div>
  </div>

  <!-- Document Editor -->
  <LegalDocumentEditor
    {documentId}
    {caseId}
    documentType={selectedDocumentType}
    title={editorTitle}
    readonly={isReadonly}
    on:save={handleSave}
    on:aiRequest={handleAIRequest}
    on:citationAdded={handleCitationAdded}
  />
</div>

<style>
  /* @unocss-include */
  /* Additional demo-specific styles */
  :global(body) {
    font-family:
      -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu,
      sans-serif;
}
</style>
