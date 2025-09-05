<!-- Enhanced Legal Document Editor with UnoCSS + Melt UI -->
<script lang="ts">
  import {
    createDialog,
    createDropdownMenu,
    createTooltip,
    melt,
  } from "@melt-ui/svelte";
  import {
    AlertCircle,
    BookOpen,
    Brain,
    ChevronDown,
    Eye,
    FileText,
    Loader2,
    Save,
    Scale,
    Search,
    Settings,
    Share2,
    X,
  } from "lucide-svelte";
  import { createEventDispatcher, onMount } from "svelte";
  import { quintOut } from "svelte/easing";
  import { fade } from "svelte/transition";

  // Props
  export let caseId: string | undefined = undefined;
  export let documentId: string | undefined = undefined;
  export let documentType: "brief" | "contract" | "motion" | "evidence" =
    "brief";
  export let title = "Legal Document";
  export let readonly = false;

  // Component state
  let content = "";
  let query = "";
  let isLoading = false;
  let isProcessingAI = false;
  let error = "";
  let loadingDocument = false;
  let documentLoadError = "";
  let citations: Array<{
    id: string;
    text: string;
    source: string;
    type: "case" | "statute" | "regulation";
  }> = [];

  // Auto-save state
  let autoSaveTimer: ReturnType<typeof setTimeout> | null = null;
  let lastSaved = "";
  let isSaving = false;
  let saveError = "";
  let hasUnsavedChanges = false;

  // Document type definitions
  interface DocumentData {
    id: string;
    title: string;
    content: string;
    type: "brief" | "contract" | "motion" | "evidence";
    caseId?: string;
    createdAt: string;
    updatedAt: string;
    citations?: Array<{
      id: string;
      text: string;
      source: string;
      type: "case" | "statute" | "regulation";
    }>;
  }

  const dispatch = createEventDispatcher<{
    save: { content: string; title: string };
    aiRequest: { query: string; context: string };
    citationAdded: { citation: unknown };
  }>();

  // Melt UI Dialog for AI Assistant
  const {
    elements: {
      trigger: aiTrigger,
      overlay: aiOverlay,
      content: aiContent,
      title: aiTitle,
      description: aiDescription,
      close: aiClose,
      portalled: aiPortalled,
    },
    states: { open: aiOpen },
  } = createDialog({
    forceVisible: true,
  });

  // Melt UI Dropdown for Document Actions
  const {
    elements: {
      trigger: actionsTrigger,
      menu: actionsMenu,
      item: actionsItem,
      separator: actionsSeparator,
    },
    states: { open: actionsOpen },
  } = createDropdownMenu();

  // Melt UI Tooltip for help
  const {
    elements: { trigger: helpTrigger, content: helpContent },
    states: { open: helpOpen },
  } = createTooltip();

  async function handleAIRequest() {
    if (!query.trim()) return;

    isProcessingAI = true;
    error = "";

    try {
      const response = await fetch("/api/ai/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: query,
          context: { content, documentType, caseId },
          options: { includeReferences: true },
        }),
      });

      if (!response.ok) throw new Error("AI request failed");

      const result = await response.json();

      // Insert AI response into document
      const aiSuggestion = `\n\n<!-- AI Suggestion -->\n${result.answer}\n`;
      content += aiSuggestion;

      query = "";
      $aiOpen = false;

      dispatch("aiRequest", { query, context: content });
    } catch (err) {
      error = err instanceof Error ? err.message : "AI request failed";
    } finally {
      isProcessingAI = false;
    }
  }

  function insertCitation(citation: (typeof citations)[0]) {
    const citationText = `[${citation.source}]`;
    content += citationText;
    citations = [...citations, citation];
    dispatch("citationAdded", { citation });
  }

  function saveDocument() {
    dispatch("save", { content, title });
  }

  // Enhanced auto-save function with debouncing
  function scheduleAutoSave() {
    if (!documentId || readonly) return;

    hasUnsavedChanges = true;

    // Clear existing timer
    if (autoSaveTimer) {
      clearTimeout(autoSaveTimer);
    }

    // Schedule new auto-save after 2 seconds of inactivity
    autoSaveTimer = setTimeout(() => {
      autoSaveDocument();
    }, 2000);
  }

  // Function to auto-save document
  async function autoSaveDocument() {
    if (!documentId || readonly || isSaving) return;

    isSaving = true;
    saveError = "";

    try {
      const response = await fetch(`/api/documents/${documentId}/auto-save`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content,
          title,
          citations,
          wordCount: content.split(/\s+/).length,
          isDirty: hasUnsavedChanges,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to auto-save document");
      }

      const result = await response.json();

      if (result.success) {
        lastSaved = new Date().toLocaleTimeString();
        hasUnsavedChanges = false;
        console.log("Document auto-saved successfully");
      } else {
        throw new Error(result.error || "Auto-save failed");
      }
    } catch (err) {
      saveError = err instanceof Error ? err.message : "Auto-save failed";
      console.error("Auto-save failed:", err);
    } finally {
      isSaving = false;
    }
  }

  // Function to manually save document
  async function manualSaveDocument() {
    if (!documentId || readonly || isSaving) return;

    isSaving = true;
    saveError = "";

    try {
      const response = await fetch(`/api/documents/${documentId}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content,
          title,
          citations,
          wordCount: content.split(/\s+/).length,
          status: "draft",
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to save document");
      }

      const result = await response.json();

      if (result.success) {
        lastSaved = new Date().toLocaleTimeString();
        hasUnsavedChanges = false;
        console.log("Document saved successfully");
      } else {
        throw new Error(result.error || "Save failed");
      }
    } catch (err) {
      saveError = err instanceof Error ? err.message : "Save failed";
      console.error("Save failed:", err);
    } finally {
      isSaving = false;
    }
  }

  // Function to get save status
  function getSaveStatus() {
    if (isSaving) return "Saving...";
    if (saveError) return "Save failed";
    if (hasUnsavedChanges) return "Unsaved changes";
    if (lastSaved) return `Last saved ${lastSaved}`;
    return "All changes saved";
  }

  function getDocumentTypeIcon() {
    switch (documentType) {
      case "brief":
        return FileText;
      case "contract":
        return BookOpen;
      case "motion":
        return Scale;
      case "evidence":
        return Search;
      default:
        return FileText;
    }
  }

  onMount(() => {
    // Load document content if documentId is provided
    if (documentId) {
      loadDocument();
    }

    // Set up auto-save on content changes
    return () => {
      if (autoSaveTimer) {
        clearTimeout(autoSaveTimer);
      }
    };
  });

  // Reactive statement to trigger auto-save when content changes
  $: if (content && documentId && !loadingDocument) {
    scheduleAutoSave();
  }

  // Reactive statement to update save status
  $: saveStatus = getSaveStatus();

  // Function to load document from API
  async function loadDocument() {
    if (!documentId) return;

    loadingDocument = true;
    documentLoadError = "";

    try {
      const response = await fetch(`/api/documents/${documentId}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to load document: ${response.statusText}`);
      }

      const documentData: DocumentData = await response.json();

      // Update component state with loaded data
      content = documentData.content || "";
      title = documentData.title || "Legal Document";
      documentType = documentData.type || "brief";

      // Load citations if available
      if (documentData.citations) {
        citations = documentData.citations;
      }

      // Set initial save status
      lastSaved = new Date(documentData.updatedAt).toLocaleTimeString();
      hasUnsavedChanges = false;

      console.log("Document loaded successfully:", documentData.title);
    } catch (err) {
      documentLoadError =
        err instanceof Error ? err.message : "Failed to load document";
      console.error("Error loading document:", err);
    } finally {
      loadingDocument = false;
    }
  }

  // Custom animation function for dialog
  function flyAndScale(
    node: Element,
    params: { duration?: number; y?: number; start?: number } = {}
  ) {
    const style = getComputedStyle(node);
    const transform = style.transform === "none" ? "" : style.transform;
    const opacity = +style.opacity;
    const scaleConversion = (
      valueA: number,
      scaleA: [number, number],
      scaleB: [number, number]
    ) => {
      const [minA, maxA] = scaleA;
      const [minB, maxB] = scaleB;
      const percentage = (valueA - minA) / (maxA - minA);
      const valueB = percentage * (maxB - minB) + minB;
      return valueB;
    };
    const styleToString = (
      style: Record<string, number | string | undefined>
    ): string => {
      return Object.keys(style).reduce((str, key) => {
        if (style[key] === undefined) return str;
        return str + `${key}:${style[key]};`;
      }, "");
    };
    return {
      duration: params.duration ?? 150,
      delay: 0,
      css: (t: number) => {
        const y = scaleConversion(t, [0, 1], [params.y ?? 0, 0]);
        const scale = scaleConversion(t, [0, 1], [params.start ?? 0.95, 1]);
        return styleToString({
          transform: `${transform} translate3d(0, ${y}px, 0) scale(${scale})`,
          opacity: t * opacity,
        });
      },
      easing: quintOut,
    };
  }
</script>

<!-- Main Document Editor Container -->
<div class="mx-auto px-4 max-w-7xl">
  <!-- Header with semantic styling -->
  <header
    class="mx-auto px-4 max-w-7xl"
  >
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <svelte:component
            this={getDocumentTypeIcon()}
            class="mx-auto px-4 max-w-7xl"
          />
          <div>
            <h1 class="mx-auto px-4 max-w-7xl">{title}</h1>
            <p class="mx-auto px-4 max-w-7xl">
              {documentType.charAt(0).toUpperCase() + documentType.slice(1)}
              {#if caseId}
                • Case {caseId.slice(0, 8)}
              {/if}
              {#if loadingDocument}
                • Loading...
              {/if}
            </p>
          </div>
        </div>

        <!-- Document Actions Dropdown -->
        <div class="mx-auto px-4 max-w-7xl">
          <button
            use:melt={$helpTrigger}
            class="mx-auto px-4 max-w-7xl"
            aria-label="Help"
          >
            <AlertCircle class="mx-auto px-4 max-w-7xl" />
          </button>

          <button
            use:melt={$actionsTrigger}
            class="mx-auto px-4 max-w-7xl"
          >
            <Settings class="mx-auto px-4 max-w-7xl" />
            <span>Actions</span>
            <ChevronDown class="mx-auto px-4 max-w-7xl" />
          </button>

          <button
            onclick={() => manualSaveDocument()}
            class="mx-auto px-4 max-w-7xl"
            disabled={readonly || loadingDocument || isSaving}
          >
            {#if isSaving}
              <Loader2 class="mx-auto px-4 max-w-7xl" />
              <span>Saving...</span>
            {:else}
              <Save class="mx-auto px-4 max-w-7xl" />
              <span>Save</span>
            {/if}
          </button>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Content Area with Grid -->
  <main class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <!-- Document Editor (2/3 width) -->
        <div class="mx-auto px-4 max-w-7xl">
          <div
            class="mx-auto px-4 max-w-7xl"
          >
            <div class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl">
                <div class="mx-auto px-4 max-w-7xl">
                  <button class="mx-auto px-4 max-w-7xl" title="Bold">
                    <strong>B</strong>
                  </button>
                  <button class="mx-auto px-4 max-w-7xl" title="Italic">
                    <em>I</em>
                  </button>
                  <button class="mx-auto px-4 max-w-7xl" title="Underline">
                    <u>U</u>
                  </button>
                  <span class="mx-auto px-4 max-w-7xl">|</span>
                  <button class="mx-auto px-4 max-w-7xl" title="Insert Citation">
                    📚
                  </button>
                  <button
                    use:melt={$aiTrigger}
                    class="mx-auto px-4 max-w-7xl"
                    title="AI Assistant"
                  >
                    <Brain class="mx-auto px-4 max-w-7xl" />
                  </button>
                </div>

                <div class="mx-auto px-4 max-w-7xl">
                  {content.length} characters
                </div>
              </div>
            </div>

            <!-- Text Editor Area -->
            <div class="mx-auto px-4 max-w-7xl">
              {#if loadingDocument}
                <div
                  class="mx-auto px-4 max-w-7xl"
                >
                  <div class="mx-auto px-4 max-w-7xl">
                    <Loader2
                      class="mx-auto px-4 max-w-7xl"
                    />
                    <p class="mx-auto px-4 max-w-7xl">Loading document...</p>
                  </div>
                </div>
              {:else if documentLoadError}
                <div
                  class="mx-auto px-4 max-w-7xl"
                >
                  <div class="mx-auto px-4 max-w-7xl">
                    <AlertCircle class="mx-auto px-4 max-w-7xl" />
                    <p class="mx-auto px-4 max-w-7xl">
                      Failed to load document
                    </p>
                    <p class="mx-auto px-4 max-w-7xl">{documentLoadError}</p>
                    <button
                      class="mx-auto px-4 max-w-7xl"
                      onclick={() => loadDocument()}
                    >
                      Try Again
                    </button>
                  </div>
                </div>
              {:else}
                <textarea
                  bind:value={content}
                  disabled={readonly}
                  placeholder="Begin drafting your legal document..."
                  class="mx-auto px-4 max-w-7xl"
                  style="font-family: 'Times New Roman', serif; font-size: 14px; line-height: 1.6;"
                ></textarea>
              {/if}
            </div>
          </div>
        </div>

        <!-- Sidebar (1/3 width) -->
        <div class="mx-auto px-4 max-w-7xl">
          <!-- Citations Panel -->
          <div
            class="mx-auto px-4 max-w-7xl"
          >
            <div class="mx-auto px-4 max-w-7xl">
              <h3
                class="mx-auto px-4 max-w-7xl"
              >
                <BookOpen class="mx-auto px-4 max-w-7xl" />
                Citations
              </h3>
            </div>
            <div class="mx-auto px-4 max-w-7xl">
              {#if citations.length === 0}
                <p class="mx-auto px-4 max-w-7xl">No citations added yet.</p>
              {:else}
                <div class="mx-auto px-4 max-w-7xl">
                  {#each citations as citation}
                    <div class="mx-auto px-4 max-w-7xl">
                      <div class="mx-auto px-4 max-w-7xl">
                        {citation.type.toUpperCase()}
                      </div>
                      <div class="mx-auto px-4 max-w-7xl">{citation.source}</div>
                    </div>
                  {/each}
                </div>
              {/if}

              <button
                class="mx-auto px-4 max-w-7xl"
                onclick={() =>
                  insertCitation({
                    id: Math.random().toString(),
                    text: "Sample Citation",
                    source: "Smith v. Jones, 123 F.3d 456 (2023)",
                    type: "case",
                  })}
              >
                Add Citation
              </button>
            </div>
          </div>

          <!-- Document Info -->
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">
              <h3 class="mx-auto px-4 max-w-7xl">
                Document Info
              </h3>
            </div>
            <div class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">Type:</span>
                <span class="mx-auto px-4 max-w-7xl">{documentType}</span>
              </div>
              <div class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">Word Count:</span>
                <span class="mx-auto px-4 max-w-7xl">{content.split(/\s+/).length}</span>
              </div>
              <div class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">Status:</span>
                <span
                  class="mx-auto px-4 max-w-7xl"
                  class:text-red-600={saveError}
                  class:text-green-600={!hasUnsavedChanges && !saveError}
                >
                  {saveStatus}
                </span>
              </div>
              {#if saveError}
                <div class="mx-auto px-4 max-w-7xl">
                  {saveError}
                </div>
              {/if}
            </div>
          </div>
        </div>
      </div>
    </div>
  </main>
</div>

<!-- AI Assistant Dialog -->
{#if $aiOpen}
  <div use:melt={$aiPortalled}>
    <div
      use:melt={$aiOverlay}
      class="mx-auto px-4 max-w-7xl"
      transition:fade={{ duration: 150 "
    ></div>
    <div
      class="mx-auto px-4 max-w-7xl"
      transition:flyAndScale={{ duration: 150, y: 8, start: 0.96 "
      use:melt={$aiContent}
    >
      <div class="mx-auto px-4 max-w-7xl">
        <h2
          use:melt={$aiTitle}
          class="mx-auto px-4 max-w-7xl"
        >
          <Brain class="mx-auto px-4 max-w-7xl" />
          AI Legal Assistant
        </h2>
        <p use:melt={$aiDescription} class="mx-auto px-4 max-w-7xl">
          Ask for help with legal research, drafting, or analysis
        </p>
        <button
          use:melt={$aiClose}
          class="mx-auto px-4 max-w-7xl"
        >
          <X class="mx-auto px-4 max-w-7xl" />
        </button>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        {#if error}
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">
              <AlertCircle class="mx-auto px-4 max-w-7xl" />
              <span class="mx-auto px-4 max-w-7xl">{error}</span>
            </div>
          </div>
        {/if}

        <div class="mx-auto px-4 max-w-7xl">
          <div>
            <label
              for="ai-query"
              class="mx-auto px-4 max-w-7xl"
            >
              What would you like help with?
            </label>
            <textarea
              id="ai-query"
              bind:value={query}
              placeholder="e.g., Help me draft a motion to dismiss based on lack of jurisdiction..."
              class="mx-auto px-4 max-w-7xl"
              disabled={isProcessingAI}
            ></textarea>
          </div>

          <div class="mx-auto px-4 max-w-7xl">
            <button
              use:melt={$aiClose}
              class="mx-auto px-4 max-w-7xl"
              disabled={isProcessingAI}
            >
              Cancel
            </button>
            <button
              onclick={() => handleAIRequest()}
              class="mx-auto px-4 max-w-7xl"
              disabled={!query.trim() || isProcessingAI}
            >
              {#if isProcessingAI}
                <Loader2 class="mx-auto px-4 max-w-7xl" />
                <span>Processing...</span>
              {:else}
                <Brain class="mx-auto px-4 max-w-7xl" />
                <span>Get Help</span>
              {/if}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
{/if}

<!-- Actions Dropdown Menu -->
{#if $actionsOpen}
  <div
    use:melt={$actionsMenu}
    class="mx-auto px-4 max-w-7xl"
  >
    <button
      use:melt={$actionsItem}
      class="mx-auto px-4 max-w-7xl"
    >
      <Eye class="mx-auto px-4 max-w-7xl" />
      Preview
    </button>
    <button
      use:melt={$actionsItem}
      class="mx-auto px-4 max-w-7xl"
    >
      <Share2 class="mx-auto px-4 max-w-7xl" />
      Share
    </button>
    <div class="mx-auto px-4 max-w-7xl"></div>
    <button
      use:melt={$actionsItem}
      class="mx-auto px-4 max-w-7xl"
    >
      <X class="mx-auto px-4 max-w-7xl" />
      Delete
    </button>
  </div>
{/if}

<!-- Help Tooltip -->
{#if $helpOpen}
  <div
    use:melt={$helpContent}
    class="mx-auto px-4 max-w-7xl"
  >
    Use the AI assistant for legal research and drafting help. Click the
    citation button to add references.
  </div>
{/if}

<style>
  .toolbar-btn {
    padding: 0.5rem 0.5rem;
    font-size: 0.875rem;
    border: 1px solid #d1d5db;
    border-radius: 0.25rem;
    transition: background-color 0.2s;
  }

  .toolbar-btn:hover {
    background-color: #f9fafb;
  }

  .toolbar-separator {
    color: #d1d5db;
    margin: 0 0.5rem;
  }

  .ai-button {
    background-color: #eff6ff;
    border-color: #93c5fd;
    color: #1d4ed8;
  }

  .ai-button:hover {
    background-color: #dbeafe;
  }
</style>
