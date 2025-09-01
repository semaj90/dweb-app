<!-- Advanced Rich Text Editor with Google Slides/Photoshop-like Features -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  import { Editor } from "@tiptap/core";
  import Color from "@tiptap/extension-color";
  import FontFamily from "@tiptap/extension-font-family";
  import Highlight from "@tiptap/extension-highlight";
  import Image from "@tiptap/extension-image";
  import Placeholder from "@tiptap/extension-placeholder";
  import Table from "@tiptap/extension-table";
  import TableCell from "@tiptap/extension-table-cell";
  import TableHeader from "@tiptap/extension-table-header";
  import TableRow from "@tiptap/extension-table-row";
  import TextAlign from "@tiptap/extension-text-align";
  import TextStyle from "@tiptap/extension-text-style";
  import Typography from "@tiptap/extension-typography";
  import StarterKit from "@tiptap/starter-kit";
  import {
    AlignCenter,
    AlignJustify,
    AlignLeft,
    AlignRight,
    Bold,
    ChevronDown,
    Code,
    Download,
    Eye,
    EyeOff,
    Grid,
    Highlighter,
    Image as ImageIcon,
    Italic,
    List,
    ListOrdered,
    Quote,
    Redo,
    Save,
    Strikethrough,
    Table as TableIcon,
    Type,
    Underline,
    Undo,
    Upload,
    ZoomIn,
    ZoomOut,
  } from "lucide-svelte";
  import { onDestroy, onMount } from "svelte";
  import { get, writable } from "svelte/store";

  // Svelte 5 props
  let {
    content = $bindable(),
    placeholder = "Start writing your legal report...",
    autosave = true,
    reportId = "",
    caseId = ""
  }: {
    content?: any;
    placeholder?: string;
    autosave?: boolean;
    reportId?: string;
    caseId?: string;
  } = $props();
let editor = $state<Editor | null >(null);
  let editorElement: HTMLElement;
let isFullscreen = $state(false);
let currentZoom = $state(100);
let showGrid = $state(false);
let showRuler = $state(true);
let wordCount = $state(0);
let characterCount = $state(0);

  // Editor state stores
  const editorState = writable({
    canUndo: false,
    canRedo: false,
    isBold: false,
    isItalic: false,
    isUnderline: false,
    isStrike: false,
    currentAlignment: "left",
    currentColor: "#000000",
    currentHighlight: "",
    currentFontFamily: "Inter",
    currentFontSize: 16,
    isTable: false,
    isCode: false,
    isList: false,
    isOrderedList: false,
    isQuote: false,
  });

  // Color palettes for quick access
  const colorPalettes = {
    text: [
      "#000000",
      "#374151",
      "#6b7280",
      "#ef4444",
      "#f97316",
      "#eab308",
      "#22c55e",
      "#3b82f6",
      "#8b5cf6",
      "#ec4899",
    ],
    highlight: [
      "transparent",
      "#fef3c7",
      "#dcfce7",
      "#dbeafe",
      "#e0e7ff",
      "#f3e8ff",
      "#fce7f3",
      "#fed7d7",
      "#f0f9ff",
    ],
    legal: [
      "#1e40af",
      "#7c2d12",
      "#991b1b",
      "#365314",
      "#581c87",
      "#831843",
      "#92400e",
      "#166534",
    ],
  };

  // Font options
  const fontFamilies = [
    "Inter",
    "Times New Roman",
    "Arial",
    "Helvetica",
    "Georgia",
    "Verdana",
    "Courier New",
    "Roboto",
    "Open Sans",
    "Lato",
    "Merriweather",
  ];

  // Auto-save functionality
let autoSaveTimeout = $state<NodeJS.Timeout;

  onMount(() >(> {
    initializeEditor());
    setupKeyboardShortcuts();
  });

  onDestroy(() => {
    if (editor) {
      editor.destroy();
    }
    if (autoSaveTimeout) {
      clearTimeout(autoSaveTimeout);
    }
  });

  function initializeEditor() {
    editor = new Editor({
      element: editorElement,
      extensions: [
        StarterKit.configure({
          history: {
            depth: 100,
          },
        }),
        Image.configure({
          inline: true,
          allowBase64: true,
        }),
        TextAlign.configure({
          types: ["heading", "paragraph"],
        }),
        Highlight.configure({
          multicolor: true,
        }),
        Typography,
        Placeholder.configure({
          placeholder: placeholder,
        }),
        Table.configure({
          resizable: true,
        }),
        TableRow,
        TableHeader,
        TableCell,
        TextStyle,
        Color,
        FontFamily.configure({
          types: ["textStyle"],
        }),
      ],
      content: content,
      onTransaction: updateEditorState,
      onUpdate: ({ editor }) => {
        updateWordCount();
        if (autosave) {
          scheduleAutoSave();
        }
      },
      editorProps: {
        attributes: {
          class:
            "prose prose-lg max-w-none focus:outline-none min-h-[400px] p-6",
        },
      },
    });
  }

  function updateEditorState() {
    if (!editor) return;

    editorState.set({
      canUndo: editor.can().undo(),
      canRedo: editor.can().redo(),
      isBold: editor.isActive("bold"),
      isItalic: editor.isActive("italic"),
      isUnderline: editor.isActive("underline"),
      isStrike: editor.isActive("strike"),
      currentAlignment: editor.isActive({ textAlign: "center" })
        ? "center"
        : editor.isActive({ textAlign: "right" })
          ? "right"
          : editor.isActive({ textAlign: "justify" })
            ? "justify"
            : "left",
      currentColor: editor.getAttributes("textStyle").color || "#000000",
      currentHighlight: editor.getAttributes("highlight").color || "",
      currentFontFamily:
        editor.getAttributes("textStyle").fontFamily || "Inter",
      currentFontSize: editor.getAttributes("textStyle").fontSize || 16,
      isTable: editor.isActive("table"),
      isCode: editor.isActive("code"),
      isList: editor.isActive("bulletList"),
      isOrderedList: editor.isActive("orderedList"),
      isQuote: editor.isActive("blockquote"),
    });
  }

  function updateWordCount() {
    if (!editor) return;
    const text = editor.getText();
    wordCount = text.split(/\s+/).filter((word) => word.length > 0).length;
    characterCount = text.length;
  }

  function scheduleAutoSave() {
    if (autoSaveTimeout) {
      clearTimeout(autoSaveTimeout);
    }
    autoSaveTimeout = setTimeout(() => {
      saveContent();
    }, 2000);
  }

  async function saveContent() {
    if (!editor) return;

    const content = editor.getJSON();
    const html = editor.getHTML();

    try {
      const response = await fetch("/api/reports/save", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          reportId,
          caseId,
          content,
          html,
          wordCount,
          characterCount,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to save");
      }

      // Show save indicator
      showSaveIndicator();
    } catch (error) {
      console.error("Auto-save failed:", error);
    }
  }

  function showSaveIndicator() {
    // Implement visual save indicator
    const indicator = document.createElement("div");
    indicator.textContent = "Saved";
    indicator.class =
      "fixed top-4 right-4 bg-green-500 text-white px-3 py-1 rounded text-sm z-50";
    document.body.appendChild(indicator);
    setTimeout(() => {
      document.body.removeChild(indicator);
    }, 2000);
  }

  function setupKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case "s":
            e.preventDefault();
            saveContent();
            break;
          case "z":
            if (e.shiftKey) {
              editor?.commands.redo();
            } else {
              editor?.commands.undo();
            }
            break;
        }
      }
    });
  }

  // Toolbar actions
  function toggleBold() {
    editor?.chain().focus().toggleBold().run();
  }
  function toggleItalic() {
    editor?.chain().focus().toggleItalic().run();
  }
  function toggleUnderline() {
    editor?.chain().focus().toggleMark("underline").run();
  }
  function toggleStrike() {
    editor?.chain().focus().toggleStrike().run();
  }

  function setAlignment(align: string) {
    editor?.chain().focus().setTextAlign(align).run();
  }

  function setTextColor(color: string) {
    editor?.chain().focus().setColor(color).run();
  }

  function setHighlight(color: string) {
    if (color === "transparent") {
      editor?.chain().focus().unsetHighlight().run();
    } else {
      editor?.chain().focus().setHighlight({ color }).run();
    }
  }

  function setFontFamily(family: string) {
    editor?.chain().focus().setFontFamily(family).run();
  }

  function insertTable() {
    editor
      ?.chain()
      .focus()
      .insertTable({ rows: 3, cols: 3, withHeaderRow: true })
      .run();
  }

  function insertImage() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.on:change=(e) => {
      const file = (e.target as HTMLInputElement)?.files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const src = e.target?.result as string;
          editor?.chain().focus().setImage({ src }).run();
        };
        reader.readAsDataURL(file);
      }
    };
    input.click();
  }

  function toggleFullscreen() {
    isFullscreen = !isFullscreen;
    if (isFullscreen) {
      document.documentElement.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  }

  function adjustZoom(delta: number) {
    currentZoom = Math.max(50, Math.min(200, currentZoom + delta));
    if (editor?.view.dom) {
      (editor.view.dom as HTMLElement).style.zoom = `${currentZoom}%`;
    }
  }

  function exportDocument(format: "html" | "json" | "pdf") {
    if (!editor) return;

    const content = format === "json" ? editor.getJSON() : editor.getHTML();
    const blob = new Blob([JSON.stringify(content, null, 2)], {
      type: format === "json" ? "application/json" : "text/html",
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `report-${reportId}.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function importDocument() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json,.html";
    input.on:change=(e) => {
      const file = (e.target as HTMLInputElement)?.files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const content = JSON.parse(e.target?.result as string);
            editor?.commands.setContent(content);
          } catch {
            // If not JSON, treat as HTML
            editor?.commands.setContent(e.target?.result as string);
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  }

  // Reactive statements
  let state = $derived(get(editorState));

  // Exported functions for parent component access
  export function setContent(content: string) {
    if (editor) {
      editor.commands.setContent(content);
    }
  }

  export function getContent() {
    return editor ? editor.getHTML() : "";
  }

  export function getJSON() {
    return editor ? editor.getJSON() : null;
  }
</script>

<div class="mx-auto px-4 max-w-7xl" class:fullscreen={isFullscreen}>
  <!-- Main Toolbar -->
  <div
    class="mx-auto px-4 max-w-7xl"
  >
    <!-- File Operations -->
    <div class="mx-auto px-4 max-w-7xl">
      <button
        class="mx-auto px-4 max-w-7xl"
        on:onclick={() => saveContent()}
        title="Save (Ctrl+S)"
      >
        <Save size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        on:onclick={() => importDocument()}
        title="Import Document"
      >
        <Upload size="18" />
      </button>
      <div class="mx-auto px-4 max-w-7xl">
        <button class="mx-auto px-4 max-w-7xl">
          <Download size="18" />
          <ChevronDown size="14" />
        </button>
        <div class="mx-auto px-4 max-w-7xl">
          <button on:onclick={() => exportDocument("html")}>Export as HTML</button
          >
          <button on:onclick={() => exportDocument("json")}>Export as JSON</button
          >
          <button on:onclick={() => exportDocument("pdf")}>Export as PDF</button>
        </div>
      </div>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- Undo/Redo -->
    <div class="mx-auto px-4 max-w-7xl">
      <button
        class="mx-auto px-4 max-w-7xl"
        class:disabled={!state.canUndo}
        on:onclick={() => editor?.commands.undo()}
        title="Undo (Ctrl+Z)"
      >
        <Undo size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        class:disabled={!state.canRedo}
        on:onclick={() => editor?.commands.redo()}
        title="Redo (Ctrl+Shift+Z)"
      >
        <Redo size="18" />
      </button>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- Text Formatting -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <select
          bind:value={state.currentFontFamily}
          change={(e) =>
            setFontFamily((e.target as HTMLSelectElement).value)}
        >
          {#each fontFamilies as font}
            <option value={font}>{font}</option>
          {/each}
        </select>
      </div>

      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.isBold}
        on:onclick={() => toggleBold()}
        title="Bold (Ctrl+B)"
      >
        <Bold size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.isItalic}
        on:onclick={() => toggleItalic()}
        title="Italic (Ctrl+I)"
      >
        <Italic size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.isUnderline}
        on:onclick={() => toggleUnderline()}
        title="Underline (Ctrl+U)"
      >
        <Underline size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.isStrike}
        on:onclick={() => toggleStrike()}
        title="Strikethrough"
      >
        <Strikethrough size="18" />
      </button>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- Color Tools -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <input
          type="color"
          bind:value={state.currentColor}
          change={(e) => setTextColor((e.target as HTMLInputElement).value)}
          title="Text Color"
        />
        <Type size="18" />
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <button class="mx-auto px-4 max-w-7xl">
          <Highlighter size="18" />
          <ChevronDown size="14" />
        </button>
        <div class="mx-auto px-4 max-w-7xl">
          {#each colorPalettes.highlight as color}
            <button
              class="mx-auto px-4 max-w-7xl"
              style="background-color: {color}"
              on:onclick={() => setHighlight(color)}
              title={color === "transparent"
                ? "Remove highlight"
                : `Highlight with ${color}`}
              aria-label={color === "transparent"
                ? "Remove highlight"
                : `Highlight with ${color}`}
            ></button>
          {/each}
        </div>
      </div>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- Alignment -->
    <div class="mx-auto px-4 max-w-7xl">
      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.currentAlignment === "left"}
        on:onclick={() => setAlignment("left")}
        title="Align Left"
      >
        <AlignLeft size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.currentAlignment === "center"}
        on:onclick={() => setAlignment("center")}
        title="Align Center"
      >
        <AlignCenter size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.currentAlignment === "right"}
        on:onclick={() => setAlignment("right")}
        title="Align Right"
      >
        <AlignRight size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.currentAlignment === "justify"}
        on:onclick={() => setAlignment("justify")}
        title="Justify"
      >
        <AlignJustify size="18" />
      </button>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- Lists and Blocks -->
    <div class="mx-auto px-4 max-w-7xl">
      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.isList}
        on:onclick={() => editor?.chain().focus().toggleBulletList().run()}
        title="Bullet List"
      >
        <List size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.isOrderedList}
        on:onclick={() => editor?.chain().focus().toggleOrderedList().run()}
        title="Numbered List"
      >
        <ListOrdered size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.isQuote}
        on:onclick={() => editor?.chain().focus().toggleBlockquote().run()}
        title="Quote"
      >
        <Quote size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={state.isCode}
        on:onclick={() => editor?.chain().focus().toggleCodeBlock().run()}
        title="Code Block"
      >
        <Code size="18" />
      </button>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- Insert -->
    <div class="mx-auto px-4 max-w-7xl">
      <button
        class="mx-auto px-4 max-w-7xl"
        on:onclick={() => insertImage()}
        title="Insert Image"
      >
        <ImageIcon size="18" />
      </button>
      <button
        class="mx-auto px-4 max-w-7xl"
        on:onclick={() => insertTable()}
        title="Insert Table"
      >
        <TableIcon size="18" />
      </button>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    <!-- View Controls -->
    <div class="mx-auto px-4 max-w-7xl">
      <button
        class="mx-auto px-4 max-w-7xl"
        on:onclick={() => adjustZoom(-10)}
        title="Zoom Out"
      >
        <ZoomOut size="18" />
      </button>
      <span class="mx-auto px-4 max-w-7xl">{currentZoom}%</span>
      <button
        class="mx-auto px-4 max-w-7xl"
        on:onclick={() => adjustZoom(10)}
        title="Zoom In"
      >
        <ZoomIn size="18" />
      </button>

      <button
        class="mx-auto px-4 max-w-7xl"
        class:active={showGrid}
        on:onclick={() => (showGrid = !showGrid)}
        title="Toggle Grid"
      >
        <Grid size="18" />
      </button>

      <button
        class="mx-auto px-4 max-w-7xl"
        on:onclick={() => toggleFullscreen()}
        title="Toggle Fullscreen"
      >
        {#if isFullscreen}
          <EyeOff size="18" />
        {:else}
          <Eye size="18" />
        {/if}
      </button>
    </div>
  </div>

  <!-- Secondary Toolbar for Advanced Features -->
  <div
    class="mx-auto px-4 max-w-7xl"
  >
    <div class="mx-auto px-4 max-w-7xl">
      Words: <span class="mx-auto px-4 max-w-7xl">{wordCount}</span> | Characters:
      <span class="mx-auto px-4 max-w-7xl">{characterCount}</span>
    </div>

    <div class="mx-auto px-4 max-w-7xl"></div>

    {#if autosave}
      <div class="mx-auto px-4 max-w-7xl">Auto-save enabled</div>
    {/if}
  </div>

  <!-- Ruler (if enabled) -->
  {#if showRuler}
    <div class="mx-auto px-4 max-w-7xl">
      {#each Array(20) as _, i}
        <div
          class="mx-auto px-4 max-w-7xl"
          style="left: {i * 50}px"
        >
          {#if i % 2 === 0}
            <span class="mx-auto px-4 max-w-7xl">{i}</span>
          {/if}
        </div>
      {/each}
    </div>
  {/if}

  <!-- Editor Container -->
  <div class="mx-auto px-4 max-w-7xl" class:show-grid={showGrid}>
    <div bind:this={editorElement} class="mx-auto px-4 max-w-7xl"></div>
  </div>
</div>

<style>
  /* Remove all @apply rules. Use Tailwind/UnoCSS classes in markup instead. */
  .advanced-editor {
    min-height: 500px;
    border: 1px solid #d1d5db;
    border-radius: 0.5rem;
    overflow: hidden;
    background: #fff;
  }
  .advanced-editor.fullscreen {
    position: fixed;
    inset: 0;
    z-index: 50;
  }
  .toolbar {
    position: sticky;
    top: 0;
    z-index: 10;
  }
  .toolbar-group {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }
  .toolbar-btn {
    padding: 0.5rem;
    border-radius: 0.375rem;
    border: none;
    background: transparent;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 36px;
    height: 36px;
    transition: background 0.2s;
  }
  .toolbar-btn:hover {
    background: #f3f4f6;
  }
  .toolbar-btn:disabled,
  .toolbar-btn.disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .toolbar-btn.active {
    background: #dbeafe;
    color: #2563eb;
  }
  .toolbar-separator {
    width: 1px;
    height: 1.5rem;
    background: #d1d5db;
    margin: 0 0.25rem;
  }
  .dropdown {
    position: relative;
  }
  .dropdown-toggle {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }
  .dropdown-menu {
    position: absolute;
    top: 100%;
    left: 0;
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 0.375rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    padding: 0.25rem 0;
    z-index: 20;
    min-width: 150px;
    display: none;
  }
  .dropdown:hover .dropdown-menu {
    display: block;
  }
  .dropdown-menu button {
    width: 100%;
    text-align: left;
    padding: 0.5rem 0.75rem;
    background: transparent;
    border: none;
    cursor: pointer;
    transition: background 0.2s;
  }
  .dropdown-menu button:hover {
    background: #f3f4f6;
  }
  .color-palette {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.25rem;
    padding: 0.5rem;
    min-width: 200px;
  }
  .color-swatch {
    width: 1.5rem;
    height: 1.5rem;
    border-radius: 0.375rem;
    border: 1px solid #d1d5db;
    cursor: pointer;
  }
  .color-picker {
    position: relative;
  }
  .color-picker input[type="color"] {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }
  .font-selector select {
    border: 1px solid #d1d5db;
    border-radius: 0.375rem;
    padding: 0.25rem 0.5rem;
    font-size: 0.875rem;
  }
  .zoom-indicator {
    font-size: 0.875rem;
    color: #4b5563;
    min-width: 40px;
    text-align: center;
  }
  .ruler {
    background: repeating-linear-gradient(
      90deg,
      transparent,
      transparent 10px,
      #e5e7eb 10px,
      #e5e7eb 11px
    );
  }
  .ruler-mark {
    height: 24px;
  }
  .editor-container {
    flex: 1;
    overflow: auto;
    min-height: 400px;
  }
  .editor-container.show-grid {
    background-image:
      linear-gradient(to right, #f3f4f6 1px, transparent 1px),
      linear-gradient(to bottom, #f3f4f6 1px, transparent 1px);
    background-size: 20px 20px;
  }
  .editor-content {
    min-height: 100%;
  }
  /* Tiptap specific styles */
  :global(.ProseMirror) {
    outline: none;
  }
  :global(.ProseMirror p.is-editor-empty:first-child::before) {
    color: #9ca3af;
    content: attr(data-placeholder);
    float: left;
    height: 0;
    pointer-events: none;
  }
  :global(.ProseMirror table) {
    border-collapse: collapse;
    border: 1px solid #d1d5db;
  }
  :global(.ProseMirror table td),
  :global(.ProseMirror table th) {
    border: 1px solid #d1d5db;
    padding: 0.75rem 1rem;
  }
  :global(.ProseMirror table th) {
    background: #f3f4f6;
    font-weight: 600;
  }
  :global(.ProseMirror blockquote) {
    border-left: 4px solid #d1d5db;
    padding-left: 1rem;
    font-style: italic;
  }
  :global(.ProseMirror code) {
    background: #f3f4f6;
    padding: 0.25rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
  }
  :global(.ProseMirror pre) {
    background: #f3f4f6;
    padding: 1rem;
    border-radius: 0.5rem;
    overflow-x: auto;
  }
  :global(.ProseMirror img) {
    max-width: 100%;
    height: auto;
    border-radius: 0.5rem;
  }
</style>

<!-- Props migrated to Svelte 5 $props() pattern -->
