<!-- Lightweight Rich Text Editor with Enhanced Image Support -->
<script lang="ts">
  interface Props {
    content: any
    placeholder?: any;
    autosave?: any;
    reportId: string
    caseId: string
    disabled?: any;
  }
  let {
    content = null,
    placeholder = "Start writing your legal report...",
    autosave = true,
    reportId = "",
    caseId = "",
    disabled = false
  } = $props();



  import { Editor } from "@tiptap/core";
  import Image from "@tiptap/extension-image";
  import Placeholder from "@tiptap/extension-placeholder";
  import TextAlign from "@tiptap/extension-text-align";
  import StarterKit from "@tiptap/starter-kit";
  import {
    AlignCenter,
    AlignLeft,
    AlignRight,
    Bold,
    Brain,
    FileText,
    Image as ImageIcon,
    Italic,
    List,
    ListOrdered,
    Quote,
    Redo,
    Save,
    Undo,
    Upload,
    Wand2,
    Sparkles,
  } from "lucide-svelte";
  import AIDropdown from '$lib/components/ui/AIDropdown.svelte';
  import { registerShortcut, setShortcutContext } from '$lib/utils/keyboard-shortcuts';
  import { onDestroy, onMount } from "svelte";
  import { writable } from "svelte/store";

            
  let editor: Editor | null = null;
  let editorElement: HTMLElement
  let wordCount = 0;
  let characterCount = 0;
  let dragCounter = 0;
  let isDragging = false;
  let uploadProgress = 0;
  let uploading = false;
  let aiGenerating = false;
  let aiProgress = 0;
  let showAIMenu = false;
  let selectedReportType = "";

  // AI-powered features
  const reportTypes = [
    { id: "case-summary", name: "Case Summary Report", icon: FileText },
    { id: "evidence-analysis", name: "Evidence Analysis", icon: Brain },
    { id: "legal-brief", name: "Legal Brief", icon: Wand2 },
    { id: "investigation-report", name: "Investigation Report", icon: Sparkles },
  ];

  // Simplified editor state
  const editorState = writable({
    canUndo: false,
    canRedo: false,
    isBold: false,
    isItalic: false,
    currentAlignment: "left",
    isList: false,
    isOrderedList: false,
    isQuote: false,
  });

  // Auto-save functionality
  let autoSaveTimeout: NodeJS.Timeout;

  onMount(() => {
    initializeEditor();
    setupKeyboardShortcuts();
    
    // Set editor context for keyboard shortcuts
    setShortcutContext('editor');
    
    // Register editor-specific shortcuts
    registerShortcut({
      key: 's',
      ctrl: true,
      description: 'Save Document',
      action: saveContent,
      preventDefault: true
    });
    
    registerShortcut({
      key: 'c',
      ctrl: true,
      shift: true,
      description: 'Generate Case Summary',
      action: () => generateAIReport('case-summary'),
      preventDefault: true
    });
    
    registerShortcut({
      key: 'e',
      ctrl: true,
      shift: true,
      description: 'Generate Evidence Analysis',
      action: () => generateAIReport('evidence-analysis'),
      preventDefault: true
    });
    
    registerShortcut({
      key: 'l',
      ctrl: true,
      shift: true,
      description: 'Generate Legal Brief',
      action: () => generateAIReport('legal-brief'),
      preventDefault: true
    });
    
    registerShortcut({
      key: 'i',
      ctrl: true,
      shift: true,
      description: 'Generate Investigation Report',
      action: () => generateAIReport('investigation-report'),
      preventDefault: true
    });
    
    registerShortcut({
      key: 's',
      ctrl: true,
      shift: true,
      description: 'Summarize Content',
      action: summarizeReport,
      preventDefault: true
    });
    
    registerShortcut({
      key: 'a',
      ctrl: true,
      shift: true,
      description: 'Analyze Report',
      action: analyzeReport,
      preventDefault: true
    });
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
        StarterKit,
        Image.configure({
          inline: true,
          allowBase64: true,
          HTMLAttributes: {
            class: 'max-w-full h-auto rounded-lg border border-gray-200',
          },
        }),
        TextAlign.configure({
          types: ["heading", "paragraph"],
        }),
        Placeholder.configure({
          placeholder: placeholder,
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
          class: "prose prose-lg max-w-none focus:outline-none min-h-[400px] p-6",
        },
        handleDrop: handleDrop,
        handlePaste: handlePaste,
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
      currentAlignment: editor.isActive({ textAlign: "center" })
        ? "center"
        : editor.isActive({ textAlign: "right" })
          ? "right"
          : "left",
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
    indicator.className =
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

  // Enhanced image handling functions
  function handleDrop(view: any, event: DragEvent, slice: any, moved: boolean): boolean {
    const files = Array.from(event.dataTransfer?.files || []);
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length > 0) {
      event.preventDefault();
      // Handle async upload without blocking the handler
      uploadImages(imageFiles).catch(console.error);
      return true;
    }
    return false;
  }

  function handlePaste(view: any, event: ClipboardEvent, slice: any): boolean {
    const items = Array.from(event.clipboardData?.items || []);
    const imageItems = items.filter(item => item.type.startsWith('image/'));
    
    if (imageItems.length > 0) {
      event.preventDefault();
      // Handle async processing without blocking
      Promise.all(
        imageItems.map(item => item.getAsFile()).filter(Boolean)
      ).then(files => uploadImages(files)).catch(console.error);
      return true;
    }
    return false;
  }

  async function uploadImages(files: File[]) {
    if (!editor) return;
    
    uploading = true;
    uploadProgress = 0;

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        uploadProgress = Math.round(((i + 1) / files.length) * 100);
        
        // For now, convert to base64 for immediate display
        const reader = new FileReader();
        const dataUrl = await new Promise<string>((resolve) => {
          reader.onload = () => resolve(reader.result as string);
          reader.readAsDataURL(file);
        });

        // Insert image at current cursor position
        editor.chain().focus().setImage({ 
          src: dataUrl,
          alt: file.name,
          title: file.name 
        }).run();

        // TODO: Upload to server and replace base64 with URL
        // const uploadedUrl = await uploadToServer(file);
        // editor.chain().focus().setImage({ src: uploadedUrl }).run();
      }
    } catch (error) {
      console.error('Image upload failed:', error);
    } finally {
      uploading = false;
      uploadProgress = 0;
    }
  }

  function handleDragEnter(e: DragEvent) {
    e.preventDefault();
    dragCounter++;
    isDragging = true;
  }

  function handleDragLeave(e: DragEvent) {
    e.preventDefault();
    dragCounter--;
    if (dragCounter === 0) {
      isDragging = false;
    }
  }

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
  }
  // Simplified toolbar actions
  function toggleBold() {
    editor?.chain().focus().toggleBold().run();
  }
  
  function toggleItalic() {
    editor?.chain().focus().toggleItalic().run();
  }
  
  function setAlignment(align: string) {
    editor?.chain().focus().setTextAlign(align).run();
  }
  
  function insertImage() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.multiple = true;
    input.onchange = async (e) => {
      const files = Array.from((e.target as HTMLInputElement)?.files || []);
      if (files.length > 0) {
        await uploadImages(files);
      }
    };
    input.click();
  }

  // AI-powered report generation
  async function generateAIReport(reportType: string) {
    if (!editor) return;
    
    aiGenerating = true;
    aiProgress = 0;
    showAIMenu = false;

    try {
      // Simulate AI generation progress
      const progressInterval = setInterval(() => {
        aiProgress = Math.min(aiProgress + 10, 90);
      }, 200);

      const response = await fetch('/api/ai/generate-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          reportType,
          caseId,
          reportId,
          existingContent: editor.getText(),
          context: {
            wordCount,
            characterCount,
          }
        }),
      });

      clearInterval(progressInterval);
      aiProgress = 100;

      if (!response.ok) {
        throw new Error('Failed to generate report');
      }

      const data = await response.json();
      
      // Insert generated content
      if (data.content) {
        editor.commands.setContent(data.content);
        updateWordCount();
      }

      // Show success notification
      showAISuccess(`${reportTypes.find(t => t.id === reportType)?.name} generated successfully!`);

    } catch (error) {
      console.error('AI generation failed:', error);
      showAIError('Failed to generate report. Please try again.');
    } finally {
      aiGenerating = false;
      aiProgress = 0;
    }
  }

  async function summarizeReport() {
    if (!editor || !editor.getText().trim()) return;

    aiGenerating = true;
    aiProgress = 0;

    try {
      const progressInterval = setInterval(() => {
        aiProgress = Math.min(aiProgress + 15, 80);
      }, 300);

      const response = await fetch('/api/ai/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: editor.getText(), // Use plain text instead of HTML
          type: 'report',
          model: 'gemma3-legal',
        }),
      });

      clearInterval(progressInterval);
      aiProgress = 100;

      if (!response.ok) {
        throw new Error('Failed to summarize report');
      }

      const data = await response.json();
      
      // Insert summary at the beginning
      if (data.summary) {
        const currentContent = editor.getHTML();
        const summaryContent = `
          <div style="background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 8px; padding: 16px; margin-bottom: 24px;">
            <h3 style="color: #0369a1; margin: 0 0 12px 0; font-size: 18px; font-weight: 600;">
              <span style="display: inline-flex; align-items: center gap: 8px;">
                🤖 AI Summary
                <small style="font-size: 12px; font-weight: normal color: #6b7280;">
                  ${data.model} • ${data.processingTime}ms
                </small>
              </span>
            </h3>
            <p style="margin: 0; color: #374151; line-height: 1.6;">${data.summary}</p>
          </div>
          ${currentContent}
        `;
        editor.commands.setContent(summaryContent);
        updateWordCount();
      }

      showAISuccess('Report summarized successfully!');

    } catch (error) {
      console.error('AI summarization failed:', error);
      showAIError('Failed to summarize report. Please try again.');
    } finally {
      aiGenerating = false;
      aiProgress = 0;
    }
  }

  async function analyzeReport() {
    if (!editor || !editor.getText().trim()) return;

    aiGenerating = true;
    aiProgress = 0;

    try {
      const progressInterval = setInterval(() => {
        aiProgress = Math.min(aiProgress + 12, 85);
      }, 250);

      const response = await fetch('/api/ai/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: editor.getText(), // Use plain text instead of HTML
          caseId,
          reportId,
          analysisType: 'comprehensive',
          model: 'gemma3-legal',
        }),
      });

      clearInterval(progressInterval);
      aiProgress = 100;

      if (!response.ok) {
        throw new Error('Failed to analyze report');
      }

      const data = await response.json();
      
      // Insert analysis at the end
      if (data.analysis || data.rawAnalysis) {
        const currentContent = editor.getHTML();
        const analysis = data.analysis || {};
        const analysisContent = `
          ${currentContent}
          <div style="background: #f0fdf4; border: 1px solid #22c55e; border-radius: 8px; padding: 16px; margin-top: 24px;">
            <h3 style="color: #15803d; margin: 0 0 12px 0; font-size: 18px; font-weight: 600;">
              <span style="display: inline-flex; align-items: center gap: 8px;">
                🧠 AI Analysis
                <small style="font-size: 12px; font-weight: normal color: #6b7280;">
                  ${data.model} • ${data.processingTime}ms • ${Math.round((data.metadata?.confidence || 0.8) * 100)}% confidence
                </small>
              </span>
            </h3>
            <div style="color: #374151; line-height: 1.6;">
              ${analysis.keyPoints?.length ? `
                <p><strong>Key Points:</strong></p>
                <ul>
                  ${analysis.keyPoints.map(point => `<li>${point}</li>`).join('')}
                </ul>
              ` : ''}
              
              ${analysis.findings?.length ? `
                <p><strong>Findings:</strong></p>
                <ul>
                  ${analysis.findings.map(finding => `<li>${finding}</li>`).join('')}
                </ul>
              ` : ''}
              
              ${analysis.recommendations?.length ? `
                <p><strong>Recommendations:</strong></p>
                <ul>
                  ${analysis.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
              ` : ''}
              
              ${analysis.risks?.length ? `
                <p><strong>Risk Assessment:</strong></p>
                <ul>
                  ${analysis.risks.map(risk => `<li style="color: #dc2626;">${risk}</li>`).join('')}
                </ul>
              ` : ''}
              
              ${data.rawAnalysis && !analysis.keyPoints?.length ? `
                <div style="white-space: pre-wrap;">${data.rawAnalysis}</div>
              ` : ''}
            </div>
          </div>
        `;
        editor.commands.setContent(analysisContent);
        updateWordCount();
      }

      showAISuccess('Report analyzed successfully!');

    } catch (error) {
      console.error('AI analysis failed:', error);
      showAIError('Failed to analyze report. Please try again.');
    } finally {
      aiGenerating = false;
      aiProgress = 0;
    }
  }

  function showAISuccess(message: string) {
    const indicator = document.createElement('div');
    indicator.textContent = message;
    indicator.className = 'fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg text-sm z-50 shadow-lg';
    document.body.appendChild(indicator);
    setTimeout(() => {
      document.body.removeChild(indicator);
    }, 3000);
  }

  function showAIError(message: string) {
    const indicator = document.createElement('div');
    indicator.textContent = message;
    indicator.className = 'fixed top-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg text-sm z-50 shadow-lg';
    document.body.appendChild(indicator);
    setTimeout(() => {
      document.body.removeChild(indicator);
    }, 4000);
  }

  // Reactive statements
  let state = $derived($editorState)

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

<!-- Lightweight Editor Container -->
<div 
  class="lightweight-editor border border-gray-200 rounded-lg bg-white"
  class:is-dragging={isDragging}
  ondragenter={handleDragEnter}
  ondragleave={handleDragLeave}
  ondragover={handleDragOver}
>
  <!-- Simplified Toolbar -->
  <div class="toolbar bg-gray-50 border-b border-gray-200 p-2 flex items-center gap-2 flex-wrap">
    <!-- Save -->
    <button
      class="toolbar-btn"
      onclick={() => saveContent()}
      title="Save (Ctrl+S)"
    >
      <Save size="16" />
    </button>

    <div class="toolbar-separator"></div>

    <!-- Undo/Redo -->
    <button
      class="toolbar-btn"
      class:disabled={!state.canUndo}
      onclick={() => editor?.commands.undo()}
      title="Undo (Ctrl+Z)"
    >
      <Undo size="16" />
    </button>
    <button
      class="toolbar-btn"
      class:disabled={!state.canRedo}
      onclick={() => editor?.commands.redo()}
      title="Redo (Ctrl+Shift+Z)"
    >
      <Redo size="16" />
    </button>

    <div class="toolbar-separator"></div>

    <!-- Text Formatting -->
    <button
      class="toolbar-btn"
      class:active={state.isBold}
      onclick={() => toggleBold()}
      title="Bold (Ctrl+B)"
    >
      <Bold size="16" />
    </button>
    <button
      class="toolbar-btn"
      class:active={state.isItalic}
      onclick={() => toggleItalic()}
      title="Italic (Ctrl+I)"
    >
      <Italic size="16" />
    </button>

    <div class="toolbar-separator"></div>

    <!-- Alignment -->
    <button
      class="toolbar-btn"
      class:active={state.currentAlignment === "left"}
      onclick={() => setAlignment("left")}
      title="Align Left"
    >
      <AlignLeft size="16" />
    </button>
    <button
      class="toolbar-btn"
      class:active={state.currentAlignment === "center"}
      onclick={() => setAlignment("center")}
      title="Align Center"
    >
      <AlignCenter size="16" />
    </button>
    <button
      class="toolbar-btn"
      class:active={state.currentAlignment === "right"}
      onclick={() => setAlignment("right")}
      title="Align Right"
    >
      <AlignRight size="16" />
    </button>

    <div class="toolbar-separator"></div>

    <!-- Lists -->
    <button
      class="toolbar-btn"
      class:active={state.isList}
      onclick={() => editor?.chain().focus().toggleBulletList().run()}
      title="Bullet List"
    >
      <List size="16" />
    </button>
    <button
      class="toolbar-btn"
      class:active={state.isOrderedList}
      onclick={() => editor?.chain().focus().toggleOrderedList().run()}
      title="Numbered List"
    >
      <ListOrdered size="16" />
    </button>
    <button
      class="toolbar-btn"
      class:active={state.isQuote}
      onclick={() => editor?.chain().focus().toggleBlockquote().run()}
      title="Quote"
    >
      <Quote size="16" />
    </button>

    <div class="toolbar-separator"></div>

    <!-- Image Upload -->
    <button
      class="toolbar-btn"
      onclick={() => insertImage()}
      title="Insert Image (supports drag & drop)"
    >
      <ImageIcon size="16" />
    </button>

    <div class="toolbar-separator"></div>

    <!-- AI Features -->
    <AIDropdown
      {disabled}
      onReportGenerate={generateAIReport}
      onSummarize={summarizeReport}
      onAnalyze={analyzeReport}
      hasContent={editor ? editor.getText().trim().length > 0 : false}
      isGenerating={aiGenerating}
    />

    <div class="flex-1"></div>

    <!-- Word Count -->
    <div class="text-sm text-gray-600">
      {wordCount} words
    </div>
  </div>

  <!-- Upload Progress -->
  {#if uploading}
    <div class="upload-progress bg-blue-50 border-b border-blue-200 p-2">
      <div class="flex items-center gap-2">
        <Upload size="16" class="text-blue-600" />
        <div class="flex-1">
          <div class="text-sm text-blue-700">Uploading images...</div>
          <div class="w-full bg-blue-200 rounded-full h-2 mt-1">
            <div 
              class="bg-blue-600 h-2 rounded-full transition-all duration-300" 
              style="width: {uploadProgress}%"
            ></div>
          </div>
        </div>
        <span class="text-sm text-blue-600">{uploadProgress}%</span>
      </div>
    </div>
  {/if}

  <!-- AI Progress -->
  {#if aiGenerating}
    <div class="ai-progress bg-purple-50 border-b border-purple-200 p-2">
      <div class="flex items-center gap-2">
        <Sparkles size="16" class="text-purple-600 animate-pulse" />
        <div class="flex-1">
          <div class="text-sm text-purple-700">AI is working on your request...</div>
          <div class="w-full bg-purple-200 rounded-full h-2 mt-1">
            <div 
              class="bg-purple-600 h-2 rounded-full transition-all duration-300" 
              style="width: {aiProgress}%"
            ></div>
          </div>
        </div>
        <span class="text-sm text-purple-600">{aiProgress}%</span>
      </div>
    </div>
  {/if}

  <!-- Drag Overlay -->
  {#if isDragging}
    <div class="drag-overlay absolute inset-0 bg-blue-50 bg-opacity-90 flex items-center justify-center z-10">
      <div class="text-center p-8">
        <ImageIcon size="48" class="mx-auto text-blue-600 mb-4" />
        <p class="text-lg font-medium text-blue-800">Drop images here</p>
        <p class="text-sm text-blue-600">PNG, JPG, GIF up to 10MB each</p>
      </div>
    </div>
  {/if}

  <!-- Editor Content -->
  <div class="editor-container bg-white min-h-[400px] relative">
    <div bind:this={editorElement} class="editor-content"></div>
  </div>

  <!-- Auto-save Indicator -->
  {#if autosave}
    <div class="status-bar bg-gray-50 border-t border-gray-200 px-3 py-1 text-xs text-gray-500 flex justify-between">
      <span>Auto-save enabled</span>
      <span>{characterCount} characters</span>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  .lightweight-editor {
    position: relative
    max-width: 100%;
  }

  .lightweight-editor.is-dragging {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }

  .toolbar-btn {
    padding: 0.5rem;
    border-radius: 0.375rem;
    border: none
    background: transparent
    cursor: pointer
    display: flex
    align-items: center
    justify-content: center
    min-width: 32px;
    height: 32px;
    transition: all 0.2s;
  }

  .toolbar-btn:hover {
    background: #e5e7eb;
  }

  .toolbar-btn:disabled,
  .toolbar-btn.disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .toolbar-btn.active {
    background: #dbeafe;
    color: #2563eb;
  }

  .toolbar-btn.ai-btn:hover {
    background: #faf5ff;
  }

  .toolbar-btn.ai-btn.active {
    background: #f3e8ff;
    color: #7c3aed;
  }

  .toolbar-separator {
    width: 1px;
    height: 1.25rem;
    background: #d1d5db;
    margin: 0 0.25rem;
  }

  .ai-menu {
    animation: fadeIn 0.15s ease-out;
  }

  .ai-menu-item:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .ai-menu-item:disabled:hover {
    background: transparent
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(-4px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .drag-overlay {
    border-radius: 0.5rem;
    border: 2px dashed #3b82f6;
  }

  .editor-content {
    padding: 1.5rem;
    min-height: 400px;
  }

  /* TipTap Editor Styles */
  :global(.ProseMirror) {
    outline: none
    line-height: 1.6;
    font-size: 16px;
  }

  :global(.ProseMirror p.is-editor-empty:first-child::before) {
    color: #9ca3af;
    content: attr(data-placeholder);
    float: left
    height: 0;
    pointer-events: none
  }

  :global(.ProseMirror blockquote) {
    border-left: 4px solid #e5e7eb;
    padding-left: 1rem;
    margin: 1rem 0;
    font-style: italic
    color: #6b7280;
  }

  :global(.ProseMirror ul),
  :global(.ProseMirror ol) {
    padding-left: 1.5rem;
    margin: 0.5rem 0;
  }

  :global(.ProseMirror img) {
    max-width: 100%;
    height: auto
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
  }

  :global(.ProseMirror img:hover) {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }

  /* Focus styles */
  :global(.ProseMirror-focused) {
    outline: none
  }

  /* Selection styles */
  :global(.ProseMirror-selectednode) {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
  }
</style>
