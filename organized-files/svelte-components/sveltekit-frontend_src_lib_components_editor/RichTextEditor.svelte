<script lang="ts">
  interface Props {
    height?: any;
    disabled?: any;
    placeholder?: any;
  }
  let {
    height = 500,
    disabled = false,
    placeholder = 'Begin writing your report...'
  } = $props();



  import { onMount, onDestroy } from 'svelte';
  import Editor from '@tinymce/tinymce-svelte';
  import { report, reportActions, editorState } from '$lib/stores/report';
  
  export const initialValue = '';
        
  let editorInstance: any
  let isInitialized = false;
  
  // TinyMCE configuration
  const editorConfig = {
    height,
    menubar: true,
    plugins: [
      'advlist', 'autolink', 'lists', 'link', 'image', 'charmap', 'preview',
      'anchor', 'searchreplace', 'visualblocks', 'code', 'fullscreen',
      'insertdatetime', 'media', 'table', 'help', 'wordcount', 'save',
      'autosave', 'paste', 'textpattern', 'emoticons', 'hr', 'pagebreak',
      'nonbreaking', 'template', 'toc', 'quickbars', 'codesample'
    ],
    toolbar: `
      undo redo | blocks fontfamily fontsize | 
      bold italic underline strikethrough | 
      link image media table | 
      alignleft aligncenter alignright alignjustify | 
      outdent indent | numlist bullist | 
      forecolor backcolor removeformat | 
      pagebreak | charmap emoticons | 
      fullscreen preview save | help
    `,
    content_style: `
      body { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
        font-size: 14px; 
        line-height: 1.6;
        color: #374151;
        background: #ffffff;
}
      p { margin-bottom: 1em; }
      h1, h2, h3, h4, h5, h6 { 
        font-weight: 600; 
        margin-top: 1.5em; 
        margin-bottom: 0.5em; 
        color: #111827;
}
      blockquote {
        border-left: 4px solid #3B82F6;
        padding-left: 1em;
        margin: 1em 0;
        font-style: italic
        background: #F8FAFC;
}
      code {
        background: #F1F5F9;
        padding: 0.2em 0.4em;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9em;
}
      pre {
        background: #1E293B;
        color: #E2E8F0;
        padding: 1em;
        border-radius: 6px;
        overflow-x: auto
}
      table {
        border-collapse: collapse
        width: 100%;
        margin: 1em 0;
}
      th, td {
        border: 1px solid #D1D5DB;
        padding: 0.5em;
        text-align: left
}
      th {
        background: #F9FAFB;
        font-weight: 600;
}
    `,
    placeholder,
    resize: true,
    autosave_ask_before_unload: true,
    autosave_interval: '30s',
    autosave_prefix: 'report-autosave-',
    quickbars_selection_toolbar: 'bold italic | quicklink h2 h3 blockquote quickimage quicktable',
    quickbars_insert_toolbar: 'quickimage quicktable | hr pagebreak',
    contextmenu: 'link image table',
    paste_data_images: true,
    paste_as_text: false,
    paste_webkit_styles: 'color font-size',
    smart_paste: true,
    // Custom save button behavior
    save_onsavecallback: () => {
      reportActions.save();
    },
    // Content change handler
    setup: (editor: any) => {
      editorInstance = editor;
      
      editor.on('init', () => {
        isInitialized = true;
        editorState.update(s => ({ ...s, isEditing: true }));
      });
      
      editor.on('input change', () => {
        if (isInitialized) {
          const content = editor.getContent();
          reportActions.updateContent(content);
          
          // Update word count
          const wordCount = editor.plugins.wordcount?.getCount() || 0;
          editorState.update(s => ({ ...s, wordCount }));
}
      });
      
      editor.on('selectionchange', () => {
        const selectedText = editor.selection.getContent({ format: 'text' });
        editorState.update(s => ({ ...s, selectedText }));
      });
      
      editor.on('focus', () => {
        editorState.update(s => ({ ...s, isEditing: true }));
      });
      
      editor.on('blur', () => {
        editorState.update(s => ({ ...s, isEditing: false }));
      });
}
  };
  
  // Reactive updates
  $effect(() => { if (editorInstance && $report.content !== editorInstance.getContent()) {
    editorInstance.setContent($report.content);
}
  // Custom methods
  export const insertContent = (content: string) => {
    if (editorInstance) {
      editorInstance.insertContent(content);
}
  };
  
  export const insertEvidence = (evidence: any) => {
    const evidenceHtml = `
      <div class="space-y-4" data-evidence-id="${evidence.id}">
        <div class="space-y-4">
          <strong>${evidence.title}</strong>
          <span class="space-y-4">(${evidence.evidenceType || evidence.type})</span>
        </div>
        ${evidence.description ? `<p class="space-y-4">${evidence.description}</p>` : ''}
        ${evidence.url ? `<a href="${evidence.url}" target="_blank">View Evidence</a>` : ''}
      </div>
    `;
    insertContent(evidenceHtml);
  };
  
  export const getWordCount = () => {
    return editorInstance?.plugins.wordcount?.getCount() || 0;
  };
  
  export const getCharCount = () => {
    return editorInstance?.plugins.wordcount?.getCharacterCount() || 0;
  };
  
  onDestroy(() => {
    editorState.update(s => ({ ...s, isEditing: false }));
  });
</script>

<div class="space-y-4">
  <Editor
    {disabled}
    bind:value={$report.content}
    conf={editorConfig}
    onchange={(e: any) => reportActions.updateContent(e.detail.level.content)}
  />
</div>

<style>
  /* @unocss-include */
  .tinymce-container {
    position: relative
    width: 100%;
}
  :global(.tox) {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
  :global(.tox-toolbar) {
    background: #F8FAFC !important;
    border-bottom: 1px solid #E2E8F0 !important;
}
  :global(.tox-menubar) {
    background: #FFFFFF !important;
    border-bottom: 1px solid #E2E8F0 !important;
}
  :global(.tox-edit-area) {
    border: none !important;
}
  :global(.tox-edit-area__iframe) {
    background: #FFFFFF !important;
}
  /* Evidence block styling */
  :global(.evidence-block) {
    background: #EFF6FF;
    border: 1px solid #DBEAFE;
    border-left: 4px solid #3B82F6;
    padding: 1em;
    margin: 1em 0;
    border-radius: 6px;
}
  :global(.evidence-header) {
    display: flex
    align-items: center
    gap: 0.5em;
    margin-bottom: 0.5em;
}
  :global(.evidence-type) {
    background: #3B82F6;
    color: white
    padding: 0.2em 0.5em;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 500;
}
  :global(.evidence-description) {
    margin: 0.5em 0;
    color: #4B5563;
    font-style: italic
}
  /* Dark theme support */
  :global([data-theme="dark"]) :global(.tox) {
    --tox-collection-toolbar-button-active-background: #374151;
    --tox-collection-toolbar-button-hover-background: #4B5563;
}
</style>
