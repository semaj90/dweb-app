<script lang="ts">
  interface Props {
    onchange?: (event?: any) => void;
  }
  let {
    content = [],
    height = '400px',
    placeholder = 'Start writing...'
  } = $props();



    import Editor from '@toast-ui/editor';
  import '@toast-ui/editor/dist/toastui-editor.css';
  import type { ContentNode } from '$lib/logic/HistoryManager';

  
        
  let editorElement: HTMLElement
  let editor: Editor
  let isInitialized = false;

  // Convert ContentNode array to markdown
  function contentToMarkdown(nodes: ContentNode[]): string {
    const nodeToMd = (node: ContentNode): string => {
      if (node.text) {
        // Apply formatting
        let text = node.text;
        if (node.bold) text = `**${text}**`;
        if (node.italic) text = `*${text}*`;
        if (node.color) text = `<span style="color: ${node.color}">${text}</span>`;
        if (node.fontSize) text = `<span style="font-size: ${node.fontSize}">${text}</span>`;
        return text;
}
      if (node.children) {
        const childText = node.children.map(nodeToMd).join('');
        
        switch (node.type) {
          case 'paragraph':
            return childText + '\n\n';
          case 'heading':
            const level = node.level || 1;
            return '#'.repeat(level) + ' ' + childText + '\n\n';
          case 'list':
            return childText;
          case 'list-item':
            return '- ' + childText + '\n';
          case 'blockquote':
            return '> ' + childText + '\n\n';
          case 'code':
            return '`' + childText + '`';
          case 'code-block':
            return '```\n' + childText + '\n```\n\n';
          case 'link':
            return `[${childText}](${node.url || '#'})`;
          case 'image':
            return `![${node.alt || ''}](${node.url || ''})`;
          default:
            return childText;
}}
      return '';
    };

    return nodes.map(nodeToMd).join('');
}
  // Convert markdown to ContentNode array (simplified)
  function markdownToContent(markdown: string): ContentNode[] {
    if (!markdown.trim()) {
      return [{ type: 'paragraph', children: [{ type: 'text', text: '' }] }];
}
    // Basic markdown parsing - in production, use a proper parser
    const lines = markdown.split('\n');
    const nodes: ContentNode[] = [];
    let currentParagraph: ContentNode | null = null;

    for (const line of lines) {
      if (line.trim() === '') {
        if (currentParagraph) {
          nodes.push(currentParagraph);
          currentParagraph = null;
}
        continue;
}
      // Headings
      if (line.startsWith('#')) {
        const level = line.match(/^#+/)?.[0].length || 1;
        const text = line.replace(/^#+\s*/, '');
        nodes.push({
          type: 'heading',
          level,
          children: [{ type: 'text', text }]
        });
        continue;
}
      // Lists
      if (line.startsWith('- ') || line.startsWith('* ')) {
        const text = line.replace(/^[-*]\s*/, '');
        nodes.push({
          type: 'list-item',
          children: [{ type: 'text', text }]
        });
        continue;
}
      // Blockquotes
      if (line.startsWith('> ')) {
        const text = line.replace(/^>\s*/, '');
        nodes.push({
          type: 'blockquote',
          children: [{ type: 'text', text }]
        });
        continue;
}
      // Regular paragraph
      if (!currentParagraph) {
        currentParagraph = {
          type: 'paragraph',
          children: []
        };
}
      // Basic inline formatting
      let text = line;
      const textNode: ContentNode = { type: 'text', text };

      // Bold
      if (text.includes('**')) {
        textNode.bold = true;
        text = text.replace(/\*\*(.*?)\*\*/g, '$1');
        textNode.text = text;
}
      // Italic
      if (text.includes('*') && !textNode.bold) {
        textNode.italic = true;
        text = text.replace(/\*(.*?)\*/g, '$1');
        textNode.text = text;
}
      currentParagraph.children!.push(textNode);
}
    if (currentParagraph) {
      nodes.push(currentParagraph);
}
    return nodes.length > 0 ? nodes : [{ type: 'paragraph', children: [{ type: 'text', text: '' }] }];
}
  onMount(() => {
    editor = new Editor({
      el: editorElement,
      initialValue: contentToMarkdown(content),
      previewStyle: 'vertical',
      height: height,
      initialEditType: 'markdown',
      placeholder: placeholder,
      usageStatistics: false,
      toolbarItems: [
        ['heading', 'bold', 'italic', 'strike'],
        ['hr', 'quote'],
        ['ul', 'ol', 'task', 'indent', 'outdent'],
        ['table', 'image', 'link'],
        ['code', 'codeblock'],
        ['scrollSync']
      ],
      hooks: {
        addImageBlobHook: (blob: Blob, callback: (url: string, alt?: string) => void) => {
          // Handle image upload
          const reader = new FileReader();
          reader.onload = (e) => {
            callback(e.target?.result as string, 'Uploaded image');
          };
          reader.readAsDataURL(blob);
}}
    });

    // Listen for content changes
    editor.on('change', () => {
      const markdown = editor.getMarkdown();
      const newContent = markdownToContent(markdown);
      onchange?.();
    });

    isInitialized = true;
  });

  onDestroy(() => {
    if (editor) {
      editor.destroy();
}
  });

  // Reactive update when content prop changes
  $effect(() => { if (editor && isInitialized && content) {
    const currentMarkdown = editor.getMarkdown();
    const newMarkdown = contentToMarkdown(content);
    
    if (currentMarkdown !== newMarkdown) {
      editor.setMarkdown(newMarkdown);
}}
  // Expose methods for parent component
  export function setContent(newContent: ContentNode[]) {
    if (editor) {
      editor.setMarkdown(contentToMarkdown(newContent));
}}
  export function getContent(): ContentNode[] {
    if (editor) {
      return markdownToContent(editor.getMarkdown());
}
    return content;
}
  export function getMarkdown(): string {
    return editor ? editor.getMarkdown() : '';
}
  export function getHTML(): string {
    return editor ? editor.getHTML() : '';
}
  export function insertText(text: string) {
    if (editor) {
      editor.insertText(text);
}}
  export function getSelectedText(): string {
    if (editor) {
      return editor.getSelectedText() || '';
}
    return '';
}
  export function focus() {
    if (editor) {
      editor.focus();
}}
  // Formatting methods
  export function toggleMark(mark: string) {
    if (!editor) return;
    
    const selectedText = editor.getSelectedText();
    if (!selectedText) return;

    let formattedText = selectedText;
    
    switch (mark) {
      case 'bold':
        formattedText = `**${selectedText}**`;
        break;
      case 'italic':
        formattedText = `*${selectedText}*`;
        break;
      case 'code':
        formattedText = `\`${selectedText}\``;
        break;
}
    editor.replaceSelection(formattedText);
}
  export function addMark(mark: string, value: string) {
    if (!editor) return;
    
    const selectedText = editor.getSelectedText();
    if (!selectedText) return;

    let formattedText = selectedText;

    switch (mark) {
      case 'color':
        formattedText = `<span style="color: ${value}">${selectedText}</span>`;
        break;
      case 'fontSize':
        formattedText = `<span style="font-size: ${value}">${selectedText}</span>`;
        break;
}
    editor.replaceSelection(formattedText);
}
  export function insertNode(node: any) {
    if (!editor) return;

    switch (node.type) {
      case 'image':
        editor.insertText(`![${node.alt || ''}](${node.url})`);
        break;
      case 'link':
        editor.insertText(`[${node.text || 'Link'}](${node.url})`);
        break;
      case 'heading':
        const level = '#'.repeat(node.level || 1);
        editor.insertText(`\n${level} ${node.text || 'Heading'}\n`);
        break;
}}
</script>

<div bind:this={editorElement} class="space-y-4"></div>

<style>
  /* @unocss-include */
  .advanced-editor {
    width: 100%;
    height: 100%;
}
  :global(.toastui-editor-defaultUI) {
    border: none !important;
}
  :global(.toastui-editor-toolbar) {
    background-color: #f8fafc !important;
    border-bottom: 1px solid #e5e7eb !important;
}
  :global(.toastui-editor-md-container) {
    background-color: white !important;
}
  :global(.toastui-editor-preview-container) {
    background-color: #fafafa !important;
}
  :global(.toastui-editor-contents) {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}
  :global(.toastui-editor-contents h1) {
    color: #1f2937 !important;
    font-weight: 700 !important;
}
  :global(.toastui-editor-contents h2) {
    color: #374151 !important;
    font-weight: 600 !important;
}
  :global(.toastui-editor-contents h3) {
    color: #4b5563 !important;
    font-weight: 600 !important;
}
  :global(.toastui-editor-contents p) {
    color: #1f2937 !important;
    margin-bottom: 16px !important;
}
  :global(.toastui-editor-contents blockquote) {
    border-left: 4px solid #3b82f6 !important;
    background-color: #f1f5f9 !important;
    padding: 12px 16px !important;
    margin: 16px 0 !important;
}
  :global(.toastui-editor-contents code) {
    background-color: #f1f5f9 !important;
    color: #be185d !important;
    padding: 2px 4px !important;
    border-radius: 4px !important;
}
  :global(.toastui-editor-contents pre) {
    background-color: #1f2937 !important;
    color: #f9fafb !important;
    padding: 16px !important;
    border-radius: 8px !important;
    overflow-x: auto !important;
}
  :global(.toastui-editor-contents table) {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 16px 0 !important;
}
  :global(.toastui-editor-contents th),
  :global(.toastui-editor-contents td) {
    border: 1px solid #e5e7eb !important;
    padding: 8px 12px !important;
    text-align: left !important;
}
  :global(.toastui-editor-contents th) {
    background-color: #f8fafc !important;
    font-weight: 600 !important;
}
</style>
