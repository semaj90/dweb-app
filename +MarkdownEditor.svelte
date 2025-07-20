<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Editor from '@toast-ui/editor';
  import '@toast-ui/editor/dist/toastui-editor.css';
  import { marked } from 'marked';

  export let content: string = '';
  export let previewStyle: 'tab' | 'vertical' = 'vertical';
  export let height: string = '500px';
  export let initialEditType: 'markdown' | 'wysiwyg' = 'markdown';
  export let showToolbar: boolean = true;
  export let readOnly: boolean = false;

  let editorElement: HTMLElement;
  let editor: Editor;

  onMount(() => {
    editor = new Editor({
      el: editorElement,
      initialValue: content,
      previewStyle: previewStyle,
      height: height,
      initialEditType: initialEditType,
      toolbarItems: showToolbar ? undefined : [],
      usageStatistics: false,
      viewer: readOnly, // Use viewer mode for readOnly
      hooks: {
        // Example: You can add custom hooks here if needed
      }
    });

    if (!readOnly) {
      editor.on('change', () => {
        content = editor.getMarkdown();
      });
    }
  });

  onDestroy(() => {
    if (editor) {
      editor.destroy();
    }
  });

  // Reactive statement to update editor content if 'content' prop changes externally
  $: if (editor && content !== editor.getMarkdown() && !readOnly) {
    editor.setMarkdown(content);
  }

  // Reactive statement to update editor readOnly status
  $: if (editor) {
    if (readOnly) {
      editor.setMarkdown(content); // Ensure content is set before switching to viewer
      editor.changeMode('viewer');
    } else {
      editor.changeMode(initialEditType);
    }
  }
</script>

<div bind:this={editorElement}></div>

<style>
  /* You can add custom styles here if needed */
</style>