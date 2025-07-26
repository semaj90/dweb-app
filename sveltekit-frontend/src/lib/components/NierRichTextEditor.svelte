<!-- NieR: Automata Themed Rich Text Editor using bits-ui -->
<script lang="ts">
  import { Editor } from "@tiptap/core";
  import StarterKit from "@tiptap/starter-kit";
  import { onMount } from "svelte";
  import { Button } from "$lib/components/ui/button";
  import { Select } from "$lib/components/ui/select";
  import { Separator } from "$lib/components/ui/separator";
  
  export let content = "";
  export let placeholder = "Initialize data input...";
  
  let editor: Editor | null = null;
  let editorElement: HTMLElement;
  
  onMount(() => {
    editor = new Editor({
      element: editorElement,
      extensions: [StarterKit],
      content,
      editorProps: {
        attributes: {
          class: "nier-editor-content focus:outline-none"
        }
      }
    });
  });
  
  const fontOptions = [
    { value: "JetBrains Mono", label: "JetBrains Mono" },
    { value: "Courier New", label: "Courier New" },
    { value: "Inter", label: "Inter" }
  ];
</script>

<div class="nier-panel">
  <!-- Toolbar -->
  <div class="nier-toolbar">
    <div class="nier-toolbar-group">
      <Button 
        variant="ghost" 
        size="sm" 
        class="nier-toolbar-btn"
        on:click={() => editor?.commands.undo()}
      >
        ↶
      </Button>
      <Button 
        variant="ghost" 
        size="sm" 
        class="nier-toolbar-btn"
        on:click={() => editor?.commands.redo()}
      >
        ↷
      </Button>
    </div>
    
    <Separator orientation="vertical" class="nier-toolbar-separator" />
    
    <div class="nier-toolbar-group">
      <Select.Root>
        <Select.Trigger class="nier-select">
          <Select.Value placeholder="Font" />
        </Select.Trigger>
        <Select.Content class="nier-dropdown-content">
          {#each fontOptions as font}
            <Select.Item 
              value={font.value} 
              class="nier-dropdown-item"
            >
              {font.label}
            </Select.Item>
          {/each}
        </Select.Content>
      </Select.Root>
    </div>
    
    <Separator orientation="vertical" class="nier-toolbar-separator" />
    
    <div class="nier-toolbar-group">
      <Button 
        variant="ghost" 
        size="sm" 
        class="nier-toolbar-btn"
        class:active={editor?.isActive('bold')}
        on:click={() => editor?.chain().focus().toggleBold().run()}
      >
        <strong>B</strong>
      </Button>
      <Button 
        variant="ghost" 
        size="sm" 
        class="nier-toolbar-btn"
        class:active={editor?.isActive('italic')}
        on:click={() => editor?.chain().focus().toggleItalic().run()}
      >
        <em>I</em>
      </Button>
    </div>
  </div>
  
  <!-- Editor -->
  <div class="nier-editor">
    <div bind:this={editorElement}></div>
  </div>
  
  <!-- Status Bar -->
  <div class="nier-status-bar">
    <span>STATUS: OPERATIONAL</span>
    <span>DATA INTEGRITY: 100%</span>
  </div>
</div>

<style>
  /* @unocss-include */
  @import '../styles/nier-design-system.css';
</style>
