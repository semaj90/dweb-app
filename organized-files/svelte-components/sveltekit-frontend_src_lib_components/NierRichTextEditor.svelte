<!-- NieR: Automata Themed Rich Text Editor using bits-ui -->
<script lang="ts">
  interface Props {
    content?: any;
    placeholder?: any;
  }
  let {
    content = "",
    placeholder = "Initialize data input..."
  } = $props();



  import { Editor } from "@tiptap/core";
  import StarterKit from "@tiptap/starter-kit";
  import { onMount } from "svelte";
  import { Button } from "$lib/components/ui/button";
  import { Select } from "$lib/components/ui/select";
  import { Separator } from "$lib/components/ui/separator";
  
      
  let editor: Editor | null = null;
  let editorElement: HTMLElement
  
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
        onclick={() => editor?.commands.undo()}
      >
        ↶
      </Button>
      <Button 
        variant="ghost" 
        size="sm" 
        class="nier-toolbar-btn"
        onclick={() => editor?.commands.redo()}
      >
        ↷
      </Button>
    </div>
    
    <Separator orientation="vertical" class="nier-toolbar-separator" />
    
    <div class="nier-toolbar-group">
      <SelectRoot>
        <SelectTrigger class="nier-select">
          <SelectValue placeholder="Font" />
        </SelectTrigger>
        <SelectContent class="nier-dropdown-content">
          {#each fontOptions as font}
            <SelectItem 
              value={font.value} 
              class="nier-dropdown-item"
            >
              {font.label}
            </SelectItem>
          {/each}
        </SelectContent>
      </SelectRoot>
    </div>
    
    <Separator orientation="vertical" class="nier-toolbar-separator" />
    
    <div class="nier-toolbar-group">
      <Button 
        variant="ghost" 
        size="sm" 
        class="nier-toolbar-btn"
        class:active={editor?.isActive('bold')}
        onclick={() => editor?.chain().focus().toggleBold().run()}
      >
        <strong>B</strong>
      </Button>
      <Button 
        variant="ghost" 
        size="sm" 
        class="nier-toolbar-btn"
        class:active={editor?.isActive('italic')}
        onclick={() => editor?.chain().focus().toggleItalic().run()}
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
