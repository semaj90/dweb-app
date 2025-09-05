# Rich Text Editor Recommendations for SvelteKit

## Requirements Met
- ✅ Lightweight & MIT licensed
- ✅ No Virtual DOM (works with Svelte's real DOM)
- ✅ SvelteKit compatible
- ✅ Works with UnoCSS
- ✅ Compatible with bits-ui/melt-ui

## Top Recommendations

### 1. **Svelte-TipTap** (RECOMMENDED)
```bash
npm install @tiptap/core @tiptap/starter-kit @tiptap/extension-placeholder
npm install svelte-tiptap
```

**Pros:**
- Built specifically for Svelte
- Uses TipTap (MIT license) 
- Excellent UnoCSS integration
- No VDOM conflicts
- Extensible with plugins
- Great for legal documents (table support, document structure)

**Example Usage:**
```typescript
// lib/components/RichTextEditor.svelte
<script lang="ts">
  import { Editor } from 'svelte-tiptap';
  import StarterKit from '@tiptap/starter-kit';
  import Placeholder from '@tiptap/extension-placeholder';
  
  export let content = '';
  
  const extensions = [
    StarterKit,
    Placeholder.configure({
      placeholder: 'Write your legal document...'
    })
  ];
</script>

<div class="prose max-w-none">
  <Editor bind:content {extensions} class="min-h-[400px] p-4" />
</div>
```

### 2. **Milkdown** 
```bash
npm install @milkdown/core @milkdown/preset-commonmark @milkdown/theme-nord
```

**Pros:**
- Plugin-based architecture
- Excellent TypeScript support
- WYSIWYG + Markdown modes
- MIT licensed
- Great for structured documents

### 3. **Quill with Svelte Wrapper**
```bash
npm install quill svelte-quill
```

**Pros:**
- Battle-tested in production
- Extensive customization
- Good performance
- Active community

## Integration with Your Stack

### UnoCSS Integration
```typescript
// In your editor component
<div class="
  prose prose-lg 
  border border-gray-200 rounded-lg
  focus-within:border-blue-500 
  transition-colors
">
  <Editor {extensions} />
</div>
```

### bits-ui/melt-ui Integration
```typescript
// Custom toolbar with bits-ui components
<script>
  import { Button } from '$lib/components/ui/button';
  import { Toggle } from '$lib/components/ui/toggle';
  
  let editor;
  
  const toggleBold = () => editor?.chain().focus().toggleBold().run();
  const toggleItalic = () => editor?.chain().focus().toggleItalic().run();
</script>

<div class="border-b p-2 flex gap-2">
  <Toggle pressed={editor?.isActive('bold')} onPressedChange={toggleBold}>
    Bold
  </Toggle>
  <Toggle pressed={editor?.isActive('italic')} onPressedChange={toggleItalic}>
    Italic
  </Toggle>
</div>
```

## Legal Document Features

For your legal app, consider these extensions:

```typescript
import { Table } from '@tiptap/extension-table';
import { TableRow } from '@tiptap/extension-table-row';
import { TableHeader } from '@tiptap/extension-table-header';
import { TableCell } from '@tiptap/extension-table-cell';
import { TextStyle } from '@tiptap/extension-text-style';
import { Color } from '@tiptap/extension-color';
import { Highlight } from '@tiptap/extension-highlight';

const extensions = [
  StarterKit,
  Table.configure({
    resizable: true,
  }),
  TableRow,
  TableHeader,
  TableCell,
  TextStyle,
  Color,
  Highlight.configure({
    multicolor: true,
  }),
];
```

## Final Recommendation

**Use Svelte-TipTap** because:
1. Purpose-built for Svelte (no VDOM issues)
2. Excellent performance and bundle size
3. Perfect UnoCSS integration
4. Extensible for legal document features
5. Active development and community
6. MIT licensed
7. Works seamlessly with your current tech stack

Would you like me to create a complete RichTextEditor.svelte component with your styling system?
