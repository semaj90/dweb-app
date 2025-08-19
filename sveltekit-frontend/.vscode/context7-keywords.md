# Context7 Keywords for Copilot

Use these keywords to trigger Context7 MCP tools and patterns:

## Core Tools

- #context7 - Access Context7 documentation
- #get-library-docs sveltekit2 - Get SvelteKit 2 docs
- #get-library-docs bitsui - Get Bits UI v2 docs
- #get-library-docs meltui - Get Melt UI docs
- #get-library-docs xstate - Get XState docs
- #resolve-library-id - Find library identifiers
- #directory_tree - Show project structure
- #read_multiple_files - Read multiple files
- #microsoft-docs - Access Microsoft documentation

## Memory Keywords

- #memory - Access memory system
- #create_entities - Create knowledge graph entities
- #create_relations - Create entity relationships
- #read_graph - Read knowledge graph
- #search_nodes - Search memory nodes
- #duplicates - Find duplicate variables/props
- #props - Analyze prop destructuring
- #consolidation - Get prop consolidation suggestions

## Svelte 5 + Bits UI v2 + Melt UI Patterns

### Component Composition
- Use `mergeProps` from bits-ui for component composition
- Use `{#snippet child({ props })}` for prop forwarding
- Props are merged automatically with internal component props
- Event handlers are chained in order
- Classes merged with `clsx` and `twMerge`

### Prop Patterns
```typescript
// ✅ Consolidated prop destructuring
interface Props {
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'destructive';
  children?: import('svelte').Snippet;
}
let { size = 'md', variant = 'default', children }: Props = $props();
```

### Bits UI Integration
```typescript
import { Button as ButtonPrimitive } from 'bits-ui';
import { mergeProps } from 'bits-ui';

const buttonProps = mergeProps(restProps, {
  class: cn(buttonVariants({ variant, size }), className)
});
```

### Melt UI Integration

**Builder Pattern:**
```typescript
import { Toggle } from "melt/builders";

const toggle = new Toggle({
  value: () => value,
  onValueChange: (v) => (value = v)
});
```

**Component Pattern:**
```typescript
import { Dialog } from "melt/components";

<Dialog bind:open>
  {#snippet children(dialog)}
    <button {...dialog.trigger}>Open</button>
    {#if dialog.open}
      <div {...dialog.overlay}>
        <div {...dialog.content}>Content</div>
      </div>
    {/if}
  {/snippet}
</Dialog>
```

### Class Merging Utility
```typescript
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

## Common Fixes

### Duplicate Variable Errors
- Consolidate multiple `let { prop } = $props()` declarations
- Use single interface with all props
- Remove `$bindable()` for regular props

### Export Function Errors  
- Convert `export function` to regular `function` inside script blocks
- Use module context (`<script context="module">`) for actual exports

### Missing Brackets/EOF
- Check for unclosed `$effect(() => { })` blocks
- Ensure all template literals are properly closed
- Match opening/closing brackets in conditional statements

### Class Merging
- Use `cn()` utility for proper class merging
- Import `clsx` and `twMerge` for component libraries
- Pass `class` prop as `className` to avoid conflicts
