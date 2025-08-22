# <¨ Modern SvelteKit 2 + Svelte 5 Design System Guide

## <× **Golden Ratio & Design Principles**

### **Golden Ratio Spacing System (Æ = 1.618)**
```css
/* Golden ratio-based spacing scale */
:root {
  --golden-base: 1rem;
  --golden-xs: calc(var(--golden-base) / 2.618);    /* ~0.382rem */
  --golden-sm: calc(var(--golden-base) / 1.618);    /* ~0.618rem */
  --golden-md: var(--golden-base);                  /* 1rem */
  --golden-lg: calc(var(--golden-base) * 1.618);   /* ~1.618rem */
  --golden-xl: calc(var(--golden-base) * 2.618);   /* ~2.618rem */
  --golden-2xl: calc(var(--golden-base) * 4.236);  /* ~4.236rem */
  --golden-3xl: calc(var(--golden-base) * 6.854);  /* ~6.854rem */
}
```

### **Modern Typography Scale**
```css
:root {
  --text-xs: 0.75rem;      /* 12px */
  --text-sm: 0.875rem;     /* 14px */
  --text-base: 1rem;       /* 16px */
  --text-lg: 1.125rem;     /* 18px */
  --text-xl: 1.25rem;      /* 20px */
  --text-2xl: 1.5rem;      /* 24px */
  --text-3xl: 1.875rem;    /* 30px */
  --text-4xl: 2.25rem;     /* 36px */
  --text-5xl: 3rem;        /* 48px */
}
```

## <¯ **Svelte 5 State Management Best Practices**

### **1. Reactive State with Runes**
```typescript
//  Modern approach - Use $state() rune
let count = $state(0);
let user = $state({ name: '', email: '' });

//  Derived state with $derived()
let doubled = $derived(count * 2);

//  Effects with $effect()
$effect(() => {
  console.log('Count changed:', count);
});
```

### **2. Context API for State Sharing**
```typescript
// lib/stores/app-context.svelte.ts
import { setContext, getContext } from 'svelte';

interface AppContext {
  theme: 'light' | 'dark';
  user: User | null;
  toggleTheme: () => void;
}

const APP_CONTEXT = Symbol('app-context');

export function setAppContext(context: AppContext) {
  setContext(APP_CONTEXT, context);
}

export function getAppContext(): AppContext {
  return getContext(APP_CONTEXT);
}
```

## <¨ **CSS Grid + Flexbox Modern Layouts**

### **1. Golden Ratio Grid System**
```css
.golden-grid {
  display: grid;
  grid-template-columns: 1fr 1.618fr; /* Golden ratio columns */
  gap: var(--golden-lg);
  min-height: 100vh;
}

.golden-grid-3 {
  display: grid;
  grid-template-columns: 1fr 1.618fr 1fr;
  gap: var(--golden-md);
}
```

### **2. Responsive Container Queries**
```css
.container {
  container-type: inline-size;
  max-width: 90rem;
  margin: 0 auto;
  padding: 0 var(--golden-lg);
}

@container (min-width: 768px) {
  .grid-responsive {
    grid-template-columns: repeat(2, 1fr);
  }
}

@container (min-width: 1024px) {
  .grid-responsive {
    grid-template-columns: repeat(3, 1fr);
  }
}
```

## >é **Modular Component Architecture**

### **1. Composition Pattern**
```svelte
<!-- ParentComponent.svelte -->
<script lang="ts">
  interface Props {
    title: string;
    children?: Snippet;
    actions?: Snippet;
  }
  
  let { title, children, actions }: Props = $props();
</script>

<div class="card">
  <header class="card-header">
    <h2>{title}</h2>
    {#if actions}
      <div class="card-actions">
        {@render actions()}
      </div>
    {/if}
  </header>
  
  <div class="card-content">
    {@render children?.()}
  </div>
</div>
```

### **2. Bits-UI + Melt-UI Integration**
```svelte
<!-- ModernDialog.svelte -->
<script lang="ts">
  import { Dialog as DialogPrimitive } from "bits-ui";
  import { createDialog, melt } from "@melt-ui/svelte";
  
  interface Props {
    open?: boolean;
    title: string;
    description?: string;
    children?: Snippet;
  }
  
  let { open = $bindable(false), title, description, children }: Props = $props();
  
  const {
    elements: { trigger, overlay, content, title: titleEl, description: descEl },
    states: { open: openState }
  } = createDialog();
  
  // Sync with bindable prop
  $effect(() => {
    openState.set(open);
  });
  
  $effect(() => {
    open = $openState;
  });
</script>

<DialogPrimitive.Root bind:open>
  <DialogPrimitive.Trigger {trigger}>
    <slot name="trigger" />
  </DialogPrimitive.Trigger>
  
  <DialogPrimitive.Portal>
    <DialogPrimitive.Overlay 
      class="dialog-overlay"
      use:melt={$overlay}
    />
    
    <DialogPrimitive.Content 
      class="dialog-content golden-spacing"
      use:melt={$content}
    >
      <DialogPrimitive.Title use:melt={$titleEl} class="dialog-title">
        {title}
      </DialogPrimitive.Title>
      
      {#if description}
        <DialogPrimitive.Description use:melt={$descEl} class="dialog-description">
          {description}
        </DialogPrimitive.Description>
      {/if}
      
      <div class="dialog-body">
        {@render children?.()}
      </div>
    </DialogPrimitive.Content>
  </DialogPrimitive.Portal>
</DialogPrimitive.Root>

<style>
  .dialog-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(4px);
    z-index: 50;
  }
  
  .dialog-content {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: var(--yorha-bg-card);
    border: 1px solid var(--yorha-border-primary);
    border-radius: 0.5rem;
    padding: var(--golden-xl);
    max-width: 32rem;
    width: calc(100vw - var(--golden-xl));
    max-height: calc(100vh - var(--golden-xl));
    overflow-y: auto;
    box-shadow: var(--yorha-shadow-xl);
    z-index: 51;
  }
  
  .golden-spacing {
    gap: var(--golden-lg);
  }
  
  .dialog-title {
    font-size: var(--text-xl);
    font-weight: 600;
    color: var(--yorha-text-primary);
    margin-bottom: var(--golden-sm);
  }
  
  .dialog-description {
    color: var(--yorha-text-secondary);
    font-size: var(--text-sm);
    margin-bottom: var(--golden-lg);
  }
</style>
```

## <¨ **UnoCSS Integration**

### **1. Configuration (uno.config.ts)**
```typescript
import { defineConfig, presetUno, presetAttributify } from 'unocss';

export default defineConfig({
  presets: [
    presetUno(),
    presetAttributify(),
  ],
  theme: {
    colors: {
      yorha: {
        bg: {
          primary: '#0a0a0a',
          secondary: '#151515',
          tertiary: '#1a1a1a',
          card: '#1f1f1f'
        },
        text: {
          primary: '#e8e8e8',
          secondary: '#b8b8b8',
          muted: '#888888'
        },
        accent: {
          gold: '#d4af37',
          blue: '#4a9eff',
          green: '#00ff88'
        }
      }
    },
    spacing: {
      'golden-xs': 'calc(1rem / 2.618)',
      'golden-sm': 'calc(1rem / 1.618)',
      'golden-md': '1rem',
      'golden-lg': 'calc(1rem * 1.618)',
      'golden-xl': 'calc(1rem * 2.618)',
      'golden-2xl': 'calc(1rem * 4.236)',
    }
  },
  shortcuts: {
    'golden-card': 'bg-yorha-bg-card border border-yorha-border-primary rounded-lg p-golden-lg shadow-lg',
    'golden-btn': 'px-golden-lg py-golden-sm rounded-md font-medium transition-all duration-200',
    'golden-grid': 'grid gap-golden-lg',
    'golden-flex': 'flex gap-golden-md',
  }
});
```

## <× **Modern Layout Patterns**

### **1. Holy Grail Layout with CSS Grid**
```svelte
<!-- +layout.svelte -->
<script lang="ts">
  interface Props {
    children?: Snippet;
  }
  
  let { children }: Props = $props();
</script>

<div class="app-grid">
  <header class="app-header">
    <Navigation />
  </header>
  
  <aside class="app-sidebar">
    <Sidebar />
  </aside>
  
  <main class="app-main">
    {@render children?.()}
  </main>
  
  <footer class="app-footer">
    <Footer />
  </footer>
</div>

<style>
  .app-grid {
    display: grid;
    grid-template-areas:
      "header header"
      "sidebar main"
      "footer footer";
    grid-template-columns: 16rem 1fr;
    grid-template-rows: auto 1fr auto;
    min-height: 100vh;
    gap: 0;
  }
  
  .app-header {
    grid-area: header;
    background: var(--yorha-bg-secondary);
    border-bottom: 1px solid var(--yorha-border-primary);
    padding: var(--golden-md) var(--golden-lg);
  }
  
  .app-sidebar {
    grid-area: sidebar;
    background: var(--yorha-bg-secondary);
    border-right: 1px solid var(--yorha-border-primary);
    padding: var(--golden-lg);
  }
  
  .app-main {
    grid-area: main;
    padding: var(--golden-xl);
    overflow-y: auto;
  }
  
  .app-footer {
    grid-area: footer;
    background: var(--yorha-bg-tertiary);
    border-top: 1px solid var(--yorha-border-primary);
    padding: var(--golden-md) var(--golden-lg);
    text-align: center;
    color: var(--yorha-text-muted);
  }
  
  @media (max-width: 768px) {
    .app-grid {
      grid-template-areas:
        "header"
        "main"
        "footer";
      grid-template-columns: 1fr;
    }
    
    .app-sidebar {
      display: none;
    }
  }
</style>
```

### **2. Card Grid with Auto-fit**
```css
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr));
  gap: var(--golden-lg);
  padding: var(--golden-xl);
}

.card-masonry {
  column-count: auto;
  column-width: 20rem;
  column-gap: var(--golden-lg);
  padding: var(--golden-xl);
}

.card-masonry > * {
  break-inside: avoid;
  margin-bottom: var(--golden-lg);
}
```

## =ñ **Responsive Design Patterns**

### **1. Mobile-first Approach**
```css
/* Mobile first */
.responsive-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--golden-md);
}

/* Tablet */
@media (min-width: 768px) {
  .responsive-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: var(--golden-lg);
  }
}

/* Desktop */
@media (min-width: 1024px) {
  .responsive-grid {
    grid-template-columns: repeat(3, 1fr);
    gap: var(--golden-xl);
  }
}

/* Large screens */
@media (min-width: 1440px) {
  .responsive-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}
```

## <¯ **Accessibility Best Practices**

### **1. Focus Management**
```css
.focus-visible {
  outline: 2px solid var(--yorha-accent-gold);
  outline-offset: 2px;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
```

### **2. ARIA Labels & Semantic HTML**
```svelte
<nav aria-label="Main navigation">
  <ul role="list">
    <li role="listitem">
      <a href="/" aria-current="page">Home</a>
    </li>
  </ul>
</nav>

<button 
  aria-expanded={isOpen}
  aria-controls="dropdown-menu"
  aria-label="Toggle menu"
>
  Menu
</button>
```

## =€ **Performance Optimization**

### **1. Component Lazy Loading**
```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  
  let LazyComponent: any = null;
  
  onMount(async () => {
    const module = await import('./LazyComponent.svelte');
    LazyComponent = module.default;
  });
</script>

{#if LazyComponent}
  <svelte:component this={LazyComponent} />
{/if}
```

### **2. Virtual Lists for Large Data**
```svelte
<script lang="ts">
  import { createVirtualizer } from '@tanstack/svelte-virtual';
  
  let items = $state(Array.from({ length: 10000 }, (_, i) => `Item ${i}`));
  let scrollElement: HTMLElement;
  
  const virtualizer = createVirtualizer({
    count: items.length,
    getScrollElement: () => scrollElement,
    estimateSize: () => 50,
    overscan: 5
  });
  
  $: virtualItems = $virtualizer.getVirtualItems();
  $: totalSize = $virtualizer.getTotalSize();
</script>

<div bind:this={scrollElement} class="virtual-list">
  <div style="height: {totalSize}px; position: relative;">
    {#each virtualItems as virtualRow (virtualRow.index)}
      <div
        class="virtual-item"
        style="
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: {virtualRow.size}px;
          transform: translateY({virtualRow.start}px);
        "
      >
        {items[virtualRow.index]}
      </div>
    {/each}
  </div>
</div>
```

This guide provides modern, production-ready patterns for SvelteKit 2 + Svelte 5 with proper golden ratio spacing, CSS Grid/Flexbox layouts, and component architecture without prop drilling.