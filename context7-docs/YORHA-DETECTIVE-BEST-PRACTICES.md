# YoRHa Detective Interface - Context7 Best Practices Guide

## 🎯 Overview

Based on Context7 MCP analysis and Svelte 5 + Bits UI best practices, this guide provides comprehensive patterns for implementing the YoRHa Detective Interface using modern SvelteKit 2 architecture.

---

## 🏗️ **Architecture Pattern: Gaming-Inspired Legal Interface**

### Core Principles
- **NieR Automata Design Language**: Cyberpunk aesthetics with professional functionality
- **Accessibility First**: ARIA compliance with screen reader support
- **Performance Optimized**: GPU-accelerated animations with reduced motion support
- **Type Safety**: Full TypeScript integration with Svelte 5 runes

---

## 🎨 **Component Architecture - Bits UI Integration**

### Dialog System Best Practices

```svelte
<!-- YoRHaDialog.svelte - Reusable Detective Interface Modal -->
<script lang="ts">
  import type { Snippet } from "svelte";
  import { Dialog, type WithoutChild } from "bits-ui";
  import { fly, scale } from "svelte/transition";
  
  type Props = Dialog.RootProps & {
    triggerText: string;
    title: Snippet;
    description: Snippet;
    variant?: 'investigation' | 'evidence' | 'alert' | 'system';
    contentProps?: WithoutChild<Dialog.ContentProps>;
  };

  let {
    open = $bindable(false),
    children,
    triggerText,
    contentProps,
    title,
    description,
    variant = 'investigation',
    ...restProps
  }: Props = $props();

  const variants = {
    investigation: 'border-yorha-blue bg-yorha-dark',
    evidence: 'border-yorha-amber bg-yorha-evidence',
    alert: 'border-yorha-red bg-yorha-alert',
    system: 'border-yorha-green bg-yorha-system'
  };
</script>

<Dialog.Root bind:open {...restProps}>
  <Dialog.Trigger
    class="yorha-button yorha-button-{variant} 
           transition-all duration-200 
           focus-visible:ring-2 focus-visible:ring-yorha-blue 
           active:scale-95"
  >
    {triggerText}
  </Dialog.Trigger>
  
  <Dialog.Portal>
    <Dialog.Overlay
      class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm
             data-[state=open]:animate-fade-in
             data-[state=closed]:animate-fade-out"
    />
    
    <Dialog.Content
      class="fixed left-1/2 top-1/2 z-50 
             w-full max-w-[600px] -translate-x-1/2 -translate-y-1/2
             {variants[variant]}
             border-2 shadow-yorha-glow
             data-[state=open]:animate-scale-fade-in
             data-[state=closed]:animate-scale-fade-out"
      forceMount={true}
      {...contentProps}
    >
      {#snippet child({ props, open })}
        {#if open}
          <div {...props} transition:scale={{ duration: 200 }}>
            <!-- YoRHa Header -->
            <div class="yorha-header border-b border-yorha-blue/30 p-4">
              <Dialog.Title class="yorha-title text-xl font-mono">
                {@render title()}
              </Dialog.Title>
              
              <Dialog.Description class="yorha-subtitle text-sm opacity-80">
                {@render description()}
              </Dialog.Description>
            </div>
            
            <!-- Content Area -->
            <div class="yorha-content p-6">
              {@render children?.()}
            </div>
            
            <!-- YoRHa Close Button -->
            <Dialog.Close
              class="absolute right-4 top-4 
                     yorha-close-button
                     focus-visible:ring-2 focus-visible:ring-yorha-blue"
              aria-label="Close Detective Interface"
            >
              <svg class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
              </svg>
            </Dialog.Close>
          </div>
        {/if}
      {/snippet}
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>
```

---

## 🔍 **Detective Interface Components**

### Investigation Search Component

```svelte
<!-- YoRHaInvestigationSearch.svelte -->
<script lang="ts">
  import { Command, Dialog } from "bits-ui";
  import { writable } from 'svelte/store';
  
  let searchOpen = $state(false);
  let searchQuery = $state('');
  let searchResults = $state<any[]>([]);
  
  // Keyboard shortcut: Ctrl/Cmd + K
  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      searchOpen = true;
    }
  }
  
  async function performInvestigation(query: string) {
    if (!query.trim()) return;
    
    // Enhanced RAG search integration
    const response = await fetch('/api/yorha/enhanced-rag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        context: 'detective-investigation',
        includeEvidence: true,
        confidenceThreshold: 0.7
      })
    });
    
    const result = await response.json();
    if (result.success) {
      searchResults = result.results;
    }
  }
</script>

<svelte:document onkeydown={handleKeydown} />

<Dialog.Root bind:open={searchOpen}>
  <Dialog.Trigger
    class="yorha-search-trigger w-full max-w-md
           bg-yorha-dark border border-yorha-blue/30
           text-left px-4 py-2 rounded-none
           hover:border-yorha-blue focus:border-yorha-blue
           transition-colors"
  >
    <span class="text-yorha-text/60">Search investigations... ⌘K</span>
  </Dialog.Trigger>
  
  <Dialog.Portal>
    <Dialog.Overlay class="fixed inset-0 z-50 bg-black/80" />
    
    <Dialog.Content
      class="fixed left-1/2 top-1/2 z-50
             w-full max-w-[700px] -translate-x-1/2 -translate-y-1/2
             bg-yorha-dark border-2 border-yorha-blue
             shadow-yorha-glow max-h-[600px]"
    >
      <Dialog.Title class="sr-only">Detective Investigation Search</Dialog.Title>
      <Dialog.Description class="sr-only">
        Search through cases, evidence, and legal documents. Use arrow keys to navigate.
      </Dialog.Description>
      
      <Command.Root class="w-full">
        <Command.Input
          class="w-full bg-transparent border-none
                 px-6 py-4 text-yorha-text
                 placeholder:text-yorha-text/60
                 focus:outline-none text-lg font-mono"
          placeholder="Enter investigation query..."
          bind:value={searchQuery}
          onkeydown={(e) => {
            if (e.key === 'Enter') {
              performInvestigation(searchQuery);
            }
          }}
        />
        
        <Command.List class="max-h-[400px] overflow-y-auto px-2 pb-2">
          <Command.Viewport>
            <Command.Empty class="py-8 text-center text-yorha-text/60">
              No results found. Try different keywords.
            </Command.Empty>
            
            <!-- Investigation Results -->
            <Command.Group>
              <Command.GroupHeading 
                class="px-4 py-2 text-xs text-yorha-blue uppercase tracking-wider"
              >
                Investigation Results
              </Command.GroupHeading>
              
              <Command.GroupItems>
                {#each searchResults as result}
                  <Command.Item
                    class="px-4 py-3 cursor-pointer
                           hover:bg-yorha-blue/10 focus:bg-yorha-blue/10
                           border-l-2 border-transparent
                           data-selected:border-yorha-amber"
                    onselect={() => {
                      // Navigate to result
                      window.location.href = `/investigation/${result.id}`;
                    }}
                  >
                    <div class="flex items-start gap-3">
                      <div class="w-2 h-2 bg-yorha-amber rounded-full mt-2 flex-shrink-0"></div>
                      <div class="flex-1 min-w-0">
                        <div class="font-mono text-yorha-text font-medium">
                          {result.title}
                        </div>
                        <div class="text-sm text-yorha-text/70 mt-1">
                          {result.summary}
                        </div>
                        <div class="flex items-center gap-4 mt-2 text-xs">
                          <span class="text-yorha-blue">
                            Confidence: {(result.confidence * 100).toFixed(1)}%
                          </span>
                          <span class="text-yorha-text/50">
                            Type: {result.type}
                          </span>
                        </div>
                      </div>
                    </div>
                  </Command.Item>
                {/each}
              </Command.GroupItems>
            </Command.Group>
          </Command.Viewport>
        </Command.List>
      </Command.Root>
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>
```

---

## 📊 **Data Grid Component**

### YoRHa Evidence Grid

```svelte
<!-- YoRHaEvidenceGrid.svelte -->
<script lang="ts">
  import type { Evidence } from '$lib/types';
  
  type Props = {
    evidence: Evidence[];
    onRowSelect?: (evidence: Evidence) => void;
    loading?: boolean;
    selectable?: boolean;
  };
  
  let { 
    evidence, 
    onRowSelect, 
    loading = false, 
    selectable = true 
  }: Props = $props();
  
  let selectedRows = $state<Set<string>>(new Set());
  
  function toggleRowSelection(id: string) {
    if (selectedRows.has(id)) {
      selectedRows.delete(id);
    } else {
      selectedRows.add(id);
    }
    selectedRows = selectedRows; // Trigger reactivity
  }
  
  const columns = [
    { key: 'yorha_id', title: 'YORHA ID', width: 120 },
    { key: 'title', title: 'EVIDENCE TITLE', width: 300 },
    { key: 'type', title: 'TYPE', width: 120 },
    { key: 'case_id', title: 'CASE', width: 150 },
    { key: 'collected_by', title: 'COLLECTED BY', width: 180 },
    { key: 'status', title: 'STATUS', width: 100 },
    { key: 'collected_at', title: 'DATE', width: 140 }
  ];
</script>

<div class="yorha-grid-container">
  <!-- Grid Header -->
  <div class="yorha-grid-header bg-yorha-dark border-b-2 border-yorha-blue">
    <div class="grid-header-content p-4">
      <h2 class="text-lg font-mono text-yorha-blue">EVIDENCE DATABASE</h2>
      <div class="text-sm text-yorha-text/70">
        {evidence.length} records • {selectedRows.size} selected
      </div>
    </div>
  </div>
  
  <!-- Grid Content -->
  <div class="yorha-grid-content">
    {#if loading}
      <div class="loading-state p-8 text-center">
        <div class="yorha-spinner"></div>
        <div class="text-yorha-text/60 mt-4">ACCESSING DATABASE...</div>
      </div>
    {:else}
      <div class="grid-table">
        <!-- Column Headers -->
        <div class="grid-header-row bg-yorha-dark/50 border-b border-yorha-blue/30">
          {#if selectable}
            <div class="grid-cell checkbox-cell">
              <input
                type="checkbox"
                class="yorha-checkbox"
                checked={selectedRows.size === evidence.length && evidence.length > 0}
                onchange={(e) => {
                  if (e.currentTarget.checked) {
                    selectedRows = new Set(evidence.map(e => e.id));
                  } else {
                    selectedRows = new Set();
                  }
                }}
              />
            </div>
          {/if}
          
          {#each columns as column}
            <div 
              class="grid-cell header-cell font-mono text-xs text-yorha-blue uppercase tracking-wider"
              style="width: {column.width}px"
            >
              {column.title}
            </div>
          {/each}
        </div>
        
        <!-- Data Rows -->
        {#each evidence as item, index}
          <div 
            class="grid-row border-b border-yorha-blue/10
                   hover:bg-yorha-blue/5 cursor-pointer
                   {selectedRows.has(item.id) ? 'bg-yorha-amber/10 border-yorha-amber/30' : ''}"
            onclick={() => onRowSelect?.(item)}
            role="row"
            tabindex="0"
            onkeydown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onRowSelect?.(item);
              }
            }}
          >
            {#if selectable}
              <div class="grid-cell checkbox-cell">
                <input
                  type="checkbox"
                  class="yorha-checkbox"
                  checked={selectedRows.has(item.id)}
                  onchange={(e) => {
                    e.stopPropagation();
                    toggleRowSelection(item.id);
                  }}
                />
              </div>
            {/if}
            
            <div class="grid-cell font-mono text-yorha-amber" style="width: 120px">
              {item.yorha_id}
            </div>
            
            <div class="grid-cell text-yorha-text" style="width: 300px">
              {item.title}
            </div>
            
            <div class="grid-cell" style="width: 120px">
              <span class="status-badge status-{item.type.toLowerCase()}">
                {item.type}
              </span>
            </div>
            
            <div class="grid-cell text-yorha-text/80" style="width: 150px">
              {item.case_id}
            </div>
            
            <div class="grid-cell text-yorha-text/80" style="width: 180px">
              {item.collected_by}
            </div>
            
            <div class="grid-cell" style="width: 100px">
              <span class="status-badge status-{item.status.toLowerCase()}">
                {item.status}
              </span>
            </div>
            
            <div class="grid-cell text-yorha-text/60 font-mono text-sm" style="width: 140px">
              {new Date(item.collected_at).toLocaleDateString()}
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .yorha-grid-container {
    @apply border-2 border-yorha-blue bg-yorha-dark shadow-yorha-glow;
  }
  
  .grid-table {
    @apply overflow-auto max-h-[600px];
  }
  
  .grid-header-row,
  .grid-row {
    @apply flex items-center min-w-max;
  }
  
  .grid-cell {
    @apply px-4 py-3 text-sm flex-shrink-0;
  }
  
  .checkbox-cell {
    @apply w-12 flex justify-center;
  }
  
  .header-cell {
    @apply py-4 font-semibold sticky top-0 bg-yorha-dark/90 backdrop-blur;
  }
  
  .yorha-checkbox {
    @apply w-4 h-4 rounded-none border-2 border-yorha-blue
           bg-transparent checked:bg-yorha-amber
           focus:ring-2 focus:ring-yorha-blue focus:ring-offset-0;
  }
  
  .status-badge {
    @apply px-2 py-1 text-xs font-mono uppercase rounded-none border;
  }
  
  .status-document { @apply bg-yorha-blue/20 border-yorha-blue text-yorha-blue; }
  .status-image { @apply bg-yorha-amber/20 border-yorha-amber text-yorha-amber; }
  .status-video { @apply bg-yorha-green/20 border-yorha-green text-yorha-green; }
  .status-audio { @apply bg-yorha-purple/20 border-yorha-purple text-yorha-purple; }
  
  .status-active { @apply bg-yorha-green/20 border-yorha-green text-yorha-green; }
  .status-pending { @apply bg-yorha-amber/20 border-yorha-amber text-yorha-amber; }
  .status-archived { @apply bg-yorha-text/20 border-yorha-text text-yorha-text; }
  
  .yorha-spinner {
    @apply w-8 h-8 border-2 border-yorha-blue border-t-transparent 
           rounded-full animate-spin mx-auto;
  }
  
  .loading-state {
    @apply flex flex-col items-center justify-center min-h-[200px];
  }
</style>
```

---

## 🎨 **CSS Architecture - YoRHa Design System**

### Core CSS Variables

```css
/* yorha-theme.css */
:root {
  /* YoRHa Color Palette */
  --yorha-black: #0a0a0a;
  --yorha-dark: #1a1a1a;
  --yorha-text: #e0e0e0;
  --yorha-blue: #4fc3f7;
  --yorha-amber: #ffbf00;
  --yorha-green: #00ff41;
  --yorha-red: #ff0041;
  --yorha-purple: #9c27b0;
  
  /* Typography */
  --yorha-font-mono: 'JetBrains Mono', 'Cascadia Code', monospace;
  --yorha-font-sans: 'Inter', system-ui, sans-serif;
  
  /* Spacing */
  --yorha-grid-unit: 8px;
  --yorha-border-width: 2px;
  
  /* Effects */
  --yorha-glow: 0 0 20px rgba(79, 195, 247, 0.3);
  --yorha-glow-strong: 0 0 30px rgba(79, 195, 247, 0.5);
}

/* YoRHa Button System */
.yorha-button {
  @apply font-mono uppercase tracking-wider text-sm
         px-6 py-3 border-2 bg-transparent
         transition-all duration-200
         hover:shadow-yorha-glow
         focus:outline-none focus:ring-2 focus:ring-offset-2;
}

.yorha-button-investigation {
  @apply border-yorha-blue text-yorha-blue
         hover:bg-yorha-blue hover:text-yorha-black
         focus:ring-yorha-blue;
}

.yorha-button-evidence {
  @apply border-yorha-amber text-yorha-amber
         hover:bg-yorha-amber hover:text-yorha-black
         focus:ring-yorha-amber;
}

.yorha-button-alert {
  @apply border-yorha-red text-yorha-red
         hover:bg-yorha-red hover:text-white
         focus:ring-yorha-red;
}

.yorha-button-system {
  @apply border-yorha-green text-yorha-green
         hover:bg-yorha-green hover:text-yorha-black
         focus:ring-yorha-green;
}

/* YoRHa Animations */
@keyframes fade-in {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes scale-fade-in {
  from { 
    opacity: 0; 
    transform: translate(-50%, -50%) scale(0.95);
  }
  to { 
    opacity: 1; 
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes glitch {
  0%, 100% { transform: translateX(0); }
  20% { transform: translateX(-2px); }
  40% { transform: translateX(2px); }
  60% { transform: translateX(-1px); }
  80% { transform: translateX(1px); }
}

.animate-fade-in { animation: fade-in 0.2s ease-out; }
.animate-fade-out { animation: fade-in 0.2s ease-out reverse; }
.animate-scale-fade-in { animation: scale-fade-in 0.2s ease-out; }
.animate-scale-fade-out { animation: scale-fade-in 0.2s ease-out reverse; }
.animate-glitch { animation: glitch 0.5s ease-in-out; }

/* Accessibility & Reduced Motion */
@media (prefers-reduced-motion: reduce) {
  .animate-fade-in,
  .animate-fade-out,
  .animate-scale-fade-in,
  .animate-scale-fade-out,
  .animate-glitch {
    animation: none;
  }
  
  .yorha-button,
  .grid-row {
    transition: none;
  }
}

/* Focus Management */
.yorha-focus {
  @apply focus:outline-none focus:ring-2 focus:ring-yorha-blue 
         focus:ring-offset-2 focus:ring-offset-yorha-black;
}

/* Screen Reader Only */
.sr-only {
  @apply absolute w-px h-px p-0 -m-px overflow-hidden 
         clip-[rect(0,0,0,0)] whitespace-nowrap border-0;
}
```

---

## ⌨️ **Keyboard Navigation & Accessibility**

### ARIA Implementation

```svelte
<!-- YoRHaAccessibleGrid.svelte -->
<script lang="ts">
  let gridRef = $state<HTMLDivElement>();
  let selectedRowIndex = $state(0);
  let focusedRowIndex = $state(0);
  
  function handleGridKeydown(e: KeyboardEvent) {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        focusedRowIndex = Math.min(focusedRowIndex + 1, evidence.length - 1);
        break;
      case 'ArrowUp':
        e.preventDefault();
        focusedRowIndex = Math.max(focusedRowIndex - 1, 0);
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        selectedRowIndex = focusedRowIndex;
        onRowSelect?.(evidence[focusedRowIndex]);
        break;
      case 'Home':
        e.preventDefault();
        focusedRowIndex = 0;
        break;
      case 'End':
        e.preventDefault();
        focusedRowIndex = evidence.length - 1;
        break;
    }
  }
</script>

<div 
  bind:this={gridRef}
  class="yorha-accessible-grid"
  role="grid"
  aria-label="Evidence Database"
  aria-rowcount={evidence.length + 1}
  aria-colcount={columns.length}
  tabindex="0"
  onkeydown={handleGridKeydown}
>
  <!-- ARIA Live Region for Updates -->
  <div 
    class="sr-only" 
    aria-live="polite" 
    aria-atomic="true"
  >
    {focusedRowIndex >= 0 ? 
      `Row ${focusedRowIndex + 1} of ${evidence.length}: ${evidence[focusedRowIndex]?.title}` : 
      'Grid focused'
    }
  </div>
  
  <!-- Grid Implementation -->
  {#each evidence as item, index}
    <div 
      role="row"
      aria-rowindex={index + 2}
      aria-selected={selectedRowIndex === index}
      class="grid-row {focusedRowIndex === index ? 'focused' : ''}"
      data-row-index={index}
    >
      {#each columns as column, colIndex}
        <div 
          role="gridcell"
          aria-colindex={colIndex + 1}
          class="grid-cell"
        >
          {item[column.key]}
        </div>
      {/each}
    </div>
  {/each}
</div>
```

---

## 🚀 **Performance Optimization**

### Lazy Loading & Virtual Scrolling

```svelte
<!-- VirtualizedEvidenceGrid.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  
  type Props = {
    evidence: Evidence[];
    itemHeight?: number;
    containerHeight?: number;
  };
  
  let { 
    evidence, 
    itemHeight = 60, 
    containerHeight = 400 
  }: Props = $props();
  
  let scrollTop = $state(0);
  let containerRef = $state<HTMLDivElement>();
  
  // Virtual scrolling calculations
  const visibleCount = $derived(Math.ceil(containerHeight / itemHeight) + 2);
  const startIndex = $derived(Math.floor(scrollTop / itemHeight));
  const endIndex = $derived(Math.min(startIndex + visibleCount, evidence.length));
  const visibleItems = $derived(evidence.slice(startIndex, endIndex));
  
  const totalHeight = $derived(evidence.length * itemHeight);
  const offsetY = $derived(startIndex * itemHeight);
  
  function handleScroll(e: Event) {
    const target = e.target as HTMLDivElement;
    scrollTop = target.scrollTop;
  }
  
  // Intersection Observer for lazy loading
  let observer: IntersectionObserver;
  
  onMount(() => {
    if (typeof IntersectionObserver !== 'undefined') {
      observer = new IntersectionObserver(
        (entries) => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              // Load additional data if needed
              loadMoreEvidence();
            }
          });
        },
        { rootMargin: '200px' }
      );
    }
    
    return () => observer?.disconnect();
  });
  
  async function loadMoreEvidence() {
    // Implement pagination logic
    console.log('Loading more evidence...');
  }
</script>

<div 
  bind:this={containerRef}
  class="virtual-scroll-container"
  style="height: {containerHeight}px"
  onscroll={handleScroll}
>
  <div style="height: {totalHeight}px; position: relative;">
    <div 
      class="virtual-content"
      style="transform: translateY({offsetY}px)"
    >
      {#each visibleItems as item, index}
        <div 
          class="virtual-item yorha-evidence-row"
          style="height: {itemHeight}px"
          data-index={startIndex + index}
        >
          <EvidenceRow {item} />
        </div>
      {/each}
    </div>
  </div>
</div>

<style>
  .virtual-scroll-container {
    @apply overflow-auto border-2 border-yorha-blue bg-yorha-dark;
  }
  
  .virtual-content {
    @apply relative;
  }
  
  .virtual-item {
    @apply border-b border-yorha-blue/10;
  }
  
  .yorha-evidence-row {
    @apply flex items-center px-4 hover:bg-yorha-blue/5 transition-colors;
  }
</style>
```

---

## 🧪 **Testing Strategy**

### Component Testing with Vitest

```typescript
// YoRHaDialog.test.ts
import { render, screen, fireEvent } from '@testing-library/svelte';
import { expect, test, describe } from 'vitest';
import YoRHaDialog from './YoRHaDialog.svelte';

describe('YoRHaDialog', () => {
  test('renders with correct ARIA attributes', () => {
    render(YoRHaDialog, {
      props: {
        triggerText: 'Open Investigation',
        title: () => 'Case Analysis',
        description: () => 'Detailed case information',
        variant: 'investigation'
      }
    });
    
    const trigger = screen.getByRole('button', { name: /open investigation/i });
    expect(trigger).toHaveAttribute('aria-haspopup', 'dialog');
  });
  
  test('opens dialog on trigger click', async () => {
    render(YoRHaDialog, {
      props: {
        triggerText: 'Open Investigation',
        title: () => 'Case Analysis',
        description: () => 'Detailed case information'
      }
    });
    
    const trigger = screen.getByRole('button');
    await fireEvent.click(trigger);
    
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('Case Analysis')).toBeInTheDocument();
  });
  
  test('closes dialog on Escape key', async () => {
    render(YoRHaDialog, {
      props: {
        triggerText: 'Open Investigation',
        title: () => 'Case Analysis',
        description: () => 'Detailed case information'
      }
    });
    
    const trigger = screen.getByRole('button');
    await fireEvent.click(trigger);
    
    await fireEvent.keyDown(document, { key: 'Escape' });
    
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
  
  test('maintains focus management', async () => {
    render(YoRHaDialog, {
      props: {
        triggerText: 'Open Investigation',
        title: () => 'Case Analysis',
        description: () => 'Detailed case information'
      }
    });
    
    const trigger = screen.getByRole('button');
    trigger.focus();
    
    await fireEvent.click(trigger);
    
    // Focus should move to dialog content
    const dialog = screen.getByRole('dialog');
    expect(document.activeElement).toBe(dialog);
  });
});
```

---

## 📱 **Responsive Design**

### Mobile-First YoRHa Interface

```svelte
<!-- ResponsiveYoRHaLayout.svelte -->
<script lang="ts">
  import { browser } from '$app/environment';
  
  let isMobile = $state(false);
  let isTablet = $state(false);
  
  if (browser) {
    const checkBreakpoints = () => {
      isMobile = window.innerWidth < 768;
      isTablet = window.innerWidth >= 768 && window.innerWidth < 1024;
    };
    
    checkBreakpoints();
    window.addEventListener('resize', checkBreakpoints);
  }
</script>

<div class="yorha-responsive-layout">
  {#if isMobile}
    <!-- Mobile Stack Layout -->
    <div class="mobile-layout">
      <header class="mobile-header">
        <h1 class="yorha-title-mobile">YoRHa Detective</h1>
        <button class="mobile-menu-toggle yorha-button-system">
          Menu
        </button>
      </header>
      
      <main class="mobile-main">
        <div class="mobile-cards">
          {#each evidenceItems as item}
            <div class="mobile-evidence-card">
              <div class="card-header">
                <span class="yorha-id">{item.yorha_id}</span>
                <span class="evidence-type">{item.type}</span>
              </div>
              <h3 class="card-title">{item.title}</h3>
              <p class="card-meta">
                Case: {item.case_id} • {item.collected_by}
              </p>
            </div>
          {/each}
        </div>
      </main>
    </div>
  {:else}
    <!-- Desktop Grid Layout -->
    <div class="desktop-layout">
      <aside class="desktop-sidebar">
        <nav class="yorha-navigation">
          <!-- Desktop navigation -->
        </nav>
      </aside>
      
      <main class="desktop-main">
        <YoRHaEvidenceGrid {evidence} />
      </main>
    </div>
  {/if}
</div>

<style>
  .yorha-responsive-layout {
    @apply min-h-screen bg-yorha-black text-yorha-text;
  }
  
  /* Mobile Styles */
  .mobile-layout {
    @apply flex flex-col;
  }
  
  .mobile-header {
    @apply flex justify-between items-center p-4 
           border-b-2 border-yorha-blue bg-yorha-dark;
  }
  
  .yorha-title-mobile {
    @apply text-lg font-mono text-yorha-blue;
  }
  
  .mobile-main {
    @apply flex-1 p-4;
  }
  
  .mobile-cards {
    @apply space-y-4;
  }
  
  .mobile-evidence-card {
    @apply border border-yorha-blue/30 bg-yorha-dark p-4
           hover:border-yorha-blue transition-colors;
  }
  
  .card-header {
    @apply flex justify-between items-center mb-2;
  }
  
  .yorha-id {
    @apply font-mono text-yorha-amber text-sm;
  }
  
  .evidence-type {
    @apply text-xs uppercase text-yorha-blue;
  }
  
  .card-title {
    @apply font-semibold text-yorha-text mb-2;
  }
  
  .card-meta {
    @apply text-sm text-yorha-text/70;
  }
  
  /* Desktop Styles */
  .desktop-layout {
    @apply flex min-h-screen;
  }
  
  .desktop-sidebar {
    @apply w-64 bg-yorha-dark border-r-2 border-yorha-blue;
  }
  
  .desktop-main {
    @apply flex-1 p-6;
  }
  
  /* Tablet Adjustments */
  @media (min-width: 768px) and (max-width: 1024px) {
    .desktop-sidebar {
      @apply w-48;
    }
    
    .mobile-evidence-card {
      @apply p-3;
    }
  }
</style>
```

---

## 🔧 **Context7 Integration Patterns**

### MCP-Aware Component Factory

```typescript
// lib/utils/yorha-component-factory.ts
import type { Component } from 'svelte';
import { mcp } from '$lib/context7/mcp-client';

export class YoRHaComponentFactory {
  static async createComponent(
    componentType: 'dialog' | 'grid' | 'search',
    context: 'investigation' | 'evidence' | 'case',
    props: Record<string, any>
  ): Promise<Component> {
    
    // Get Context7 recommendations
    const mcpRecommendations = await mcp.analyzeStack({
      component: componentType,
      context: `yorha-${context}`,
      framework: 'svelte5-bits-ui'
    });
    
    // Apply performance optimizations
    const optimizedProps = this.applyPerformanceOptimizations(
      props,
      mcpRecommendations.performance
    );
    
    // Enhance accessibility
    const accessibleProps = this.enhanceAccessibility(
      optimizedProps,
      mcpRecommendations.accessibility
    );
    
    return this.buildComponent(componentType, accessibleProps);
  }
  
  private static applyPerformanceOptimizations(
    props: Record<string, any>,
    perfRecommendations: any
  ) {
    // Apply virtual scrolling for large datasets
    if (perfRecommendations.virtualScrolling && props.data?.length > 100) {
      props.virtualized = true;
      props.itemHeight = perfRecommendations.optimalItemHeight;
    }
    
    // Enable lazy loading
    if (perfRecommendations.lazyLoading) {
      props.lazyLoad = true;
      props.loadThreshold = perfRecommendations.loadThreshold;
    }
    
    return props;
  }
  
  private static enhanceAccessibility(
    props: Record<string, any>,
    a11yRecommendations: any
  ) {
    // Add ARIA labels
    props.ariaLabel = a11yRecommendations.ariaLabel;
    props.ariaDescription = a11yRecommendations.ariaDescription;
    
    // Enable keyboard navigation
    props.keyboardNavigation = true;
    props.focusManagement = a11yRecommendations.focusStrategy;
    
    return props;
  }
}
```

---

## 📚 **Documentation & Style Guide**

### Component Usage Examples

```svelte
<!-- Example: Investigation Search -->
<YoRHaInvestigationSearch
  placeholder="Search cases, evidence, documents..."
  onSearch={handleInvestigationSearch}
  shortcuts={{ open: 'cmd+k', filter: 'cmd+f' }}
  filters={['cases', 'evidence', 'documents']}
  resultLimit={50}
/>

<!-- Example: Evidence Analysis Dialog -->
<YoRHaDialog variant="evidence" bind:open={evidenceDialogOpen}>
  {#snippet title()}
    Evidence Analysis: {selectedEvidence?.title}
  {/snippet}
  
  {#snippet description()}
    Detailed forensic analysis and metadata for evidence #{selectedEvidence?.yorha_id}
  {/snippet}
  
  <EvidenceAnalysisPanel evidence={selectedEvidence} />
  <EvidenceMetadata evidence={selectedEvidence} />
  
  <div class="dialog-actions">
    <button class="yorha-button yorha-button-system">Export Report</button>
    <button class="yorha-button yorha-button-evidence">Mark as Key Evidence</button>
  </div>
</YoRHaDialog>

<!-- Example: Case Timeline -->
<YoRHaTimeline
  events={caseEvents}
  variant="investigation"
  interactive={true}
  onEventSelect={handleEventSelect}
  filters={{
    dateRange: { start: caseStartDate, end: new Date() },
    eventTypes: ['evidence_collected', 'witness_interview', 'analysis_complete']
  }}
/>
```

---

## 🚀 **Production Deployment Checklist**

### Performance Audit
- [ ] Bundle size analysis (< 100KB for core components)
- [ ] Lighthouse accessibility score > 95
- [ ] Virtual scrolling for datasets > 100 items
- [ ] Image optimization and lazy loading
- [ ] Service Worker implementation

### Accessibility Compliance
- [ ] WCAG 2.1 AA compliance
- [ ] Screen reader testing (NVDA, JAWS, VoiceOver)
- [ ] Keyboard navigation testing
- [ ] Color contrast validation (4.5:1 minimum)
- [ ] Focus management audit

### Browser Testing
- [ ] Chrome/Edge (Chromium)
- [ ] Firefox
- [ ] Safari (WebKit)
- [ ] Mobile browsers (iOS Safari, Chrome Mobile)
- [ ] Reduced motion preferences

### Security Review
- [ ] XSS prevention (no innerHTML usage)
- [ ] CSRF token validation
- [ ] Content Security Policy headers
- [ ] Input sanitization
- [ ] Authentication state management

---

## 📖 **Context7 MCP Commands**

### Quick Reference
```bash
# Analyze YoRHa components
claude: "analyze svelte with context yorha-detective-interface"

# Get performance recommendations
claude: "generate best practices for performance"

# Integration guidance
claude: "suggest integration for evidence-grid with requirements accessibility virtualization"

# Library documentation
claude: "get library docs for bits-ui topic dialog accessibility"
```

---

This comprehensive guide provides production-ready patterns for implementing the YoRHa Detective Interface using Context7 best practices, Svelte 5 runes, and Bits UI components. The architecture emphasizes performance, accessibility, and maintainability while delivering the unique gaming-inspired aesthetic of the NieR Automata universe.