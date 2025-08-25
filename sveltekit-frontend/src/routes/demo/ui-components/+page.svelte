<script lang="ts">
  import Button from '$lib/components/ui/Button.svelte';
  import { createDialog } from 'bits-ui';
  import { createAccordion, melt } from 'melt';
  
  // Bits UI Dialog
  const {
    elements: { trigger: dialogTrigger, content: dialogContent, overlay: dialogOverlay, close: dialogClose, title: dialogTitle },
    states: { open: dialogOpen }
  } = createDialog();
  
  // Melt UI Accordion  
  const {
    elements: { root: accordionRoot, item: accordionItem, trigger: accordionTrigger, content: accordionContent },
    helpers: { isSelected }
  } = createAccordion();
  
  let selectedDemo = 'overview';
  
  const demoItems = [
    { id: 'overview', label: 'Overview', description: 'Component library integration summary' },
    { id: 'buttons', label: 'Buttons', description: 'bits-ui + shadcn button variants' },
    { id: 'dialogs', label: 'Dialogs', description: 'bits-ui modal components' },
    { id: 'accordion', label: 'Accordion', description: 'melt-ui accordion with animations' },
    { id: 'forms', label: 'Forms', description: 'Combined form components' }
  ];
</script>

<div class="container mx-auto p-8 space-y-8">
  <div class="text-center space-y-4">
    <h1 class="text-4xl font-bold text-primary">Advanced UI Components</h1>
    <p class="text-xl text-muted-foreground">
      bits-ui v2.9.4 + melt v0.39.0 + shadcn-svelte on Svelte 5, SvelteKit 2
    </p>
    <div class="flex items-center justify-center space-x-4 text-sm">
      <span class="bg-green-100 text-green-800 px-2 py-1 rounded">✅ Svelte 5 Compatible</span>
      <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded">✅ SvelteKit 2</span>
      <span class="bg-purple-100 text-purple-800 px-2 py-1 rounded">✅ TypeScript</span>
    </div>
  </div>

  <!-- Demo Navigation -->
  <div class="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
    {#each demoItems as item}
      <button
        class="p-4 rounded-lg border-2 transition-all hover:shadow-md {selectedDemo === item.id 
          ? 'border-primary bg-primary/5 shadow-md' 
          : 'border-border hover:border-primary/50'}"
        onclick={() => selectedDemo = item.id}
      >
        <h3 class="font-semibold">{item.label}</h3>
        <p class="text-sm text-muted-foreground mt-1">{item.description}</p>
      </button>
    {/each}
  </div>

  <!-- Demo Content -->
  <div class="min-h-[500px] border rounded-lg p-6 bg-card">
    {#if selectedDemo === 'overview'}
      <div class="space-y-6">
        <h2 class="text-2xl font-bold">Integration Overview</h2>
        
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div class="border rounded-lg p-6">
            <h3 class="text-lg font-semibold mb-3">bits-ui v2.9.4</h3>
            <ul class="space-y-2 text-sm">
              <li>✅ Svelte 5 compatible</li>
              <li>✅ Accessible by default</li>
              <li>✅ Headless components</li>
              <li>✅ TypeScript support</li>
            </ul>
          </div>
          
          <div class="border rounded-lg p-6">
            <h3 class="text-lg font-semibold mb-3">melt v0.39.0</h3>
            <ul class="space-y-2 text-sm">
              <li>✅ Svelte 5 native</li>
              <li>✅ Builder pattern API</li>
              <li>✅ Animation support</li>
              <li>✅ Event handling</li>
            </ul>
          </div>
          
          <div class="border rounded-lg p-6">
            <h3 class="text-lg font-semibold mb-3">shadcn-svelte</h3>
            <ul class="space-y-2 text-sm">
              <li>✅ Tailwind CSS</li>
              <li>✅ Design tokens</li>
              <li>✅ Dark mode ready</li>
              <li>✅ Customizable</li>
            </ul>
          </div>
        </div>
        
        <div class="bg-muted p-4 rounded-lg">
          <h4 class="font-semibold mb-2">Architecture Benefits:</h4>
          <p class="text-sm">This combination provides the best of all worlds: bits-ui handles accessibility and behavior, melt provides advanced interactions, and shadcn-svelte delivers beautiful styling - all optimized for Svelte 5 and SvelteKit 2.</p>
        </div>
      </div>

    {:else if selectedDemo === 'buttons'}
      <div class="space-y-6">
        <h2 class="text-2xl font-bold">Button Components</h2>
        
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Button variant="default">Default</Button>
          <Button variant="secondary">Secondary</Button>
          <Button variant="outline">Outline</Button>
          <Button variant="ghost">Ghost</Button>
          <Button variant="legal">Legal</Button>
          <Button variant="evidence">Evidence</Button>
          <Button variant="case">Case</Button>
          <Button variant="destructive">Destructive</Button>
        </div>
        
        <div class="grid grid-cols-3 gap-4">
          <Button size="sm">Small</Button>
          <Button size="default">Default</Button>
          <Button size="lg">Large</Button>
        </div>
        
        <div class="bg-muted p-4 rounded-lg">
          <h4 class="font-semibold mb-2">Implementation:</h4>
          <p class="text-sm">Uses class-variance-authority for variant management and integrates with XState for state management and analytics tracking.</p>
        </div>
      </div>

    {:else if selectedDemo === 'dialogs'}
      <div class="space-y-6">
        <h2 class="text-2xl font-bold">Dialog Components (bits-ui)</h2>
        
        <div class="flex space-x-4">
          <button 
            use:melt={$dialogTrigger}
            class="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
          >
            Open Dialog
          </button>
        </div>
        
        <div class="bg-muted p-4 rounded-lg">
          <h4 class="font-semibold mb-2">Features:</h4>
          <ul class="text-sm space-y-1">
            <li>• Accessible keyboard navigation</li>
            <li>• Focus management</li>
            <li>• Portal rendering</li>
            <li>• Escape key handling</li>
          </ul>
        </div>
      </div>

    {:else if selectedDemo === 'accordion'}
      <div class="space-y-6">
        <h2 class="text-2xl font-bold">Accordion Component (melt-ui)</h2>
        
        <div use:melt={$accordionRoot} class="space-y-2">
          <div use:melt={$accordionItem('item-1')}>
            <button 
              use:melt={$accordionTrigger('item-1')}
              class="w-full text-left p-4 bg-muted hover:bg-muted/80 rounded-lg font-medium"
            >
              What is bits-ui?
            </button>
            <div 
              use:melt={$accordionContent('item-1')}
              class="px-4 pb-4 text-sm text-muted-foreground"
            >
              bits-ui is a headless component library for Svelte that provides accessible, customizable UI primitives.
            </div>
          </div>
          
          <div use:melt={$accordionItem('item-2')}>
            <button 
              use:melt={$accordionTrigger('item-2')}
              class="w-full text-left p-4 bg-muted hover:bg-muted/80 rounded-lg font-medium"
            >
              What is melt-ui?
            </button>
            <div 
              use:melt={$accordionContent('item-2')}
              class="px-4 pb-4 text-sm text-muted-foreground"
            >
              melt-ui provides builder functions that create accessible, keyboard-navigable components with smooth animations.
            </div>
          </div>
          
          <div use:melt={$accordionItem('item-3')}>
            <button 
              use:melt={$accordionTrigger('item-3')}
              class="w-full text-left p-4 bg-muted hover:bg-muted/80 rounded-lg font-medium"
            >
              How do they work together?
            </button>
            <div 
              use:melt={$accordionContent('item-3')}
              class="px-4 pb-4 text-sm text-muted-foreground"
            >
              bits-ui and melt-ui complement each other - bits-ui for complex components, melt-ui for interactive behaviors and animations.
            </div>
          </div>
        </div>
      </div>

    {:else if selectedDemo === 'forms'}
      <div class="space-y-6">
        <h2 class="text-2xl font-bold">Form Components</h2>
        
        <form class="space-y-4 max-w-md">
          <div class="space-y-2">
            <label for="name" class="text-sm font-medium">Name</label>
            <input 
              id="name"
              type="text" 
              placeholder="Enter your name"
              class="w-full p-2 border border-input rounded-md focus:ring-2 focus:ring-primary focus:border-transparent"
            />
          </div>
          
          <div class="space-y-2">
            <label for="email" class="text-sm font-medium">Email</label>
            <input 
              id="email"
              type="email" 
              placeholder="Enter your email"
              class="w-full p-2 border border-input rounded-md focus:ring-2 focus:ring-primary focus:border-transparent"
            />
          </div>
          
          <div class="space-y-2">
            <label for="message" class="text-sm font-medium">Message</label>
            <textarea 
              id="message"
              rows="4"
              placeholder="Enter your message"
              class="w-full p-2 border border-input rounded-md focus:ring-2 focus:ring-primary focus:border-transparent"
            ></textarea>
          </div>
          
          <Button type="submit" class="w-full">Submit Form</Button>
        </form>
        
        <div class="bg-muted p-4 rounded-lg">
          <h4 class="font-semibold mb-2">Form Features:</h4>
          <p class="text-sm">Integrates with sveltekit-superforms for validation and XState for form state management.</p>
        </div>
      </div>
    {/if}
  </div>
  
  <!-- Package Info -->
  <div class="bg-card border rounded-lg p-6">
    <h3 class="text-lg font-semibold mb-4">Current Package Versions</h3>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
      <div class="flex justify-between">
        <span>bits-ui:</span>
        <span class="font-mono">v2.9.4</span>
      </div>
      <div class="flex justify-between">
        <span>melt:</span>
        <span class="font-mono">v0.39.0</span>
      </div>
      <div class="flex justify-between">
        <span>shadcn-svelte:</span>
        <span class="font-mono">v1.0.7</span>
      </div>
    </div>
  </div>
</div>

<!-- Dialog Portal -->
{#if $dialogOpen}
  <div use:melt={$dialogOverlay} class="fixed inset-0 bg-black/50 z-50">
    <div 
      use:melt={$dialogContent}
      class="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg p-6 shadow-lg max-w-md w-full mx-4"
    >
      <h2 use:melt={$dialogTitle} class="text-lg font-semibold mb-4">
        bits-ui Dialog Example
      </h2>
      <p class="text-sm text-muted-foreground mb-4">
        This is a fully accessible dialog component built with bits-ui v2.9.4 for Svelte 5.
      </p>
      <div class="flex justify-end space-x-2">
        <button 
          use:melt={$dialogClose}
          class="px-4 py-2 border border-input rounded-md hover:bg-muted"
        >
          Cancel
        </button>
        <button 
          use:melt={$dialogClose}
          class="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
        >
          OK
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .container {
    max-width: 1200px;
  }
</style>