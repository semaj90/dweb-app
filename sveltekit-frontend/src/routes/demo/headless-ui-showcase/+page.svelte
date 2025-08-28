<script lang="ts">
  import { createDialog, melt, type CreateDialogProps } from '@melt-ui/svelte';
  import { fade, fly } from 'svelte/transition';
  import Button from '$lib/components/ui/Button.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  
  // Melt UI Dialog
  const {
    elements: { trigger, overlay, content, title, description, close },
    states: { open }
  } = createDialog();
  
  // Bits UI components (imported from your existing library)
  let selectedValue = $state('option1');
  let inputValue = $state('');
  let isLoading = $state(false);
  let progress = $state(0);
  let notifications = $state<Array<{id: number, message: string, type: 'success' | 'warning' | 'error'}>>([]);
  
  function addNotification(type: 'success' | 'warning' | 'error', message: string) {
    const id = Date.now();
    notifications = [...notifications, { id, message, type }];
    
    // Auto remove after 3 seconds
    setTimeout(() => {
      notifications = notifications.filter(n => n.id !== id);
    }, 3000);
  }
  
  function simulateAsyncOperation() {
    isLoading = true;
    progress = 0;
    
    const interval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress >= 100) {
        clearInterval(interval);
        progress = 100;
        setTimeout(() => {
          isLoading = false;
          progress = 0;
          addNotification('success', 'Operation completed successfully! 🎮');
        }, 500);
      }
    }, 200);
  }
</script>

<svelte:head>
  <title>Headless UI Showcase - Bits UI + Melt UI + NES.css + UnoCSS</title>
</svelte:head>

<div class="headless-ui-showcase">
  <!-- Header -->
  <div class="yorha-card p-6 mb-6">
    <h1 class="text-4xl font-bold mb-4 flex items-center gap-4">
      <span class="nes-icon star"></span>
      Headless UI Showcase
    </h1>
    <p class="text-nier-text-secondary mb-4">
      Combining <strong>Bits UI v2</strong>, <strong>Melt UI</strong>, <strong>NES.css</strong>, and <strong>UnoCSS</strong> 
      for the ultimate flexible component system.
    </p>
    
    <div class="flex flex-wrap gap-2">
      <span class="bits-badge-default">Bits UI v2</span>
      <span class="bits-badge-secondary">Melt UI</span>
      <span class="bits-badge-outline">NES.css</span>
      <span class="vector-confidence-badge bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400">UnoCSS</span>
    </div>
  </div>

  <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
    
    <!-- Column 1: Melt UI Components -->
    <div class="space-y-6">
      
      <!-- Melt UI Dialog with NES.css styling -->
      <Card class="nes-container with-title">
        <CardHeader>
          <CardTitle class="nes-text is-primary">🎭 Melt UI Dialog</CardTitle>
        </CardHeader>
        <CardContent class="space-y-4">
          <p class="text-sm text-nier-text-secondary">
            Melt UI provides the behavior, NES.css provides the retro styling.
          </p>
          
          <button 
            use:melt={$trigger}
            class="nes-btn is-primary w-full"
          >
            Open Retro Dialog
          </button>
        </CardContent>
      </Card>

      <!-- Hybrid Form Example -->
      <div class="nes-container with-title is-rounded">
        <p class="title">📝 Hybrid Form Components</p>
        
        <div class="space-y-4">
          <!-- NES Input with UnoCSS layout -->
          <div class="nes-field">
            <label for="hybrid-input" class="text-nier-text-primary">Command Input</label>
            <input 
              type="text" 
              id="hybrid-input"
              class="nes-input bg-nier-bg-secondary border-nier-border-primary text-nier-text-primary"
              placeholder="Enter command..."
              bind:value={inputValue}
            >
          </div>
          
          <!-- NES Select with custom styling -->
          <div class="nes-field">
            <label class="text-nier-text-primary">System Mode</label>
            <div class="nes-select">
              <select bind:value={selectedValue} class="bg-nier-bg-secondary text-nier-text-primary">
                <option value="option1">🎮 Gaming Mode</option>
                <option value="option2">🤖 AI Assistant</option>
                <option value="option3">⚖️ Legal Analysis</option>
                <option value="option4">🔍 Evidence Search</option>
              </select>
            </div>
          </div>
          
          <!-- Action buttons -->
          <div class="flex gap-2 mt-4">
            <button 
              class="nes-btn is-success flex-1"
              onclick={() => addNotification('success', `Command executed: ${inputValue || 'default'}`)}
            >
              Execute
            </button>
            <button 
              class="nes-btn is-warning flex-1"
              onclick={simulateAsyncOperation}
              disabled={isLoading}
            >
              {isLoading ? 'Processing...' : 'Async Process'}
            </button>
          </div>
          
          <!-- Progress indicator -->
          {#if isLoading}
            <div class="mt-4">
              <progress class="nes-progress is-success" value={progress} max="100"></progress>
              <p class="nes-text text-center text-xs mt-1">{Math.round(progress)}% Complete</p>
            </div>
          {/if}
        </div>
      </div>
    </div>

    <!-- Column 2: Bits UI Components -->
    <div class="space-y-6">
      
      <!-- Bits UI Components with NES styling -->
      <Card class="nes-container with-title">
        <CardHeader>
          <CardTitle class="nes-text is-primary">🎯 Bits UI + Styling</CardTitle>
        </CardHeader>
        <CardContent class="space-y-4">
          <p class="text-sm text-nier-text-secondary">
            Using your existing Bits UI components with NES.css and YoRHa styling.
          </p>
          
          <!-- Your existing Button component with mixed styling -->
          <div class="flex flex-wrap gap-2">
            <Button 
              variant="default" 
              class="nes-btn is-primary"
              onclick={() => addNotification('success', 'Bits UI Button clicked! 🎮')}
            >
              Hybrid Button
            </Button>
            
            <Button 
              variant="outline" 
              class="nes-btn"
              onclick={() => addNotification('warning', 'Warning: Retro mode activated!')}
            >
              Outline + NES
            </Button>
          </div>
          
          <!-- Status indicators -->
          <div class="space-y-2">
            <div class="flex items-center gap-2">
              <span class="ai-status-online"></span>
              <span class="nes-text is-success text-sm">Systems Online</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="ai-status-processing"></span>
              <span class="nes-text text-sm">Processing: {selectedValue}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <!-- Interactive Gaming Panel -->
      <div class="nes-container is-dark with-title">
        <p class="title">🎮 Gaming Interface</p>
        
        <div class="space-y-4">
          <!-- NES Icons -->
          <div class="flex justify-center gap-4 text-2xl">
            <i class="nes-icon trophy"></i>
            <i class="nes-icon coin"></i>
            <i class="nes-icon heart"></i>
            <i class="nes-icon star"></i>
          </div>
          
          <!-- Game-style controls -->
          <div class="grid grid-cols-2 gap-2">
            <button class="nes-btn is-success text-xs">▲ UP</button>
            <button class="nes-btn is-error text-xs">● ACTION</button>
            <button class="nes-btn is-warning text-xs">▼ DOWN</button>
            <button class="nes-btn is-primary text-xs">■ MENU</button>
          </div>
          
          <!-- Health/Status bar -->
          <div class="space-y-2">
            <div class="flex justify-between text-xs">
              <span>Health</span>
              <span>85/100</span>
            </div>
            <progress class="nes-progress is-pattern" value="85" max="100"></progress>
          </div>
        </div>
      </div>
    </div>

    <!-- Column 3: Advanced Examples -->
    <div class="space-y-6">
      
      <!-- Notification System -->
      <div class="nes-container with-title is-rounded">
        <p class="title">📢 Live Notifications</p>
        
        <div class="space-y-2 max-h-40 overflow-y-auto">
          {#each notifications as notification (notification.id)}
            <div 
              class="nes-container is-rounded text-xs p-2"
              class:is-success={notification.type === 'success'}
              class:is-warning={notification.type === 'warning'}
              class:is-error={notification.type === 'error'}
              transition:fly={{ y: -20, duration: 300 }}
            >
              {notification.message}
            </div>
          {/each}
          
          {#if notifications.length === 0}
            <p class="nes-text text-center text-xs text-nier-text-muted">
              No notifications
            </p>
          {/if}
        </div>
      </div>

      <!-- Data Table with mixed styling -->
      <div class="nes-container with-title">
        <p class="title">📊 Component Matrix</p>
        
        <div class="nes-table-responsive">
          <table class="nes-table is-bordered is-centered text-xs">
            <thead>
              <tr>
                <th>Component</th>
                <th>Library</th>
                <th>Styling</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Dialog</td>
                <td><span class="bits-badge-secondary text-xs">Melt UI</span></td>
                <td><span class="nes-badge is-splited"><span class="is-dark">NES</span><span class="is-primary">CSS</span></span></td>
              </tr>
              <tr>
                <td>Button</td>
                <td><span class="bits-badge-default text-xs">Bits UI</span></td>
                <td><span class="vector-confidence-badge bg-green-100 text-green-800 text-xs">Mixed</span></td>
              </tr>
              <tr>
                <td>Layout</td>
                <td><span class="bits-badge-outline text-xs">Custom</span></td>
                <td><span class="vector-confidence-badge bg-blue-100 text-blue-800 text-xs">UnoCSS</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Advanced Styling Example -->
      <Card class="bg-nier-bg-secondary border-nier-border-primary">
        <CardHeader class="border-b border-nier-border-secondary">
          <CardTitle class="text-nier-accent-warm">🚀 Advanced Integration</CardTitle>
        </CardHeader>
        <CardContent class="p-4 space-y-3">
          <p class="text-xs text-nier-text-secondary">
            This card uses Bits UI structure + YoRHa colors + UnoCSS utilities.
          </p>
          
          <div class="flex items-center gap-2">
            <div class="w-3 h-3 rounded-full bg-ai-status-online animate-pulse"></div>
            <span class="text-xs">Multi-library integration active</span>
          </div>
          
          <div class="flex justify-between text-xs">
            <span>Compatibility:</span>
            <span class="text-nier-accent-warm font-bold">100%</span>
          </div>
        </CardContent>
      </Card>
    </div>
  </div>
</div>

<!-- Melt UI Dialog Implementation -->
{#if $open}
  <div use:melt={$overlay} 
       class="fixed inset-0 z-50 bg-black/70"
       transition:fade={{ duration: 150 }}>
  </div>
  
  <div
    use:melt={$content}
    class="nes-dialog fixed left-1/2 top-1/2 z-50 w-full max-w-md transform -translate-x-1/2 -translate-y-1/2"
    transition:fly={{ y: -50, duration: 200 }}
  >
    <div class="nes-container with-title is-rounded bg-nier-bg-primary">
      <button
        use:melt={$close}
        class="absolute top-2 right-2 nes-btn is-error text-xs w-8 h-8 p-0"
      >
        ✕
      </button>
      
      <h2 use:melt={$title} class="title text-nier-accent-warm">🎮 System Dialog</h2>
      
      <div use:melt={$description} class="space-y-4">
        <p class="nes-text text-sm">
          This dialog demonstrates Melt UI's headless behavior combined with NES.css retro styling 
          and YoRHa color theming.
        </p>
        
        <div class="nes-balloon from-left">
          <p class="text-xs">The behavior is from Melt UI, the styling is from NES.css!</p>
        </div>
        
        <div class="flex gap-2">
          <button 
            class="nes-btn is-success flex-1"
            onclick={() => {
              addNotification('success', 'Dialog action confirmed! 🎯');
              $open = false;
            }}
          >
            Confirm
          </button>
          <button 
            use:melt={$close}
            class="nes-btn flex-1"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .headless-ui-showcase {
    padding: 2rem;
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%);
  }

  /* Custom dialog styling to integrate NES + YoRHa */
  .nes-dialog .nes-container {
    border-color: var(--color-nier-border-primary);
    box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
  }

  /* Enhanced button hover effects */
  .nes-btn:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  }

  /* Notification animations */
  .nes-container.is-success {
    border-color: #00ff41;
    background-color: rgba(0, 255, 65, 0.1);
  }

  .nes-container.is-warning {
    border-color: #ffaa00;
    background-color: rgba(255, 170, 0, 0.1);
  }

  .nes-container.is-error {
    border-color: #ff0041;
    background-color: rgba(255, 0, 65, 0.1);
  }

  /* Responsive adjustments */
  @media (max-width: 768px) {
    .headless-ui-showcase {
      padding: 1rem;
    }
    
    .grid-cols-1.xl\\:grid-cols-3 {
      gap: 1rem;
    }
    
    .nes-dialog {
      max-width: 90vw;
    }
  }

  /* Smooth animations */
  * {
    transition: all 0.15s ease;
  }

  /* Status indicator animations */
  .ai-status-online {
    animation: pulse 2s infinite;
  }

  .ai-status-processing {
    animation: bounce 1s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  @keyframes bounce {
    0%, 20%, 53%, 80%, 100% { transform: translate3d(0,0,0); }
    40%, 43% { transform: translate3d(0, -5px, 0); }
    70% { transform: translate3d(0, -3px, 0); }
    90% { transform: translate3d(0, -1px, 0); }
  }
</style>