<script lang="ts">
  import { goto } from '$app/navigation';
  
  let message = $state('');
  let selectedOS = $state('windows');
  let isDialogOpen = $state(false);
  let isLoading = $state(false);
  let progress = $state(0);
  let radioSelected = $state('option1');
  
  function startProcess() {
    isLoading = true;
    progress = 0;
    
    const interval = setInterval(() => {
      progress += Math.random() * 20;
      if (progress >= 100) {
        clearInterval(interval);
        progress = 100;
        setTimeout(() => {
          isLoading = false;
          progress = 0;
          message = '🎮 Retro processing complete! YoRHa systems online.';
        }, 500);
      }
    }, 200);
  }
</script>

<div class="nes-yorha-demo">
  <!-- Page Header -->
  <div class="yorha-card p-6 mb-6">
    <h1 class="text-3xl font-bold mb-4 flex items-center gap-4">
      <span class="nes-text is-primary">🎮</span>
      NES.css + YoRHa + UnoCSS Demo
    </h1>
    <p class="text-nier-text-secondary mb-4">
      A hybrid interface combining retro 8-bit NES.css components with YoRHa's futuristic design system and UnoCSS utilities.
    </p>
    
    <div class="flex flex-wrap gap-2 mb-4">
      <span class="bits-badge-secondary">NES.css v2.2.1</span>
      <span class="bits-badge-secondary">YoRHa Theme</span>
      <span class="bits-badge-secondary">UnoCSS v66.4.2</span>
      <span class="bits-badge-secondary">SvelteKit 2</span>
    </div>
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    
    <!-- Left Column: NES.css Components -->
    <div class="space-y-6">
      <div class="nes-container with-title is-rounded">
        <p class="title">🎮 Classic NES.css Components</p>
        
        <!-- NES Buttons -->
        <div class="mb-4">
          <h3 class="nes-text is-primary mb-2">Buttons</h3>
          <div class="flex flex-wrap gap-2">
            <button type="button" class="nes-btn">Normal</button>
            <button type="button" class="nes-btn is-primary">Primary</button>
            <button type="button" class="nes-btn is-success">Success</button>
            <button type="button" class="nes-btn is-warning">Warning</button>
            <button type="button" class="nes-btn is-error">Error</button>
            <button type="button" class="nes-btn is-disabled">Disabled</button>
          </div>
        </div>

        <!-- NES Input -->
        <div class="mb-4">
          <h3 class="nes-text is-primary mb-2">Input Field</h3>
          <div class="nes-field">
            <label for="name_field">Your name</label>
            <input type="text" id="name_field" class="nes-input" placeholder="Enter retro message...">
          </div>
        </div>

        <!-- NES Select -->
        <div class="mb-4">
          <h3 class="nes-text is-primary mb-2">Select Dropdown</h3>
          <div class="nes-select">
            <select bind:value={selectedOS}>
              <option value="windows">Windows 95</option>
              <option value="mac">Mac OS</option>
              <option value="linux">Linux</option>
              <option value="dos">MS-DOS</option>
            </select>
          </div>
        </div>

        <!-- NES Radio -->
        <div class="mb-4">
          <h3 class="nes-text is-primary mb-2">Radio Buttons</h3>
          <label>
            <input type="radio" class="nes-radio" name="answer" bind:group={radioSelected} value="option1">
            <span>8-bit Graphics</span>
          </label>
          <label>
            <input type="radio" class="nes-radio" name="answer" bind:group={radioSelected} value="option2">
            <span>16-bit Sound</span>
          </label>
          <label>
            <input type="radio" class="nes-radio" name="answer" bind:group={radioSelected} value="option3">
            <span>32-bit Processing</span>
          </label>
        </div>
      </div>

      <!-- NES Progress -->
      <div class="nes-container with-title is-rounded">
        <p class="title">🎯 Progress & Loading</p>
        
        <div class="mb-4">
          <button type="button" class="nes-btn is-primary" on:onclick={startProcess} disabled={isLoading}>
            {isLoading ? 'Processing...' : 'Start Retro Process'}
          </button>
        </div>
        
        {#if isLoading}
          <progress class="nes-progress is-primary" value={progress} max="100"></progress>
          <p class="nes-text text-center mt-2">{Math.round(progress)}% Complete</p>
        {/if}
        
        {#if message}
          <div class="nes-container is-rounded mt-4">
            <p class="nes-text">{message}</p>
          </div>
        {/if}
      </div>
    </div>

    <!-- Right Column: Hybrid YoRHa + NES -->
    <div class="space-y-6">
      
      <!-- Hybrid Card -->
      <div class="yorha-card p-6">
        <h3 class="text-xl font-bold mb-4 flex items-center gap-2">
          <span class="nes-icon trophy"></span>
          YoRHa × NES Hybrid
        </h3>
        
        <!-- Mixed styling example -->
        <div class="space-y-4">
          <div class="flex items-center gap-4">
            <button type="button" class="yorha-button" on:onclick={() => goto('/evidenceboard')}>
              YoRHa Evidence Board
            </button>
            <button type="button" class="nes-btn is-success" on:onclick={() => goto('/demo')}>
              🎮 NES Demo Hub
            </button>
          </div>
          
          <!-- Status indicators -->
          <div class="flex items-center gap-4">
            <div class="flex items-center gap-2">
              <i class="nes-icon heart is-small"></i>
              <span class="text-nier-text-primary">YoRHa Systems:</span>
              <span class="ai-status-online">Online</span>
            </div>
          </div>
          
          <!-- Hybrid input -->
          <div class="nes-field">
            <label for="hybrid_input" class="text-nier-text-primary">Hybrid Command Input</label>
            <input 
              type="text" 
              id="hybrid_input" 
              class="nes-input yorha-input" 
              placeholder="Enter YoRHa command..."
              bind:value={message}
            >
          </div>
        </div>
      </div>

      <!-- NES Dialogs -->
      <div class="nes-container with-title is-rounded">
        <p class="title">💬 Dialogs & Modals</p>
        
        <button type="button" class="nes-btn is-warning mb-4" on:onclick={() => isDialogOpen = true}>
          Open Retro Dialog
        </button>
        
        <!-- Balloons -->
        <div class="nes-balloon from-left mb-4">
          <p>This is a speech balloon from the left!</p>
        </div>
        
        <div class="nes-balloon from-right">
          <p>And this one is from the right! Perfect for retro gaming interfaces.</p>
        </div>
      </div>

      <!-- Lists -->
      <div class="nes-container with-title is-rounded">
        <p class="title">📝 Lists & Text</p>
        
        <div class="nes-list is-disc">
          <ul>
            <li>Retro 8-bit styling with NES.css</li>
            <li>Modern YoRHa futuristic theme</li>
            <li>UnoCSS utility classes</li>
            <li>Svelte 5 runes reactivity</li>
            <li>Perfect hybrid combination</li>
          </ul>
        </div>
        
        <p class="nes-text is-success mt-4">
          🎮 Combining the nostalgia of classic gaming with the sleek design of NieR: Automata!
        </p>
      </div>
    </div>
  </div>

  <!-- Bottom Section: Tables -->
  <div class="mt-6">
    <div class="nes-container with-title is-rounded">
      <p class="title">📊 Data Tables</p>
      
      <div class="nes-table-responsive">
        <table class="nes-table is-bordered is-centered">
          <thead>
            <tr>
              <th>Feature</th>
              <th>NES.css</th>
              <th>YoRHa</th>
              <th>UnoCSS</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>8-bit Styling</td>
              <td>✅</td>
              <td>❌</td>
              <td>❌</td>
              <td><span class="nes-text is-success">Active</span></td>
            </tr>
            <tr>
              <td>Futuristic Theme</td>
              <td>❌</td>
              <td>✅</td>
              <td>❌</td>
              <td><span class="nes-text is-success">Active</span></td>
            </tr>
            <tr>
              <td>Utility Classes</td>
              <td>❌</td>
              <td>✅</td>
              <td>✅</td>
              <td><span class="nes-text is-success">Active</span></td>
            </tr>
            <tr>
              <td>Reactivity</td>
              <td>❌</td>
              <td>❌</td>
              <td>❌</td>
              <td><span class="nes-text is-primary">Svelte 5</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- Modal Dialog -->
{#if isDialogOpen}
  <div class="nes-dialog" id="dialog-rounded">
    <form method="dialog">
      <p class="title">🎮 Retro System Message</p>
      <p>This is a classic 8-bit style dialog powered by NES.css! It integrates perfectly with your existing YoRHa theme and UnoCSS utilities.</p>
      <div class="flex gap-2 justify-end mt-4">
        <button type="button" class="nes-btn" on:onclick={() => isDialogOpen = false}>Cancel</button>
        <button type="button" class="nes-btn is-primary" on:onclick={() => {isDialogOpen = false; message = '🎮 Dialog confirmed!';}}>OK</button>
      </div>
    </form>
  </div>
{/if}

<style>
  .nes-yorha-demo {
    padding: 2rem;
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #2a2a2a 100%);
  }

  /* Hybrid styling for NES + YoRHa */
  .nes-input.yorha-input {
    background: var(--color-nier-bg-secondary);
    border-color: var(--color-nier-border-primary);
    color: var(--color-nier-text-primary);
  }

  .nes-input.yorha-input:focus {
    border-color: var(--color-nier-accent-warm);
    box-shadow: 0 0 0 2px rgba(255, 215, 0, 0.2);
  }

  /* Custom dialog styling */
  .nes-dialog {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1000;
    background: var(--color-nier-bg-primary);
    border: 4px solid var(--color-nier-border-primary);
    padding: 2rem;
    max-width: 400px;
    width: 90vw;
  }

  .nes-dialog::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    z-index: -1;
  }

  /* Enhanced table styling */
  .nes-table {
    background: var(--color-nier-bg-secondary);
    color: var(--color-nier-text-primary);
  }

  .nes-table th {
    background: var(--color-nier-bg-tertiary);
    color: var(--color-nier-accent-warm);
    font-weight: bold;
  }

  .nes-table td {
    border-color: var(--color-nier-border-secondary);
  }

  /* Animated elements */
  .nes-container {
    animation: fadeInUp 0.5s ease-out;
  }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  /* Responsive adjustments */
  @media (max-width: 768px) {
    .nes-yorha-demo {
      padding: 1rem;
    }
    
    .grid-cols-1.lg\\:grid-cols-2 {
      gap: 1rem;
    }
  }
</style>