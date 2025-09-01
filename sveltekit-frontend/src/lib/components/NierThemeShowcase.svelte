<script lang="ts">
  import { Button } from 'bits-ui';
  // import { createDialog, melt } from 'melt'; // Removed melt dependency
  import { fly, fade } from 'svelte/transition';
  import { writable } from 'svelte/store';
  
  // Demo states
  let isDarkMode = $state(false);
  let showModal = $state(false);
  let activeTab = $state('overview');
  let inputValue = $state('');
  let selectedStatus = $state('active');
  
  // Create melt-ui dialog
  // const {
  //   elements: { trigger, overlay, content, title, description, close },
  //   states: { open }
  // } = createDialog({
  //   forceVisible: true,
  // })
  
  // Demo data
  const demoCase = {
    id: 'CASE-2025-001',
    title: 'Project 2B Investigation',
    status: 'active',
    priority: 'critical',
    created: '2025-07-16',
    evidence: 12,
    witnesses: 5
  }
  
  const statuses = ['active', 'pending', 'closed', 'archived']
  const priorities = ['critical', 'high', 'medium', 'low']
</script>

<div class="min-h-screen bg-nier-white dark:bg-nier-black transition-colors duration-500">
  <!-- Header -->
  <nav class="nier-nav px-6 py-4">
    <div class="max-w-7xl mx-auto flex items-center justify-between">
      <h1 class="text-2xl font-display nier-heading">NieR: Legal System</h1>
      
      <div class="flex items-center gap-6">
        <a href="#" class="nav-item">Cases</a>
        <a href="#" class="nav-item">Evidence</a>
        <a href="#" class="nav-item">Analytics</a>
        <a href="#" class="nav-item">AI Assistant</a>
        
        <button
          on:onclick={() => isDarkMode = !isDarkMode}
          class="nier-button-outline px-4 py-2 rounded-lg"
          class:dark={isDarkMode}
        >
          {isDarkMode ? '☀️' : '🌙'}
        </button>
      </div>
    </div>
  </nav>

  <!-- Main Content -->
  <main class="max-w-7xl mx-auto p-6 space-y-8">
    <!-- Hero Section -->
    <section class="text-center py-12">
      <h2 class="text-5xl font-display nier-heading mb-4">
        Legal Intelligence System
      </h2>
      <p class="text-xl text-nier-gray dark:text-nier-silver">
        Powered by YoRHa Combat Data Analysis
      </p>
    </section>

    <!-- Tabs -->
    <div class="flex gap-2 border-b nier-divider">
      {#each ['overview', 'components', 'forms', 'cards'] as tab}
        <button
          on:onclick={() => activeTab = tab}
          class="px-6 py-3 font-medium capitalize nier-transition"
          class:text-harvard-crimson={activeTab === tab}
          class:dark:text-digital-green={activeTab === tab}
          class:border-b-2={activeTab === tab}
          class:border-harvard-crimson={activeTab === tab}
          class:dark:border-digital-green={activeTab === tab}
        >
          {tab}
        </button>
      {/each}
    </div>

    <!-- Tab Content -->
    <div class="space-y-8">
      {#if activeTab === 'overview'}
        <!-- Color Palette -->
        <section class="nier-card p-8">
          <h3 class="text-2xl font-display nier-heading mb-6">Color System</h3>
          
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <!-- NieR Colors -->
            <div class="space-y-2">
              <h4 class="font-semibold text-nier-gray dark:text-nier-silver">NieR Palette</h4>
              <div class="space-y-2">
                <div class="flex items-center gap-3">
                  <div class="w-12 h-12 bg-nier-black rounded-lg nier-shadow"></div>
                  <span class="text-sm">Black</span>
                </div>
                <div class="flex items-center gap-3">
                  <div class="w-12 h-12 bg-nier-gray rounded-lg nier-shadow"></div>
                  <span class="text-sm">Gray</span>
                </div>
                <div class="flex items-center gap-3">
                  <div class="w-12 h-12 bg-nier-gold rounded-lg nier-shadow"></div>
                  <span class="text-sm">Gold</span>
                </div>
              </div>
            </div>
            
            <!-- Harvard Colors -->
            <div class="space-y-2">
              <h4 class="font-semibold text-nier-gray dark:text-nier-silver">Harvard Palette</h4>
              <div class="space-y-2">
                <div class="flex items-center gap-3">
                  <div class="w-12 h-12 bg-harvard-crimson rounded-lg nier-shadow"></div>
                  <span class="text-sm">Crimson</span>
                </div>
                <div class="flex items-center gap-3">
                  <div class="w-12 h-12 bg-harvard-crimson-light rounded-lg nier-shadow"></div>
                  <span class="text-sm">Light</span>
                </div>
                <div class="flex items-center gap-3">
                  <div class="w-12 h-12 bg-harvard-crimson-pale rounded-lg nier-shadow"></div>
                  <span class="text-sm">Pale</span>
                </div>
              </div>
            </div>
            
            <!-- Digital Colors -->
            <div class="space-y-2">
              <h4 class="font-semibold text-nier-gray dark:text-nier-silver">Digital Accent</h4>
              <div class="space-y-2">
                <div class="flex items-center gap-3">
                  <div class="w-12 h-12 bg-digital-green rounded-lg nier-glow"></div>
                  <span class="text-sm">Green</span>
                </div>
                <div class="flex items-center gap-3">
                  <div class="w-12 h-12 bg-digital-blue rounded-lg nier-shadow"></div>
                  <span class="text-sm">Blue</span>
                </div>
                <div class="flex items-center gap-3">
                  <div class="w-12 h-12 bg-digital-purple rounded-lg nier-shadow"></div>
                  <span class="text-sm">Purple</span>
                </div>
              </div>
            </div>
            
            <!-- Gradients -->
            <div class="space-y-2">
              <h4 class="font-semibold text-nier-gray dark:text-nier-silver">Gradients</h4>
              <div class="space-y-2">
                <div class="w-full h-12 nier-gradient-dark rounded-lg nier-shadow"></div>
                <div class="w-full h-12 nier-gradient-crimson rounded-lg nier-shadow"></div>
                <div class="w-full h-12 nier-gradient-digital rounded-lg nier-glow"></div>
              </div>
            </div>
          </div>
        </section>
      {/if}

      {#if activeTab === 'components'}
        <!-- Buttons -->
        <section class="nier-card p-8">
          <h3 class="text-2xl font-display nier-heading mb-6">Button Components</h3>
          
          <div class="flex flex-wrap gap-4">
            <button class="nier-button-primary">Primary Action</button>
            <button class="nier-button-crimson">Crimson Action</button>
            <button class="nier-button-gold">Gold Action</button>
            <button class="nier-button-digital">Digital Action</button>
            <button class="nier-button-outline text-nier-black dark:text-nier-white">
              Outline Action
            </button>
          </div>
          
          <div class="mt-6 p-4 bg-nier-white/50 dark:bg-nier-black/50 rounded-lg">
            <code class="text-sm font-mono text-digital-green">
              class="nier-button-primary"
            </code>
          </div>
        </section>

        <!-- Badges -->
        <section class="nier-card p-8">
          <h3 class="text-2xl font-display nier-heading mb-6">Status Badges</h3>
          
          <div class="flex flex-wrap gap-3">
            <span class="nier-badge-success">Active</span>
            <span class="nier-badge-warning">Pending</span>
            <span class="nier-badge-error">Critical</span>
            <span class="nier-badge-info">Information</span>
          </div>
        </section>

        <!-- Interactive Elements -->
        <section class="nier-card p-8">
          <h3 class="text-2xl font-display nier-heading mb-6">Interactive Elements</h3>
          
          <!-- Melt UI Dialog Example -->
          <button
            <!-- <!-- <!-- use:melt={$trigger} -->
            class="nier-button-crimson"
            on:on:on:click={() => showModal = true}
          >
            Open Modal Dialog
          </button>
          
          {#if showModal}
            <div
              <!-- <!-- <!-- use:melt={$overlay} -->
              class="nier-modal-overlay"
              transitifade={{ duration: 200 }}
></div>
            
            <div
              <!-- <!-- <!-- use:melt={$content} -->
              class="nier-modal"
              transitifly={{ y: 20, duration: 300 }}
            >
              <h2 <!-- <!-- <!-- use:melt={$title} --> class="text-2xl font-display nier-heading mb-4">
                System Alert
              </h2>
              <p <!-- <!-- <!-- use:melt={$description} --> class="text-nier-gray dark:text-nier-silver mb-6">
                This is a NieR: Automata styled modal dialog using Melt UI.
              </p>
              
              <div class="flex gap-4 justify-end">
                <button <!-- <!-- <!-- use:melt={$close} --> class="nier-button-outline px-4 py-2" on:on:on:click={() => showModal = false}>
                  Cancel
                </button>
                <button class="nier-button-digital px-4 py-2">
                  Confirm
                </button>
              </div>
            </div>
          {/if}
        </section>
      {/if}

      {#if activeTab === 'forms'}
        <!-- Form Elements -->
        <section class="nier-card p-8">
          <h3 class="text-2xl font-display nier-heading mb-6">Form Elements</h3>
          
          <div class="space-y-6 max-w-md">
            <!-- Input Field -->
            <div>
              <label class="block text-sm font-medium mb-2 text-nier-gray dark:text-nier-silver">
                Case Title
              </label>
              <input
                type="text"
                bind:value={inputValue}
                placeholder="Enter case title..."
                class="nier-input"
              />
            </div>
            
            <!-- Select Dropdown -->
            <div>
              <label class="block text-sm font-medium mb-2 text-nier-gray dark:text-nier-silver">
                Status
              </label>
              <select
                bind:value={selectedStatus}
                class="nier-input"
              >
                {#each statuses as status}
                  <option value={status}>{status}</option>
                {/each}
              </select>
            </div>
            
            <!-- Textarea -->
            <div>
              <label class="block text-sm font-medium mb-2 text-nier-gray dark:text-nier-silver">
                Case Description
              </label>
              <textarea
                rows="4"
                placeholder="Describe the case details..."
                class="nier-input resize-none"
              ></textarea>
            </div>
            
            <!-- Checkbox Group -->
            <div>
              <label class="block text-sm font-medium mb-2 text-nier-gray dark:text-nier-silver">
                Evidence Types
              </label>
              <div class="space-y-2">
                {#each ['Documents', 'Photos', 'Videos', 'Audio'] as type}
                  <label class="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      class="w-5 h-5 rounded border-nier-gray focus:ring-2 focus:ring-digital-green"
                    />
                    <span class="text-sm">{type}</span>
                  </label>
                {/each}
              </div>
            </div>
          </div>
        </section>
      {/if}

      {#if activeTab === 'cards'}
        <!-- Case Card -->
        <section class="grid md:grid-cols-2 gap-6">
          <!-- Standard Case Card -->
          <div class="case-card priority-{demoCase.priority}">
            <div class="flex justify-between items-start mb-4">
              <div>
                <h4 class="text-lg font-semibold nier-heading">{demoCase.title}</h4>
                <p class="text-sm text-nier-gray dark:text-nier-silver">{demoCase.id}</p>
              </div>
              <span class="nier-badge-success">{demoCase.status}</span>
            </div>
            
            <div class="grid grid-cols-3 gap-4 mb-4">
              <div class="text-center">
                <p class="text-2xl font-bold text-harvard-crimson dark:text-digital-green">
                  {demoCase.evidence}
                </p>
                <p class="text-xs text-nier-gray dark:text-nier-silver">Evidence</p>
              </div>
              <div class="text-center">
                <p class="text-2xl font-bold text-harvard-crimson dark:text-digital-green">
                  {demoCase.witnesses}
                </p>
                <p class="text-xs text-nier-gray dark:text-nier-silver">Witnesses</p>
              </div>
              <div class="text-center">
                <p class="text-sm font-medium">{demoCase.created}</p>
                <p class="text-xs text-nier-gray dark:text-nier-silver">Created</p>
              </div>
            </div>
            
            <div class="flex gap-2">
              <button class="nier-button-primary text-sm px-4 py-2">
                View Details
              </button>
              <button class="nier-button-outline text-sm px-4 py-2">
                Edit
              </button>
            </div>
          </div>
          
          <!-- Evidence Card -->
          <div class="evidence-item">
            <div class="flex items-start gap-4">
              <div class="w-16 h-16 bg-nier-gradient-digital rounded-lg flex items-center justify-center text-2xl">
                📄
              </div>
              <div class="flex-1">
                <h5 class="font-semibold nier-heading">Witness Statement #1</h5>
                <p class="text-sm text-nier-gray dark:text-nier-silver mt-1">
                  Uploaded 2 hours ago • PDF • 2.4MB
                </p>
                <div class="flex gap-2 mt-3">
                  <span class="nier-badge-info text-xs">Verified</span>
                  <span class="nier-badge-success text-xs">Original</span>
                </div>
              </div>
              <button class="opacity-0 group-hover:opacity-100 nier-transition">
                <span class="text-2xl">⋮</span>
              </button>
            </div>
          </div>
        </section>
        
        <!-- AI Assistant Card -->
        <section class="nier-panel p-6">
          <h3 class="text-xl font-display nier-heading mb-4">AI Legal Assistant</h3>
          
          <div class="space-y-4">
            <!-- Assistant Message -->
            <div class="ai-chat-assistant">
              <p class="text-sm">
                I've analyzed the case files. There are 3 key patterns in the evidence 
                that suggest a connection to the previous investigation.
              </p>
            </div>
            
            <!-- User Message -->
            <div class="ai-chat-user">
              <p class="text-sm">
                Can you elaborate on the connection patterns?
              </p>
            </div>
            
            <!-- Thinking Indicator -->
            <div class="ai-thinking">
              <div class="flex gap-1">
                <div class="w-2 h-2 bg-digital-green rounded-full animate-bounce"></div>
                <div class="w-2 h-2 bg-digital-green rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                <div class="w-2 h-2 bg-digital-green rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
              </div>
              <span class="text-sm">AI is analyzing...</span>
            </div>
          </div>
          
          <div class="mt-6 flex gap-2">
            <input
              type="text"
              placeholder="Ask about the case..."
              class="nier-input flex-1"
            />
            <button class="nier-button-digital px-6">
              Send
            </button>
          </div>
        </section>
      {/if}
    </div>

    <!-- Footer -->
    <footer class="mt-16 py-8 border-t nier-divider text-center">
      <p class="text-sm text-nier-gray dark:text-nier-silver">
        YoRHa Legal System v2.0 • For the Glory of Mankind
      </p>
    </footer>
  </main>
</div>

<style>
  /* @unocss-include */
  /* Add any component-specific styles here */
  :global(body) {
    transition: background-color 0.5s ease;
  }
  
  :global(.dark) {
    color-scheme: dark;
  }
</style>


