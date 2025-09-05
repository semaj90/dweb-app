<script lang="ts">
  import { createDialog, createPopover, createDropdownMenu, melt } from '@melt-ui/svelte';
  import { fade } from 'svelte/transition';
  
  // Create dialog state
  const {
    elements: { trigger: dialogTrigger, overlay: dialogOverlay, content: dialogContent, title: dialogTitle, description: dialogDescription, close: dialogClose, portalled: dialogPortalled },
    states: { open: dialogOpen }
  } = createDialog();
  
  // Create popover state
  const {
    elements: { trigger: popoverTrigger, content: popoverContent, arrow: popoverArrow },
    states: { open: popoverOpen }
  } = createPopover();
  
  // Create dropdown state
  const {
    elements: { trigger: dropdownTrigger, menu: dropdownMenu, item: dropdownItem, separator: dropdownSeparator },
    states: { open: dropdownOpen }
  } = createDropdownMenu();
</script>

<div class="mx-auto px-4 max-w-7xl">
  <!-- Header -->
  <div class="mx-auto px-4 max-w-7xl">
    <h1 class="mx-auto px-4 max-w-7xl">
      Melt UI + UnoCSS Integration Demo
    </h1>
    <p class="mx-auto px-4 max-w-7xl">
      Showcasing seamless integration between Melt UI components and UnoCSS utilities
    </p>

    <!-- Main Demo Grid -->
    <div class="mx-auto px-4 max-w-7xl">
      
      <!-- Dialog Demo -->
      <div class="mx-auto px-4 max-w-7xl">
        <h3 class="mx-auto px-4 max-w-7xl">
          <span class="mx-auto px-4 max-w-7xl">💬</span>
          Dialog Component
        </h3>
        <p class="mx-auto px-4 max-w-7xl">Modal dialogs with UnoCSS styling</p>
        
        <button 
          use:melt={$dialogTrigger}
          class="mx-auto px-4 max-w-7xl"
        >
          Open Dialog
        </button>
        
        <div use:melt={$dialogPortalled}>
          {#if $dialogOpen}
            <div use:melt={$dialogOverlay} class="mx-auto px-4 max-w-7xl" in:fade={{ duration: 150 " out:fade={{ duration: 150 "></div>
            <div 
              use:melt={$dialogContent}
              class="mx-auto px-4 max-w-7xl"
              in:fade={{ duration: 150 "
              out:fade={{ duration: 150  }}
            >
              <h2 use:melt={$dialogTitle} class="mx-auto px-4 max-w-7xl">
                UnoCSS Styled Dialog
              </h2>
              <p use:melt={$dialogDescription} class="mx-auto px-4 max-w-7xl">
                This dialog showcases Melt UI components styled with UnoCSS utility classes for a modern, responsive design.
              </p>
              
              <div class="mx-auto px-4 max-w-7xl">
                <button 
                  use:melt={$dialogClose}
                  class="mx-auto px-4 max-w-7xl"
                >
                  Cancel
                </button>
                <button 
                  use:melt={$dialogClose}
                  class="mx-auto px-4 max-w-7xl"
                >
                  Confirm
                </button>
              </div>
            </div>
          {/if}
        </div>
      </div>

      <!-- Popover Demo -->
      <div class="mx-auto px-4 max-w-7xl">
        <h3 class="mx-auto px-4 max-w-7xl">
          <span class="mx-auto px-4 max-w-7xl">🎯</span>
          Popover Component
        </h3>
        <p class="mx-auto px-4 max-w-7xl">Context menus and tooltips</p>
        
        <button 
          use:melt={$popoverTrigger}
          class="mx-auto px-4 max-w-7xl"
        >
          Show Popover
        </button>
        
        {#if $popoverOpen}
          <div 
            use:melt={$popoverContent}
            class="mx-auto px-4 max-w-7xl"
            in:fade={{ duration: 150 "
            out:fade={{ duration: 150  }}
          >
            <div class="mx-auto px-4 max-w-7xl">
              <h4 class="mx-auto px-4 max-w-7xl">UnoCSS Features</h4>
              <ul class="mx-auto px-4 max-w-7xl">
                <li class="mx-auto px-4 max-w-7xl">
                  <span class="mx-auto px-4 max-w-7xl"></span>
                  Utility-first CSS framework
                </li>
                <li class="mx-auto px-4 max-w-7xl">
                  <span class="mx-auto px-4 max-w-7xl"></span>
                  On-demand compilation
                </li>
                <li class="mx-auto px-4 max-w-7xl">
                  <span class="mx-auto px-4 max-w-7xl"></span>
                  Highly customizable
                </li>
              </ul>
            </div>
          </div>
        {/if}
      </div>

      <!-- Dropdown Demo -->
      <div class="mx-auto px-4 max-w-7xl">
        <h3 class="mx-auto px-4 max-w-7xl">
          <span class="mx-auto px-4 max-w-7xl">📋</span>
          Dropdown Menu
        </h3>
        <p class="mx-auto px-4 max-w-7xl">Navigation and action menus</p>
        
        <button 
          use:melt={$dropdownTrigger}
          class="mx-auto px-4 max-w-7xl"
        >
          Open Menu
          <span class="mx-auto px-4 max-w-7xl">▼</span>
        </button>
        
        {#if $dropdownOpen}
          <div 
            use:melt={$dropdownMenu}
            class="mx-auto px-4 max-w-7xl"
            in:fade={{ duration: 150 "
            out:fade={{ duration: 150  }}
          >
            <div use:melt={$dropdownItem} class="mx-auto px-4 max-w-7xl">
              <span class="mx-auto px-4 max-w-7xl">⚙️</span>
              Settings
            </div>
            <div use:melt={$dropdownItem} class="mx-auto px-4 max-w-7xl">
              <span class="mx-auto px-4 max-w-7xl">👤</span>
              Profile
            </div>
            <div use:melt={$dropdownSeparator} class="mx-auto px-4 max-w-7xl"></div>
            <div use:melt={$dropdownItem} class="mx-auto px-4 max-w-7xl">
              <span class="mx-auto px-4 max-w-7xl">🚪</span>
              Sign out
            </div>
          </div>
        {/if}
      </div>
    </div>

    <!-- Feature Showcase -->
    <div class="mx-auto px-4 max-w-7xl">
      <h3 class="mx-auto px-4 max-w-7xl">
        Integration Highlights
      </h3>
      
      <div class="mx-auto px-4 max-w-7xl">
        <div>
          <h4 class="mx-auto px-4 max-w-7xl">Bits UI Features</h4>
          <ul class="mx-auto px-4 max-w-7xl">
            <li class="mx-auto px-4 max-w-7xl">
              <span class="mx-auto px-4 max-w-7xl">✓</span>
              Accessible components out of the box
            </li>
            <li class="mx-auto px-4 max-w-7xl">
              <span class="mx-auto px-4 max-w-7xl">✓</span>
              Keyboard navigation support
            </li>
            <li class="mx-auto px-4 max-w-7xl">
              <span class="mx-auto px-4 max-w-7xl">✓</span>
              Customizable with any CSS framework
            </li>
            <li class="mx-auto px-4 max-w-7xl">
              <span class="mx-auto px-4 max-w-7xl">✓</span>
              Built for Svelte with TypeScript
            </li>
          </ul>
        </div>
        
        <div>
          <h4 class="mx-auto px-4 max-w-7xl">UnoCSS Benefits</h4>
          <ul class="mx-auto px-4 max-w-7xl">
            <li class="mx-auto px-4 max-w-7xl">
              <span class="mx-auto px-4 max-w-7xl">✓</span>
              Instant CSS generation
            </li>
            <li class="mx-auto px-4 max-w-7xl">
              <span class="mx-auto px-4 max-w-7xl">✓</span>
              Minimal bundle size
            </li>
            <li class="mx-auto px-4 max-w-7xl">
              <span class="mx-auto px-4 max-w-7xl">✓</span>
              Flexible theming system
            </li>
            <li class="mx-auto px-4 max-w-7xl">
              <span class="mx-auto px-4 max-w-7xl">✓</span>
              IDE autocompletion support
            </li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Status Indicator -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <span class="mx-auto px-4 max-w-7xl"></span>
        Bits UI + UnoCSS Integration Active
      </div>
    </div>
  </div>
</div>
