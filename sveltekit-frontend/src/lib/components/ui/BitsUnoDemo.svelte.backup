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

<div class="container mx-auto px-4">
  <!-- Header -->
  <div class="container mx-auto px-4">
    <h1 class="container mx-auto px-4">
      Melt UI + UnoCSS Integration Demo
    </h1>
    <p class="container mx-auto px-4">
      Showcasing seamless integration between Melt UI components and UnoCSS utilities
    </p>

    <!-- Main Demo Grid -->
    <div class="container mx-auto px-4">

      <!-- Dialog Demo -->
      <div class="container mx-auto px-4">
        <h3 class="container mx-auto px-4">
          <span class="container mx-auto px-4">💬</span>
          Dialog Component
        </h3>
        <p class="container mx-auto px-4">Modal dialogs with UnoCSS styling</p>

        <button
          use:melt={$dialogTrigger}
          class="container mx-auto px-4"
        >
          Open Dialog
        </button>

        <div use:melt={$dialogPortalled}>
          {#if $dialogOpen}
            <div use:melt={$dialogOverlay} class="container mx-auto px-4" in:fade={{ duration: 150 }} out:fade={{ duration: 150 }}></div>
            <div
              use:melt={$dialogContent}
              class="container mx-auto px-4"
              in:fade={{ duration: 150 }}
              out:fade={{ duration: 150 }}
            >
              <h2 use:melt={$dialogTitle} class="container mx-auto px-4">
                UnoCSS Styled Dialog
              </h2>
              <p use:melt={$dialogDescription} class="container mx-auto px-4">
                This dialog showcases Melt UI components styled with UnoCSS utility classes for a modern, responsive design.
              </p>

              <div class="container mx-auto px-4">
                <button
                  use:melt={$dialogClose}
                  class="container mx-auto px-4"
                >
                  Cancel
                </button>
                <button
                  use:melt={$dialogClose}
                  class="container mx-auto px-4"
                >
                  Confirm
                </button>
              </div>
            </div>
          {/if}
        </div>
      </div>

      <!-- Popover Demo -->
      <div class="container mx-auto px-4">
        <h3 class="container mx-auto px-4">
          <span class="container mx-auto px-4">🎯</span>
          Popover Component
        </h3>
        <p class="container mx-auto px-4">Context menus and tooltips</p>

        <button
          use:melt={$popoverTrigger}
          class="container mx-auto px-4"
        >
          Show Popover
        </button>

        {#if $popoverOpen}
          <div
            use:melt={$popoverContent}
            class="container mx-auto px-4"
            in:fade={{ duration: 150 }}
            out:fade={{ duration: 150 }}
          >
            <div class="container mx-auto px-4">
              <h4 class="container mx-auto px-4">UnoCSS Features</h4>
              <ul class="container mx-auto px-4">
                <li class="container mx-auto px-4">
                  <span class="container mx-auto px-4"></span>
                  Utility-first CSS framework
                </li>
                <li class="container mx-auto px-4">
                  <span class="container mx-auto px-4"></span>
                  On-demand compilation
                </li>
                <li class="container mx-auto px-4">
                  <span class="container mx-auto px-4"></span>
                  Highly customizable
                </li>
              </ul>
            </div>
          </div>
        {/if}
      </div>

      <!-- Dropdown Demo -->
      <div class="container mx-auto px-4">
        <h3 class="container mx-auto px-4">
          <span class="container mx-auto px-4">📋</span>
          Dropdown Menu
        </h3>
        <p class="container mx-auto px-4">Navigation and action menus</p>

        <button
          use:melt={$dropdownTrigger}
          class="container mx-auto px-4"
        >
          Open Menu
          <span class="container mx-auto px-4">▼</span>
        </button>

        {#if $dropdownOpen}
          <div
            use:melt={$dropdownMenu}
            class="container mx-auto px-4"
            in:fade={{ duration: 150 }}
            out:fade={{ duration: 150  }}
          >
            <div use:melt={$dropdownItem} class="container mx-auto px-4">
              <span class="container mx-auto px-4">⚙️</span>
              Settings
            </div>
            <div use:melt={$dropdownItem} class="container mx-auto px-4">
              <span class="container mx-auto px-4">👤</span>
              Profile
            </div>
            <div use:melt={$dropdownSeparator} class="container mx-auto px-4"></div>
            <div use:melt={$dropdownItem} class="container mx-auto px-4">
              <span class="container mx-auto px-4">🚪</span>
              Sign out
            </div>
          </div>
        {/if}
      </div>
    </div>

    <!-- Feature Showcase -->
    <div class="container mx-auto px-4">
      <h3 class="container mx-auto px-4">
        Integration Highlights
      </h3>

      <div class="container mx-auto px-4">
        <div>
          <h4 class="container mx-auto px-4">Bits UI Features</h4>
          <ul class="container mx-auto px-4">
            <li class="container mx-auto px-4">
              <span class="container mx-auto px-4">✓</span>
              Accessible components out of the box
            </li>
            <li class="container mx-auto px-4">
              <span class="container mx-auto px-4">✓</span>
              Keyboard navigation support
            </li>
            <li class="container mx-auto px-4">
              <span class="container mx-auto px-4">✓</span>
              Customizable with any CSS framework
            </li>
            <li class="container mx-auto px-4">
              <span class="container mx-auto px-4">✓</span>
              Built for Svelte with TypeScript
            </li>
          </ul>
        </div>

        <div>
          <h4 class="container mx-auto px-4">UnoCSS Benefits</h4>
          <ul class="container mx-auto px-4">
            <li class="container mx-auto px-4">
              <span class="container mx-auto px-4">✓</span>
              Instant CSS generation
            </li>
            <li class="container mx-auto px-4">
              <span class="container mx-auto px-4">✓</span>
              Minimal bundle size
            </li>
            <li class="container mx-auto px-4">
              <span class="container mx-auto px-4">✓</span>
              Flexible theming system
            </li>
            <li class="container mx-auto px-4">
              <span class="container mx-auto px-4">✓</span>
              IDE autocompletion support
            </li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Status Indicator -->
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <span class="container mx-auto px-4"></span>
        Bits UI + UnoCSS Integration Active
      </div>
    </div>
  </div>
</div>
