<script lang="ts">
  import { Dialog, Button, Badge, Card, Checkbox, Select, Tabs, Toast, Popover, Tooltip } from 'bits-ui';
  import 'nes.css/css/nes.min.css';
  
  let dialogOpen = $state(false);
  let toastOpen = $state(false);
  let popoverOpen = $state(false);
  let tooltipOpen = $state(false);
  let selectedValue = $state('option1');
  let checkboxChecked = $state(false);
  let activeTab = $state('tab1');
  let counterValue = $state(0);

  function showToast() {
    toastOpen = true;
    setTimeout(() => toastOpen = false, 3000);
  }

  function incrementCounter() {
    counterValue++;
  }

  function resetCounter() {
    counterValue = 0;
  }
</script>

<svelte:head>
  <title>NES.css + bits-ui Component Showcase</title>
  <link href="https://unpkg.com/nes.css@2.3.0/css/nes.min.css" rel="stylesheet" />
  <link href="https://fonts.googleapis.com/css?family=Press+Start+2P" rel="stylesheet">
</svelte:head>

<div class="nes-container with-title is-centered">
  <p class="title">🎮 NES.css + bits-ui Showcase</p>
  
  <div class="showcase-grid">
    <!-- Hero Section -->
    <section class="hero-section">
      <div class="nes-container is-dark with-title">
        <p class="title">Welcome, Player!</p>
        <p>Experience the perfect fusion of modern bits-ui components with retro NES.css styling.</p>
        <div class="nes-text is-success">
          <p>✓ Accessibility-first components</p>
          <p>✓ Retro gaming aesthetics</p>
          <p>✓ TypeScript support</p>
        </div>
      </div>
    </section>

    <!-- Button Components -->
    <section class="component-section">
      <h2 class="nes-text is-primary">🎯 Buttons</h2>
      
      <div class="button-showcase">
        <Button.Root 
          class="nes-btn is-primary"
          on:on:on:click={incrementCounter}
        >
          Primary Button
        </Button.Root>

        <Button.Root 
          class="nes-btn is-success"
          on:on:on:click={showToast}
        >
          Success Button
        </Button.Root>

        <Button.Root 
          class="nes-btn is-warning"
          on:on:on:click={resetCounter}
        >
          Warning Button
        </Button.Root>

        <Button.Root 
          class="nes-btn is-error"
          disabled
        >
          Error Button
        </Button.Root>

        <Button.Root 
          class="nes-btn"
          on:on:on:click={() => dialogOpen = true}
        >
          Open Dialog
        </Button.Root>
      </div>

      <div class="counter-display">
        <div class="nes-container is-rounded">
          <p>Counter: <span class="nes-text is-success">{counterValue}</span></p>
        </div>
      </div>
    </section>

    <!-- Dialog Component -->
    <Dialog.Root bind:open={dialogOpen}>
      <Dialog.Portal>
        <Dialog.Overlay class="dialog-overlay" />
        <Dialog.Content class="nes-dialog dialog-content">
          <form method="dialog">
            <Dialog.Title class="nes-text is-primary">
              🏆 Achievement Unlocked!
            </Dialog.Title>
            <Dialog.Description>
              <div class="nes-container">
                <p>You've successfully opened a NES.css styled dialog!</p>
                <div class="lists">
                  <ul class="nes-list is-disc">
                    <li>Modern accessibility features</li>
                    <li>Retro gaming aesthetics</li>
                    <li>Perfect keyboard navigation</li>
                  </ul>
                </div>
              </div>
            </Dialog.Description>
            <menu class="dialog-actions">
              <Dialog.Close class="nes-btn">Cancel</Dialog.Close>
              <Dialog.Close class="nes-btn is-primary">Accept</Dialog.Close>
            </menu>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>

    <!-- Badge Components -->
    <section class="component-section">
      <h2 class="nes-text is-success">🏷️ Badges</h2>
      
      <div class="badge-showcase">
        <Badge.Root class="nes-badge">
          <span class="is-dark">Default</span>
        </Badge.Root>

        <Badge.Root class="nes-badge">
          <span class="is-primary">Primary</span>
        </Badge.Root>

        <Badge.Root class="nes-badge">
          <span class="is-success">Success</span>
        </Badge.Root>

        <Badge.Root class="nes-badge">
          <span class="is-warning">Warning</span>
        </Badge.Root>

        <Badge.Root class="nes-badge">
          <span class="is-error">Error</span>
        </Badge.Root>
      </div>
    </section>

    <!-- Card Components -->
    <section class="component-section">
      <h2 class="nes-text is-warning">🎴 Cards</h2>
      
      <div class="card-grid">
        <Card.Root class="nes-container">
          <Card.Header>
            <Card.Title class="nes-text is-primary">Game Stats</Card.Title>
          </Card.Header>
          <Card.Content>
            <div class="stats">
              <p>Level: <span class="nes-text is-success">42</span></p>
              <p>Score: <span class="nes-text is-warning">12,345</span></p>
              <p>Lives: <span class="nes-text is-error">❤️❤️❤️</span></p>
            </div>
          </Card.Content>
        </Card.Root>

        <Card.Root class="nes-container with-title">
          <p class="title">Inventory</p>
          <Card.Content>
            <div class="inventory-items">
              <div class="nes-badge"><span class="is-success">🗡️ Sword</span></div>
              <div class="nes-badge"><span class="is-primary">🛡️ Shield</span></div>
              <div class="nes-badge"><span class="is-warning">💎 Gem</span></div>
            </div>
          </Card.Content>
        </Card.Root>

        <Card.Root class="nes-container is-dark">
          <Card.Header>
            <Card.Title class="nes-text is-primary">Dark Mode</Card.Title>
          </Card.Header>
          <Card.Content>
            <p>This card uses the dark theme variant.</p>
            <Button.Root class="nes-btn is-success">
              Action
            </Button.Root>
          </Card.Content>
        </Card.Root>
      </div>
    </section>

    <!-- Form Components -->
    <section class="component-section">
      <h2 class="nes-text is-error">📝 Forms</h2>
      
      <div class="form-showcase">
        <div class="nes-field">
          <label for="name_field" class="nes-text is-primary">Player Name</label>
          <input type="text" id="name_field" class="nes-input" placeholder="Enter your name">
        </div>

        <div class="nes-field">
          <label for="textarea_field" class="nes-text is-primary">Game Description</label>
          <textarea id="textarea_field" class="nes-textarea" placeholder="Describe your adventure..."></textarea>
        </div>

        <Checkbox.Root 
          bind:checked={checkboxChecked}
          class="checkbox-wrapper"
        >
          <Checkbox.Input class="nes-checkbox" />
          <Checkbox.Label class="nes-text">
            Accept terms and conditions
            {#if checkboxChecked}
              <span class="nes-text is-success">✓</span>
            {/if}
          </Checkbox.Label>
        </Checkbox.Root>

        <div class="nes-select">
          <Select.Root bind:selected={selectedValue}>
            <Select.Trigger class="nes-select-trigger">
              <Select.Value placeholder="Choose difficulty" />
            </Select.Trigger>
            <Select.Content class="nes-container">
              <Select.Item value="easy" class="nes-text is-success">🟢 Easy</Select.Item>
              <Select.Item value="normal" class="nes-text is-warning">🟡 Normal</Select.Item>
              <Select.Item value="hard" class="nes-text is-error">🔴 Hard</Select.Item>
              <Select.Item value="expert" class="nes-text is-primary">💜 Expert</Select.Item>
            </Select.Content>
          </Select.Root>
        </div>
      </div>
    </section>

    <!-- Tabs Component -->
    <section class="component-section">
      <h2 class="nes-text is-success">📂 Tabs</h2>
      
      <Tabs.Root bind:value={activeTab} class="tabs-wrapper">
        <Tabs.List class="tab-list">
          <Tabs.Trigger value="tab1" class="nes-btn {activeTab === 'tab1' ? 'is-primary' : ''}">
            🎮 Games
          </Tabs.Trigger>
          <Tabs.Trigger value="tab2" class="nes-btn {activeTab === 'tab2' ? 'is-primary' : ''}">
            ⚙️ Settings
          </Tabs.Trigger>
          <Tabs.Trigger value="tab3" class="nes-btn {activeTab === 'tab3' ? 'is-primary' : ''}">
            👤 Profile
          </Tabs.Trigger>
        </Tabs.List>
        
        <Tabs.Content value="tab1" class="nes-container">
          <h3 class="nes-text is-primary">Game Library</h3>
          <div class="game-list">
            <div class="nes-container is-rounded">
              <p>🎯 Super Mario Bros - <span class="nes-text is-success">Completed</span></p>
            </div>
            <div class="nes-container is-rounded">
              <p>🏰 Legend of Zelda - <span class="nes-text is-warning">In Progress</span></p>
            </div>
            <div class="nes-container is-rounded">
              <p>🚀 Metroid - <span class="nes-text is-error">Not Started</span></p>
            </div>
          </div>
        </Tabs.Content>
        
        <Tabs.Content value="tab2" class="nes-container">
          <h3 class="nes-text is-primary">Game Settings</h3>
          <div class="settings-grid">
            <label class="nes-text">
              <input type="radio" class="nes-radio" name="volume" checked>
              <span>🔊 Sound On</span>
            </label>
            <label class="nes-text">
              <input type="radio" class="nes-radio" name="volume">
              <span>🔇 Sound Off</span>
            </label>
          </div>
        </Tabs.Content>
        
        <Tabs.Content value="tab3" class="nes-container">
          <h3 class="nes-text is-primary">Player Profile</h3>
          <div class="profile-info">
            <p>🏆 High Score: 99,999</p>
            <p>⭐ Level: Master</p>
            <p>🎖️ Achievements: 47/50</p>
          </div>
        </Tabs.Content>
      </Tabs.Root>
    </section>

    <!-- Toast Component -->
    <Toast.Root bind:open={toastOpen} class="nes-container is-success toast-container">
      <Toast.Title class="nes-text">🎉 Success!</Toast.Title>
      <Toast.Description>
        You've successfully triggered a NES.css styled toast notification!
      </Toast.Description>
      <Toast.Close class="nes-btn is-small">×</Toast.Close>
    </Toast.Root>

    <!-- Interactive Elements -->
    <section class="component-section">
      <h2 class="nes-text is-primary">🎲 Interactive Elements</h2>
      
      <div class="interactive-grid">
        <!-- Tooltip -->
        <Tooltip.Root bind:open={tooltipOpen}>
          <Tooltip.Trigger class="nes-btn is-warning">
            Hover for Tooltip
          </Tooltip.Trigger>
          <Tooltip.Content class="nes-balloon from-left">
            <p>This is a NES.css styled tooltip! 🎮</p>
          </Tooltip.Content>
        </Tooltip.Root>

        <!-- Popover -->
        <Popover.Root bind:open={popoverOpen}>
          <Popover.Trigger class="nes-btn is-success">
            Open Popover
          </Popover.Trigger>
          <Popover.Content class="nes-container popover-content">
            <h4 class="nes-text is-primary">Game Menu</h4>
            <div class="menu-options">
              <Button.Root class="nes-btn menu-btn">Continue</Button.Root>
              <Button.Root class="nes-btn menu-btn">Save Game</Button.Root>
              <Button.Root class="nes-btn menu-btn">Load Game</Button.Root>
              <Button.Root class="nes-btn is-error menu-btn">Quit</Button.Root>
            </div>
          </Popover.Content>
        </Popover.Root>

        <!-- Progress Indicator -->
        <div class="nes-container">
          <p class="nes-text is-primary">Loading Progress</p>
          <progress class="nes-progress is-success" value="75" max="100"></progress>
          <p>75% Complete</p>
        </div>
      </div>
    </section>

    <!-- Footer -->
    <footer class="footer-section">
      <div class="nes-container is-dark with-title">
        <p class="title">🏁 Game Over</p>
        <p>Thank you for exploring the NES.css + bits-ui component showcase!</p>
        <div class="footer-stats">
          <span class="nes-text is-success">Components: 12</span> |
          <span class="nes-text is-warning">Styles: Retro</span> |
          <span class="nes-text is-primary">Accessibility: ✓</span>
        </div>
      </div>
    </footer>
  </div>
</div>

<style>
  @import 'nes.css/css/nes.min.css';
  
  /* Global styling */
  :global(body) {
    font-family: "Press Start 2P", cursive;
    background-color: #212529;
    color: white;
    min-height: 100vh;
    padding: 1rem;
  }

  /* Layout */
  .showcase-grid {
    display: grid;
    gap: 2rem;
    max-width: 1200px;
    margin: 0 auto;
  }

  .hero-section {
    text-align: center;
    margin-bottom: 2rem;
  }

  .component-section {
    margin-bottom: 3rem;
  }

  .component-section h2 {
    margin-bottom: 1rem;
    font-size: 1.2rem;
  }

  /* Button showcase */
  .button-showcase {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .counter-display {
    margin-top: 1rem;
  }

  /* Badge showcase */
  .badge-showcase {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
  }

  .nes-badge span {
    padding: 0.5rem;
    border-radius: 4px;
  }

  /* Card grid */
  .card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
  }

  .stats p, .profile-info p {
    margin: 0.5rem 0;
  }

  .inventory-items {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  /* Form showcase */
  .form-showcase {
    max-width: 500px;
  }

  .nes-field {
    margin-bottom: 1rem;
  }

  .checkbox-wrapper {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1rem 0;
  }

  .nes-select-trigger {
    width: 100%;
    padding: 8px;
  }

  /* Tabs */
  .tabs-wrapper {
    width: 100%;
  }

  .tab-list {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
  }

  .game-list {
    display: grid;
    gap: 0.5rem;
  }

  .settings-grid {
    display: grid;
    gap: 1rem;
  }

  .settings-grid label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* Interactive elements */
  .interactive-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
  }

  .popover-content {
    min-width: 200px;
    z-index: 1000;
  }

  .menu-options {
    display: grid;
    gap: 0.5rem;
  }

  .menu-btn {
    width: 100%;
    text-align: left;
  }

  /* Dialog styling */
  .dialog-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.8);
    z-index: 999;
  }

  .dialog-content {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    max-width: 500px;
    width: 90vw;
    max-height: 80vh;
    overflow-y: auto;
    z-index: 1000;
    background: white;
    color: #212529;
  }

  .dialog-actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    margin-top: 1rem;
    padding: 0;
  }

  .lists {
    margin: 1rem 0;
  }

  /* Toast styling */
  .toast-container {
    position: fixed;
    bottom: 1rem;
    right: 1rem;
    max-width: 400px;
    z-index: 1000;
  }

  /* Footer */
  .footer-section {
    margin-top: 3rem;
    text-align: center;
  }

  .footer-stats {
    margin-top: 1rem;
    font-size: 0.8rem;
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .showcase-grid {
      padding: 0 1rem;
    }
    
    .button-showcase {
      flex-direction: column;
    }
    
    .card-grid {
      grid-template-columns: 1fr;
    }
    
    .tab-list {
      flex-direction: column;
    }
    
    .interactive-grid {
      grid-template-columns: 1fr;
    }
    
    .dialog-content {
      width: 95vw;
      max-height: 90vh;
    }
  }

  /* Custom NES.css enhancements */
  .nes-progress.is-success::-webkit-progress-value {
    background-color: #92cc41;
  }

  .nes-progress.is-success::-moz-progress-bar {
    background-color: #92cc41;
  }

  .nes-balloon.from-left::after {
    border-color: transparent white transparent transparent;
  }

  /* Component-specific styling */
  :global(.nes-btn) {
    margin: 0.25rem;
    min-width: auto;
  }

  :global(.nes-container) {
    margin: 0.5rem 0;
  }

  /* Accessibility improvements */
  @media (prefers-reduced-motion: reduce) {
    * {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }
  }

  /* High contrast mode */
  @media (prefers-contrast: high) {
    :global(body) {
      background-color: #000000;
      color: #ffffff;
    }
  }
</style>