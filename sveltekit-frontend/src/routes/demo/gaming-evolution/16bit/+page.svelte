<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { Card } from 'bits-ui';
  import { Dialog } from 'bits-ui';
  import { Checkbox } from 'bits-ui';
  import { Select } from 'bits-ui';
  import { Tabs } from 'bits-ui';
  import { Toast } from 'bits-ui';
  import { Popover } from 'bits-ui';
  import { Tooltip } from 'bits-ui';

  // SNES-era state management
  let dialogOpen = $state(false);
  let toastOpen = $state(false);
  let popoverOpen = $state(false);
  let selectedValue = $state('super-mario-world');
  let checked = $state(false);
  let activeTab = $state('graphics');
  
  // SNES color palette showcase
  const snesColors = [
    '#000000', '#800000', '#008000', '#808000', '#000080', '#800080', '#008080', '#C0C0C0',
    '#808080', '#FF0000', '#00FF00', '#FFFF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFFFFF',
    '#FFB6C1', '#FFA500', '#32CD32', '#87CEEB', '#DDA0DD', '#F0E68C', '#B0C4DE', '#FFF8DC'
  ];

  // Mode 7 transformation matrix simulation
  let mode7Rotation = $state(0);
  let mode7Scale = $state(1);
  
  onMount(() => {
    // Animate mode 7 demo
    const interval = setInterval(() => {
      mode7Rotation += 1;
      mode7Scale = 1 + Math.sin(mode7Rotation * 0.1) * 0.2;
    }, 50);
    
    return () => clearInterval(interval);
  });

  const gameOptions = [
    { value: 'super-mario-world', label: '🍄 Super Mario World' },
    { value: 'zelda-lttp', label: '⚔️ Zelda: Link to the Past' },
    { value: 'super-metroid', label: '🚀 Super Metroid' },
    { value: 'chrono-trigger', label: '⏰ Chrono Trigger' },
    { value: 'ff6', label: '⚡ Final Fantasy VI' }
  ];
</script>

<svelte:head>
  <title>16-bit SNES Gaming Evolution - Legal AI Platform</title>
</svelte:head>

<div class="snes-container">
  <!-- SNES Header -->
  <header class="snes-header">
    <div class="snes-logo">
      <span class="logo-text">SUPER LEGAL AI</span>
      <div class="snes-cartridge"></div>
    </div>
    <div class="power-indicator">
      <div class="power-led"></div>
      <span>POWER</span>
    </div>
  </header>

  <!-- Main Content Grid -->
  <main class="snes-main">
    
    <!-- Technical Analysis Panel -->
    <section class="analysis-panel">
      <Card.Root class="snes-card">
        <Card.Header>
          <Card.Title class="snes-title">
            🎮 16-bit SNES Era Analysis
          </Card.Title>
          <Card.Description class="snes-subtitle">
            Super Nintendo Entertainment System Visual Characteristics
          </Card.Description>
        </Card.Header>
        <Card.Content class="technical-specs">
          
          <div class="spec-grid">
            <div class="spec-item">
              <strong>PPU Capabilities:</strong>
              <p>Background layers: 4, Sprites: 128, Colors: 32,768 palette</p>
            </div>
            <div class="spec-item">
              <strong>Mode 7:</strong>
              <p>Rotation, scaling, pseudo-3D effects</p>
            </div>
            <div class="spec-item">
              <strong>Color Depth:</strong>
              <p>15-bit color (32,768 colors), up to 256 on screen</p>
            </div>
            <div class="spec-item">
              <strong>Resolution:</strong>
              <p>256×224 to 512×448 (interlaced)</p>
            </div>
          </div>

          <!-- SNES Color Palette Demo -->
          <div class="color-showcase">
            <h4 class="snes-subtitle">Enhanced Color Palette</h4>
            <div class="palette-grid">
              {#each snesColors as color}
                <div class="color-swatch" style="background-color: {color};" 
                     title="{color}"></div>
              {/each}
            </div>
          </div>

          <!-- Mode 7 Demo -->
          <div class="mode7-demo">
            <h4 class="snes-subtitle">Mode 7 Transformation</h4>
            <div class="mode7-plane" 
                 style="transform: perspective(200px) rotateX(45deg) rotateZ({mode7Rotation}deg) scale({mode7Scale});">
              <div class="checkerboard"></div>
            </div>
          </div>

        </Card.Content>
      </Card.Root>
    </section>

    <!-- Interactive Components Panel -->
    <section class="components-panel">
      <Card.Root class="snes-card">
        <Card.Header>
          <Card.Title class="snes-title">
            🕹️ 16-bit UI Components
          </Card.Title>
          <Card.Description class="snes-subtitle">
            bits-ui Components with SNES Aesthetics
          </Card.Description>
        </Card.Header>
        <Card.Content class="components-grid">

          <!-- Buttons Section -->
          <div class="component-section">
            <h4 class="section-title">Action Buttons</h4>
            <div class="button-group">
              <Button.Root class="snes-btn snes-btn-a">A</Button.Root>
              <Button.Root class="snes-btn snes-btn-b">B</Button.Root>
              <Button.Root class="snes-btn snes-btn-x">X</Button.Root>
              <Button.Root class="snes-btn snes-btn-y">Y</Button.Root>
            </div>
            <div class="shoulder-buttons">
              <Button.Root class="snes-btn snes-btn-l">L</Button.Root>
              <Button.Root class="snes-btn snes-btn-r">R</Button.Root>
            </div>
          </div>

          <!-- D-Pad -->
          <div class="component-section">
            <h4 class="section-title">D-Pad Navigation</h4>
            <div class="dpad">
              <Button.Root class="dpad-btn dpad-up">▲</Button.Root>
              <div class="dpad-middle">
                <Button.Root class="dpad-btn dpad-left">◄</Button.Root>
                <div class="dpad-center"></div>
                <Button.Root class="dpad-btn dpad-right">►</Button.Root>
              </div>
              <Button.Root class="dpad-btn dpad-down">▼</Button.Root>
            </div>
          </div>

          <!-- Game Selection -->
          <div class="component-section">
            <h4 class="section-title">Game Cartridge</h4>
            <Select.Root bind:value={selectedValue}>
              <Select.Trigger class="snes-select-trigger">
                <Select.Value placeholder="Insert Cartridge..." />
              </Select.Trigger>
              <Select.Content class="snes-select-content">
                {#each gameOptions as option}
                  <Select.Item class="snes-select-item" value={option.value}>
                    {option.label}
                  </Select.Item>
                {/each}
              </Select.Content>
            </Select.Root>
          </div>

          <!-- Status Indicators -->
          <div class="component-section">
            <h4 class="section-title">Status Badges</h4>
            <div class="badge-collection">
              <Badge.Root class="snes-badge snes-badge-lives">Lives: 3</Badge.Root>
              <Badge.Root class="snes-badge snes-badge-score">Score: 12,450</Badge.Root>
              <Badge.Root class="snes-badge snes-badge-world">World 1-1</Badge.Root>
              <Badge.Root class="snes-badge snes-badge-time">Time: 400</Badge.Root>
            </div>
          </div>

        </Card.Content>
      </Card.Root>
    </section>

    <!-- Game UI Recreation -->
    <section class="game-ui-panel">
      <Card.Root class="snes-card">
        <Card.Header>
          <Card.Title class="snes-title">
            👾 16-bit Game Interface
          </Card.Title>
          <Card.Description class="snes-subtitle">
            Classic SNES Menu & Dialog Recreation
          </Card.Description>
        </Card.Header>
        <Card.Content class="game-interface">

          <!-- RPG-style Tabs -->
          <Tabs.Root bind:value={activeTab} class="snes-tabs">
            <Tabs.List class="snes-tabs-list">
              <Tabs.Trigger class="snes-tab" value="graphics">Graphics</Tabs.Trigger>
              <Tabs.Trigger class="snes-tab" value="audio">Audio</Tabs.Trigger>
              <Tabs.Trigger class="snes-tab" value="controls">Controls</Tabs.Trigger>
            </Tabs.List>
            
            <Tabs.Content class="snes-tab-content" value="graphics">
              <div class="settings-grid">
                <div class="setting-item">
                  <label>Transparency Effects</label>
                  <Checkbox.Root class="snes-checkbox" bind:checked>
                    <Checkbox.Indicator class="snes-checkbox-indicator">✓</Checkbox.Indicator>
                  </Checkbox.Root>
                </div>
                <div class="setting-item">
                  <label>Sprite Scaling</label>
                  <div class="mode7-toggle">ON</div>
                </div>
                <div class="setting-item">
                  <label>Layer Priority</label>
                  <div class="priority-slider">
                    <div class="slider-track"></div>
                    <div class="slider-thumb"></div>
                  </div>
                </div>
              </div>
            </Tabs.Content>
            
            <Tabs.Content class="snes-tab-content" value="audio">
              <div class="audio-visualizer">
                <div class="channel" style="height: 60%;">Ch1</div>
                <div class="channel" style="height: 80%;">Ch2</div>
                <div class="channel" style="height: 40%;">Ch3</div>
                <div class="channel" style="height: 90%;">Ch4</div>
                <div class="channel" style="height: 70%;">Ch5</div>
                <div class="channel" style="height: 50%;">Ch6</div>
                <div class="channel" style="height: 85%;">Ch7</div>
                <div class="channel" style="height: 45%;">Ch8</div>
              </div>
            </Tabs.Content>
            
            <Tabs.Content class="snes-tab-content" value="controls">
              <div class="control-layout">
                <div class="controller-diagram">
                  <div class="snes-controller">
                    <div class="controller-body"></div>
                    <div class="controller-dpad"></div>
                    <div class="controller-buttons"></div>
                  </div>
                </div>
              </div>
            </Tabs.Content>
          </Tabs.Root>

          <!-- Interactive Dialog -->
          <div class="dialog-section">
            <Button.Root 
              class="snes-btn snes-btn-primary" 
              on:on:on:click={() => dialogOpen = true}
            >
              🗨️ Open Message Box
            </Button.Root>
            
            <Dialog.Root bind:open={dialogOpen}>
              <Dialog.Portal>
                <Dialog.Overlay class="snes-dialog-overlay" />
                <Dialog.Content class="snes-dialog">
                  <div class="dialog-border">
                    <Dialog.Title class="dialog-title">
                      System Message
                    </Dialog.Title>
                    <Dialog.Description class="dialog-text">
                      The princess is in another castle! But first, let me tell you about 
                      the incredible 16-bit capabilities of the Super Nintendo Entertainment System...
                    </Dialog.Description>
                    <div class="dialog-actions">
                      <Button.Root 
                        class="snes-btn snes-btn-confirm" 
                        on:on:on:click={() => dialogOpen = false}
                      >
                        ✓ OK
                      </Button.Root>
                    </div>
                  </div>
                </Dialog.Content>
              </Dialog.Portal>
            </Dialog.Root>
          </div>

        </Card.Content>
      </Card.Root>
    </section>

  </main>

  <!-- Footer with Credits -->
  <footer class="snes-footer">
    <div class="credits">
      <p>© 2025 Legal AI Platform - 16-bit Gaming Evolution Showcase</p>
      <p>Powered by bits-ui + SvelteKit + SNES Aesthetics</p>
    </div>
  </footer>

</div>

<!-- Toast Notification -->
<Toast.Root bind:open={toastOpen} class="snes-toast">
  <Toast.Title class="toast-title">Achievement Unlocked!</Toast.Title>
  <Toast.Description class="toast-description">
    16-bit Graphics Mode Activated
  </Toast.Description>
</Toast.Root>

<style>
  /* SNES 16-bit Color Palette */
  :root {
    /* Enhanced SNES colors */
    --snes-purple: #5A4FCF;
    --snes-blue: #3366CC;
    --snes-teal: #00AA88;
    --snes-green: #228B22;
    --snes-yellow: #FFD700;
    --snes-orange: #FF8C00;
    --snes-red: #DC143C;
    --snes-pink: #FF69B4;
    
    /* Gradients for depth */
    --snes-bg: linear-gradient(135deg, #2C1810 0%, #4A2C17 50%, #2C1810 100%);
    --snes-panel: linear-gradient(145deg, #E6E6FA 0%, #D8BFD8 50%, #DDA0DD 100%);
    --snes-button: linear-gradient(145deg, #C0C0C0 0%, #A9A9A9 50%, #808080 100%);
    
    /* Shadows and highlights */
    --snes-shadow: rgba(0, 0, 0, 0.4);
    --snes-highlight: rgba(255, 255, 255, 0.8);
    --snes-glow: rgba(255, 215, 0, 0.6);
  }

  /* Base Container */
  .snes-container {
    min-height: 100vh;
    background: var(--snes-bg);
    font-family: 'Courier New', monospace;
    font-weight: bold;
    color: #FFFFFF;
    padding: 1rem;
    
    /* Enhanced visual effects */
    background-attachment: fixed;
    position: relative;
  }

  .snes-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      radial-gradient(circle at 20% 80%, rgba(138, 43, 226, 0.1) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(255, 215, 0, 0.1) 0%, transparent 50%);
    pointer-events: none;
  }

  /* Header */
  .snes-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    background: var(--snes-panel);
    padding: 1rem 2rem;
    border-radius: 12px;
    border: 3px solid #8A2BE2;
    box-shadow: 
      0 8px 16px var(--snes-shadow),
      inset 0 2px 4px var(--snes-highlight);
    color: #4B0082;
  }

  .snes-logo {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .logo-text {
    font-size: 2rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    background: linear-gradient(45deg, #8A2BE2, #FF69B4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 2px 2px 4px var(--snes-shadow);
  }

  .snes-cartridge {
    width: 60px;
    height: 40px;
    background: linear-gradient(145deg, #C0C0C0, #808080);
    border-radius: 4px;
    border: 2px solid #696969;
    position: relative;
  }

  .snes-cartridge::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 40px;
    height: 8px;
    background: #4B0082;
    border-radius: 2px;
  }

  .power-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .power-led {
    width: 12px;
    height: 12px;
    background: #00FF00;
    border-radius: 50%;
    box-shadow: 0 0 8px #00FF00;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  /* Main Layout */
  .snes-main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 2rem;
    margin-bottom: 2rem;
  }

  .game-ui-panel {
    grid-column: 1 / -1;
  }

  /* Card Styling */
  .snes-card {
    background: var(--snes-panel);
    border: 4px solid #8A2BE2;
    border-radius: 16px;
    box-shadow: 
      0 12px 24px var(--snes-shadow),
      inset 0 4px 8px var(--snes-highlight);
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    color: #2F1B69;
  }

  .snes-card:hover {
    transform: translateY(-4px);
    box-shadow: 
      0 16px 32px var(--snes-shadow),
      0 0 20px var(--snes-glow),
      inset 0 4px 8px var(--snes-highlight);
  }

  .snes-title {
    font-size: 1.5rem;
    font-weight: 900;
    text-transform: uppercase;
    color: #4B0082;
    text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
    margin-bottom: 0.5rem;
  }

  .snes-subtitle {
    color: #663399;
    font-size: 0.9rem;
    font-style: italic;
  }

  /* Technical Specifications */
  .technical-specs {
    padding: 1.5rem;
  }

  .spec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .spec-item {
    background: rgba(255, 255, 255, 0.9);
    padding: 1rem;
    border-radius: 8px;
    border: 2px solid #DDA0DD;
  }

  .spec-item strong {
    color: #8A2BE2;
    display: block;
    margin-bottom: 0.5rem;
  }

  .spec-item p {
    color: #4B0082;
    margin: 0;
    font-size: 0.85rem;
  }

  /* Color Palette */
  .color-showcase {
    margin: 2rem 0;
  }

  .palette-grid {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    gap: 4px;
    max-width: 400px;
  }

  .color-swatch {
    width: 40px;
    height: 40px;
    border: 2px solid rgba(0, 0, 0, 0.3);
    border-radius: 4px;
    cursor: pointer;
    transition: transform 0.2s ease;
  }

  .color-swatch:hover {
    transform: scale(1.1);
    border-color: #FFFFFF;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
  }

  /* Mode 7 Demo */
  .mode7-demo {
    margin: 2rem 0;
    text-align: center;
  }

  .mode7-plane {
    width: 200px;
    height: 200px;
    margin: 2rem auto;
    transform-style: preserve-3d;
    transition: transform 0.1s linear;
  }

  .checkerboard {
    width: 100%;
    height: 100%;
    background: 
      repeating-conic-gradient(#8A2BE2 0deg 90deg, #DDA0DD 90deg 180deg) 
      0 0 / 20px 20px;
    border: 3px solid #4B0082;
    border-radius: 8px;
  }

  /* Component Sections */
  .components-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    padding: 1.5rem;
  }

  .component-section {
    background: rgba(255, 255, 255, 0.95);
    padding: 1.5rem;
    border-radius: 12px;
    border: 3px solid #DDA0DD;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }

  .section-title {
    color: #8A2BE2;
    font-size: 1.1rem;
    font-weight: bold;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  /* SNES Buttons */
  .snes-btn {
    border: none;
    border-radius: 50%;
    width: 48px;
    height: 48px;
    font-weight: bold;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 
      0 4px 8px rgba(0, 0, 0, 0.3),
      inset 0 2px 4px rgba(255, 255, 255, 0.8);
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    color: white;
    font-family: inherit;
  }

  .snes-btn:hover {
    transform: translateY(-2px);
    box-shadow: 
      0 6px 12px rgba(0, 0, 0, 0.4),
      0 0 12px rgba(255, 255, 255, 0.6),
      inset 0 2px 4px rgba(255, 255, 255, 0.9);
  }

  .snes-btn:active {
    transform: translateY(0);
    box-shadow: 
      0 2px 4px rgba(0, 0, 0, 0.3),
      inset 0 2px 8px rgba(0, 0, 0, 0.2);
  }

  .snes-btn-a { background: linear-gradient(145deg, #FF4444, #CC0000); }
  .snes-btn-b { background: linear-gradient(145deg, #FFFF44, #CCCC00); }
  .snes-btn-x { background: linear-gradient(145deg, #4444FF, #0000CC); }
  .snes-btn-y { background: linear-gradient(145deg, #44FF44, #00CC00); }
  .snes-btn-l, .snes-btn-r { 
    background: linear-gradient(145deg, #AAAAAA, #666666);
    border-radius: 8px;
    width: 60px;
    height: 32px;
  }

  .button-group {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
    max-width: 120px;
    margin: 0 auto 1rem;
  }

  .shoulder-buttons {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
  }

  /* D-Pad */
  .dpad {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    width: 120px;
    margin: 0 auto;
  }

  .dpad-middle {
    display: flex;
    align-items: center;
    gap: 2px;
  }

  .dpad-btn {
    width: 36px;
    height: 36px;
    background: linear-gradient(145deg, #666666, #333333);
    border: 2px solid #222222;
    border-radius: 4px;
    color: #CCCCCC;
    font-weight: bold;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
  }

  .dpad-btn:hover {
    background: linear-gradient(145deg, #777777, #444444);
    color: white;
  }

  .dpad-center {
    width: 36px;
    height: 36px;
    background: radial-gradient(circle, #333333, #111111);
    border: 2px solid #222222;
    border-radius: 50%;
  }

  /* Select Component */
  .snes-select-trigger {
    background: var(--snes-panel);
    border: 3px solid #8A2BE2;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #4B0082;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
  }

  .snes-select-trigger:hover {
    border-color: #FF69B4;
    box-shadow: 0 4px 8px rgba(138, 43, 226, 0.3);
  }

  .snes-select-content {
    background: var(--snes-panel);
    border: 3px solid #8A2BE2;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
  }

  .snes-select-item {
    padding: 0.75rem 1rem;
    color: #4B0082;
    font-weight: bold;
    cursor: pointer;
    transition: background-color 0.2s ease;
  }

  .snes-select-item:hover {
    background: rgba(138, 43, 226, 0.2);
  }

  /* Badges */
  .badge-collection {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .snes-badge {
    padding: 0.5rem 1rem;
    border-radius: 16px;
    font-weight: bold;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 2px solid;
  }

  .snes-badge-lives {
    background: linear-gradient(145deg, #90EE90, #32CD32);
    color: #006400;
    border-color: #228B22;
  }

  .snes-badge-score {
    background: linear-gradient(145deg, #FFD700, #FFA500);
    color: #8B4513;
    border-color: #FF8C00;
  }

  .snes-badge-world {
    background: linear-gradient(145deg, #87CEEB, #4169E1);
    color: #000080;
    border-color: #0000CD;
  }

  .snes-badge-time {
    background: linear-gradient(145deg, #FFB6C1, #FF69B4);
    color: #8B008B;
    border-color: #FF1493;
  }

  /* Game Interface */
  .game-interface {
    padding: 1.5rem;
  }

  /* Tabs */
  .snes-tabs {
    width: 100%;
  }

  .snes-tabs-list {
    display: flex;
    background: rgba(138, 43, 226, 0.1);
    border-radius: 12px;
    padding: 4px;
    margin-bottom: 1rem;
  }

  .snes-tab {
    flex: 1;
    padding: 0.75rem 1rem;
    background: transparent;
    border: none;
    border-radius: 8px;
    color: #8A2BE2;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .snes-tab[data-state="active"] {
    background: var(--snes-panel);
    color: #4B0082;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }

  .snes-tab:hover:not([data-state="active"]) {
    background: rgba(138, 43, 226, 0.2);
  }

  .snes-tab-content {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    padding: 1.5rem;
    border: 3px solid #DDA0DD;
  }

  /* Settings Grid */
  .settings-grid {
    display: grid;
    gap: 1rem;
  }

  .setting-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    background: rgba(138, 43, 226, 0.1);
    border-radius: 8px;
    border: 2px solid #DDA0DD;
  }

  .setting-item label {
    color: #4B0082;
    font-weight: bold;
  }

  /* Checkbox */
  .snes-checkbox {
    width: 24px;
    height: 24px;
    background: white;
    border: 3px solid #8A2BE2;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }

  .snes-checkbox[data-state="checked"] {
    background: #8A2BE2;
  }

  .snes-checkbox-indicator {
    color: white;
    font-weight: bold;
  }

  /* Toggle */
  .mode7-toggle {
    background: linear-gradient(145deg, #90EE90, #32CD32);
    color: #006400;
    padding: 0.5rem 1rem;
    border-radius: 16px;
    font-weight: bold;
    border: 2px solid #228B22;
  }

  /* Slider */
  .priority-slider {
    position: relative;
    width: 100px;
    height: 8px;
    background: #DDA0DD;
    border-radius: 4px;
    border: 2px solid #8A2BE2;
  }

  .slider-thumb {
    position: absolute;
    top: -6px;
    left: 60%;
    width: 16px;
    height: 16px;
    background: #8A2BE2;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  }

  /* Audio Visualizer */
  .audio-visualizer {
    display: flex;
    align-items: end;
    justify-content: space-between;
    height: 120px;
    padding: 1rem;
    background: #1a1a1a;
    border-radius: 8px;
    border: 3px solid #333;
  }

  .channel {
    width: 20px;
    background: linear-gradient(to top, #8A2BE2, #FF69B4);
    border-radius: 2px;
    display: flex;
    align-items: end;
    justify-content: center;
    color: white;
    font-size: 0.7rem;
    font-weight: bold;
    padding-bottom: 4px;
    animation: audioChannel 2s infinite;
  }

  @keyframes audioChannel {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
  }

  /* Controller Layout */
  .control-layout {
    display: flex;
    justify-content: center;
    padding: 2rem;
  }

  .controller-diagram {
    position: relative;
  }

  .snes-controller {
    width: 240px;
    height: 120px;
    background: linear-gradient(145deg, #E6E6FA, #C0C0C0);
    border-radius: 60px;
    border: 4px solid #8A2BE2;
    position: relative;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
  }

  /* Dialog */
  .snes-dialog-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    z-index: 50;
  }

  .snes-dialog {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 51;
    max-width: 500px;
    width: calc(100vw - 2rem);
  }

  .dialog-border {
    background: var(--snes-panel);
    border: 6px solid #8A2BE2;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.5);
  }

  .dialog-title {
    color: #4B0082;
    font-size: 1.25rem;
    font-weight: bold;
    margin-bottom: 1rem;
    text-align: center;
  }

  .dialog-text {
    color: #663399;
    line-height: 1.6;
    margin-bottom: 1.5rem;
    text-align: center;
  }

  .dialog-actions {
    display: flex;
    justify-content: center;
  }

  .snes-btn-primary,
  .snes-btn-confirm {
    background: linear-gradient(145deg, #8A2BE2, #663399);
    color: white;
    padding: 0.75rem 2rem;
    border-radius: 8px;
    border: none;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
  }

  .snes-btn-primary:hover,
  .snes-btn-confirm:hover {
    background: linear-gradient(145deg, #9932CC, #8A2BE2);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    transform: translateY(-2px);
  }

  /* Toast */
  .snes-toast {
    background: var(--snes-panel);
    border: 4px solid #32CD32;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    color: #006400;
  }

  .toast-title {
    font-weight: bold;
    margin-bottom: 0.5rem;
  }

  .toast-description {
    font-size: 0.9rem;
  }

  /* Footer */
  .snes-footer {
    background: var(--snes-panel);
    padding: 2rem;
    border-radius: 12px;
    border: 3px solid #8A2BE2;
    text-align: center;
    color: #4B0082;
  }

  .credits p {
    margin: 0.25rem 0;
    font-size: 0.9rem;
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .snes-main {
      grid-template-columns: 1fr;
    }
    
    .components-grid {
      grid-template-columns: 1fr;
    }
    
    .spec-grid {
      grid-template-columns: 1fr;
    }
    
    .snes-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }
    
    .logo-text {
      font-size: 1.5rem;
    }
  }

  /* Performance optimizations */
  .snes-container * {
    will-change: transform;
  }
  
  .mode7-plane {
    backface-visibility: hidden;
  }
</style>