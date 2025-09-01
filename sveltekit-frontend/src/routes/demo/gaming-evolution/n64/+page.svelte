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
  import { Slider } from 'bits-ui';

  // N64-era state management with 3D context
  let dialogOpen = $state(false);
  let toastOpen = $state(false);
  let selectedGame = $state('super-mario-64');
  let antiAliasing = $state(true);
  let textureFiltering = $state('trilinear');
  let activeTab = $state('graphics');
  let z64Rotation = $state(0);
  let polygonCount = $state(0);
  
  // N64 texture filtering simulation
  let filteringStrength = $state([0.75]);
  let mipmapLevel = $state([2]);
  let anisotropicLevel = $state([4]);
  
  // 3D transformation matrices
  let cameraAngle = $state(0);
  let perspectiveDepth = $state(800);
  
  onMount(() => {
    // Animate 3D elements
    const interval = setInterval(() => {
      z64Rotation += 1.5;
      cameraAngle += 0.8;
      polygonCount = Math.floor(Math.random() * 2000) + 1000;
    }, 60); // ~16fps like N64
    
    return () => clearInterval(interval);
  });

  const n64Games = [
    { value: 'super-mario-64', label: '🍄 Super Mario 64' },
    { value: 'zelda-oot', label: '🗡️ The Legend of Zelda: OoT' },
    { value: 'mario-kart-64', label: '🏎️ Mario Kart 64' },
    { value: 'goldeneye', label: '🔫 GoldenEye 007' },
    { value: 'super-smash-bros', label: '👊 Super Smash Bros.' }
  ];

  const textureFilters = [
    { value: 'point', label: 'Point (Nearest)' },
    { value: 'bilinear', label: 'Bilinear' },
    { value: 'trilinear', label: 'Trilinear' }
  ];
</script>

<svelte:head>
  <title>N64 3D Gaming Evolution - Legal AI Platform</title>
</svelte:head>

<div class="n64-container">
  
  <!-- N64 Header with 3D branding -->
  <header class="n64-header">
    <div class="n64-logo">
      <div class="logo-3d">
        <span class="logo-text" style="transform: perspective({perspectiveDepth}px) rotateY({cameraAngle}deg);">
          LEGAL AI 64
        </span>
        <div class="n64-controller"></div>
      </div>
    </div>
    <div class="system-status">
      <div class="status-item">
        <span class="status-label">Polygons/Sec:</span>
        <span class="status-value">{polygonCount.toLocaleString()}</span>
      </div>
      <div class="status-item">
        <span class="status-label">Z-Buffer:</span>
        <span class="status-value">16-bit</span>
      </div>
    </div>
  </header>

  <!-- Main 3D Layout -->
  <main class="n64-main">
    
    <!-- Technical Analysis with 3D elements -->
    <section class="tech-panel">
      <Card.Root class="n64-card">
        <Card.Header>
          <Card.Title class="n64-title">
            🎮 Nintendo 64 3D Architecture
          </Card.Title>
          <Card.Description class="n64-subtitle">
            Reality Co-Processor (RCP) with Anti-Aliasing & Texture Filtering
          </Card.Description>
        </Card.Header>
        <Card.Content class="n64-content">
          
          <!-- 3D Cube Demo -->
          <div class="cube-demo">
            <div class="scene-3d">
              <div 
                class="n64-cube" 
                style="transform: perspective({perspectiveDepth}px) rotateX({z64Rotation * 0.5}deg) rotateY({z64Rotation}deg) rotateZ({z64Rotation * 0.3}deg);"
              >
                <div class="cube-face cube-front">64</div>
                <div class="cube-face cube-back">AI</div>
                <div class="cube-face cube-right">3D</div>
                <div class="cube-face cube-left">AA</div>
                <div class="cube-face cube-top">TF</div>
                <div class="cube-face cube-bottom">RCP</div>
              </div>
            </div>
          </div>

          <!-- Technical Specifications -->
          <div class="tech-specs">
            <div class="spec-section">
              <h4>Reality Co-Processor (RCP)</h4>
              <div class="spec-grid">
                <div class="spec-item">
                  <strong>RSP:</strong> Reality Signal Processor
                </div>
                <div class="spec-item">
                  <strong>RDP:</strong> Reality Display Processor
                </div>
                <div class="spec-item">
                  <strong>Polygons:</strong> 150,000+ flat-shaded/sec
                </div>
                <div class="spec-item">
                  <strong>Z-Buffer:</strong> 16-bit depth testing
                </div>
              </div>
            </div>

            <div class="spec-section">
              <h4>Texture Filtering Features</h4>
              <div class="feature-list">
                <div class="feature-item">
                  <span class="feature-icon">🔍</span>
                  <span>Bilinear texture filtering</span>
                </div>
                <div class="feature-item">
                  <span class="feature-icon">🌊</span>
                  <span>Trilinear mipmap filtering</span>
                </div>
                <div class="feature-item">
                  <span class="feature-icon">⚡</span>
                  <span>Hardware anti-aliasing</span>
                </div>
                <div class="feature-item">
                  <span class="feature-icon">🎯</span>
                  <span>Perspective-correct texturing</span>
                </div>
              </div>
            </div>

            <!-- Texture Filtering Controls -->
            <div class="filtering-controls">
              <h4>Real-time Filtering Adjustment</h4>
              
              <div class="control-group">
                <label>Texture Filtering:</label>
                <Select.Root bind:value={textureFiltering}>
                  <Select.Trigger class="n64-select-trigger">
                    <Select.Value />
                  </Select.Trigger>
                  <Select.Content class="n64-select-content">
                    {#each textureFilters as filter}
                      <Select.Item class="n64-select-item" value={filter.value}>
                        {filter.label}
                      </Select.Item>
                    {/each}
                  </Select.Content>
                </Select.Root>
              </div>

              <div class="control-group">
                <label>Filtering Strength:</label>
                <Slider.Root class="n64-slider" bind:value={filteringStrength} max={1} step={0.1}>
                  <Slider.Track class="slider-track">
                    <Slider.Range class="slider-range" />
                  </Slider.Track>
                  <Slider.Thumb class="slider-thumb" />
                </Slider.Root>
                <span class="slider-value">{(filteringStrength[0] * 100).toFixed(0)}%</span>
              </div>

              <div class="control-group">
                <label>Mipmap Level:</label>
                <Slider.Root class="n64-slider" bind:value={mipmapLevel} max={8} step={1}>
                  <Slider.Track class="slider-track">
                    <Slider.Range class="slider-range" />
                  </Slider.Track>
                  <Slider.Thumb class="slider-thumb" />
                </Slider.Root>
                <span class="slider-value">Level {mipmapLevel[0]}</span>
              </div>

              <div class="control-group">
                <label>Anti-Aliasing:</label>
                <Checkbox.Root class="n64-checkbox" bind:checked={antiAliasing}>
                  <Checkbox.Indicator class="checkbox-indicator">✓</Checkbox.Indicator>
                </Checkbox.Root>
                <span class="aa-status">{antiAliasing ? 'Enabled' : 'Disabled'}</span>
              </div>

            </div>

          </Card.Content>
        </Card.Root>
      </Card.Root>
    </section>

    <!-- Interactive 3D Components -->
    <section class="components-panel">
      <Card.Root class="n64-card">
        <Card.Header>
          <Card.Title class="n64-title">
            🎮 3D Interface Elements
          </Card.Title>
          <Card.Description class="n64-subtitle">
            bits-ui Components with N64 3D Aesthetics & Texture Filtering
          </Card.Description>
        </Card.Header>
        <Card.Content class="components-3d">

          <!-- N64 Controller -->
          <div class="controller-section">
            <h4 class="section-title">N64 Controller</h4>
            <div class="n64-controller-layout">
              
              <!-- Left section -->
              <div class="controller-left">
                <div class="analog-stick">
                  <div class="stick-base"></div>
                  <div class="stick-top"></div>
                </div>
              </div>

              <!-- Center section -->
              <div class="controller-center">
                <div class="dpad-3d">
                  <Button.Root class="dpad-btn-3d dpad-up-3d">▲</Button.Root>
                  <div class="dpad-middle-3d">
                    <Button.Root class="dpad-btn-3d dpad-left-3d">◄</Button.Root>
                    <div class="dpad-center-3d"></div>
                    <Button.Root class="dpad-btn-3d dpad-right-3d">►</Button.Root>
                  </div>
                  <Button.Root class="dpad-btn-3d dpad-down-3d">▼</Button.Root>
                </div>
                
                <Button.Root class="start-button-3d">START</Button.Root>
              </div>

              <!-- Right section -->
              <div class="controller-right">
                <div class="c-buttons">
                  <Button.Root class="c-btn c-up">C↑</Button.Root>
                  <div class="c-middle">
                    <Button.Root class="c-btn c-left">C←</Button.Root>
                    <Button.Root class="c-btn c-right">C→</Button.Root>
                  </div>
                  <Button.Root class="c-btn c-down">C↓</Button.Root>
                </div>
                
                <div class="face-buttons-3d">
                  <Button.Root class="n64-btn-3d n64-btn-a">A</Button.Root>
                  <Button.Root class="n64-btn-3d n64-btn-b">B</Button.Root>
                </div>
                
                <div class="shoulder-buttons-3d">
                  <Button.Root class="n64-btn-3d n64-btn-z">Z</Button.Root>
                  <Button.Root class="n64-btn-3d n64-btn-r">R</Button.Root>
                </div>
              </div>

            </div>
          </div>

          <!-- Game Selection with 3D Cards -->
          <div class="game-selection">
            <h4 class="section-title">Game Cartridge Selection</h4>
            <div class="cartridge-rack">
              {#each n64Games as game}
                <div class="cartridge-3d" class:selected={selectedGame === game.value}>
                  <div class="cartridge-front">
                    <div class="cartridge-label">{game.label}</div>
                  </div>
                  <div class="cartridge-top"></div>
                  <div class="cartridge-side"></div>
                </div>
              {/each}
            </div>
          </div>

          <!-- Status Badges with 3D effects -->
          <div class="badges-section">
            <h4 class="section-title">Game Status</h4>
            <div class="badges-3d">
              <Badge.Root class="n64-badge n64-badge-stars">★★★ 120 Stars</Badge.Root>
              <Badge.Root class="n64-badge n64-badge-time">Best Time: 1:23:45</Badge.Root>
              <Badge.Root class="n64-badge n64-badge-completion">100% Complete</Badge.Root>
            </div>
          </div>

        </Card.Content>
      </Card.Root>
    </section>

    <!-- 3D Game Interface -->
    <section class="game-interface-panel">
      <Card.Root class="n64-card">
        <Card.Header>
          <Card.Title class="n64-title">
            🌟 3D Game Menu System
          </Card.Title>
          <Card.Description class="n64-subtitle">
            N64-style Menus with Hardware Anti-Aliasing
          </Card.Description>
        </Card.Header>
        <Card.Content class="interface-3d">

          <!-- 3D Tabs -->
          <Tabs.Root bind:value={activeTab} class="n64-tabs">
            <Tabs.List class="n64-tabs-list">
              <Tabs.Trigger class="n64-tab-3d" value="graphics">Graphics</Tabs.Trigger>
              <Tabs.Trigger class="n64-tab-3d" value="audio">Audio</Tabs.Trigger>
              <Tabs.Trigger class="n64-tab-3d" value="controller">Controller</Tabs.Trigger>
            </Tabs.List>
            
            <Tabs.Content class="n64-tab-content" value="graphics">
              <div class="graphics-settings">
                
                <!-- Texture filtering demo -->
                <div class="texture-demo">
                  <h5>Texture Filtering Comparison</h5>
                  <div class="texture-samples">
                    <div class="texture-sample texture-point">
                      <div class="sample-label">Point Filtering</div>
                      <div class="sample-texture point-filter"></div>
                    </div>
                    <div class="texture-sample texture-bilinear">
                      <div class="sample-label">Bilinear</div>
                      <div class="sample-texture bilinear-filter"></div>
                    </div>
                    <div class="texture-sample texture-trilinear">
                      <div class="sample-label">Trilinear</div>
                      <div class="sample-texture trilinear-filter"></div>
                    </div>
                  </div>
                </div>

                <!-- 3D Settings -->
                <div class="settings-3d">
                  <div class="setting-card">
                    <span>Resolution</span>
                    <div class="resolution-display">320×240</div>
                  </div>
                  <div class="setting-card">
                    <span>Frame Rate</span>
                    <div class="fps-display">30 FPS</div>
                  </div>
                  <div class="setting-card">
                    <span>Z-Buffer</span>
                    <div class="zbuffer-display">16-bit</div>
                  </div>
                </div>

              </div>
            </Tabs.Content>
            
            <Tabs.Content class="n64-tab-content" value="audio">
              <div class="audio-3d">
                <div class="audio-visualizer-3d">
                  {#each Array(8) as _, i}
                    <div class="audio-bar-3d" style="height: {Math.random() * 80 + 20}%;">
                      <div class="bar-face bar-front"></div>
                      <div class="bar-face bar-top"></div>
                      <div class="bar-face bar-right"></div>
                    </div>
                  {/each}
                </div>
                <div class="audio-info">
                  <p>64-channel digital audio processing</p>
                  <p>Compressed audio samples with real-time decompression</p>
                </div>
              </div>
            </Tabs.Content>
            
            <Tabs.Content class="n64-tab-content" value="controller">
              <div class="controller-config">
                <div class="control-mapping">
                  <div class="mapping-item">
                    <span class="control-name">Analog Stick</span>
                    <span class="control-function">360° Movement</span>
                  </div>
                  <div class="mapping-item">
                    <span class="control-name">C Buttons</span>
                    <span class="control-function">Camera Control</span>
                  </div>
                  <div class="mapping-item">
                    <span class="control-name">Z Trigger</span>
                    <span class="control-function">Z-Targeting</span>
                  </div>
                </div>
              </div>
            </Tabs.Content>
          </Tabs.Root>

          <!-- 3D Dialog Demo -->
          <div class="dialog-section-3d">
            <Button.Root 
              class="n64-btn-primary" 
              on:on:on:click={() => dialogOpen = true}
            >
              🌟 Show N64 Dialog
            </Button.Root>
            
            <Dialog.Root bind:open={dialogOpen}>
              <Dialog.Portal>
                <Dialog.Overlay class="n64-dialog-overlay" />
                <Dialog.Content class="n64-dialog-3d">
                  <div class="dialog-box-3d">
                    <Dialog.Title class="dialog-title-3d">
                      System Message
                    </Dialog.Title>
                    <Dialog.Description class="dialog-text-3d">
                      Welcome to the N64 era of 3D gaming! Experience hardware-accelerated 
                      texture filtering, anti-aliasing, and true 3D graphics rendering. 
                      The Reality Co-Processor delivers unprecedented visual quality.
                    </Dialog.Description>
                    <div class="dialog-actions-3d">
                      <Button.Root 
                        class="n64-btn-dialog" 
                        on:on:on:click={() => dialogOpen = false}
                      >
                        ✓ Acknowledge
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

  <!-- Footer with 3D elements -->
  <footer class="n64-footer">
    <div class="footer-3d">
      <div class="credits-3d">
        <p>© 2025 Legal AI Platform - N64 3D Gaming Evolution</p>
        <p>Reality Co-Processor • Anti-Aliasing • Texture Filtering</p>
        <p>Powered by bits-ui + SvelteKit + N64 Aesthetics</p>
      </div>
    </div>
  </footer>

</div>

<!-- Toast with 3D styling -->
<Toast.Root bind:open={toastOpen} class="n64-toast">
  <Toast.Title class="toast-title-3d">Achievement Unlocked!</Toast.Title>
  <Toast.Description class="toast-description-3d">
    3D Acceleration Enabled - Texture Filtering Active
  </Toast.Description>
</Toast.Root>

<style>
  /* N64 3D Color Palette & Variables */
  :root {
    /* N64 signature colors */
    --n64-blue: #0066CC;
    --n64-red: #FF3333;
    --n64-green: #00CC33;
    --n64-yellow: #FFCC00;
    --n64-purple: #9933CC;
    --n64-gray: #666666;
    
    /* 3D gradients */
    --n64-bg: radial-gradient(ellipse at center, #001122 0%, #000811 50%, #000408 100%);
    --n64-surface: linear-gradient(145deg, #E8E8E8 0%, #D0D0D0 50%, #B8B8B8 100%);
    --n64-metal: linear-gradient(145deg, #C0C0C0 0%, #A0A0A0 25%, #808080 50%, #606060 75%, #404040 100%);
    
    /* 3D lighting */
    --n64-highlight: rgba(255, 255, 255, 0.9);
    --n64-shadow: rgba(0, 0, 0, 0.6);
    --n64-depth-1: 0 2px 4px rgba(0, 0, 0, 0.2);
    --n64-depth-2: 0 4px 8px rgba(0, 0, 0, 0.3);
    --n64-depth-3: 0 8px 16px rgba(0, 0, 0, 0.4);
    --n64-depth-4: 0 16px 32px rgba(0, 0, 0, 0.5);
    
    /* Texture filtering effects */
    --point-filter: pixelated;
    --bilinear-filter: auto;
    --trilinear-filter: high-quality;
  }

  /* Base Container with 3D environment */
  .n64-container {
    min-height: 100vh;
    background: var(--n64-bg);
    background-attachment: fixed;
    font-family: 'Arial', sans-serif;
    color: #FFFFFF;
    padding: 1rem;
    perspective: 1000px;
    position: relative;
    overflow-x: hidden;
  }

  .n64-container::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      radial-gradient(circle at 30% 40%, rgba(0, 102, 204, 0.1) 0%, transparent 50%),
      radial-gradient(circle at 70% 80%, rgba(255, 204, 0, 0.1) 0%, transparent 50%);
    pointer-events: none;
    z-index: -1;
  }

  /* Header with 3D branding */
  .n64-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    background: var(--n64-surface);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    border: 4px solid var(--n64-blue);
    box-shadow: 
      var(--n64-depth-3),
      inset 0 4px 8px var(--n64-highlight);
    transform: perspective(800px) rotateX(5deg);
    color: #333;
  }

  .n64-logo {
    display: flex;
    align-items: center;
    gap: 1.5rem;
  }

  .logo-3d {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .logo-text {
    font-size: 2.5rem;
    font-weight: 900;
    color: var(--n64-blue);
    text-shadow: 
      2px 2px 0 var(--n64-shadow),
      4px 4px 8px rgba(0, 0, 0, 0.3);
    transform-style: preserve-3d;
    transition: transform 0.1s linear;
  }

  .n64-controller {
    width: 80px;
    height: 50px;
    background: var(--n64-metal);
    border-radius: 25px;
    border: 3px solid #444;
    position: relative;
    box-shadow: var(--n64-depth-2);
    transform: perspective(400px) rotateX(30deg) rotateY(-10deg);
  }

  .system-status {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .status-item {
    display: flex;
    justify-content: space-between;
    min-width: 150px;
    font-size: 0.9rem;
    font-weight: bold;
  }

  .status-label {
    color: #666;
  }

  .status-value {
    color: var(--n64-blue);
    font-family: 'Courier New', monospace;
  }

  /* Main Layout with 3D transforms */
  .n64-main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 2rem;
    margin-bottom: 2rem;
  }

  .game-interface-panel {
    grid-column: 1 / -1;
  }

  /* 3D Cards */
  .n64-card {
    background: var(--n64-surface);
    border: 4px solid var(--n64-blue);
    border-radius: 20px;
    box-shadow: 
      var(--n64-depth-3),
      inset 0 4px 8px var(--n64-highlight);
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    transform-style: preserve-3d;
    color: #333;
  }

  .n64-card:hover {
    transform: perspective(800px) rotateX(5deg) rotateY(5deg) translateZ(20px);
    box-shadow: 
      var(--n64-depth-4),
      0 0 30px rgba(0, 102, 204, 0.4),
      inset 0 4px 8px var(--n64-highlight);
  }

  .n64-title {
    font-size: 1.6rem;
    font-weight: 900;
    color: var(--n64-blue);
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    margin-bottom: 0.5rem;
  }

  .n64-subtitle {
    color: #666;
    font-size: 0.95rem;
    font-style: italic;
  }

  .n64-content {
    padding: 2rem;
  }

  /* 3D Cube Demo */
  .cube-demo {
    display: flex;
    justify-content: center;
    margin: 2rem 0;
  }

  .scene-3d {
    perspective: 800px;
    width: 200px;
    height: 200px;
  }

  .n64-cube {
    position: relative;
    width: 100%;
    height: 100%;
    transform-style: preserve-3d;
    transition: transform 0.1s linear;
  }

  .cube-face {
    position: absolute;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    font-weight: bold;
    color: white;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    border: 3px solid rgba(255, 255, 255, 0.3);
  }

  .cube-front { background: linear-gradient(135deg, var(--n64-blue), #004499); transform: rotateY(0deg) translateZ(100px); }
  .cube-back { background: linear-gradient(135deg, var(--n64-red), #CC0000); transform: rotateY(180deg) translateZ(100px); }
  .cube-right { background: linear-gradient(135deg, var(--n64-green), #009900); transform: rotateY(90deg) translateZ(100px); }
  .cube-left { background: linear-gradient(135deg, var(--n64-yellow), #CC9900); transform: rotateY(-90deg) translateZ(100px); }
  .cube-top { background: linear-gradient(135deg, var(--n64-purple), #7700AA); transform: rotateX(90deg) translateZ(100px); }
  .cube-bottom { background: linear-gradient(135deg, var(--n64-gray), #444444); transform: rotateX(-90deg) translateZ(100px); }

  /* Technical Specifications */
  .tech-specs {
    display: grid;
    gap: 2rem;
  }

  .spec-section h4 {
    color: var(--n64-blue);
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .spec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }

  .spec-item {
    background: rgba(255, 255, 255, 0.9);
    padding: 1rem;
    border-radius: 12px;
    border: 3px solid rgba(0, 102, 204, 0.3);
    box-shadow: var(--n64-depth-1);
    transition: all 0.3s ease;
  }

  .spec-item:hover {
    transform: translateY(-2px);
    box-shadow: var(--n64-depth-2);
  }

  .spec-item strong {
    color: var(--n64-blue);
    display: block;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
  }

  /* Feature List */
  .feature-list {
    display: grid;
    gap: 0.75rem;
  }

  .feature-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(255, 255, 255, 0.9);
    padding: 1rem;
    border-radius: 12px;
    border: 3px solid rgba(0, 102, 204, 0.2);
    font-weight: bold;
    color: #333;
    transition: all 0.3s ease;
  }

  .feature-item:hover {
    border-color: var(--n64-blue);
    transform: translateX(5px);
  }

  .feature-icon {
    font-size: 1.5rem;
  }

  /* Filtering Controls */
  .filtering-controls {
    background: rgba(0, 102, 204, 0.1);
    padding: 2rem;
    border-radius: 16px;
    border: 3px solid rgba(0, 102, 204, 0.3);
    margin-top: 2rem;
  }

  .filtering-controls h4 {
    color: var(--n64-blue);
    font-size: 1.2rem;
    margin-bottom: 1.5rem;
    text-transform: uppercase;
  }

  .control-group {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
  }

  .control-group label {
    min-width: 140px;
    font-weight: bold;
    color: #333;
  }

  /* Select Component */
  .n64-select-trigger {
    background: var(--n64-surface);
    border: 3px solid var(--n64-blue);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    color: #333;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    min-width: 160px;
  }

  .n64-select-trigger:hover {
    border-color: var(--n64-purple);
    box-shadow: var(--n64-depth-2);
  }

  .n64-select-content {
    background: var(--n64-surface);
    border: 3px solid var(--n64-blue);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: var(--n64-depth-3);
  }

  .n64-select-item {
    padding: 0.75rem 1rem;
    color: #333;
    font-weight: bold;
    cursor: pointer;
    transition: background-color 0.2s ease;
  }

  .n64-select-item:hover {
    background: rgba(0, 102, 204, 0.2);
  }

  /* Slider Component */
  .n64-slider {
    flex: 1;
    max-width: 200px;
  }

  .slider-track {
    background: #CCC;
    height: 8px;
    border-radius: 4px;
    border: 2px solid #999;
    position: relative;
  }

  .slider-range {
    background: linear-gradient(90deg, var(--n64-blue), var(--n64-purple));
    height: 100%;
    border-radius: 2px;
  }

  .slider-thumb {
    width: 20px;
    height: 20px;
    background: var(--n64-surface);
    border: 3px solid var(--n64-blue);
    border-radius: 50%;
    cursor: pointer;
    box-shadow: var(--n64-depth-2);
    transition: all 0.2s ease;
  }

  .slider-thumb:hover {
    transform: scale(1.1);
    box-shadow: var(--n64-depth-3);
  }

  .slider-value {
    min-width: 50px;
    text-align: center;
    font-weight: bold;
    color: var(--n64-blue);
    font-family: 'Courier New', monospace;
  }

  /* Checkbox */
  .n64-checkbox {
    width: 28px;
    height: 28px;
    background: white;
    border: 3px solid var(--n64-blue);
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .n64-checkbox[data-state="checked"] {
    background: var(--n64-blue);
    box-shadow: var(--n64-depth-2);
  }

  .checkbox-indicator {
    color: white;
    font-weight: bold;
    font-size: 1.2rem;
  }

  .aa-status {
    font-weight: bold;
    color: #333;
  }

  /* 3D Components Section */
  .components-3d {
    padding: 2rem;
    display: grid;
    gap: 3rem;
  }

  .section-title {
    color: var(--n64-blue);
    font-size: 1.3rem;
    font-weight: bold;
    margin-bottom: 1.5rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    text-align: center;
  }

  /* N64 Controller Layout */
  .controller-section {
    background: rgba(255, 255, 255, 0.95);
    padding: 2rem;
    border-radius: 20px;
    border: 4px solid var(--n64-blue);
    box-shadow: var(--n64-depth-2);
  }

  .n64-controller-layout {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 2rem;
    perspective: 800px;
  }

  .controller-left,
  .controller-center,
  .controller-right {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }

  /* Analog Stick */
  .analog-stick {
    position: relative;
    width: 60px;
    height: 60px;
  }

  .stick-base {
    width: 100%;
    height: 100%;
    background: var(--n64-metal);
    border-radius: 50%;
    border: 4px solid #444;
    box-shadow: var(--n64-depth-2);
  }

  .stick-top {
    position: absolute;
    top: 10px;
    left: 10px;
    width: 40px;
    height: 40px;
    background: #333;
    border-radius: 50%;
    border: 2px solid #555;
    cursor: pointer;
    transition: transform 0.2s ease;
  }

  .stick-top:hover {
    transform: translate(2px, 2px);
  }

  /* 3D D-Pad */
  .dpad-3d {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }

  .dpad-middle-3d {
    display: flex;
    align-items: center;
    gap: 2px;
  }

  .dpad-btn-3d {
    width: 40px;
    height: 40px;
    background: var(--n64-metal);
    border: 3px solid #444;
    border-radius: 6px;
    color: #FFF;
    font-weight: bold;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    box-shadow: var(--n64-depth-1);
    transform: perspective(200px) rotateX(10deg);
  }

  .dpad-btn-3d:hover {
    background: #999;
    transform: perspective(200px) rotateX(10deg) translateZ(5px);
  }

  .dpad-center-3d {
    width: 40px;
    height: 40px;
    background: radial-gradient(circle, #444, #222);
    border: 3px solid #333;
    border-radius: 50%;
    box-shadow: inset var(--n64-depth-1);
  }

  /* Start Button */
  .start-button-3d {
    background: var(--n64-red);
    color: white;
    padding: 0.5rem 1rem;
    border: 3px solid #AA0000;
    border-radius: 12px;
    font-weight: bold;
    font-size: 0.8rem;
    cursor: pointer;
    box-shadow: var(--n64-depth-2);
    transition: all 0.2s ease;
    transform: perspective(200px) rotateX(15deg);
  }

  .start-button-3d:hover {
    background: #FF5555;
    transform: perspective(200px) rotateX(15deg) translateZ(5px);
  }

  /* C Buttons */
  .c-buttons {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .c-middle {
    display: flex;
    gap: 4px;
  }

  .c-btn {
    width: 32px;
    height: 32px;
    background: var(--n64-yellow);
    border: 2px solid #CC9900;
    border-radius: 6px;
    color: #664400;
    font-weight: bold;
    font-size: 0.7rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    box-shadow: var(--n64-depth-1);
    transform: perspective(150px) rotateX(20deg);
  }

  .c-btn:hover {
    background: #FFDD33;
    transform: perspective(150px) rotateX(20deg) translateZ(3px);
  }

  /* Face Buttons */
  .face-buttons-3d {
    display: flex;
    gap: 1rem;
  }

  .n64-btn-3d {
    width: 50px;
    height: 50px;
    border: none;
    border-radius: 50%;
    font-weight: bold;
    font-size: 1.2rem;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: var(--n64-depth-2);
    color: white;
    transform: perspective(200px) rotateX(20deg);
  }

  .n64-btn-3d:hover {
    transform: perspective(200px) rotateX(20deg) translateZ(8px);
    box-shadow: var(--n64-depth-3);
  }

  .n64-btn-a { background: linear-gradient(145deg, var(--n64-blue), #004499); }
  .n64-btn-b { background: linear-gradient(145deg, var(--n64-green), #009900); }
  .n64-btn-z { background: linear-gradient(145deg, var(--n64-purple), #7700AA); width: 60px; height: 30px; border-radius: 15px; }
  .n64-btn-r { background: linear-gradient(145deg, var(--n64-gray), #444444); width: 60px; height: 30px; border-radius: 15px; }

  .shoulder-buttons-3d {
    display: flex;
    gap: 0.5rem;
  }

  /* Game Cartridge Selection */
  .game-selection {
    background: rgba(255, 255, 255, 0.95);
    padding: 2rem;
    border-radius: 20px;
    border: 4px solid var(--n64-blue);
    box-shadow: var(--n64-depth-2);
  }

  .cartridge-rack {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    perspective: 800px;
  }

  .cartridge-3d {
    position: relative;
    width: 80px;
    height: 120px;
    transform-style: preserve-3d;
    cursor: pointer;
    transition: all 0.4s ease;
  }

  .cartridge-3d:hover {
    transform: rotateY(15deg) rotateX(10deg) translateZ(10px);
  }

  .cartridge-3d.selected {
    transform: rotateY(15deg) rotateX(10deg) translateZ(20px);
    filter: brightness(1.2);
  }

  .cartridge-front {
    position: absolute;
    width: 100%;
    height: 100%;
    background: linear-gradient(145deg, var(--n64-surface), #B0B0B0);
    border: 3px solid var(--n64-blue);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: 0.7rem;
    font-weight: bold;
    color: #333;
    transform: translateZ(20px);
    box-shadow: var(--n64-depth-2);
  }

  .cartridge-top {
    position: absolute;
    width: 100%;
    height: 40px;
    background: linear-gradient(145deg, #D0D0D0, #A0A0A0);
    border: 2px solid #999;
    border-radius: 8px 8px 0 0;
    transform: rotateX(90deg) translateZ(20px);
  }

  .cartridge-side {
    position: absolute;
    width: 40px;
    height: 100%;
    background: linear-gradient(145deg, #C0C0C0, #909090);
    border: 2px solid #888;
    transform: rotateY(90deg) translateZ(20px);
  }

  .cartridge-label {
    padding: 0.5rem;
    line-height: 1.2;
  }

  /* Badges with 3D effects */
  .badges-section {
    background: rgba(255, 255, 255, 0.95);
    padding: 2rem;
    border-radius: 20px;
    border: 4px solid var(--n64-blue);
    box-shadow: var(--n64-depth-2);
  }

  .badges-3d {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    justify-content: center;
  }

  .n64-badge {
    padding: 0.75rem 1.5rem;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 3px solid;
    box-shadow: var(--n64-depth-2);
    transition: all 0.3s ease;
    transform: perspective(300px) rotateX(10deg);
  }

  .n64-badge:hover {
    transform: perspective(300px) rotateX(10deg) translateZ(5px);
    box-shadow: var(--n64-depth-3);
  }

  .n64-badge-stars {
    background: linear-gradient(145deg, var(--n64-yellow), #CC9900);
    color: #664400;
    border-color: #AA7700;
  }

  .n64-badge-time {
    background: linear-gradient(145deg, var(--n64-green), #009900);
    color: white;
    border-color: #007700;
  }

  .n64-badge-completion {
    background: linear-gradient(145deg, var(--n64-purple), #7700AA);
    color: white;
    border-color: #5500AA;
  }

  /* 3D Game Interface */
  .interface-3d {
    padding: 2rem;
  }

  /* 3D Tabs */
  .n64-tabs {
    width: 100%;
  }

  .n64-tabs-list {
    display: flex;
    background: rgba(0, 102, 204, 0.1);
    border-radius: 16px;
    padding: 6px;
    margin-bottom: 2rem;
    transform: perspective(600px) rotateX(5deg);
  }

  .n64-tab-3d {
    flex: 1;
    padding: 1rem 2rem;
    background: transparent;
    border: none;
    border-radius: 12px;
    color: var(--n64-blue);
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    transform: perspective(400px) rotateX(-5deg);
  }

  .n64-tab-3d[data-state="active"] {
    background: var(--n64-surface);
    color: #333;
    box-shadow: var(--n64-depth-2);
    transform: perspective(400px) rotateX(-5deg) translateZ(5px);
  }

  .n64-tab-3d:hover:not([data-state="active"]) {
    background: rgba(0, 102, 204, 0.2);
    transform: perspective(400px) rotateX(-5deg) translateZ(2px);
  }

  .n64-tab-content {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 2rem;
    border: 4px solid rgba(0, 102, 204, 0.3);
    box-shadow: var(--n64-depth-2);
    transform: perspective(800px) rotateX(3deg);
  }

  /* Texture Filtering Demo */
  .texture-demo {
    margin-bottom: 2rem;
  }

  .texture-demo h5 {
    color: var(--n64-blue);
    font-size: 1.1rem;
    font-weight: bold;
    margin-bottom: 1rem;
    text-align: center;
  }

  .texture-samples {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
  }

  .texture-sample {
    text-align: center;
    background: white;
    padding: 1rem;
    border-radius: 12px;
    border: 3px solid rgba(0, 102, 204, 0.2);
    box-shadow: var(--n64-depth-1);
  }

  .sample-label {
    font-weight: bold;
    color: #333;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
  }

  .sample-texture {
    width: 80px;
    height: 80px;
    margin: 0 auto;
    border: 2px solid #CCC;
    background: 
      repeating-conic-gradient(#0066CC 0deg 45deg, #FFCC00 45deg 90deg) 
      0 0 / 10px 10px;
    border-radius: 8px;
  }

  .point-filter {
    image-rendering: pixelated;
    image-rendering: -moz-crisp-edges;
    image-rendering: crisp-edges;
  }

  .bilinear-filter {
    image-rendering: auto;
    filter: blur(0.5px);
  }

  .trilinear-filter {
    image-rendering: high-quality;
    filter: blur(0.2px) brightness(1.05);
  }

  /* 3D Settings */
  .settings-3d {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 2rem;
  }

  .setting-card {
    background: white;
    padding: 1.5rem 1rem;
    border-radius: 12px;
    border: 3px solid rgba(0, 102, 204, 0.2);
    text-align: center;
    box-shadow: var(--n64-depth-1);
    transform: perspective(400px) rotateX(10deg);
    transition: transform 0.3s ease;
  }

  .setting-card:hover {
    transform: perspective(400px) rotateX(10deg) translateZ(5px);
  }

  .setting-card span {
    display: block;
    font-weight: bold;
    color: #666;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
  }

  .resolution-display,
  .fps-display,
  .zbuffer-display {
    font-family: 'Courier New', monospace;
    font-size: 1.1rem;
    font-weight: bold;
    color: var(--n64-blue);
  }

  /* 3D Audio Visualizer */
  .audio-3d {
    text-align: center;
  }

  .audio-visualizer-3d {
    display: flex;
    justify-content: center;
    align-items: end;
    height: 150px;
    gap: 8px;
    perspective: 600px;
    margin-bottom: 2rem;
  }

  .audio-bar-3d {
    position: relative;
    width: 30px;
    transform-style: preserve-3d;
    transition: height 0.3s ease;
    animation: audioBar3D 2s infinite ease-in-out;
  }

  .audio-bar-3d:nth-child(odd) {
    animation-delay: -0.5s;
  }

  .audio-bar-3d:nth-child(even) {
    animation-delay: -1s;
  }

  @keyframes audioBar3D {
    0%, 100% { transform: rotateX(0deg) rotateY(5deg); }
    50% { transform: rotateX(10deg) rotateY(-5deg); }
  }

  .bar-face {
    position: absolute;
    background: linear-gradient(135deg, var(--n64-blue), var(--n64-purple));
  }

  .bar-front {
    width: 30px;
    height: 100%;
    transform: translateZ(15px);
    border-radius: 4px 4px 0 0;
  }

  .bar-top {
    width: 30px;
    height: 30px;
    transform: rotateX(90deg) translateZ(15px);
    background: linear-gradient(135deg, #0088FF, #AA44FF);
  }

  .bar-right {
    width: 30px;
    height: 100%;
    transform: rotateY(90deg) translateZ(15px);
    background: linear-gradient(135deg, #004499, #770088);
  }

  .audio-info {
    color: #666;
    font-style: italic;
  }

  .audio-info p {
    margin: 0.25rem 0;
  }

  /* Controller Configuration */
  .controller-config {
    padding: 1rem;
  }

  .control-mapping {
    display: grid;
    gap: 1rem;
  }

  .mapping-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: white;
    padding: 1rem;
    border-radius: 12px;
    border: 3px solid rgba(0, 102, 204, 0.2);
    box-shadow: var(--n64-depth-1);
    transform: perspective(400px) rotateX(5deg);
  }

  .control-name {
    font-weight: bold;
    color: var(--n64-blue);
  }

  .control-function {
    color: #666;
    font-style: italic;
  }

  /* 3D Dialog */
  .dialog-section-3d {
    text-align: center;
    margin-top: 2rem;
  }

  .n64-btn-primary {
    background: linear-gradient(145deg, var(--n64-blue), #004499);
    color: white;
    padding: 1rem 2rem;
    border-radius: 16px;
    border: 3px solid #003366;
    font-weight: bold;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: var(--n64-depth-2);
    transform: perspective(400px) rotateX(10deg);
  }

  .n64-btn-primary:hover {
    background: linear-gradient(145deg, #0077FF, #0055CC);
    transform: perspective(400px) rotateX(10deg) translateZ(8px);
    box-shadow: var(--n64-depth-3);
  }

  .n64-dialog-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.8);
    backdrop-filter: blur(4px);
    z-index: 50;
  }

  .n64-dialog-3d {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 51;
    max-width: 600px;
    width: calc(100vw - 2rem);
    perspective: 800px;
  }

  .dialog-box-3d {
    background: var(--n64-surface);
    border: 6px solid var(--n64-blue);
    border-radius: 24px;
    padding: 3rem;
    box-shadow: var(--n64-depth-4);
    transform: perspective(600px) rotateX(5deg) rotateY(2deg);
    transform-style: preserve-3d;
  }

  .dialog-title-3d {
    color: var(--n64-blue);
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 1.5rem;
    text-align: center;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
  }

  .dialog-text-3d {
    color: #333;
    line-height: 1.6;
    margin-bottom: 2rem;
    text-align: center;
    font-size: 1rem;
  }

  .dialog-actions-3d {
    display: flex;
    justify-content: center;
  }

  .n64-btn-dialog {
    background: linear-gradient(145deg, var(--n64-green), #009900);
    color: white;
    padding: 1rem 2rem;
    border-radius: 16px;
    border: 3px solid #007700;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: var(--n64-depth-2);
    transform: perspective(300px) rotateX(15deg);
  }

  .n64-btn-dialog:hover {
    background: linear-gradient(145deg, #00CC44, #00AA33);
    transform: perspective(300px) rotateX(15deg) translateZ(5px);
    box-shadow: var(--n64-depth-3);
  }

  /* Footer with 3D styling */
  .n64-footer {
    margin-top: 3rem;
  }

  .footer-3d {
    background: var(--n64-surface);
    padding: 3rem 2rem;
    border-radius: 20px;
    border: 4px solid var(--n64-blue);
    text-align: center;
    color: #333;
    box-shadow: var(--n64-depth-3);
    transform: perspective(800px) rotateX(5deg);
  }

  .credits-3d p {
    margin: 0.5rem 0;
    font-size: 0.95rem;
  }

  .credits-3d p:first-child {
    font-weight: bold;
    color: var(--n64-blue);
    font-size: 1.1rem;
  }

  /* Toast with 3D styling */
  .n64-toast {
    background: var(--n64-surface);
    border: 4px solid var(--n64-green);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    box-shadow: var(--n64-depth-3);
    color: #333;
    transform: perspective(400px) rotateX(10deg);
  }

  .toast-title-3d {
    font-weight: bold;
    color: var(--n64-green);
    margin-bottom: 0.5rem;
  }

  .toast-description-3d {
    font-size: 0.95rem;
    color: #666;
  }

  /* Responsive Design */
  @media (max-width: 1024px) {
    .n64-main {
      grid-template-columns: 1fr;
    }
    
    .n64-controller-layout {
      flex-direction: column;
      gap: 1rem;
    }
    
    .cartridge-rack {
      justify-content: center;
    }
    
    .texture-samples {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 768px) {
    .n64-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
      transform: none;
    }
    
    .logo-text {
      font-size: 2rem;
    }
    
    .spec-grid {
      grid-template-columns: 1fr;
    }
    
    .settings-3d {
      grid-template-columns: 1fr;
    }
    
    .control-group {
      flex-direction: column;
      align-items: stretch;
    }
    
    .control-group label {
      min-width: auto;
      text-align: center;
    }
  }

  /* Performance optimizations for 3D */
  .n64-container * {
    will-change: transform;
  }
  
  .n64-cube,
  .cartridge-3d,
  .audio-bar-3d {
    backface-visibility: hidden;
  }
  
  /* Hardware acceleration hints */
  .n64-card,
  .n64-btn-3d,
  .dialog-box-3d {
    transform-style: preserve-3d;
  }
</style>