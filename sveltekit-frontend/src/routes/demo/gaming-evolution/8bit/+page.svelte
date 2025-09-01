<script lang="ts">
  import { Dialog, Button, Badge, Card, Checkbox } from 'bits-ui';
  import 'nes.css/css/nes.min.css';
  
  let dialogOpen = $state(false);
  let health = $state(3);
  let score = $state(1250);
  let level = $state(1);

  function decreaseHealth() {
    if (health > 0) health--;
  }

  function increaseScore() {
    score += 100;
  }
</script>

<svelte:head>
  <title>8-Bit Gaming Components - NES Era</title>
  <link href="https://unpkg.com/nes.css@2.3.0/css/nes.min.css" rel="stylesheet" />
  <link href="https://fonts.googleapis.com/css?family=Press+Start+2P" rel="stylesheet">
</svelte:head>

<div class="nes-container with-title console-container">
  <p class="title">🎮 8-BIT GAMING ERA (1983-1995)</p>
  
  <!-- Analysis of 8-bit characteristics -->
  <section class="analysis-section">
    <div class="nes-container is-dark">
      <h2 class="nes-text is-primary">What makes it 8-bit?</h2>
      <div class="characteristics-grid">
        <div class="characteristic">
          <h3 class="nes-text is-success">🎨 Color Palette</h3>
          <div class="color-palette">
            <div class="color-box" style="background: #212529;"></div>
            <div class="color-box" style="background: #ffffff;"></div>
            <div class="color-box" style="background: #92cc41;"></div>
            <div class="color-box" style="background: #f7d51d;"></div>
            <div class="color-box" style="background: #e76e55;"></div>
            <div class="color-box" style="background: #0084ff;"></div>
          </div>
          <p>Limited 6-color palette typical of NES hardware constraints</p>
        </div>
        
        <div class="characteristic">
          <h3 class="nes-text is-warning">📐 Pixel Perfect</h3>
          <div class="pixel-demo">
            <div class="pixel-art-sprite"></div>
          </div>
          <p>Hard edges, no anti-aliasing, pixelated borders</p>
        </div>
        
        <div class="characteristic">
          <h3 class="nes-text is-error">🔤 Typography</h3>
          <div class="font-demo">
            <p class="bitmap-font">PRESS START 2P</p>
          </div>
          <p>Bitmap fonts, fixed-width, chunky appearance</p>
        </div>
      </div>
    </div>
  </section>

  <!-- Game UI Recreation -->
  <section class="game-ui-section">
    <h2 class="nes-text is-primary">Classic Game UI</h2>
    
    <div class="game-hud">
      <!-- Player Stats -->
      <div class="nes-container stats-panel">
        <div class="player-info">
          <p>MARIO</p>
          <p>SCORE: {score.toLocaleString()}</p>
          <p>LEVEL: {level}-1</p>
        </div>
        
        <div class="health-display">
          <p>HEALTH:</p>
          <div class="hearts">
            {#each Array(3) as _, i}
              <span class="heart {i < health ? 'full' : 'empty'}">
                {i < health ? '❤️' : '🖤'}
              </span>
            {/each}
          </div>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="controls-panel">
        <div class="nes-container">
          <p class="nes-text is-primary">GAME CONTROLS</p>
          <div class="button-grid">
            <Button.Root class="nes-btn action-btn" on:on:on:click={increaseScore}>
              A BUTTON
            </Button.Root>
            <Button.Root class="nes-btn action-btn" on:on:on:click={decreaseHealth}>
              B BUTTON
            </Button.Root>
            <Button.Root class="nes-btn start-btn" on:on:on:click={() => dialogOpen = true}>
              START
            </Button.Root>
            <Button.Root class="nes-btn select-btn">
              SELECT
            </Button.Root>
          </div>
        </div>
      </div>
    </div>

    <!-- D-Pad Recreation -->
    <div class="dpad-container">
      <div class="nes-container">
        <p class="nes-text is-primary">DIRECTIONAL PAD</p>
        <div class="dpad">
          <Button.Root class="nes-btn dpad-btn dpad-up">▲</Button.Root>
          <div class="dpad-middle">
            <Button.Root class="nes-btn dpad-btn dpad-left">◄</Button.Root>
            <div class="dpad-center"></div>
            <Button.Root class="nes-btn dpad-btn dpad-right">►</Button.Root>
          </div>
          <Button.Root class="nes-btn dpad-btn dpad-down">▼</Button.Root>
        </div>
      </div>
    </div>
  </section>

  <!-- Technical Specifications -->
  <section class="specs-section">
    <div class="nes-container with-title">
      <p class="title">NES TECHNICAL SPECS</p>
      <div class="specs-grid">
        <div class="spec-item">
          <Badge.Root class="nes-badge spec-badge">
            <span class="is-primary">CPU</span>
          </Badge.Root>
          <p>6502 @ 1.79MHz</p>
        </div>
        
        <div class="spec-item">
          <Badge.Root class="nes-badge spec-badge">
            <span class="is-success">COLORS</span>
          </Badge.Root>
          <p>25 on screen / 64 total</p>
        </div>
        
        <div class="spec-item">
          <Badge.Root class="nes-badge spec-badge">
            <span class="is-warning">RESOLUTION</span>
          </Badge.Root>
          <p>256 × 240 pixels</p>
        </div>
        
        <div class="spec-item">
          <Badge.Root class="nes-badge spec-badge">
            <span class="is-error">MEMORY</span>
          </Badge.Root>
          <p>2KB RAM</p>
        </div>
      </div>
    </div>
  </section>

  <!-- Game Selection Menu -->
  <Dialog.Root bind:open={dialogOpen}>
    <Dialog.Portal>
      <Dialog.Overlay class="dialog-overlay" />
      <Dialog.Content class="nes-dialog game-menu">
        <Dialog.Title class="nes-text is-primary">
          GAME SELECT
        </Dialog.Title>
        <Dialog.Description>
          <div class="game-list">
            <Card.Root class="nes-container game-card">
              <Card.Content>
                <div class="game-preview">
                  <div class="game-icon mario-icon"></div>
                  <div class="game-info">
                    <h3>SUPER MARIO BROS</h3>
                    <p>WORLD 1-1</p>
                    <Badge.Root class="nes-badge">
                      <span class="is-success">CLASSIC</span>
                    </Badge.Root>
                  </div>
                </div>
              </Card.Content>
            </Card.Root>

            <Card.Root class="nes-container game-card">
              <Card.Content>
                <div class="game-preview">
                  <div class="game-icon zelda-icon"></div>
                  <div class="game-info">
                    <h3>THE LEGEND OF ZELDA</h3>
                    <p>ADVENTURE</p>
                    <Badge.Root class="nes-badge">
                      <span class="is-warning">RPG</span>
                    </Badge.Root>
                  </div>
                </div>
              </Card.Content>
            </Card.Root>

            <Card.Root class="nes-container game-card">
              <Card.Content>
                <div class="game-preview">
                  <div class="game-icon metroid-icon"></div>
                  <div class="game-info">
                    <h3>METROID</h3>
                    <p>PLANET ZEBES</p>
                    <Badge.Root class="nes-badge">
                      <span class="is-error">ACTION</span>
                    </Badge.Root>
                  </div>
                </div>
              </Card.Content>
            </Card.Root>
          </div>
        </Dialog.Description>
        <div class="menu-actions">
          <Dialog.Close class="nes-btn is-primary">START GAME</Dialog.Close>
          <Dialog.Close class="nes-btn">CANCEL</Dialog.Close>
        </div>
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>

  <!-- 8-bit Color Analysis -->
  <section class="color-analysis">
    <div class="nes-container with-title">
      <p class="title">8-BIT COLOR THEORY</p>
      <div class="color-theory-grid">
        <div class="color-theory-item">
          <h3 class="nes-text is-primary">Hardware Limitations</h3>
          <ul class="nes-list is-disc">
            <li>NES PPU: 64 colors total</li>
            <li>25 colors on screen max</li>
            <li>4 colors per sprite</li>
            <li>No color blending</li>
          </ul>
        </div>
        
        <div class="color-theory-item">
          <h3 class="nes-text is-success">Design Principles</h3>
          <ul class="nes-list is-disc">
            <li>High contrast ratios</li>
            <li>Distinct color separation</li>
            <li>Symbolic color usage</li>
            <li>Readability first</li>
          </ul>
        </div>
      </div>
    </div>
  </section>
</div>

<style>
  @import 'nes.css/css/nes.min.css';

  :global(body) {
    font-family: "Press Start 2P", cursive;
    background: linear-gradient(45deg, #1a1a2e, #16213e);
    color: white;
    image-rendering: pixelated;
    image-rendering: -moz-crisp-edges;
    image-rendering: crisp-edges;
  }

  .console-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
  }

  /* Analysis Section */
  .analysis-section {
    margin: 2rem 0;
  }

  .characteristics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-top: 1rem;
  }

  .characteristic {
    text-align: center;
  }

  .characteristic h3 {
    margin-bottom: 1rem;
  }

  .color-palette {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin: 1rem 0;
  }

  .color-box {
    width: 30px;
    height: 30px;
    border: 2px solid white;
    image-rendering: pixelated;
  }

  .pixel-demo {
    display: flex;
    justify-content: center;
    margin: 1rem 0;
  }

  .pixel-art-sprite {
    width: 64px;
    height: 64px;
    background: 
      conic-gradient(from 0deg, 
        #92cc41 0deg 90deg,
        #f7d51d 90deg 180deg,
        #e76e55 180deg 270deg,
        #0084ff 270deg 360deg
      );
    image-rendering: pixelated;
    border: 2px solid white;
  }

  .bitmap-font {
    font-size: 1.2rem;
    color: #f7d51d;
    text-shadow: 2px 2px 0px #000;
  }

  /* Game UI Section */
  .game-ui-section {
    margin: 3rem 0;
  }

  .game-hud {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin: 2rem 0;
  }

  .stats-panel {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .player-info p {
    margin: 0.25rem 0;
    font-size: 0.8rem;
  }

  .health-display {
    text-align: center;
  }

  .hearts {
    display: flex;
    gap: 0.25rem;
    margin-top: 0.5rem;
  }

  .heart {
    font-size: 1.2rem;
  }

  .controls-panel .button-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 1rem;
  }

  .action-btn, .start-btn, .select-btn {
    font-size: 0.7rem;
    padding: 0.5rem;
  }

  /* D-Pad */
  .dpad-container {
    text-align: center;
    margin: 2rem 0;
  }

  .dpad {
    display: inline-grid;
    grid-template-rows: auto auto auto;
    grid-template-columns: auto auto auto;
    gap: 2px;
    margin-top: 1rem;
  }

  .dpad-up {
    grid-column: 2;
    grid-row: 1;
  }

  .dpad-middle {
    grid-column: 1 / 4;
    grid-row: 2;
    display: flex;
    align-items: center;
  }

  .dpad-left {
    grid-column: 1;
  }

  .dpad-center {
    width: 40px;
    height: 40px;
    background: #666;
    border: 2px solid #333;
  }

  .dpad-right {
    grid-column: 3;
  }

  .dpad-down {
    grid-column: 2;
    grid-row: 3;
  }

  .dpad-btn {
    width: 40px;
    height: 40px;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
  }

  /* Specifications */
  .specs-section {
    margin: 2rem 0;
  }

  .specs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }

  .spec-item {
    text-align: center;
    padding: 1rem;
  }

  .spec-badge {
    margin-bottom: 0.5rem;
  }

  /* Dialog Styling */
  .dialog-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.9);
    z-index: 999;
  }

  .game-menu {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    max-width: 600px;
    width: 90vw;
    max-height: 80vh;
    overflow-y: auto;
    z-index: 1000;
    background: #212529;
    color: white;
    border: 4px solid white;
  }

  .game-list {
    margin: 1rem 0;
  }

  .game-card {
    margin: 1rem 0;
  }

  .game-preview {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .game-icon {
    width: 48px;
    height: 48px;
    image-rendering: pixelated;
    border: 2px solid white;
  }

  .mario-icon {
    background: linear-gradient(45deg, #e76e55, #f7d51d);
  }

  .zelda-icon {
    background: linear-gradient(45deg, #92cc41, #0084ff);
  }

  .metroid-icon {
    background: linear-gradient(45deg, #e76e55, #92cc41);
  }

  .game-info h3 {
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
  }

  .game-info p {
    font-size: 0.6rem;
    margin-bottom: 0.5rem;
  }

  .menu-actions {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1rem;
  }

  /* Color Analysis */
  .color-analysis {
    margin: 2rem 0;
  }

  .color-theory-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-top: 1rem;
  }

  .color-theory-item ul {
    text-align: left;
    margin: 1rem 0;
  }

  .color-theory-item li {
    margin: 0.5rem 0;
    font-size: 0.7rem;
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .game-hud {
      grid-template-columns: 1fr;
    }
    
    .characteristics-grid {
      grid-template-columns: 1fr;
    }
    
    .game-preview {
      flex-direction: column;
      text-align: center;
    }
    
    .dpad-middle {
      flex-direction: column;
    }
  }

  /* 8-bit specific enhancements */
  :global(.nes-btn) {
    image-rendering: pixelated;
    text-shadow: 1px 1px 0px rgba(0,0,0,0.5);
  }

  :global(.nes-container) {
    image-rendering: pixelated;
    box-shadow: 4px 4px 0px rgba(0,0,0,0.3);
  }

  :global(.nes-badge span) {
    text-shadow: 1px 1px 0px rgba(0,0,0,0.5);
  }
</style>