<script lang="ts">
  import { ModernCard, ModernButton, ModernDialog } from '$lib/components/ui/modern';
  import {
    Shield,
    Settings,
    Bell,
    User,
    FileText,
    Search,
    Brain,
    Database
  } from 'lucide-svelte';

  // Demo state
  let dialogOpen = $state(false);
  let notifications = $state(3);
  let loadingCard = $state(false);

  function handleCardClick() {
    console.log('Card clicked!');
  }

  function handleLoadingDemo() {
    loadingCard = true;
    setTimeout(() => {
      loadingCard = false;
    }, 3000);
  }
</script>

<svelte:head>
  <title>Modern Components Demo - YoRHa Legal AI</title>
</svelte:head>

<div class="modern-demo">
  <header class="demo-header">
    <ModernCard title="Modern SvelteKit 2 Demo" subtitle="Golden Ratio Design System with Bits-UI + Melt-UI">
      {#snippet children()}
        <p class="demo-description">
          This page demonstrates the modern component library built with Svelte 5,
          golden ratio spacing, CSS Grid layouts, and modular architecture.
        </p>
      {/snippet}
    </ModernCard>
  </header>

  <!-- Button Variants Demo -->
  <section class="demo-section">
    <h2 class="section-title">Button Components</h2>
    <div class="button-grid">
      <ModernButton variant="primary" tooltip="Primary action button">
        {#snippet icon()}<Shield class="w-4 h-4" />{/snippet}
        {#snippet children()}Primary Action{/snippet}
      </ModernButton>

      <ModernButton variant="secondary" tooltip="Secondary action button">
        {#snippet icon()}<Settings class="w-4 h-4" />{/snippet}
        {#snippet children()}Secondary{/snippet}
      </ModernButton>

      <ModernButton variant="ghost" tooltip="Ghost style button">
        {#snippet icon()}<User class="w-4 h-4" />{/snippet}
        {#snippet children()}Ghost{/snippet}
      </ModernButton>

      <ModernButton variant="outline" tooltip="Outlined button">
        {#snippet icon()}<FileText class="w-4 h-4" />{/snippet}
        {#snippet children()}Outline{/snippet}
      </ModernButton>

      <ModernButton
        variant="danger"
        size="lg"
        on:on:click={() => handleLoadingDemo()}
        tooltip="Danger action with loading demo"
      >
        {#snippet children()}Danger Large{/snippet}
      </ModernButton>

      <ModernButton
        variant="success"
        size="xl"
        tooltip="Success extra large button"
      >
        {#snippet icon()}<Brain class="w-4 h-4" />{/snippet}
        {#snippet children()}Success XL{/snippet}
      </ModernButton>
    </div>
  </section>

  <!-- Card Variants Demo -->
  <section class="demo-section">
    <h2 class="section-title">Card Components</h2>
    <div class="cards-grid">
      <ModernCard
        title="Default Card"
        subtitle="Standard card with actions"
        variant="default"
        hoverable
        clickable
        on:on:click={handleCardClick}
        tooltip="This is a clickable card with hover effects"
      >
        {#snippet children()}
          <p>This is a default card with golden ratio spacing and modern styling.</p>
        {/snippet}
        {#snippet actions()}
          <ModernButton variant="ghost" size="sm">
            {#snippet icon()}<Bell class="w-4 h-4" />{/snippet}
          </ModernButton>
          <span class="notification-badge">{notifications}</span>
        {/snippet}
      </ModernCard>

      <ModernCard
        title="Elevated Card"
        subtitle="Enhanced shadow and depth"
        variant="elevated"
        size="lg"
        loading={loadingCard}
      >
        {#snippet children()}
          <p>This elevated card demonstrates loading states and larger sizing.</p>
          <div class="feature-list">
            <div class="feature-item">
              <Search class="w-5 h-5 text-yorha-accent-gold" />
              <span>Advanced Search</span>
            </div>
            <div class="feature-item">
              <Database class="w-5 h-5 text-yorha-accent-blue" />
              <span>Vector Database</span>
            </div>
            <div class="feature-item">
              <Brain class="w-5 h-5 text-yorha-accent-green" />
              <span>AI Processing</span>
            </div>
          </div>
        {/snippet}
      </ModernCard>

      <ModernCard
        variant="outline"
        size="xl"
      >
        {#snippet header()}
          <div class="custom-header">
            <Shield class="w-6 h-6 text-yorha-accent-gold" />
            <div>
              <h3 class="custom-title">Custom Header</h3>
              <p class="custom-subtitle">With custom layout and icons</p>
            </div>
          </div>
        {/snippet}
        {#snippet children()}
          <p>This card uses a custom header snippet with flexible layout options.</p>
        {/snippet}
        {#snippet footer()}
          <div class="card-footer-actions">
            <ModernButton variant="outline" size="sm">Cancel</ModernButton>
            <ModernButton variant="primary" size="sm">Continue</ModernButton>
          </div>
        {/snippet}
      </ModernCard>

      <ModernCard
        variant="ghost"
        hoverable
      >
        {#snippet children()}
          <div class="ghost-content">
            <h3>Ghost Card</h3>
            <p>Transparent background that appears on hover</p>
            <ModernButton
              variant="primary"
              on:on:click={() => dialogOpen = true}
            >
              Open Dialog
            </ModernButton>
          </div>
        {/snippet}
      </ModernCard>
    </div>
  </section>

  <!-- Dialog Demo -->
  <ModernDialog
    bind:open={dialogOpen}
    title="Modern Dialog Example"
    description="This dialog demonstrates the modern component with proper focus management and accessibility."
    size="lg"
  >
    {#snippet children()}
      <div class="dialog-content">
        <p>This is a modern dialog built with Melt-UI and Bits-UI primitives.</p>
        <div class="dialog-features">
          <div class="feature">✅ Proper focus management</div>
          <div class="feature">✅ Keyboard navigation</div>
          <div class="feature">✅ Screen reader support</div>
          <div class="feature">✅ Golden ratio spacing</div>
        </div>
      </div>
    {/snippet}
    {#snippet footer()}
      <div class="dialog-footer">
        <ModernButton variant="ghost" on:on:click={() => dialogOpen = false}>
          Cancel
        </ModernButton>
        <ModernButton variant="primary" on:on:click={() => dialogOpen = false}>
          Confirm
        </ModernButton>
      </div>
    {/snippet}
  </ModernDialog>

  <!-- Layout Demo -->
  <section class="demo-section">
    <h2 class="section-title">Layout System</h2>
    <ModernCard title="Golden Ratio Grid System" subtitle="CSS Grid with φ = 1.618 spacing">
      {#snippet children()}
        <div class="layout-demo">
          <div class="grid-item">Item 1</div>
          <div class="grid-item">Item 2</div>
          <div class="grid-item">Item 3</div>
          <div class="grid-item">Item 4</div>
        </div>
      {/snippet}
    </ModernCard>
  </section>
</div>

<style>
  .modern-demo {
    max-width: 1200px;
    margin: 0 auto;
    padding: var(--golden-xl);
    display: grid;
    gap: var(--golden-xl);
  }

  .demo-header {
    text-align: center;
    margin-bottom: var(--golden-lg);
  }

  .demo-description {
    color: var(--yorha-text-secondary);
    line-height: 1.6;
    font-size: var(--text-lg);
    max-width: 600px;
    margin: 0 auto;
  }

  .demo-section {
    display: grid;
    gap: var(--golden-lg);
  }

  .section-title {
    font-size: var(--text-2xl);
    font-weight: 600;
    color: var(--yorha-text-primary);
    text-transform: uppercase;
    letter-spacing: 0.025em;
    border-bottom: 2px solid var(--yorha-border-accent);
    padding-bottom: var(--golden-sm);
  }

  .button-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: var(--golden-md);
    align-items: start;
  }

  .cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: var(--golden-lg);
  }

  .feature-list {
    display: grid;
    gap: var(--golden-sm);
    margin-top: var(--golden-md);
  }

  .feature-item {
    display: flex;
    align-items: center;
    gap: var(--golden-sm);
    color: var(--yorha-text-secondary);
  }

  .notification-badge {
    background: var(--yorha-accent-gold);
    color: var(--yorha-bg-primary);
    border-radius: 50%;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
  }

  .custom-header {
    display: flex;
    align-items: center;
    gap: var(--golden-md);
  }

  .custom-title {
    font-size: var(--text-lg);
    font-weight: 600;
    color: var(--yorha-text-primary);
    margin: 0;
  }

  .custom-subtitle {
    font-size: var(--text-sm);
    color: var(--yorha-text-muted);
    margin: 0;
  }

  .card-footer-actions {
    display: flex;
    gap: var(--golden-sm);
    justify-content: flex-end;
  }

  .ghost-content {
    text-align: center;
    padding: var(--golden-lg);
  }

  .ghost-content h3 {
    font-size: var(--text-xl);
    color: var(--yorha-text-primary);
    margin-bottom: var(--golden-sm);
  }

  .ghost-content p {
    color: var(--yorha-text-secondary);
    margin-bottom: var(--golden-md);
  }

  .dialog-content {
    display: grid;
    gap: var(--golden-lg);
  }

  .dialog-features {
    display: grid;
    gap: var(--golden-sm);
  }

  .feature {
    padding: var(--golden-sm);
    background: var(--yorha-bg-secondary);
    border-radius: 0.375rem;
    font-size: var(--text-sm);
    color: var(--yorha-text-primary);
  }

  .dialog-footer {
    display: flex;
    gap: var(--golden-sm);
    justify-content: flex-end;
  }

  .layout-demo {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: var(--golden-md);
    margin-top: var(--golden-md);
  }

  .grid-item {
    background: var(--yorha-bg-tertiary);
    border: 1px solid var(--yorha-border-primary);
    border-radius: 0.5rem;
    padding: var(--golden-lg);
    text-align: center;
    color: var(--yorha-text-primary);
    font-weight: 500;
    aspect-ratio: 1.618; /* Golden ratio */
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 200ms ease;
  }

  .grid-item:hover {
    background: var(--yorha-bg-hover);
    border-color: var(--yorha-border-accent);
    transform: translateY(-2px);
    box-shadow: var(--yorha-shadow-md);
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .modern-demo {
      padding: var(--golden-lg);
    }

    .cards-grid {
      grid-template-columns: 1fr;
    }

    .button-grid {
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    }

    .layout-demo {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>
