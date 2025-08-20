<!-- YoRHa Interface Home Page -->
<script lang="ts">
  import { goto } from '$app/navigation';
  import { Button } from 'bits-ui';
  import { 
    Play, 
    Terminal, 
    Settings, 
    Monitor, 
    ChevronRight,
    Home,
    Gamepad2,
    Activity
  } from 'lucide-svelte';

  function navigateTo(path: string) {
    goto(path);
  }
</script>

<svelte:head>
  <title>YoRHa Interface - Legal AI Command Center</title>
  <meta name="description" content="YoRHa-themed interface for Legal AI system access and demonstrations." />
</svelte:head>

<div class="yorha-interface">
  <!-- Header -->
  <section class="yorha-header">
    <div class="yorha-header-content">
      <div class="yorha-nav">
        <Button.Root 
          class="yorha-nav-btn"
          onclick={() => navigateTo('/')}
        >
          <Home size={16} />
          MAIN SYSTEM
        </Button.Root>
      </div>
      
      <div class="yorha-title-section">
        <h1 class="yorha-main-title">
          <Terminal size={48} class="yorha-title-icon" />
          YoRHa INTERFACE
        </h1>
        <div class="yorha-subtitle">LEGAL AI COMMAND CENTER</div>
        <div class="yorha-status">
          <Activity size={16} />
          SYSTEM OPERATIONAL
        </div>
      </div>
    </div>
  </section>

  <!-- Main Interface Options -->
  <section class="yorha-main-section">
    <div class="yorha-interface-grid">
      <!-- Demo Center -->
      <div class="yorha-interface-card yorha-card-primary" onclick={() => navigateTo('/demos')}>
        <div class="yorha-card-header">
          <Gamepad2 size={32} class="yorha-card-icon" />
          <h2 class="yorha-card-title">DEMO CENTER</h2>
        </div>
        <p class="yorha-card-description">
          Access comprehensive interactive demonstrations of all Legal AI system capabilities
        </p>
        <div class="yorha-card-stats">
          <span>27+ INTERACTIVE DEMOS</span>
          <span>8 CATEGORIES</span>
        </div>
        <div class="yorha-card-footer">
          <span class="yorha-card-path">/demos</span>
          <ChevronRight size={20} class="yorha-card-arrow" />
        </div>
      </div>

      <!-- YoRHa Demo -->
      <div class="yorha-interface-card" onclick={() => navigateTo('/yorha-demo')}>
        <div class="yorha-card-header">
          <Play size={32} class="yorha-card-icon" />
          <h2 class="yorha-card-title">YoRHa THEME DEMO</h2>
        </div>
        <p class="yorha-card-description">
          Experience the complete YoRHa interface theming and component showcase
        </p>
        <div class="yorha-card-stats">
          <span>THEME SHOWCASE</span>
          <span>UI COMPONENTS</span>
        </div>
        <div class="yorha-card-footer">
          <span class="yorha-card-path">/yorha-demo</span>
          <ChevronRight size={20} class="yorha-card-arrow" />
        </div>
      </div>

      <!-- YoRHa Dashboard -->
      <div class="yorha-interface-card" onclick={() => navigateTo('/yorha-dashboard')}>
        <div class="yorha-card-header">
          <Monitor size={32} class="yorha-card-icon" />
          <h2 class="yorha-card-title">YoRHa DASHBOARD</h2>
        </div>
        <p class="yorha-card-description">
          Complete YoRHa-themed dashboard interface with system monitoring
        </p>
        <div class="yorha-card-stats">
          <span>SYSTEM MONITORING</span>
          <span>REAL-TIME DATA</span>
        </div>
        <div class="yorha-card-footer">
          <span class="yorha-card-path">/yorha-dashboard</span>
          <ChevronRight size={20} class="yorha-card-arrow" />
        </div>
      </div>

      <!-- YoRHa Terminal -->
      <div class="yorha-interface-card" onclick={() => navigateTo('/yorha-terminal')}>
        <div class="yorha-card-header">
          <Terminal size={32} class="yorha-card-icon" />
          <h2 class="yorha-card-title">YoRHa TERMINAL</h2>
        </div>
        <p class="yorha-card-description">
          Command line interface with YoRHa styling and AI command integration
        </p>
        <div class="yorha-card-stats">
          <span>CLI INTERFACE</span>
          <span>AI COMMANDS</span>
        </div>
        <div class="yorha-card-footer">
          <span class="yorha-card-path">/yorha-terminal</span>
          <ChevronRight size={20} class="yorha-card-arrow" />
        </div>
      </div>
    </div>
  </section>

  <!-- Footer Info -->
  <section class="yorha-footer">
    <div class="yorha-footer-content">
      <div class="yorha-system-info">
        <h3>SYSTEM INFORMATION</h3>
        <div class="yorha-info-grid">
          <div class="yorha-info-item">
            <strong>Frontend:</strong> SvelteKit 2 + Svelte 5
          </div>
          <div class="yorha-info-item">
            <strong>UI Framework:</strong> bits-ui + melt-ui + shadcn-svelte
          </div>
          <div class="yorha-info-item">
            <strong>Theme:</strong> YoRHa Cyberpunk Interface
          </div>
          <div class="yorha-info-item">
            <strong>Status:</strong> Production Ready
          </div>
        </div>
      </div>
    </div>
  </section>
</div>

<style>
  .yorha-interface {
    @apply min-h-screen bg-black text-amber-400 font-mono;
    font-family: 'Courier New', monospace;
    background-image: 
      radial-gradient(circle at 20% 50%, rgba(255, 191, 0, 0.03) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(255, 191, 0, 0.03) 0%, transparent 50%);
  }

  /* Header */
  .yorha-header {
    @apply py-16 px-6 border-b border-amber-400 border-opacity-30;
    background: linear-gradient(135deg, transparent 0%, rgba(255, 191, 0, 0.05) 100%);
  }

  .yorha-header-content {
    @apply max-w-6xl mx-auto;
  }

  .yorha-nav {
    @apply mb-8;
  }

  .yorha-nav-btn {
    @apply px-4 py-2 bg-amber-400 text-black font-mono text-sm tracking-wider;
    @apply hover:bg-amber-300 transition-colors flex items-center gap-2;
  }

  .yorha-title-section {
    @apply text-center space-y-6;
  }

  .yorha-main-title {
    @apply text-6xl md:text-8xl font-bold tracking-wider flex items-center justify-center gap-4;
    text-shadow: 0 0 20px rgba(255, 191, 0, 0.5);
  }

  .yorha-title-icon {
    @apply text-amber-400;
  }

  .yorha-subtitle {
    @apply text-2xl text-amber-300 tracking-wide opacity-80;
  }

  .yorha-status {
    @apply flex items-center justify-center gap-2 text-green-400 font-bold;
  }

  /* Main Section */
  .yorha-main-section {
    @apply py-16 px-6;
  }

  .yorha-interface-grid {
    @apply grid grid-cols-1 md:grid-cols-2 gap-8 max-w-6xl mx-auto;
  }

  .yorha-interface-card {
    @apply bg-gray-900 border border-amber-400 border-opacity-30 p-8 cursor-pointer;
    @apply hover:border-opacity-60 transition-all duration-300 hover:bg-amber-900 hover:bg-opacity-10;
    @apply space-y-6;
  }

  .yorha-card-primary {
    @apply border-amber-400 border-opacity-60;
    box-shadow: 0 0 20px rgba(255, 191, 0, 0.2);
  }

  .yorha-card-header {
    @apply flex items-center gap-4;
  }

  .yorha-card-icon {
    @apply text-amber-400;
  }

  .yorha-card-title {
    @apply text-xl font-bold text-amber-400 tracking-wider;
  }

  .yorha-card-description {
    @apply text-amber-300 leading-relaxed;
  }

  .yorha-card-stats {
    @apply flex gap-4 text-xs text-amber-400 opacity-60;
  }

  .yorha-card-footer {
    @apply flex items-center justify-between pt-4 border-t border-amber-400 border-opacity-20;
  }

  .yorha-card-path {
    @apply text-xs text-amber-400 opacity-60 font-mono;
  }

  .yorha-card-arrow {
    @apply text-amber-300 opacity-60;
  }

  /* Footer */
  .yorha-footer {
    @apply border-t border-amber-400 border-opacity-30 bg-gray-900 bg-opacity-50 px-6 py-12;
  }

  .yorha-footer-content {
    @apply max-w-6xl mx-auto;
  }

  .yorha-system-info h3 {
    @apply text-xl font-bold text-amber-400 mb-6 tracking-wider;
  }

  .yorha-info-grid {
    @apply grid grid-cols-1 md:grid-cols-2 gap-4;
  }

  .yorha-info-item {
    @apply text-amber-300 border-l-2 border-amber-400 border-opacity-30 pl-4;
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .yorha-main-title {
      @apply text-4xl;
    }
    
    .yorha-interface-grid {
      @apply grid-cols-1 gap-6;
    }
    
    .yorha-info-grid {
      @apply grid-cols-1 gap-3;
    }
  }

  /* Hover animations */
  .yorha-interface-card:hover .yorha-card-arrow {
    @apply text-amber-400 opacity-100;
    transform: translateX(4px);
  }

  .yorha-interface-card:hover .yorha-card-icon {
    text-shadow: 0 0 10px rgba(255, 191, 0, 0.5);
  }
</style>