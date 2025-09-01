<script lang="ts">
  import type {     Snippet     } from 'svelte';
  import { useIsMobile, useIsTablet, useIsDesktop, useIsDark } from '$lib/utils/media-query.svelte';
  
  // Props using Svelte 5 children pattern
  interface Props {
    title?: string;
    children?: Snippet;
    header?: Snippet<[{ isMobile: boolean }]>; // Snippet with parameters
    sidebar?: Snippet;
    footer?: Snippet;
  }
  
  let { 
    title = 'Svelte 5 Example',
    children, 
    header, 
    sidebar, 
    footer 
  }: Props = $props();
  
  // Reactive media queries using Svelte 5 runes
  const mobile = useIsMobile();
  const tablet = useIsTablet();
  const desktop = useIsDesktop();
  const darkMode = useIsDark();
  
  // Derived responsive layout
  let layoutClass = $derived(() => {
    if (mobile.matches) return 'layout-mobile';
    if (tablet.matches) return 'layout-tablet';
    if (desktop.matches) return 'layout-desktop';
    return 'layout-default';
  });
  
  let themeClass = $derived(darkMode.matches ? 'theme-dark' : 'theme-light');
  
  // Reactive breakpoint info
  let breakpointInfo = $derived(() => ({
    isMobile: mobile.matches,
    isTablet: tablet.matches,
    isDesktop: desktop.matches,
    isDark: darkMode.matches,
    currentBreakpoint: mobile.matches ? 'mobile' : tablet.matches ? 'tablet' : 'desktop'
  }));
</script>

<!-- Responsive layout using media query reactivity -->
<div class="responsive-container {layoutClass} {themeClass}">
  
  <!-- Header with snippet parameters -->
  {#if header}
    <header class="header">
      {@render header({ isMobile: mobile.matches })}
    </header>
  {:else}
    <header class="header">
      <h1>{title}</h1>
      <div class="breakpoint-indicator">
        Current: {breakpointInfo.currentBreakpoint}
        {#if darkMode.matches}🌙{:else}☀️{/if}
      </div>
    </header>
  {/if}
  
  <!-- Main content area -->
  <main class="main-content">
    
    <!-- Sidebar (hidden on mobile) -->
    {#if sidebar && !mobile.matches}
      <aside class="sidebar">
        {@render sidebar()}
      </aside>
    {/if}
    
    <!-- Main content -->
    <div class="content">
      {#if children}
        {@render children()}
      {:else}
        <div class="example-content">
          <h2>Responsive Behavior</h2>
          <div class="info-grid">
            <div class="info-card" class:active={mobile.matches}>
              <strong>Mobile</strong>
              <span>{mobile.matches ? '✅' : '❌'}</span>
            </div>
            <div class="info-card" class:active={tablet.matches}>
              <strong>Tablet</strong>
              <span>{tablet.matches ? '✅' : '❌'}</span>
            </div>
            <div class="info-card" class:active={desktop.matches}>
              <strong>Desktop</strong>
              <span>{desktop.matches ? '✅' : '❌'}</span>
            </div>
            <div class="info-card" class:active={darkMode.matches}>
              <strong>Dark Mode</strong>
              <span>{darkMode.matches ? '🌙' : '☀️'}</span>
            </div>
          </div>
          
          <p>Resize your window or change your theme to see reactive updates!</p>
        </div>
      {/if}
    </div>
  </main>
  
  <!-- Footer -->
  {#if footer}
    <footer class="footer">
      {@render footer()}
    </footer>
  {/if}
</div>

<style>
  .responsive-container {
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr auto;
    transition: all 0.3s ease;
  }
  
  .header {
    padding: 1rem;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .breakpoint-indicator {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.875rem;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    background: var(--accent-bg);
    color: var(--accent-text);
  }
  
  .main-content {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1rem;
    padding: 1rem;
  }
  
  .sidebar {
    background: var(--sidebar-bg);
    padding: 1rem;
    border-radius: 0.5rem;
  }
  
  .content {
    background: var(--content-bg);
    padding: 1.5rem;
    border-radius: 0.5rem;
  }
  
  .info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
  }
  
  .info-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1rem;
    border: 2px solid var(--border-color);
    border-radius: 0.5rem;
    transition: all 0.3s ease;
  }
  
  .info-card.active {
    border-color: var(--accent-color);
    background: var(--accent-bg);
    transform: scale(1.05);
  }
  
  .footer {
    padding: 1rem;
    border-top: 1px solid var(--border-color);
    text-align: center;
  }
  
  /* Responsive layouts */
  .layout-mobile .main-content {
    grid-template-columns: 1fr;
    padding: 0.5rem;
  }
  
  .layout-tablet .main-content {
    grid-template-columns: 200px 1fr;
  }
  
  .layout-desktop .main-content {
    grid-template-columns: 250px 1fr;
  }
  
  /* Theme variations */
  .theme-light {
    --border-color: #e5e5e5;
    --accent-color: #3b82f6;
    --accent-bg: #dbeafe;
    --accent-text: #1e40af;
    --sidebar-bg: #f8fafc;
    --content-bg: #ffffff;
  }
  
  .theme-dark {
    --border-color: #374151;
    --accent-color: #60a5fa;
    --accent-bg: #1e3a8a;
    --accent-text: #dbeafe;
    --sidebar-bg: #1f2937;
    --content-bg: #111827;
  }
</style>