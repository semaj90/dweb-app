<!-- YoRHa Interface Layout -->
<script lang="ts">
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import type { LayoutData } from './$types';
  
  export let data: LayoutData;
  
  $: user = data.user;
  
  // YoRHa color scheme
  const colors = {
    bg: '#D4D3A7',
    text: '#454138',
    accent: '#BAA68C',
    border: '#8B8680',
    highlight: '#CDC8B0',
    error: '#8B4513',
    success: '#6B7353'
  };
  
  // Handle logout
  async function handleLogout() {
    const response = await fetch('/api/auth/logout', {
      method: 'POST'
    });
    
    if (response.ok) {
      goto('/');
    }
  }
  
  // Navigation items
  $: navigationItems = user ? [
    { href: '/dashboard', label: 'COMMAND CENTER', icon: '⚡' },
    { href: '/profile', label: 'UNIT PROFILE', icon: '👤' },
    { href: '/missions', label: 'MISSIONS', icon: '🎯' },
    { href: '/equipment', label: 'EQUIPMENT', icon: '🛡️' },
    { href: '/achievements', label: 'ACHIEVEMENTS', icon: '🏆' },
    { href: '/settings', label: 'SETTINGS', icon: '⚙️' }
  ] : [
    { href: '/login', label: 'ACCESS TERMINAL', icon: '🔐' },
    { href: '/register', label: 'UNIT REGISTRATION', icon: '📝' },
    { href: '/about', label: 'ABOUT YORHA', icon: 'ℹ️' }
  ];
  
  // Check if current route is active
  function isActive(href: string): boolean {
    return $page.url.pathname === href || $page.url.pathname.startsWith(href + '/');
  }
</script>

<div class="min-h-screen" style="background-color: {colors.bg}">
  <!-- Scan lines overlay -->
  <div class="scan-lines"></div>
  
  <!-- Header -->
  <header class="yorha-header" style="border-color: {colors.border}">
    <div class="container">
      <div class="header-content">
        <a href="/" class="logo">
          <h1 style="color: {colors.text}">YoRHa</h1>
          <span class="version" style="color: {colors.text}">v13.2.7</span>
        </a>
        
        <nav class="main-nav">
          {#each navigationItems as item}
            <a 
              href={item.href}
              class="nav-item"
              class:active={isActive(item.href)}
              style="color: {colors.text}"
            >
              <span class="icon">{item.icon}</span>
              <span class="label">{item.label}</span>
            </a>
          {/each}
        </nav>
        
        {#if user}
          <div class="user-menu">
            <div class="user-info" style="color: {colors.text}">
              <div class="name">{user.name}</div>
              <div class="stats">LVL {user.level} • {user.xp} XP</div>
            </div>
            <button 
              on:click={handleLogout}
              class="logout-btn"
              style="border-color: {colors.border}"
            >
              <span style="color: {colors.text}">⏻</span>
            </button>
          </div>
        {:else}
          <div class="auth-buttons">
            <a href="/login" class="btn-secondary" style="border-color: {colors.border}; color: {colors.text}">
              LOGIN
            </a>
            <a href="/register" class="btn-primary" style="background-color: {colors.text}; color: {colors.bg}">
              REGISTER
            </a>
          </div>
        {/if}
      </div>
    </div>
  </header>
  
  <!-- Main Content -->
  <main class="main-content">
    <slot />
  </main>
  
  <!-- Footer -->
  <footer class="yorha-footer" style="border-color: {colors.border}">
    <div class="container">
      <div class="footer-content">
        <div class="copyright" style="color: {colors.text}">
          © YoRHa • FOR THE GLORY OF MANKIND
        </div>
        <div class="status">
          <span class="status-indicator"></span>
          <span style="color: {colors.text}">SYSTEM OPERATIONAL</span>
        </div>
      </div>
    </div>
  </footer>
</div>

<style>
  :global(body) {
    margin: 0;
    font-family: 'Courier New', monospace;
  }
  
  .scan-lines {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    pointer-events: none;
    opacity: 0.05;
    background-image: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0, 0, 0, 0.3) 2px,
      rgba(0, 0, 0, 0.3) 4px
    );
    z-index: 1000;
  }
  
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
  }
  
  .yorha-header {
    border-bottom: 2px solid;
    padding: 1rem 0;
    position: relative;
    z-index: 100;
  }
  
  .header-content {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
  }
  
  .logo {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    text-decoration: none;
  }
  
  .logo h1 {
    font-size: 2rem;
    font-weight: bold;
    margin: 0;
  }
  
  .version {
    font-size: 0.75rem;
    opacity: 0.7;
  }
  
  .main-nav {
    display: flex;
    gap: 1rem;
    flex: 1;
    justify-content: center;
  }
  
  .nav-item {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.5rem 1rem;
    text-decoration: none;
    border: 1px solid transparent;
    transition: all 0.2s;
    font-size: 0.875rem;
  }
  
  .nav-item:hover {
    border-color: currentColor;
    transform: translate(1px, 1px);
  }
  
  .nav-item.active {
    background-color: #454138;
    color: #D4D3A7 !important;
  }
  
  .icon {
    font-size: 1rem;
  }
  
  .label {
    font-family: monospace;
  }
  
  .user-menu {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  
  .user-info {
    text-align: right;
  }
  
  .name {
    font-weight: bold;
    font-size: 0.875rem;
  }
  
  .stats {
    font-size: 0.75rem;
    opacity: 0.7;
  }
  
  .logout-btn {
    padding: 0.5rem;
    border: 1px solid;
    background: transparent;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .logout-btn:hover {
    background-color: rgba(0, 0, 0, 0.1);
  }
  
  .auth-buttons {
    display: flex;
    gap: 0.5rem;
  }
  
  .btn-secondary,
  .btn-primary {
    padding: 0.5rem 1rem;
    text-decoration: none;
    font-size: 0.875rem;
    transition: all 0.2s;
    font-family: monospace;
  }
  
  .btn-secondary {
    border: 1px solid;
  }
  
  .btn-secondary:hover,
  .btn-primary:hover {
    transform: translate(1px, 1px);
  }
  
  .main-content {
    min-height: calc(100vh - 200px);
    position: relative;
    z-index: 10;
  }
  
  .yorha-footer {
    border-top: 2px solid;
    padding: 1.5rem 0;
    margin-top: 3rem;
    position: relative;
    z-index: 100;
  }
  
  .footer-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .copyright {
    font-size: 0.75rem;
    font-family: monospace;
  }
  
  .status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
    font-family: monospace;
  }
  
  .status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #6B7353;
    animation: pulse 2s infinite;
  }
  
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }
  
  @media (max-width: 768px) {
    .header-content {
      flex-direction: column;
      gap: 1rem;
    }
    
    .main-nav {
      order: 3;
      width: 100%;
      justify-content: flex-start;
      overflow-x: auto;
    }
    
    .footer-content {
      flex-direction: column;
      gap: 1rem;
    }
  }
</style>