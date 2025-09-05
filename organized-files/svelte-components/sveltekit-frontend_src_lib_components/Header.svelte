<script lang="ts">
  interface Props {
    user: User | null ;
  }
  let {
    user = null
  } = $props();



  import { goto } from "$app/navigation";
  import { page } from "$app/stores";
  import SearchInput from "./SearchInput.svelte";
  import type { User } from "$lib/types/user";

  import {
    FolderOpen,
    Home,
    LogOut,
    MoreVertical,
    Palette,
    Settings,
    Shield,
    User as UserIcon,
  } from "lucide-svelte";

  
  let searchQuery = "";
  let userMenuOpen = false;

  function handleSearch(event: CustomEvent) {
    searchQuery = event.detail.query;
    // Handle global search
    console.log("Global search:", searchQuery);
  }

  function handleLogout() {
    goto("/logout");
  }

  function handleNavigation(path: string) {
    goto(path);
    userMenuOpen = false;
  }

  function toggleUserMenu() {
    userMenuOpen = !userMenuOpen;
  }

  function closeUserMenu() {
    userMenuOpen = false;
  }

  // Progressive enhancement: Close menu on Escape key
  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === 'Escape' && userMenuOpen) {
      closeUserMenu();
    }
  }

  // Check if current route is active
  function isActiveRoute(path: string): boolean {
    return $page.url.pathname === path || $page.url.pathname.startsWith(path + '/');
  }
</script>

<svelte:window onkeydown={handleKeyDown} />

<header class="app-header">
  <div class="header-content">
    <!-- Logo and Brand -->
    <div class="brand-section">
      <button
        class="brand-button"
        onclick={() => handleNavigation("/")}
        aria-label="Go to homepage"
      >
        <Palette size={24} aria-hidden="true" />
        <span class="brand-text">Prosecutor Canvas</span>
      </button>
    </div>

    <!-- Navigation -->
    <nav class="main-nav" aria-label="Main navigation">
      <button
        class="nav-button"
        class:active={isActiveRoute('/dashboard')}
        onclick={() => handleNavigation("/dashboard")}
        aria-label="Dashboard"
        aria-current={isActiveRoute('/dashboard') ? 'page' : undefined}
      >
        <Home size={18} aria-hidden="true" />
        <span>Dashboard</span>
      </button>

      <button
        class="nav-button"
        class:active={isActiveRoute('/cases')}
        onclick={() => handleNavigation("/cases")}
        aria-label="Cases"
        aria-current={isActiveRoute('/cases') ? 'page' : undefined}
      >
        <FolderOpen size={18} aria-hidden="true" />
        <span>Cases</span>
      </button>

      <button
        class="nav-button"
        class:active={isActiveRoute('/interactive-canvas')}
        onclick={() => handleNavigation("/interactive-canvas")}
        aria-label="Interactive Canvas"
        aria-current={isActiveRoute('/interactive-canvas') ? 'page' : undefined}
      >
        <Palette size={18} aria-hidden="true" />
        <span>Canvas</span>
      </button>

      <button
        class="nav-button"
        class:active={isActiveRoute('/evidence/hash')}
        onclick={() => handleNavigation("/evidence/hash")}
        aria-label="Hash Verification"
        aria-current={isActiveRoute('/evidence/hash') ? 'page' : undefined}
        title="Verify evidence file integrity"
      >
        <Shield size={18} aria-hidden="true" />
        <span>Hash Verify</span>
      </button>
    </nav>

    <!-- Search -->
    <div class="search-section">
      <SearchInput
        placeholder="Search cases, evidence, notes..."
        value={searchQuery}
        onsearch={handleSearch}
      />
    </div>

    <!-- User Menu -->
    <div class="user-section">
      {#if user}
        <div class="user-menu-container">
          <button
            class="user-button"
            onclick={toggleUserMenu}
            aria-label="User menu"
            aria-expanded={userMenuOpen}
            aria-haspopup="menu"
          >
            <div class="user-avatar">
              {#if user.avatarUrl}
                <img src={user.avatarUrl} alt="" />
              {:else}
                <span class="avatar-fallback">
                  {user.name?.charAt(0)?.toUpperCase() || "U"}
                </span>
              {/if}
            </div>
            <span class="user-name">{user.name}</span>
            <MoreVertical size={16} aria-hidden="true" />
          </button>

          {#if userMenuOpen}
            <div class="user-menu" role="menu" aria-labelledby="user-button">
              <button
                class="menu-item"
                onclick={() => handleNavigation("/profile")}
                role="menuitem"
                tabindex="0"
              >
                <UserIcon size={16} aria-hidden="true" />
                Profile
              </button>

              <button
                class="menu-item"
                onclick={() => handleNavigation("/settings")}
                role="menuitem"
                tabindex="0"
              >
                <Settings size={16} aria-hidden="true" />
                Settings
              </button>

              <hr class="menu-separator" />

              <button
                class="menu-item"
                onclick={handleLogout}
                role="menuitem"
                tabindex="0"
              >
                <LogOut size={16} aria-hidden="true" />
                Sign Out
              </button>
            </div>
          {/if}
        </div>
      {:else}
        <button
          class="sign-in-button"
          onclick={() => handleNavigation("/login")}
          aria-label="Sign in"
        >
          Sign In
        </button>
      {/if}
    </div>
  </div>
</header>

<!-- Click outside to close menu -->
{#if userMenuOpen}
  <div
    class="menu-overlay"
    onclick={closeUserMenu}
    onkeydown={(e) => e.key === "Escape" && closeUserMenu()}
    role="button"
    tabindex="-1"
    aria-label="Close user menu"
  ></div>
{/if}

<style>
  /* @unocss-include */
  .app-header {
    @apply fixed top-0 left-0 right-0 h-15 bg-card border-b border-border z-30;
    backdrop-filter: blur(8px);
  }

  .header-content {
    @apply flex items-center h-full px-4 max-w-7xl mx-auto gap-4;
  }

  .brand-section {
    @apply flex items-center flex-shrink-0;
  }

  .brand-button {
    @apply flex items-center gap-3 px-4 py-2 font-semibold text-blue-600 bg-transparent border-none cursor-pointer rounded-md transition-colors duration-200 hover:bg-accent focus:outline-none focus:ring-2 focus:ring-blue-600/50;
  }

  .brand-text {
    @apply text-lg font-bold;
  }

  .main-nav {
    @apply flex items-center gap-1 flex-shrink-0;
  }

  .nav-button {
    @apply flex items-center gap-2 px-4 py-2 text-muted-foreground bg-transparent border-none cursor-pointer rounded-md transition-all duration-200 hover:text-foreground hover:bg-accent focus:outline-none focus:ring-2 focus:ring-blue-600/50;
  }

  .nav-button.active {
    @apply text-blue-600 bg-accent;
  }

  .search-section {
    @apply flex-1 max-w-md mx-8;
  }

  .user-section {
    @apply flex items-center flex-shrink-0;
  }

  .user-menu-container {
    @apply relative;
  }

  .user-button {
    @apply flex items-center gap-3 px-4 py-2 bg-transparent border-none cursor-pointer rounded-md transition-colors duration-200 text-foreground hover:bg-accent focus:outline-none focus:ring-2 focus:ring-blue-600/50;
  }

  .user-avatar {
    @apply w-8 h-8 rounded-full overflow-hidden flex items-center justify-center bg-secondary text-blue-600;
  }

  .user-avatar img {
    @apply w-full h-full object-cover;
  }

  .avatar-fallback {
    @apply font-semibold text-sm;
  }

  .user-name {
    @apply font-medium text-foreground;
  }

  .user-menu {
    @apply absolute top-full right-0 min-w-45 bg-card border border-border rounded-lg shadow-lg p-2 z-1000 mt-2;
  }

  .menu-item {
    @apply flex items-center gap-3 p-2 w-full bg-transparent border-none cursor-pointer rounded text-foreground text-left transition-colors duration-200 hover:bg-accent focus:outline-none focus:ring-2 focus:ring-blue-600/50;
  }

  .menu-separator {
    @apply border-none border-t border-border my-2;
  }

  .sign-in-button {
    @apply px-4 py-2 bg-transparent border border-blue-600 text-blue-600 rounded-md cursor-pointer transition-all duration-200 hover:bg-blue-600 hover:text-white focus:outline-none focus:ring-2 focus:ring-blue-600/50;
  }

  .menu-overlay {
    @apply fixed top-0 left-0 right-0 bottom-0 z-999 bg-transparent;
  }

  /* Responsive */
  @media (max-width: 768px) {
    .header-content {
      @apply px-2 gap-2;
    }

    .brand-text {
      @apply hidden;
    }

    .search-section {
      @apply mx-4;
    }

    .nav-button span {
      @apply hidden;
    }

    .user-name {
      @apply hidden;
    }
  }

  @media (max-width: 480px) {
    .main-nav {
      @apply gap-0;
    }

    .search-section {
      @apply max-w-50 mx-2;
    }
  }
</style>