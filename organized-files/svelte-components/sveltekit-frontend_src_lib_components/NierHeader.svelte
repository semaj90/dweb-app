import type { User } from '$lib/types';


import type { User } from '$lib/types/user';


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

  import { FolderOpen, Home, LogOut, MoreVertical, Palette, Settings, Shield, User as UserIcon } from "lucide-svelte";






import type { User } from '$lib/types/user';


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
</script>

<header class="space-y-4">
  <div class="space-y-4">
    <!-- Logo and Brand -->
    <div class="space-y-4">
      <button
        class="space-y-4"
        onclick={() => handleNavigation("/")}
        aria-label="Go to homepage"
      >
        <Palette size={24} />
        <span class="space-y-4">Prosecutor Canvas</span>
      </button>
    </div>

    <!-- Navigation -->
    <nav class="space-y-4" aria-label="Main navigation">
      <button
        class="space-y-4"
        onclick={() => handleNavigation("/dashboard")}
        aria-label="Dashboard"
      >
        <Home size={18} />
        <span>Dashboard</span>
      </button>

      <button
        class="space-y-4"
        onclick={() => handleNavigation("/cases")}
        aria-label="Cases"
      >
        <FolderOpen size={18} />
        <span>Cases</span>
      </button>

      <button
        class="space-y-4"
        onclick={() => handleNavigation("/interactive-canvas")}
        aria-label="Interactive Canvas"
      >
        <Palette size={18} />
        <span>Canvas</span>
      </button>

      <button
        class="space-y-4"
        onclick={() => handleNavigation("/evidence/hash")}
        aria-label="Hash Verification"
        title="Verify evidence file integrity"
      >
        <Shield size={18} />
        <span>Hash Verify</span>
      </button>
    </nav>

    <!-- Search -->
    <div class="space-y-4">
      <SearchInput
        placeholder="Search cases, evidence, notes..."
        value={searchQuery}
        onsearch={handleSearch}
      />
    </div>

    <!-- User Menu -->
    <div class="space-y-4">
      {#if user}
        <div class="space-y-4">
          <button
            class="space-y-4"
            onclick={() => toggleUserMenu()}
            aria-label="User menu"
            aria-expanded={userMenuOpen}
          >
            <div class="space-y-4">
              {#if user.avatarUrl}
                <img src={user.avatarUrl} alt={user.name} />
              {:else}
                <span class="space-y-4">
                  {user.name?.charAt(0)?.toUpperCase() || "U"}
                </span>
              {/if}
            </div>
            <span class="space-y-4">{user.name}</span>
            <MoreVertical size={16} />
          </button>

          {#if userMenuOpen}
            <div class="space-y-4" role="menu">
              <button
                class="space-y-4"
                onclick={() => handleNavigation("/profile")}
                role="menuitem"
              >
                <UserIcon size={16} />
                Profile
              </button>

              <button
                class="space-y-4"
                onclick={() => handleNavigation("/settings")}
                role="menuitem"
              >
                <Settings size={16} />
                Settings
              </button>

              <hr class="space-y-4" />

              <button
                class="space-y-4"
                onclick={() => handleLogout()}
                role="menuitem"
              >
                <LogOut size={16} />
                Sign Out
              </button>
            </div>
          {/if}
        </div>
      {:else}
        <button
          class="space-y-4"
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
    class="space-y-4"
    onclick={() => closeUserMenu()}
    onkeydown={(e) => e.key === "Escape" && closeUserMenu()}
    role="button"
    tabindex={-1}
    aria-label="Close user menu"
  ></div>
{/if}

<style>
  /* @unocss-include */
  .app-header {
    position: fixed
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-light);
    z-index: 30;
    backdrop-filter: blur(8px);
}
  .header-content {
    display: flex
    align-items: center
    height: 100%;
    padding: 0 1rem;
    max-width: 1400px;
    margin: 0 auto;
    gap: 1rem;
}
  .brand-section {
    display: flex
    align-items: center
    flex-shrink: 0;
}
  .brand-button {
    display: flex
    align-items: center
    gap: 0.75rem;
    padding: 0.5rem 1rem;
    font-weight: 600;
    color: var(--harvard-crimson);
    background: transparent
    border: none
    cursor: pointer
    border-radius: 6px;
    transition: background 0.2s ease;
}
  .brand-button:hover {
    background: var(--bg-tertiary);
}
  .brand-text {
    font-size: 1.1rem;
    font-weight: 700;
}
  .main-nav {
    display: flex
    align-items: center
    gap: 0.25rem;
    flex-shrink: 0;
}
  .nav-button {
    display: flex
    align-items: center
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    color: var(--text-muted);
    background: transparent
    border: none
    cursor: pointer
    border-radius: 6px;
    transition: all 0.2s ease;
}
  .nav-button:hover {
    color: var(--text-primary);
    background: var(--bg-tertiary);
}
  .nav-button.active {
    color: var(--harvard-crimson);
    background: var(--bg-secondary);
}
  .search-section {
    flex: 1;
    max-width: 400px;
    margin: 0 2rem;
}
  .user-section {
    display: flex
    align-items: center
    flex-shrink: 0;
}
  .user-menu-container {
    position: relative
}
  .user-button {
    display: flex
    align-items: center
    gap: 0.75rem;
    padding: 0.5rem 1rem;
    background: transparent
    border: none
    cursor: pointer
    border-radius: 6px;
    transition: background 0.2s ease;
    color: var(--text-primary);
}
  .user-button:hover {
    background: var(--bg-tertiary);
}
  .user-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    overflow: hidden
    display: flex
    align-items: center
    justify-content: center
    background: var(--bg-secondary);
    color: var(--harvard-crimson);
}
  .user-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover
}
  .avatar-fallback {
    font-weight: 600;
    font-size: 0.875rem;
}
  .user-name {
    font-weight: 500;
    color: var(--text-primary);
}
  .user-menu {
    position: absolute
    top: 100%;
    right: 0;
    min-width: 180px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    padding: 0.5rem;
    z-index: 1000;
    margin-top: 0.5rem;
}
  .menu-item {
    display: flex
    align-items: center
    gap: 0.75rem;
    padding: 0.5rem;
    width: 100%;
    background: transparent
    border: none
    cursor: pointer
    border-radius: 4px;
    transition: background 0.2s ease;
    color: var(--text-primary);
    text-align: left
}
  .menu-item:hover {
    background: var(--bg-tertiary);
}
  .menu-separator {
    border: none
    border-top: 1px solid var(--border-light);
    margin: 0.5rem 0;
}
  .sign-in-button {
    padding: 0.5rem 1rem;
    background: transparent
    border: 1px solid var(--harvard-crimson);
    color: var(--harvard-crimson);
    border-radius: 6px;
    cursor: pointer
    transition: all 0.2s ease;
}
  .sign-in-button:hover {
    background: var(--harvard-crimson);
    color: var(--text-inverse);
}
  .menu-overlay {
    position: fixed
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 999;
    background: transparent
}
  /* Responsive */
  @media (max-width: 768px) {
    .header-content {
      padding: 0 0.5rem;
      gap: 0.5rem;
}
    .brand-text {
      display: none
}
    .search-section {
      margin: 0 1rem;
}
    .nav-button span {
      display: none
}
    .user-name {
      display: none
}}
  @media (max-width: 480px) {
    .main-nav {
      gap: 0;
}
    .search-section {
      max-width: 200px;
      margin: 0 0.5rem;
}}
</style>
