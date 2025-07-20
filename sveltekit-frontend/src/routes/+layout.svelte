<script lang="ts">
  import type { LayoutData } from "./$types";
  export let data: LayoutData;

  import { goto } from "$app/navigation";
  import AuthForm from "$lib/components/auth/AuthForm.svelte";
  import Modal from "$lib/components/ui/Modal.svelte";
  import { setAuthContext } from "$lib/stores/auth";
  import { AccessibilityValidator } from "$lib/utils/accessibility";
  import { onMount } from "svelte";
  import "uno.css";
  import "../app.css";
  import ErrorBoundary from '$lib/components/ErrorBoundary.svelte';
  import Settings from '$lib/components/Settings.svelte';
  import KeyboardShortcuts from '$lib/components/keyboard/KeyboardShortcuts.svelte';
  import EnhancedNotificationContainer from '$lib/components/notifications/EnhancedNotificationContainer.svelte';
  import ModalManager from '$lib/components/ui/ModalManager.svelte';
  import { clearUser, user } from '$lib/stores/user';
  import { initializeTauri } from '$lib/tauri';

  // Create and set the auth context
  const auth = setAuthContext();

  let showSettings = false;
  let showKeyboardShortcuts = false;
  let showLoginModal = false;
  let showRegisterModal = false;
  let mobileMenuOpen = false;

  // Settings state
  let settings: {
    theme: string;
    language: string;
    ttsEngine: string;
    voiceLanguage: string;
    enableSuggestions: boolean;
    enableMasking: boolean;
    enableAutoSave: boolean;
    maxHistoryItems: number;
    enableNotifications: boolean;
    fontFamily: string;
    fontSize: string;
    enableAccessibilityFeatures: boolean;
    enableHighContrast: boolean;
    enableReducedMotion: boolean;
    enableSounds: boolean;
  } = {
    theme: "light",
    language: "en",
    ttsEngine: "",
    voiceLanguage: "",
    enableSuggestions: true,
    enableMasking: false,
    enableAutoSave: true,
    maxHistoryItems: 50,
    enableNotifications: true,
    fontFamily: "Inter",
    fontSize: "16px",
    enableAccessibilityFeatures: true,
    enableHighContrast: false,
    enableReducedMotion: false,
    enableSounds: true,
  };

  function handleSettingsClose() {
    showSettings = false;
}
  function handleSettingsSave(event: CustomEvent<any>) {
    settings = { ...event.detail };
    // Apply theme immediately
    document.documentElement.setAttribute("data-theme", settings.theme);

    // Apply accessibility settings
    if (settings.enableHighContrast) {
      document.documentElement.classList.add("high-contrast");
    } else {
      document.documentElement.classList.remove("high-contrast");
}
    if (settings.enableReducedMotion) {
      document.documentElement.classList.add("reduce-motion");
    } else {
      document.documentElement.classList.remove("reduce-motion");
}
    // Apply font settings
    document.documentElement.style.setProperty(
      "--app-font-family",
      settings.fontFamily
    );
    document.documentElement.style.setProperty(
      "--app-font-size",
      settings.fontSize
    );

    showSettings = false;

    // Save settings to localStorage
    localStorage.setItem("app-settings", JSON.stringify(settings));
}
  // Logout function
  async function logout() {
    try {
      const response = await fetch("/api/auth/logout", { method: "POST" });
      if (response.ok) {
        clearUser();
        goto("/login");
}
    } catch (error) {
      console.error("Logout error:", error);
      // Clear the store anyway
      clearUser();
      goto("/login");
}
}
  // Initialize Tauri and auth when component mounts
  onMount(() => {
    initializeTauri();
    // Check authentication status on app load
    auth.checkAuth();

    // Load saved settings
    const savedSettings = localStorage.getItem("app-settings");
    if (savedSettings) {
      try {
        settings = { ...settings, ...JSON.parse(savedSettings) };
        // Apply loaded settings
        document.documentElement.setAttribute("data-theme", settings.theme);

        if (settings.enableHighContrast) {
          document.documentElement.classList.add("high-contrast");
}
        if (settings.enableReducedMotion) {
          document.documentElement.classList.add("reduce-motion");
}
        document.documentElement.style.setProperty(
          "--app-font-family",
          settings.fontFamily
        );
        document.documentElement.style.setProperty(
          "--app-font-size",
          settings.fontSize
        );
      } catch (error) {
        console.error("Error loading settings:", error);
}
}
    // Run accessibility validation in development
    if (settings.enableAccessibilityFeatures && import.meta.env.DEV) {
      setTimeout(() => {
        const errors = AccessibilityValidator.validateForm(
          document.body as any
        );
        if (errors.length > 0) {
          console.warn("Accessibility issues found:", errors);
}
      }, 1000);
}
    // Listen for global close modals event
    document.addEventListener("close-modals", () => {
      showSettings = false;
      showKeyboardShortcuts = false;
    });
  });

  // Access user from data prop
  $: currentUser = data.user;

  // Navigation links data
  const navLinks = [
    { href: "/dashboard", label: "Dashboard", icon: "📊" },
    { href: "/search", label: "Search", icon: "🔍" },
    { href: "/cases", label: "Cases", icon: "📁" },
    { href: "/evidence", label: "Evidence", icon: "🔍" },
    { href: "/legal/documents", label: "Documents", icon: "📄" },
    { href: "/interactive-canvas", label: "Canvas", icon: "🎨" },
    { href: "/reports", label: "Reports", icon: "📈" },
    { href: "/analytics", label: "Analytics", icon: "📊" },
    { href: "/ai-assistant", label: "AI Assistant", icon: "🤖" },
    { href: "/help", label: "Help", icon: "❓" }
  ];
</script>

<!-- Skip Links for accessibility -->
<div class="skip-links">
  <a href="#main-content" class="skip-link">Skip to main content</a>
  <a href="#navigation" class="skip-link">Skip to navigation</a>
</div>

<!-- Main Navigation -->
<nav class="nav" id="navigation" aria-label="Main navigation">
  <div class="container mx-auto px-4">
    <div class="nav-content">
      <!-- Brand -->
      <div class="nav-brand">
        <a href="/" class="brand-link" aria-label="Legal Case Management - Home">
          <span class="brand-icon">⚖️</span>
          <span class="brand-text font-display">Detective Mode</span>
        </a>
      </div>

      <!-- Desktop Navigation Links -->
      <div class="nav-links desktop-only" role="menubar">
        {#each navLinks as link}
          <a 
            href={link.href} 
            role="menuitem" 
            class="nav-link"
            aria-label={link.label}
          >
            <span class="nav-icon">{link.icon}</span>
            <span class="nav-text">{link.label}</span>
          </a>
        {/each}
        
        <!-- Special CRUD Dashboard Link -->
        <a 
          href="/crud-dashboard" 
          role="menuitem" 
          class="nav-link special-link"
          aria-label="CRUD Dashboard"
        >
          <span class="nav-icon">🔄</span>
          <span class="nav-text">CRUD Dashboard</span>
        </a>
      </div>

      <!-- Right Side Actions -->
      <div class="nav-actions">
        <!-- Theme Toggle -->
        <button
          type="button"
          class="btn btn-ghost btn-sm"
          on:click={() => {
            const newTheme = settings.theme === 'light' ? 'dark' : 'light';
            settings.theme = newTheme;
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('app-settings', JSON.stringify(settings));
          }}
          title="Toggle theme"
          aria-label="Toggle theme"
        >
          {settings.theme === 'light' ? '🌙' : '☀️'}
        </button>

        <!-- Keyboard Shortcuts -->
        <button
          type="button"
          class="btn btn-ghost btn-sm"
          on:click={() => (showKeyboardShortcuts = true)}
          title="Keyboard shortcuts (Ctrl+/)"
          aria-label="Show keyboard shortcuts"
        >
          ⌨️
        </button>

        <!-- Settings -->
        <button
          type="button"
          class="btn btn-ghost btn-sm"
          on:click={() => (showSettings = true)}
          title="Settings"
          aria-label="Open application settings"
        >
          ⚙️
        </button>

        <!-- User Menu or Auth Buttons -->
        {#if $user}
          <div class="user-menu">
            <div class="user-info">
              <div class="avatar">
                {$user.email.charAt(0).toUpperCase()}
              </div>
              <span class="user-email">{$user.email}</span>
            </div>
            <form action="?/logout" method="POST" style="display: inline;">
              <button type="submit" class="btn btn-outline btn-sm">
                Logout
              </button>
            </form>
          </div>
        {:else}
          <div class="auth-buttons">
            <button 
              class="btn btn-ghost btn-sm" 
              on:click={() => (showLoginModal = true)}
            >
              Login
            </button>
            <button 
              class="btn btn-secondary btn-sm" 
              on:click={() => (showRegisterModal = true)}
            >
              Register
            </button>
          </div>
        {/if}

        <!-- Mobile Menu Toggle -->
        <button
          type="button"
          class="btn btn-ghost btn-sm mobile-only"
          on:click={() => (mobileMenuOpen = !mobileMenuOpen)}
          aria-label="Toggle mobile menu"
        >
          {mobileMenuOpen ? '✕' : '☰'}
        </button>
      </div>
    </div>

    <!-- Mobile Navigation Menu -->
    {#if mobileMenuOpen}
      <div class="mobile-menu" role="menu">
        {#each navLinks as link}
          <a 
            href={link.href} 
            role="menuitem" 
            class="mobile-link"
            on:click={() => (mobileMenuOpen = false)}
          >
            <span class="nav-icon">{link.icon}</span>
            <span class="nav-text">{link.label}</span>
          </a>
        {/each}
        <a 
          href="/crud-dashboard" 
          role="menuitem" 
          class="mobile-link special-link"
          on:click={() => (mobileMenuOpen = false)}
        >
          <span class="nav-icon">🔄</span>
          <span class="nav-text">CRUD Dashboard</span>
        </a>
      </div>
    {/if}
  </div>
</nav>

<!-- Main Content -->
<main class="main-content container" id="main-content">
  <slot />
</main>

<!-- Settings Modal -->
<Settings
  bind:isOpen={showSettings}
  bind:settings
  on:close={handleSettingsClose}
  on:save={handleSettingsSave}
/>

<!-- Global UI Components -->
<EnhancedNotificationContainer />
<ModalManager />
<ErrorBoundary showInline={false} />
<KeyboardShortcuts bind:open={showKeyboardShortcuts} />

<!-- Authentication Modals -->
{#if showLoginModal}
  <Modal bind:open={showLoginModal}>
    <div slot="title">Welcome Back</div>
    <AuthForm data={{ formType: "login" }} />
  </Modal>
{/if}

{#if showRegisterModal}
  <Modal bind:open={showRegisterModal}>
    <div slot="title">Create Your Account</div>
    <AuthForm data={{ formType: "register" }} />
  </Modal>
{/if}

<style>
  /* @unocss-include */
  /* Skip links for accessibility */
  .skip-links {
    position: absolute;
    top: -100px;
    left: 0;
    z-index: var(--z-debug);
}
  .skip-link {
    position: absolute;
    top: 0;
    left: 0;
    background: var(--nier-black);
    color: var(--nier-white);
    padding: var(--space-2) var(--space-4);
    text-decoration: none;
    font-weight: 600;
    border-radius: 0 0 var(--radius-md) 0;
    transform: translateY(-100%);
    transition: transform var(--transition-fast);
}
  .skip-link:focus {
    transform: translateY(0);
    outline: 2px solid var(--nier-white);
    outline-offset: 2px;
}
  /* Navigation Styles */
  .nav {
    background: var(--bg-glass);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border-light);
    position: sticky;
    top: 0;
    z-index: var(--z-sticky);
}
  .nav-content {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-4) 0;
    gap: var(--space-8);
}
  .nav-brand .brand-link {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    text-decoration: none;
    color: var(--text-primary);
    font-size: 1.5rem;
    font-weight: 700;
    transition: color var(--transition-fast);
}
  .nav-brand .brand-link:hover {
    color: var(--text-accent);
}
  .brand-icon {
    font-size: 1.75rem;
}
  .brand-text {
    background: linear-gradient(135deg, var(--nier-black), var(--harvard-crimson));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
  /* Desktop Navigation Links */
  .nav-links {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex: 1;
    justify-content: center;
}
  .nav-link {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    padding: var(--space-2) var(--space-3);
    color: var(--text-secondary);
    text-decoration: none;
    border-radius: var(--radius-lg);
    font-weight: 500;
    font-size: 0.875rem;
    transition: all var(--transition-fast);
    position: relative;
}
  .nav-link:hover,
  .nav-link:focus {
    color: var(--text-primary);
    background: var(--bg-secondary);
    transform: translateY(-1px);
}
  .nav-link.special-link {
    background: linear-gradient(135deg, var(--harvard-crimson), var(--harvard-crimson-dark));
    color: var(--nier-white);
    font-weight: 600;
    box-shadow: var(--shadow-crimson);
}
  .nav-link.special-link:hover {
    background: linear-gradient(135deg, var(--harvard-crimson-dark), var(--harvard-crimson));
    box-shadow: var(--shadow-crimson), var(--shadow-moderate);
}
  .nav-icon {
    font-size: 1rem;
}
  .nav-text {
    font-size: 0.875rem;
}
  /* Navigation Actions */
  .nav-actions {
    display: flex;
    align-items: center;
    gap: var(--space-2);
}
  .user-menu {
    display: flex;
    align-items: center;
    gap: var(--space-3);
}
  .user-info {
    display: flex;
    align-items: center;
    gap: var(--space-2);
}
  .avatar {
    width: 32px;
    height: 32px;
    border-radius: var(--radius-full);
    background: linear-gradient(135deg, var(--harvard-crimson), var(--harvard-crimson-dark));
    color: var(--nier-white);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 0.875rem;
    box-shadow: var(--shadow-subtle);
}
  .user-email {
    font-size: 0.875rem;
    color: var(--text-secondary);
    font-weight: 500;
}
  .auth-buttons {
    display: flex;
    align-items: center;
    gap: var(--space-2);
}
  /* Mobile Menu */
  .mobile-menu {
    display: none;
    flex-direction: column;
    padding: var(--space-4) 0;
    border-top: 1px solid var(--border-light);
    background: var(--bg-primary);
}
  .mobile-link {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    color: var(--text-secondary);
    text-decoration: none;
    border-radius: var(--radius-lg);
    margin: 0 var(--space-4);
    transition: all var(--transition-fast);
}
  .mobile-link:hover,
  .mobile-link:focus {
    color: var(--text-primary);
    background: var(--bg-secondary);
}
  .mobile-link.special-link {
    background: linear-gradient(135deg, var(--harvard-crimson), var(--harvard-crimson-dark));
    color: var(--nier-white);
    font-weight: 600;
}
  /* Main Content */
  .main-content {
    min-height: calc(100vh - 80px);
    padding-top: var(--space-8);
    padding-bottom: var(--space-8);
}
  /* Responsive Design */
  .desktop-only {
    display: flex;
}
  .mobile-only {
    display: none;
}
  @media (max-width: 1024px) {
    .nav-links {
      gap: var(--space-1);
}
    .nav-text {
      display: none;
}
    .nav-link {
      padding: var(--space-2);
}
}
  @media (max-width: 768px) {
    .desktop-only {
      display: none;
}
    .mobile-only {
      display: flex;
}
    .mobile-menu {
      display: flex;
}
    .nav-content {
      gap: var(--space-4);
}
    .user-email {
      display: none;
}
    .main-content {
      padding-top: var(--space-4);
}
}
  @media (max-width: 640px) {
    .nav-content {
      padding: var(--space-3) 0;
}
    .brand-text {
      font-size: 1.25rem;
}
    .auth-buttons {
      flex-direction: column;
      gap: var(--space-1);
}
}
  /* Dark Mode Enhancements */
  [data-theme="dark"] .nav {
    background: var(--bg-panel);
    border-bottom-color: var(--border-digital);
}
  [data-theme="dark"] .brand-text {
    background: linear-gradient(135deg, var(--nier-white), var(--digital-green));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
  [data-theme="dark"] .mobile-menu {
    background: var(--bg-secondary);
}
  /* Print Styles */
  @media print {
    .nav,
    .skip-links {
      display: none;
}
    .main-content {
      padding-top: 0;
}
}
  /* High Contrast Mode */
  :global(.high-contrast) .nav {
    border-bottom: 3px solid var(--text-primary);
}
  :global(.high-contrast) .nav-link {
    border: 2px solid transparent;
}
  :global(.high-contrast) .nav-link:focus {
    border-color: var(--text-primary);
}
  /* Reduced Motion */
  :global(.reduce-motion) .nav-link {
    transition: none;
}
  :global(.reduce-motion) .skip-link {
    transition: none;
}
</style>
