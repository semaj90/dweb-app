<!-- Keyboard Shortcut Provider - Initialize shortcuts system -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { initializeKeyboardShortcuts, registerShortcut } from '$lib/utils/keyboard-shortcuts';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';

  // Initialize the keyboard shortcut system
  let cleanup: (() => void) | null = null;

  onMount(() => {
    // Initialize the global keyboard shortcut listener
    cleanup = initializeKeyboardShortcuts();

    // Register global application shortcuts
    const globalShortcuts = [
      // Navigation shortcuts
      registerShortcut({
        id: 'nav-dashboard',
        keys: ['ctrl', 'h'],
        description: 'Go to Dashboard',
        category: 'navigation',
        action: () => goto('/'),
        global: true,
      }),

      registerShortcut({
        id: 'nav-cases',
        keys: ['ctrl', 'k', 'c'],
        description: 'Go to Cases',
        category: 'navigation',
        action: () => goto('/cases'),
        global: true,
      }),

      registerShortcut({
        id: 'nav-evidence',
        keys: ['ctrl', 'k', 'e'],
        description: 'Go to Evidence',
        category: 'navigation',
        action: () => goto('/evidence'),
        global: true,
      }),

      registerShortcut({
        id: 'nav-reports',
        keys: ['ctrl', 'k', 'r'],
        description: 'Go to Reports',
        category: 'navigation',
        action: () => goto('/reports'),
        global: true,
      }),

      // Interface shortcuts
      registerShortcut({
        id: 'toggle-theme',
        keys: ['ctrl', 'shift', 't'],
        description: 'Toggle Theme',
        category: 'interface',
        action: toggleTheme,
        global: true,
      }),

      registerShortcut({
        id: 'global-search',
        keys: ['ctrl', 'k'],
        description: 'Global Search',
        category: 'search',
        action: openGlobalSearch,
        global: true,
        priority: 100, // High priority to override other Ctrl+K shortcuts
      }),

      registerShortcut({
        id: 'open-command-palette',
        keys: ['ctrl', 'shift', 'p'],
        description: 'Open Command Palette',
        category: 'interface',
        action: openCommandPalette,
        global: true,
        priority: 100,
      }),

      // Creation shortcuts
      registerShortcut({
        id: 'create-new-case',
        keys: ['ctrl', 'n', 'c'],
        description: 'Create New Case',
        category: 'creation',
        action: () => goto('/cases/new'),
        global: true,
      }),

      registerShortcut({
        id: 'create-new-report',
        keys: ['ctrl', 'n', 'r'],
        description: 'Create New Report',
        category: 'creation',
        action: () => goto('/reports/new'),
        global: true,
      }),

      // Accessibility shortcuts
      registerShortcut({
        id: 'skip-to-main',
        keys: ['alt', 'm'],
        description: 'Skip to Main Content',
        category: 'accessibility',
        action: skipToMain,
        global: true,
      }),

      registerShortcut({
        id: 'focus-search',
        keys: ['alt', 's'],
        description: 'Focus Search Field',
        category: 'accessibility',
        action: focusSearch,
        global: true,
      }),

      registerShortcut({
        id: 'show-shortcuts-help',
        keys: ['alt', '?'],
        description: 'Show Keyboard Shortcuts Help',
        category: 'accessibility',
        action: showShortcutsHelp,
        global: true,
      }),
    ];

    return () => {
      // Cleanup global shortcuts
      globalShortcuts.forEach(unregister => unregister());
      
      // Cleanup keyboard event listeners
      if (cleanup) {
        cleanup();
      }
    };
  });

  // Action implementations
  function toggleTheme() {
    const currentTheme = document.documentElement.classList.contains('yorha-theme') 
      ? 'yorha' 
      : 'default';
    
    const newTheme = currentTheme === 'yorha' ? 'default' : 'yorha';
    
    if (newTheme === 'yorha') {
      document.documentElement.classList.add('yorha-theme');
    } else {
      document.documentElement.classList.remove('yorha-theme');
    }
    
    // Store theme preference
    localStorage.setItem('theme', newTheme);
    
    // Announce theme change for accessibility
    announceToScreenReader(`Theme changed to ${newTheme} theme`);
  }

  function openGlobalSearch() {
    // Focus on global search if it exists
    const searchInput = document.querySelector('[data-global-search]') as HTMLInputElement;
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
    } else {
      // Create a temporary search modal or navigate to search page
      goto('/search');
    }
  }

  function openCommandPalette() {
    // Trigger command palette modal
    const event = new CustomEvent('open-command-palette');
    document.dispatchEvent(event);
  }

  function skipToMain() {
    const mainContent = document.querySelector('main') || document.querySelector('[role="main"]');
    if (mainContent && mainContent instanceof HTMLElement) {
      mainContent.focus();
      mainContent.scrollIntoView({ behavior: 'smooth' });
      announceToScreenReader('Skipped to main content');
    }
  }

  function focusSearch() {
    const searchSelectors = [
      '[data-search]',
      '[type="search"]',
      '[placeholder*="search" i]',
      '.search-input',
      '#search'
    ];
    
    for (const selector of searchSelectors) {
      const searchInput = document.querySelector(selector) as HTMLInputElement;
      if (searchInput && searchInput.offsetParent !== null) {
        searchInput.focus();
        searchInput.select();
        announceToScreenReader('Search field focused');
        return;
      }
    }
    
    announceToScreenReader('No search field found');
  }

  function showShortcutsHelp() {
    // Create or show keyboard shortcuts help modal
    const event = new CustomEvent('show-shortcuts-help');
    document.dispatchEvent(event);
  }

  function announceToScreenReader(message: string) {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    
    document.body.appendChild(announcement);
    
    setTimeout(() => {
      document.body.removeChild(announcement);
    }, 1000);
  }

  // Initialize theme on mount
  onMount(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'yorha') {
      document.documentElement.classList.add('yorha-theme');
    }
  });
</script>

<!-- This component doesn't render anything visible -->
<div class="sr-only" aria-hidden="true">
  Keyboard shortcuts initialized
</div>

<style>
  .sr-only {
    position: absolute
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden
    clip: rect(0, 0, 0, 0);
    white-space: nowrap
    border: 0;
  }
</style>