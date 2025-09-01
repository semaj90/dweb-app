<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { get, type Unsubscriber } from 'svelte/store';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { AuthStore, isAuthenticated, currentUser } from '$lib/auth/auth-store';
  import { AccessControl } from '$lib/auth/roles';
  import type { LayoutData } from './$types';
  import type { Permission } from '$lib/auth/roles';
  export let data: LayoutData;

  let isLoading = true;
  let hasAdminAccess = false;
  let glitchEffect = '';
  const glitchChars = ['#', '%', '@', '◈', '◉', '◎', '⧨', '◐', '⬢'];
  // Use a cross-environment interval type to avoid Node vs DOM return-type mismatches
  let glitchInterval: ReturnType<typeof setInterval> | null = null;

  // Local snapshot of current user (subscribe to store to avoid $ pref in code)
  let currentUserValue: { email?: string; role?: string } | null = null;
  let unsubscribeCurrentUser: Unsubscriber | null = null;

  // Explicit subscription for page store to avoid using $page in script reactive context
  let unsubscribePage: Unsubscriber | null = null;

  // YoRHa Terminal styling classes
  const yorhaClasses = {
    container: 'bg-[#0a0a0a] text-[#ffffff] font-mono min-h-screen',
    header: 'border-b border-[#333333] bg-[#111111] p-4',
    nav: 'flex space-x-6',
    navLink: 'px-4 py-2 border border-[#333333] bg-[#1a1a1a] hover:bg-[#2a2a2a] transition-colors',
    navLinkActive: 'px-4 py-2 border border-[#00ff88] bg-[#002211] text-[#00ff88]',
    sidebar: 'w-64 bg-[#111111] border-r border-[#333333] p-4',
    content: 'flex-1 p-6',
    glitch: 'relative overflow-hidden'
  };

  // Navigation items for admin panel
  const navItems: { path: string; label: string; icon: string; permission: Permission }[] = [
    {
      path: '/admin',
      label: 'DASHBOARD',
      icon: '◈',
      permission: 'access_admin_panel'
    },
    {
      path: '/admin/users',
      label: 'USER MANAGEMENT',
      icon: '◉',
      permission: 'manage_users'
    },
    {
      path: '/admin/roles',
      label: 'ROLE MANAGEMENT',
      icon: '◎',
      permission: 'manage_users'
    },
    {
      path: '/admin/system',
      label: 'SYSTEM CONFIG',
      icon: '⧨',
      permission: 'configure_system'
    },
    {
      path: '/admin/audit',
      label: 'AUDIT LOGS',
      icon: '◐',
      permission: 'view_audit_logs'
    },
    {
      path: '/admin/integrations',
      label: 'INTEGRATIONS',
      icon: '⬢',
      permission: 'manage_integrations'
    }
  ];

  // Derive visible nav items from current user permissions
  let visibleNavItems: { path: string; label: string; icon: string; permission: Permission }[] = [];

  // Derived display values for safer template usage
  $: userEmail = currentUserValue?.email ?? '';
  $: userRoleLabel = currentUserValue?.role ? currentUserValue.role.toUpperCase().replace(/_/g, ' ') : 'UNKNOWN';

  $: visibleNavItems = currentUserValue && currentUserValue.role
    ? navItems.filter(item => AccessControl.hasPermission(currentUserValue.role, item.permission))
    : [];

  // Maintain current path via explicit subscription to avoid $page usage in TS logic
  let currentPath = '';

  function isActivePath(itemPath: string) {
    return currentPath === itemPath || (itemPath !== '/admin' && currentPath.startsWith(itemPath + '/'));
  }

  function navClass(item: { path: string; label: string; icon: string; permission: Permission }) {
    return (isActivePath(item.path) ? yorhaClasses.navLinkActive : yorhaClasses.navLink) + ' w-full block text-left';
  }

  onMount(async () => {
    try {
      // Initialize auth if not already done
      await AuthStore.initialize();

      // Initialize current user from data or store snapshot and subscribe for updates
      const initialUser = data.user || get(currentUser);
      currentUserValue = initialUser ?? null;
      unsubscribeCurrentUser = currentUser.subscribe(v => { currentUserValue = v; });

      // subscribe to page store to keep currentPath updated (safe in TS/SSR)
      unsubscribePage = page.subscribe(p => {
        currentPath = p?.url?.pathname ?? '';
      });

      // Check if user is authenticated and has admin access
      const user = currentUserValue;

      if (!user || !get(isAuthenticated)) {
        goto('/login?redirect=/admin');
        return;
      }

      // Check for admin panel access permission (guard role access)
      const userRole = user?.role ?? '';
      hasAdminAccess = userRole ? AccessControl.hasPermission(userRole, 'access_admin_panel') : false;

      if (!hasAdminAccess) {
        goto('/unauthorized');
        return;
      }

      // Start a small decorative glitch effect
      glitchInterval = setInterval(() => {
        glitchEffect = Math.random() < 0.08 ? glitchChars[Math.floor(Math.random() * glitchChars.length)] : '';
      }, 120);

    } catch (error) {
      console.error('Admin layout initialization error:', error);
      goto('/login');
    } finally {
      isLoading = false;
    }
  });

  onDestroy(() => {
    if (glitchInterval !== null) {
      clearInterval(glitchInterval);
    }
    if (unsubscribeCurrentUser) {
      unsubscribeCurrentUser();
    }
    if (unsubscribePage) {
      unsubscribePage();
    }
  });
</script>

<!-- Admin Layout with YoRHa Styling -->
{#if !isLoading}
  <div class={yorhaClasses.container}>
    <!-- Header -->
    <header class={yorhaClasses.header}>
      <div class="flex items-center justify-between">
        <div class="text-sm">
          <div class="opacity-60">OPERATOR:</div>
          <div class="font-bold">{userEmail ? userEmail.toUpperCase() : (currentUserValue?.email ? currentUserValue.email.toUpperCase() : '')}</div>
        </div>
        <div class="text-sm">
          <div class="opacity-60">ROLE:</div>
          <div class="font-bold text-[#00ff88]">{userRoleLabel !== 'UNKNOWN' ? userRoleLabel : (currentUserValue?.role ? currentUserValue.role.toUpperCase().replace(/_/g, ' ') : '')}</div>
        </div>
        <div>
          <button
            onclick={() => AuthStore.logout()}
            class="px-3 py-1 border border-red-500 text-red-500 hover:bg-red-500 hover:text-black transition-colors text-xs"
          >
            LOGOUT
          </button>
        </div>
      </div>
    </header>

    <div class="flex">
      <aside class={yorhaClasses.sidebar}>
        <nav class="space-y-2">
          {#each visibleNavItems as item}
            <a href={item.path} class={navClass(item)}>
              <span class="mr-3">{item.icon}</span>
              {item.label}
            </a>
          {/each}
        </nav>
        <div class="mt-8 pt-4 border-t border-[#333333]">
          <div class="text-xs opacity-60 mb-4 tracking-wider">QUICK ACTIONS</div>
          <div class="space-y-2 text-sm">
            <button class="w-full text-left px-2 py-1 hover:bg-[#1a1a1a] transition-colors">
              ◈ SYSTEM STATUS
            </button>
            <button class="w-full text-left px-2 py-1 hover:bg-[#1a1a1a] transition-colors">
              ◉ BACKUP SYSTEM
            </button>
            <button class="w-full text-left px-2 py-1 hover:bg-[#1a1a1a] transition-colors">
              ◎ CLEAR CACHE
            </button>
          </div>
        </div>
      </aside>

      <!-- Main Content Area -->
      <main class={yorhaClasses.content}>
        <div class="space-y-2 text-xs mb-4">
          <div>REQUIRED: ADMIN OR MANAGEMENT ROLE</div>
          <div>CURRENT: {currentUserValue?.role ? currentUserValue.role.toUpperCase().replace(/_/g, ' ') : 'UNKNOWN'}</div>
        </div>
        <slot />
      </main>
    </div>
  </div>

{:else}
  <div class="text-center max-w-md mx-auto py-12">
    <div class="text-6xl mb-4 text-red-500">⚠</div>
    <div class="space-y-2 text-xs mb-4">
      <div>REQUIRED: ADMIN OR MANAGEMENT ROLE</div>
      <div>CURRENT: {currentUserValue?.role ? currentUserValue.role.toUpperCase().replace(/_/g, ' ') : 'UNKNOWN'}</div>
    </div>
    <button
      onclick={() => goto('/')}
      class="mt-6 px-6 py-2 border border-[#333333] hover:bg-[#1a1a1a] transition-colors"
    >
      RETURN TO MAIN INTERFACE
    </button>
  </div>
{/if}

<style>
  /* YoRHa Terminal Effects */
  @keyframes glitch {
    0% { transform: translateX(0); }
    20% { transform: translateX(-2px); }
    40% { transform: translateX(2px); }
    60% { transform: translateX(-1px); }
    80% { transform: translateX(1px); }
    100% { transform: translateX(0); }
  }

  .glitch {
    animation: glitch 0.3s ease-in-out infinite;
  }

  /* Custom scrollbar for YoRHa theme */
  :global(::-webkit-scrollbar) {
    width: 8px;
  }

  :global(::-webkit-scrollbar-track) {
    background: #111111;
  }

  :global(::-webkit-scrollbar-thumb) {
    background: #333333;
    border-radius: 0;
  }

  :global(::-webkit-scrollbar-thumb:hover) {
    background: #555555;
  }

  /* Terminal cursor effect */
  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }

  .cursor {
    animation: blink 1s infinite;
  }
</style>