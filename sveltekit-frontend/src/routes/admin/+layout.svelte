<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { AuthStore, isAuthenticated, currentUser } from '$lib/auth/auth-store';
  import { AccessControl } from '$lib/auth/roles';
  import type { LayoutData } from './$types';
  
  export let data: LayoutData;
  
  let isLoading = true;
  let hasAdminAccess = false;
  
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
  
  onMount(async () => {
    try {
      // Initialize auth if not already done
      await AuthStore.initialize();
      
      // Check if user is authenticated and has admin access
      const user = data.user || $currentUser;
      
      if (!user || !$isAuthenticated) {
        goto('/login?redirect=/admin');
        return;
      }
      
      // Check for admin panel access permission
      hasAdminAccess = AccessControl.hasPermission(user.role, 'access_admin_panel');
      
      if (!hasAdminAccess) {
        goto('/unauthorized');
        return;
      }
      
    } catch (error) {
      console.error('Admin layout initialization error:', error);
      goto('/login');
    } finally {
      isLoading = false;
    }
  });
  
  // Navigation items for admin panel
  const navItems = [
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
  
  // Filter nav items based on user permissions
  $: visibleNavItems = navItems.filter(item => {
    const user = $currentUser;
    return user && AccessControl.hasPermission(user.role, item.permission);
  });
  
  // Check if current path is active
  function isActivePath(itemPath: string): boolean {
    return $page.url.pathname === itemPath || 
           ($page.url.pathname.startsWith(itemPath + '/') && itemPath !== '/admin');
  }
  
  // YoRHa terminal effect
  let glitchEffect = '';
  onMount(() => {
    const glitchChars = ['◈', '◉', '◎', '⧨', '◐', '⬢', '◇', '◆'];
    setInterval(() => {
      glitchEffect = Math.random() < 0.1 ? glitchChars[Math.floor(Math.random() * glitchChars.length)] : '';
    }, 100);
  });
</script>

<!-- Admin Layout with YoRHa Styling -->
{#if isLoading}
  <div class="{yorhaClasses.container} flex items-center justify-center">
    <div class="text-center">
      <div class="text-6xl mb-4">◈</div>
      <div class="text-xl tracking-wider">INITIALIZING ADMIN INTERFACE</div>
      <div class="mt-2 text-sm opacity-60">VERIFYING CREDENTIALS...</div>
    </div>
  </div>
{:else if hasAdminAccess}
  <div class="{yorhaClasses.container}">
    <!-- Header -->
    <header class="{yorhaClasses.header}">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-4">
          <div class="text-2xl font-bold tracking-wider">
            YORHA ADMIN{#if glitchEffect}<span class="text-red-500">{glitchEffect}</span>{/if}
          </div>
          <div class="text-sm opacity-60">
            LEGAL AI PLATFORM ADMINISTRATION
          </div>
        </div>
        
        <!-- User Info -->
        <div class="flex items-center space-x-4">
          <div class="text-sm">
            <div class="opacity-60">OPERATOR:</div>
            <div class="font-bold">{$currentUser?.email?.toUpperCase()}</div>
          </div>
          <div class="text-sm">
            <div class="opacity-60">ROLE:</div>
            <div class="font-bold text-[#00ff88]">{$currentUser?.role?.toUpperCase().replace('_', ' ')}</div>
          </div>
          <button 
            on:click={() => AuthStore.logout()}
            class="px-3 py-1 border border-red-500 text-red-500 hover:bg-red-500 hover:text-black transition-colors text-xs"
          >
            LOGOUT
          </button>
        </div>
      </div>
    </header>

    <div class="flex h-[calc(100vh-80px)]">
      <!-- Sidebar Navigation -->
      <aside class="{yorhaClasses.sidebar}">
        <nav class="space-y-2">
          <div class="text-xs opacity-60 mb-4 tracking-wider">ADMIN FUNCTIONS</div>
          {#each visibleNavItems as item}
            <a
              href={item.path}
              class={isActivePath(item.path) ? yorhaClasses.navLinkActive : yorhaClasses.navLink}
              class:w-full={true}
              class:block={true}
              class:text-left={true}
            >
              <span class="mr-3">{item.icon}</span>
              {item.label}
            </a>
          {/each}
        </nav>
        
        <!-- Quick Actions -->
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
      <main class="{yorhaClasses.content}">
        <!-- Breadcrumb -->
        <div class="mb-6 text-sm opacity-60">
          <span>ADMIN</span>
          {#if $page.url.pathname !== '/admin'}
            <span class="mx-2">></span>
            <span class="text-[#00ff88]">
              {$page.url.pathname.split('/').pop()?.toUpperCase().replace('-', ' ')}
            </span>
          {/if}
        </div>
        
        <!-- Page Content -->
        <div class="bg-[#111111] border border-[#333333] p-6 rounded-none min-h-[calc(100%-100px)]">
          <slot />
        </div>
      </main>
    </div>
  </div>
{:else}
  <!-- Access Denied -->
  <div class="{yorhaClasses.container} flex items-center justify-center">
    <div class="text-center max-w-md">
      <div class="text-6xl mb-4 text-red-500">⚠</div>
      <div class="text-2xl mb-4 tracking-wider">ACCESS DENIED</div>
      <div class="text-sm opacity-60 mb-6">
        INSUFFICIENT PRIVILEGES FOR ADMIN INTERFACE ACCESS
      </div>
      <div class="space-y-2 text-xs">
        <div>REQUIRED: ADMIN OR MANAGEMENT ROLE</div>
        <div>CURRENT: {$currentUser?.role?.toUpperCase().replace('_', ' ') || 'UNKNOWN'}</div>
      </div>
      <button 
        on:click={() => goto('/')}
        class="mt-6 px-6 py-2 border border-[#333333] hover:bg-[#1a1a1a] transition-colors"
      >
        RETURN TO MAIN INTERFACE
      </button>
    </div>
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