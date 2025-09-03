<!--
  All Routes Index - Dynamic Route Discovery and Navigation
  Enhanced with N64 3D Gaming UI Components and Comprehensive Testing
-->
<script lang="ts">
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';
  import type { PageData } from './$types';
  
  // Comprehensive bits-ui v2 components integration
  import { Button } from 'bits-ui';
  import { Menubar } from 'bits-ui';
  import { Combobox } from 'bits-ui';
  import { Popover } from 'bits-ui';
  import { Select } from 'bits-ui';
  import { Separator } from 'bits-ui';
  import { Dialog } from 'bits-ui';
  import { NavigationMenu } from 'bits-ui';
  import { Tooltip } from 'bits-ui';
  import { Label } from 'bits-ui';
  import { Toggle } from 'bits-ui';
  import { Slider } from 'bits-ui';
  import { ScrollArea } from 'bits-ui';
  import { Toolbar } from 'bits-ui';
  import { Progress } from 'bits-ui';
  import { Input } from 'bits-ui';
  import { DropdownMenu } from 'bits-ui';
  
  // Gaming UI Components
  import N643DButton from '$lib/components/ui/gaming/n64/N643DButton.svelte';
  // import N64TextureFilteringCache from '$lib/components/ui/gaming/n64/N64TextureFilteringCache.svelte';
  import NES8BitButton from '$lib/components/ui/gaming/8bit/NES8BitButton.svelte';
  import NES8BitContainer from '$lib/components/ui/gaming/8bit/NES8BitContainer.svelte';
  
  // Standard UI Components
  import Card from '$lib/components/ui/Card.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';

  // Import route configuration
  import { allRoutes, routeCategories } from '$lib/data/routes-config';
  
  // Get data from server loader
  let { data }: { data: PageData } = $props();
  // Discover all route modules
  const modules = import.meta.glob('/src/routes/**/+page.svelte');

  // Enhanced state management with bits-ui integration
  let searchValue = $state('');
  let categoryValue = $state('all');
  let sortValue = $state<'name' | 'category' | 'status'>('name');
  let viewMode = $state<'grid' | 'list' | 'terminal'>('grid');
  let gamingMode = $state(true);
  let testingMode = $state(false);
  
  // bits-ui v2 component states
  let showApiDialog = $state(false);
  let selectedRoutes = $state<string[]>([]);
  let bulkOperationProgress = $state(0);
  let showSettings = $state(false);
  let searchComboValue = $state('');
  let batchSize = $state(10);
  let testTimeout = $state(5000);
  let enableParallelTesting = $state(true);
  let apiEndpointTest = $state<any>(null);
  
  // Enhanced route testing state with comprehensive API integration
  let testResults = $state<Record<string, 'pending' | 'success' | 'error' | 'timeout'>>({});
  let testingProgress = $state(0);
  let totalTests = $state(0);
  let currentlyTesting = $state<string | null>(null);
  let systemHealth = $state<any>(null);
  let apiOperationResults = $state<Record<string, any>>({});
  let recentOperations = $state<any[]>([]);
  
  // Authentication state management - initialize from server data
  let showAuthDialog = $state(false);
  let authMode = $state<'login' | 'register'>('login');
  let authLoading = $state(false);
  let authError = $state<string | null>(null);
  let currentUser = $state<any>(data.userSession.user);
  let isAuthenticated = $state(data.userSession.isAuthenticated);
  
  // Auth form state
  let authForm = $state({
    email: '',
    password: '',
    firstName: '',
    lastName: '',
    role: 'user' as 'attorney' | 'paralegal' | 'investigator' | 'user',
    rememberMe: false
  });

  // Build discovered routes from file system with enhanced tracking
  let filteredRoutes = $state<any[]>([]);
  let allTestedRoutes = $state<any[]>([]);
  let favoriteRoutes = $state<string[]>([]);
  let routeHistory = $state<string[]>([]);

  // Update discovered routes
  let discoveredRoutes = $derived(Object.keys(modules)
    .map((path) => {
      let route = path
        .replace('/src/routes', '')
        .replace('/+page.svelte', '')
        .replace('/+layout.svelte', '');

      // Handle root route
      if (route === '') route = '/';

      // Handle dynamic routes
      route = route.replace(/\[([^\]]+)\]/g, ':$1');

      return route;
    })
    .filter((r, i, arr) => arr.indexOf(r) === i) // unique
    .filter(r => r !== '/+error') // filter out error pages
    .sort());

  // Combine configured routes with discovered routes
  let allAvailableRoutes = $derived([
    ...allRoutes.map(route => ({
      ...route,
      type: 'configured',
      available: discoveredRoutes.includes(route.route)
    })),
    ...discoveredRoutes
      .filter(route => !allRoutes.some(r => r.route === route))
      .map(route => ({
        id: route.replace(/[\/\:]/g, '-').slice(1) || 'home',
        label: formatRouteLabel(route),
        route,
        icon: inferRouteIcon(route),
        description: `Discovered route: ${route}`,
        category: inferRouteCategory(route),
        status: 'discovered',
        tags: [],
        type: 'discovered',
        available: true
      }))
  ]);

  // Filter and sort routes
  let filteredRoutesComputed = $derived(allAvailableRoutes
    .filter(route => {
      // Search filter
      if (searchValue) {
        const searchLower = searchValue.toLowerCase();
        return (
          route.label.toLowerCase().includes(searchLower) ||
          route.route.toLowerCase().includes(searchLower) ||
          route.description.toLowerCase().includes(searchLower) ||
          (route.tags && route.tags.some((tag: string) => tag.toLowerCase().includes(searchLower)))
        );
      }
      return true;
    })
    .filter(route => {
      // Category filter
      if (categoryValue === 'all') return true;
      if (categoryValue === 'available') return route.available;
      if (categoryValue === 'configured') return route.type === 'configured';
      if (categoryValue === 'discovered') return route.type === 'discovered';
      return route.category === categoryValue;
    })
    .sort((a, b) => {
      switch (sortValue) {
        case 'category':
          return a.category.localeCompare(b.category) || a.label.localeCompare(b.label);
        case 'status':
          return a.status.localeCompare(b.status) || a.label.localeCompare(b.label);
        default:
          return a.label.localeCompare(b.label);
      }
    }));

  // Update filteredRoutes reactively
  $effect(() => {
    filteredRoutes = filteredRoutesComputed;
  });

  function formatRouteLabel(route: string): string {
    if (route === '/') return 'Home';

    return route
      .split('/')
      .filter(Boolean)
      .map(segment =>
        segment
          .replace(/^:/, '') // Remove parameter prefix
          .split('-')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ')
      )
      .join(' → ');
  }

  function inferRouteIcon(route: string): string {
    if (route === '/') return '🏠';
    if (route.includes('demo')) return '🎯';
    if (route.includes('dev')) return '🔧';
    if (route.includes('ai')) return '🤖';
    if (route.includes('legal')) return '⚖️';
    if (route.includes('admin')) return '⚙️';
    if (route.includes('case')) return '📁';
    if (route.includes('evidence')) return '🔍';
    if (route.includes('chat')) return '💬';
    if (route.includes('search')) return '🔎';
    if (route.includes('upload')) return '📤';
    if (route.includes('report')) return '📊';
    if (route.includes('setting')) return '⚙️';
    if (route.includes('profile')) return '👤';
    if (route.includes('help')) return '❓';
    if (route.includes('security')) return '🛡️';
    return '📄';
  }

  function inferRouteCategory(route: string): string {
    if (route.includes('/demo/')) return 'demo';
    if (route.includes('/dev/')) return 'dev';
    if (route.includes('/ai/') || route.includes('/chat/')) return 'ai';
    if (route.includes('/legal/')) return 'legal';
    if (route.includes('/admin/') || route.includes('/settings/')) return 'admin';
    return 'main';
  }

  // Enhanced route testing and navigation
  async function testRoute(route: string): Promise<'success' | 'error' | 'timeout'> {
    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        resolve('timeout');
      }, 5000);

      // Create iframe to test route
      const iframe = document.createElement('iframe');
      iframe.style.display = 'none';
      iframe.src = route;
      
      iframe.onload = () => {
        clearTimeout(timeout);
        document.body.removeChild(iframe);
        resolve('success');
      };
      
      iframe.onerror = () => {
        clearTimeout(timeout);
        document.body.removeChild(iframe);
        resolve('error');
      };
      
      document.body.appendChild(iframe);
    });
  }

  async function testAllRoutes() {
    testingMode = true;
    testResults = {};
    testingProgress = 0;
    
    const routesToTest = allAvailableRoutes.filter(r => r.available);
    totalTests = routesToTest.length;

    for (let i = 0; i < routesToTest.length; i++) {
      const route = routesToTest[i];
      currentlyTesting = route.route;
      testResults[route.route] = 'pending';
      
      const result = await testRoute(route.route);
      testResults[route.route] = result;
      testingProgress = i + 1;
      
      // Small delay to show progress
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    currentlyTesting = null;
    testingMode = false;
  }

  async function navigateToRoute(route: string) {
    try {
      if (gamingMode) {
        // Add gaming transition effect
        document.body.style.transition = 'all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
        document.body.style.transform = 'scale(0.95) rotateX(2deg)';
        
        setTimeout(async () => {
          await goto(route);
          document.body.style.transform = '';
        }, 150);
      } else {
        await goto(route);
      }
    } catch (error) {
      console.error('Navigation failed:', error);
      if (gamingMode) {
        // Gaming-style error notification
        showN64ErrorDialog(`Failed to navigate to ${route}: ${error.message}`);
      } else {
        alert(`Failed to navigate to ${route}`);
      }
    }
  }

  function showN64ErrorDialog(message: string) {
    const dialog = document.createElement('div');
    dialog.className = 'n64-error-dialog';
    dialog.innerHTML = `
      <div class="n64-dialog-content">
        <div class="n64-dialog-header">⚠️ NAVIGATION ERROR</div>
        <div class="n64-dialog-body">${message}</div>
        <button class="n64-dialog-button" on:on:onclick="this.parentElement.parentElement.remove()">
          ACKNOWLEDGE
        </button>
      </div>
    `;
    document.body.appendChild(dialog);
    
    setTimeout(() => {
      if (dialog.parentElement) {
        dialog.remove();
      }
    }, 5000);
  }

  function getRouteStatusColor(status: string): string {
    switch (status) {
      case 'active': return 'bg-green-500/20 text-green-300 border-green-500/30';
      case 'experimental': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
      case 'beta': return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
      case 'deprecated': return 'bg-red-500/20 text-red-300 border-red-500/30';
      case 'discovered': return 'bg-purple-500/20 text-purple-300 border-purple-500/30';
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30';
    }
  }

  // Enhanced API integration functions
  async function fetchSystemHealth() {
    try {
      const response = await fetch('/api/comprehensive-integration', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        systemHealth = await response.json();
      }
    } catch (error) {
      console.error('System health fetch error:', error);
    }
  }
  
  async function performApiOperation(operation: string, data?: any) {
    try {
      const response = await fetch('/api/comprehensive-integration', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ operation, data })
      });
      
      if (response.ok) {
        const result = await response.json();
        apiOperationResults[operation] = result;
        recentOperations = [{ operation, result, timestamp: Date.now() }, ...recentOperations.slice(0, 9)];
        return result;
      }
    } catch (error) {
      console.error(`API operation ${operation} error:`, error);
      throw error;
    }
  }
  
  async function testApiEndpoint(endpoint: string) {
    try {
      const response = await fetch(endpoint, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const result = {
        endpoint,
        status: response.status,
        ok: response.ok,
        data: response.ok ? await response.json() : await response.text(),
        timestamp: Date.now()
      };
      
      apiEndpointTest = result;
      return result;
    } catch (error) {
      const result = {
        endpoint,
        status: 0,
        ok: false,
        error: String(error),
        timestamp: Date.now()
      };
      apiEndpointTest = result;
      return result;
    }
  }
  
  function toggleRouteSelection(route: string) {
    if (selectedRoutes.includes(route)) {
      selectedRoutes = selectedRoutes.filter(r => r !== route);
    } else {
      selectedRoutes = [...selectedRoutes, route];
    }
  }
  
  function toggleFavorite(route: string) {
    if (favoriteRoutes.includes(route)) {
      favoriteRoutes = favoriteRoutes.filter(r => r !== route);
    } else {
      favoriteRoutes = [...favoriteRoutes, route];
    }
  }
  
  async function bulkTestRoutes() {
    if (selectedRoutes.length === 0) return;
    
    testingMode = true;
    bulkOperationProgress = 0;
    
    const routesToTest = selectedRoutes.filter(r => 
      allAvailableRoutes.find(route => route.route === r)?.available
    );
    
    for (let i = 0; i < routesToTest.length; i++) {
      const route = routesToTest[i];
      currentlyTesting = route;
      testResults[route] = 'pending';
      
      const result = await testRoute(route);
      testResults[route] = result;
      bulkOperationProgress = ((i + 1) / routesToTest.length) * 100;
      
      if (enableParallelTesting && i < routesToTest.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
    
    currentlyTesting = null;
    testingMode = false;
    selectedRoutes = [];
  }
  
  // Authentication functions
  async function handleLogin() {
    authLoading = true;
    authError = null;
    
    try {
      // Multi-protocol login with intelligent protocol selection
      const loginPayload = {
        email: authForm.email,
        password: authForm.password,
        rememberMe: authForm.rememberMe,
        protocol: 'quic', // Try QUIC first for best performance
        enableAutoTagging: true
      };

      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(loginPayload)
      });
      
      const result = await response.json();
      
      if (result.success) {
        currentUser = result.data.user;
        isAuthenticated = true;
        showAuthDialog = false;
        
        // Show protocol and performance info
        const protocolInfo = result.protocol ? 
          ` via ${result.protocol.used.toUpperCase()} (${result.protocol.processingTime})` : '';
        
        // Reset form
        authForm = {
          email: '',
          password: '',
          firstName: '',
          lastName: '',
          role: 'user',
          rememberMe: false
        };
        
        // Add to recent operations with protocol info
        recentOperations = [
          { 
            operation: `User Login${protocolInfo}`, 
            timestamp: new Date().toISOString(), 
            status: 'success',
            protocol: result.protocol?.used || 'rest',
            autoTagging: result.protocol?.autoTagging || false
          },
          ...recentOperations.slice(0, 4)
        ];
        
        console.log('🚀 Login successful:', {
          user: currentUser,
          protocol: result.protocol,
          autoTagging: result.protocol?.autoTagging
        });
      } else {
        authError = result.message || 'Login failed';
      }
    } catch (error) {
      authError = 'Network error during login';
      console.error('❌ Login error:', error);
    } finally {
      authLoading = false;
    }
  }
  
  async function handleRegister() {
    authLoading = true;
    authError = null;
    
    try {
      // Multi-protocol registration with comprehensive profile data
      const registrationPayload = {
        email: authForm.email,
        password: authForm.password,
        firstName: authForm.firstName,
        lastName: authForm.lastName,
        role: authForm.role,
        protocol: 'quic', // Try QUIC first for best performance
        enableAutoTagging: true,
        profileData: {
          preferences: {
            theme: 'light',
            language: 'en',
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            notifications: {
              email: true,
              push: true,
              sms: false
            },
            aiAssistance: {
              autoSummarize: true,
              suggestCitations: true,
              riskAnalysis: true
            }
          }
        }
      };

      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(registrationPayload)
      });
      
      const result = await response.json();
      
      if (result.success) {
        // Show protocol and performance info
        const protocolInfo = result.protocol ? 
          ` via ${result.protocol.used.toUpperCase()} (${result.protocol.processingTime})` : '';
        
        // Add registration to recent operations
        recentOperations = [
          { 
            operation: `User Registration${protocolInfo}`, 
            timestamp: new Date().toISOString(), 
            status: 'success',
            protocol: result.protocol?.used || 'rest',
            autoTagging: result.protocol?.autoTagging || false
          },
          ...recentOperations.slice(0, 4)
        ];

        console.log('🚀 Registration successful:', {
          user: result.data.user,
          protocol: result.protocol,
          autoTagging: result.protocol?.autoTagging
        });

        // Auto-login after successful registration
        authForm.firstName = '';
        authForm.lastName = '';
        await handleLogin();
      } else {
        authError = result.message || 'Registration failed';
      }
    } catch (error) {
      authError = 'Network error during registration';
      console.error('❌ Registration error:', error);
    } finally {
      authLoading = false;
    }
  }
  
  async function handleLogout() {
    try {
      await fetch('/api/auth/logout', { method: 'POST' });
    } catch (error) {
      console.error('Logout error:', error);
    }
    
    currentUser = null;
    isAuthenticated = false;
    
    // Add to recent operations
    recentOperations = [
      { operation: 'User Logout', timestamp: new Date().toISOString(), status: 'success' },
      ...recentOperations.slice(0, 4)
    ];
  }
  
  function openAuthDialog(mode: 'login' | 'register') {
    authMode = mode;
    authError = null;
    showAuthDialog = true;
  }
  
  // Initialize system on mount - use server data as initial state
  onMount(async () => {
    // Initialize with server-loaded data
    systemHealth = data.systemHealth;
    recentOperations = data.recentOperations;
    
    // Only fetch if server data is missing
    if (!systemHealth) {
      await fetchSystemHealth();
    }
  });
  
  // Log derived values when they change
  $effect(() => {
    console.log('Discovered routes:', discoveredRoutes);
    console.log('All available routes:', allAvailableRoutes);
  });
</script>

<svelte:head>
  <title>YoRHa Routes Index - Legal AI System</title>
</svelte:head>

<NES8BitContainer class="min-h-screen">
  <div class="max-w-7xl mx-auto space-y-6 p-6">
    <!-- Enhanced Header with bits-ui v2 Toolbar Integration -->
    <div class="text-center border-b border-amber-500/30 pb-6 relative">
      <!-- <N64TextureFilteringCache /> -->
      
      <div class="relative z-10">
        <h1 class="text-5xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600">
          🎮 BITS-UI V2 ROUTES CENTER
        </h1>
        <p class="text-xl mb-2 text-amber-100">
          Comprehensive bits-ui v2 + SvelteKit 2 + Svelte 5 Integration
        </p>
        <p class="text-amber-300/70">
          {allAvailableRoutes.length} total routes • {filteredRoutes.length} displayed • {selectedRoutes.length} selected
        </p>
        
        <!-- bits-ui v2 Toolbar -->
        <Toolbar.Root class="flex justify-center items-center gap-2 mt-4 p-2 bg-black/20 rounded-lg border border-amber-500/30">
          <!-- Gaming Mode Toggle -->
          <Toggle.Root
            bind:pressed={gamingMode}
            class="px-4 py-2 rounded bg-purple-600/20 border border-purple-500/40 text-purple-200 hover:bg-purple-600/30 data-[state=on]:bg-purple-600 data-[state=on]:text-white"
            title="Toggle between gaming and classic UI modes"
          >
            {gamingMode ? '🎮 Gaming ON' : '📱 Classic'}
          </Toggle.Root>
          
          <Separator.Root orientation="vertical" class="h-6 w-px bg-amber-500/40" />
          
          <!-- View Mode Select -->
          <Select.Root bind:selected={viewMode}>
            <Select.Trigger class="px-4 py-2 bg-blue-600/20 border border-blue-500/40 text-blue-200 rounded hover:bg-blue-600/30">
              {viewMode === 'grid' ? '🔲 Grid' : viewMode === 'list' ? '📋 List' : '💻 Terminal'}
            </Select.Trigger>
            <Select.Content class="bg-black/90 border border-amber-500/50 rounded">
              <Select.Item value="grid" class="p-2 text-white hover:bg-amber-600/20">🔲 Grid View</Select.Item>
              <Select.Item value="list" class="p-2 text-white hover:bg-amber-600/20">📋 List View</Select.Item>
              <Select.Item value="terminal" class="p-2 text-white hover:bg-amber-600/20">💻 Terminal View</Select.Item>
            </Select.Content>
          </Select.Root>
          
          <Separator.Root orientation="vertical" class="h-6 w-px bg-amber-500/40" />
          
          <!-- Test Operations -->
          <Button.Root
            on:on:on:click={testAllRoutes}
            disabled={testingMode}
            class="px-4 py-2 bg-green-600/20 border border-green-500/40 text-green-200 rounded hover:bg-green-600/30 disabled:opacity-50"
          >
            {testingMode ? '⏳ Testing...' : '🧪 Test All'}
          </Button.Root>
          
          <Button.Root
            on:on:on:click={bulkTestRoutes}
            disabled={testingMode || selectedRoutes.length === 0}
            class="px-4 py-2 bg-orange-600/20 border border-orange-500/40 text-orange-200 rounded hover:bg-orange-600/30 disabled:opacity-50"
          >
            🎯 Test Selected ({selectedRoutes.length})
          </Button.Root>
          
          <Separator.Root orientation="vertical" class="h-6 w-px bg-amber-500/40" />
          
          <!-- Authentication Section -->
          {#if isAuthenticated && currentUser}
            <!-- User Avatar Dropdown -->
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild let:builder>
                <Button.Root
                  builders={[builder]}
                  class="flex items-center gap-2 px-4 py-2 bg-emerald-600/20 border border-emerald-500/40 text-emerald-200 rounded hover:bg-emerald-600/30"
                >
                  <div class="w-6 h-6 bg-emerald-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
                    {currentUser.firstName?.[0]?.toUpperCase() || currentUser.email[0].toUpperCase()}
                  </div>
                  <span class="hidden sm:inline">
                    {currentUser.firstName || currentUser.email.split('@')[0]}
                  </span>
                </Button.Root>
              </DropdownMenu.Trigger>
              <DropdownMenu.Content class="w-56 bg-gray-900/95 border border-amber-500/50 rounded p-2 z-50">
                <DropdownMenu.Item class="w-full p-3 text-left rounded hover:bg-amber-600/20 text-white" on:on:on:click={() => goto('/profile')}>
                  👤 Profile
                </DropdownMenu.Item>
                <DropdownMenu.Item class="w-full p-3 text-left rounded hover:bg-amber-600/20 text-white" on:on:on:click={() => goto('/settings')}>
                  ⚙️ Settings
                </DropdownMenu.Item>
                <DropdownMenu.Separator class="my-1 h-px bg-amber-500/30" />
                <DropdownMenu.Item class="w-full p-3 text-left rounded hover:bg-red-600/20 text-red-300" on:on:on:click={handleLogout}>
                  🚪 Logout
                </DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Root>
          {:else}
            <!-- Login/Register Buttons -->
            <Button.Root
              on:on:on:click={() => openAuthDialog('login')}
              class="px-4 py-2 bg-blue-600/20 border border-blue-500/40 text-blue-200 rounded hover:bg-blue-600/30"
            >
              🔑 Login
            </Button.Root>
            
            <Button.Root
              on:on:on:click={() => openAuthDialog('register')}
              class="px-4 py-2 bg-green-600/20 border border-green-500/40 text-green-200 rounded hover:bg-green-600/30"
            >
              📝 Register
            </Button.Root>
          {/if}
          
          <Separator.Root orientation="vertical" class="h-6 w-px bg-amber-500/40" />
          
          <!-- API Operations Dialog -->
          <Dialog.Root bind:open={showApiDialog}>
            <Button.Root
              on:on:on:click={() => showApiDialog = true}
              class="px-4 py-2 bg-cyan-600/20 border border-cyan-500/40 text-cyan-200 rounded hover:bg-cyan-600/30"
            >
              🔗 API Ops
            </Button.Root>
            <Dialog.Portal>
              <Dialog.Overlay class="fixed inset-0 bg-black/70 z-50" />
              <Dialog.Content class="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-gray-900 border border-amber-500/50 rounded-lg p-6 z-51 max-w-2xl w-full max-h-[80vh]">
                <ScrollArea.Root class="h-full">
                  <ScrollArea.Viewport class="h-full">
                    <Dialog.Title class="text-xl font-bold text-amber-300 mb-4">
                      🚀 API Operations Center
                    </Dialog.Title>
                    
                    <div class="space-y-4">
                      <!-- System Health -->
                      {#if systemHealth}
                      <div class="p-4 bg-black/40 rounded border border-green-500/30">
                        <h3 class="font-semibold text-green-300 mb-2">System Health</h3>
                        <div class="text-sm text-green-200">
                          Services: {systemHealth.system_overview.healthy_services}/{systemHealth.system_overview.total_services} healthy
                        </div>
                        <Progress.Root value={systemHealth.system_overview.healthy_services} max={systemHealth.system_overview.total_services} class="mt-2">
                          <Progress.Indicator class="h-2 bg-green-500 rounded transition-all" />
                        </Progress.Root>
                      </div>
                      {/if}
                      
                      <!-- API Operations -->
                      <div class="grid grid-cols-2 gap-2">
                        <Button.Root
                          on:on:on:click={() => performApiOperation('system_optimization')}
                          class="p-3 bg-blue-600/20 border border-blue-500/40 text-blue-200 rounded hover:bg-blue-600/30"
                        >
                          🔧 System Optimization
                        </Button.Root>
                        
                        <Button.Root
                          on:on:on:click={() => performApiOperation('context7_integration')}
                          class="p-3 bg-purple-600/20 border border-purple-500/40 text-purple-200 rounded hover:bg-purple-600/30"
                        >
                          🧠 Context7 Integration
                        </Button.Root>
                        
                        <Button.Root
                          on:on:on:click={() => performApiOperation('real_time_analysis')}
                          class="p-3 bg-yellow-600/20 border border-yellow-500/40 text-yellow-200 rounded hover:bg-yellow-600/30"
                        >
                          ⚡ Real-time Analysis
                        </Button.Root>
                        
                        <Button.Root
                          on:on:on:click={() => performApiOperation('legal_research')}
                          class="p-3 bg-red-600/20 border border-red-500/40 text-red-200 rounded hover:bg-red-600/30"
                        >
                          ⚖️ Legal Research
                        </Button.Root>
                      </div>
                      
                      <!-- Recent Operations -->
                      {#if recentOperations.length > 0}
                      <div class="p-4 bg-black/40 rounded border border-gray-500/30">
                        <h3 class="font-semibold text-gray-300 mb-2">Recent Operations</h3>
                        <div class="space-y-2 max-h-32 overflow-y-auto">
                          {#each recentOperations as op}
                          <div class="text-xs p-2 bg-gray-800/50 rounded">
                            <span class="text-cyan-300">{op.operation}</span>
                            <span class="text-gray-400 ml-2">{new Date(op.timestamp).toLocaleTimeString()}</span>
                          </div>
                          {/each}
                        </div>
                      </div>
                      {/if}
                    </div>
                  </ScrollArea.Viewport>
                  <ScrollArea.Scrollbar orientation="vertical">
                    <ScrollArea.Thumb />
                  </ScrollArea.Scrollbar>
                </ScrollArea.Root>
                
                <Button.Root
                  on:on:on:click={() => showApiDialog = false}
                  class="absolute top-2 right-2 p-2 text-gray-400 hover:text-white"
                >
                  ✕
                </Button.Root>
              </Dialog.Content>
            </Dialog.Portal>
          </Dialog.Root>
          
          <!-- Authentication Dialog -->
          <Dialog.Root bind:open={showAuthDialog}>
            <Dialog.Portal>
              <Dialog.Overlay class="fixed inset-0 bg-black/70 z-50" />
              <Dialog.Content class="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-gray-900 border border-amber-500/50 rounded-lg p-6 z-51 max-w-md w-full">
                <Dialog.Title class="text-xl font-bold text-amber-300 mb-4">
                  {authMode === 'login' ? '🔑 Login' : '📝 Register'}
                </Dialog.Title>
                
                <form onsubmit={e => { e.preventDefault(); authMode === 'login' ? handleLogin() : handleRegister(); }} class="space-y-4">
                  {#if authMode === 'register'}
                    <div class="grid grid-cols-2 gap-2">
                      <div>
                        <Label.Root class="block text-sm font-medium text-gray-300 mb-1">First Name</Label.Root>
                        <Input.Root
                          bind:value={authForm.firstName}
                          placeholder="John"
                          required
                          class="w-full px-3 py-2 bg-black/40 border border-gray-600 rounded text-white placeholder-gray-400 focus:border-amber-500 focus:ring-1 focus:ring-amber-500"
                        />
                      </div>
                      <div>
                        <Label.Root class="block text-sm font-medium text-gray-300 mb-1">Last Name</Label.Root>
                        <Input.Root
                          bind:value={authForm.lastName}
                          placeholder="Doe"
                          required
                          class="w-full px-3 py-2 bg-black/40 border border-gray-600 rounded text-white placeholder-gray-400 focus:border-amber-500 focus:ring-1 focus:ring-amber-500"
                        />
                      </div>
                    </div>
                  {/if}
                  
                  <div>
                    <Label.Root class="block text-sm font-medium text-gray-300 mb-1">Email</Label.Root>
                    <Input.Root
                      bind:value={authForm.email}
                      type="email"
                      placeholder="user@example.com"
                      required
                      class="w-full px-3 py-2 bg-black/40 border border-gray-600 rounded text-white placeholder-gray-400 focus:border-amber-500 focus:ring-1 focus:ring-amber-500"
                    />
                  </div>
                  
                  <div>
                    <Label.Root class="block text-sm font-medium text-gray-300 mb-1">Password</Label.Root>
                    <Input.Root
                      bind:value={authForm.password}
                      type="password"
                      placeholder="••••••••"
                      required={authMode === 'login'}
                      minlength={authMode === 'register' ? 8 : undefined}
                      class="w-full px-3 py-2 bg-black/40 border border-gray-600 rounded text-white placeholder-gray-400 focus:border-amber-500 focus:ring-1 focus:ring-amber-500"
                    />
                  </div>
                  
                  {#if authMode === 'register'}
                    <div>
                      <Label.Root class="block text-sm font-medium text-gray-300 mb-1">Role</Label.Root>
                      <Select.Root bind:selected={authForm.role}>
                        <Select.Trigger class="w-full px-3 py-2 bg-black/40 border border-gray-600 rounded text-white focus:border-amber-500 focus:ring-1 focus:ring-amber-500">
                          {authForm.role === 'attorney' ? '⚖️ Attorney' : 
                           authForm.role === 'paralegal' ? '📋 Paralegal' :
                           authForm.role === 'investigator' ? '🔍 Investigator' : '👤 User'}
                        </Select.Trigger>
                        <Select.Content class="bg-gray-900 border border-gray-600 rounded">
                          <Select.Item value="user" class="p-2 text-white hover:bg-amber-600/20">👤 User</Select.Item>
                          <Select.Item value="attorney" class="p-2 text-white hover:bg-amber-600/20">⚖️ Attorney</Select.Item>
                          <Select.Item value="paralegal" class="p-2 text-white hover:bg-amber-600/20">📋 Paralegal</Select.Item>
                          <Select.Item value="investigator" class="p-2 text-white hover:bg-amber-600/20">🔍 Investigator</Select.Item>
                        </Select.Content>
                      </Select.Root>
                    </div>
                  {/if}
                  
                  {#if authMode === 'login'}
                    <div class="flex items-center">
                      <input
                        type="checkbox"
                        bind:checked={authForm.rememberMe}
                        class="mr-2 rounded bg-black/40 border-gray-600 text-amber-500 focus:ring-amber-500"
                      />
                      <Label.Root class="text-sm text-gray-300">Remember me</Label.Root>
                    </div>
                  {/if}
                  
                  {#if authError}
                    <div class="p-3 bg-red-900/30 border border-red-500/50 rounded text-red-300 text-sm">
                      {authError}
                    </div>
                  {/if}
                  
                  <div class="flex gap-2">
                    <Button.Root
                      type="submit"
                      disabled={authLoading}
                      class="flex-1 px-4 py-2 bg-amber-600 hover:bg-amber-700 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {authLoading ? '⏳ Processing...' : authMode === 'login' ? '🔑 Login' : '📝 Register'}
                    </Button.Root>
                    
                    <Button.Root
                      type="button"
                      on:on:on:click={() => {
                        authMode = authMode === 'login' ? 'register' : 'login';
                        authError = null;
                      }}
                      class="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded"
                    >
                      {authMode === 'login' ? 'Register' : 'Login'}
                    </Button.Root>
                  </div>
                </form>
                
                <Button.Root
                  on:on:on:click={() => showAuthDialog = false}
                  class="absolute top-2 right-2 p-2 text-gray-400 hover:text-white"
                >
                  ✕
                </Button.Root>
              </Dialog.Content>
            </Dialog.Portal>
          </Dialog.Root>
          
          <!-- Settings Popover -->
          <Popover.Root bind:open={showSettings}>
            <Button.Root
              on:on:on:click={() => showSettings = true}
              class="px-4 py-2 bg-gray-600/20 border border-gray-500/40 text-gray-200 rounded hover:bg-gray-600/30"
            >
              ⚙️ Settings
            </Button.Root>
            <Popover.Content class="p-4 bg-gray-900 border border-amber-500/50 rounded-lg min-w-[300px]">
              <h3 class="font-semibold text-amber-300 mb-3">Testing Configuration</h3>
              
              <div class="space-y-4">
                <div>
                  <Label.Root class="text-sm text-gray-300">Batch Size</Label.Root>
                  <Slider.Root bind:value={batchSize} min={1} max={50} step={1} class="mt-1">
                    <Slider.Track class="h-2 bg-gray-700 rounded">
                      <Slider.Range class="h-2 bg-blue-500 rounded" />
                    </Slider.Track>
                    <Slider.Thumb class="w-4 h-4 bg-blue-400 rounded-full border-2 border-blue-600" />
                  </Slider.Root>
                  <div class="text-xs text-gray-400 mt-1">Current: {batchSize}</div>
                </div>
                
                <div>
                  <Label.Root class="text-sm text-gray-300">Test Timeout (ms)</Label.Root>
                  <Slider.Root bind:value={testTimeout} min={1000} max={30000} step={1000} class="mt-1">
                    <Slider.Track class="h-2 bg-gray-700 rounded">
                      <Slider.Range class="h-2 bg-green-500 rounded" />
                    </Slider.Track>
                    <Slider.Thumb class="w-4 h-4 bg-green-400 rounded-full border-2 border-green-600" />
                  </Slider.Root>
                  <div class="text-xs text-gray-400 mt-1">Current: {testTimeout}ms</div>
                </div>
                
                <div class="flex items-center space-x-2">
                  <Toggle.Root bind:pressed={enableParallelTesting} class="text-sm text-gray-300 data-[state=on]:text-green-300">
                    Parallel Testing
                  </Toggle.Root>
                </div>
              </div>
            </Popover.Content>
          </Popover.Root>
        </Toolbar.Root>
      </div>
    </div>
    
    <!-- Enhanced Testing Progress with bits-ui Progress -->
    {#if testingMode || bulkOperationProgress > 0}
      <Card class="p-6 bg-gradient-to-r from-blue-900/50 to-purple-900/50 border-2 border-cyan-500/30">
        <div class="text-center">
          <h3 class="text-xl font-bold text-cyan-300 mb-4">
            {testingMode ? '🔄 ROUTE TESTING IN PROGRESS' : '🎯 BULK OPERATION PROGRESS'}
          </h3>
          
          <div class="mb-4">
            <Progress.Root 
              value={testingMode ? (testingProgress / totalTests) * 100 : bulkOperationProgress}
              max={100}
              class="h-4 bg-gray-800 rounded-full overflow-hidden"
            >
              <Progress.Indicator 
                class="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-300" 
                style="transform: translateX(-{100 - (testingMode ? (testingProgress / totalTests) * 100 : bulkOperationProgress)}%)"
              />
            </Progress.Root>
          </div>
          
          <div class="space-y-2">
            <p class="text-cyan-200">
              {#if testingMode}
                Testing: {currentlyTesting || 'Initializing...'} ({testingProgress}/{totalTests})
              {:else}
                Bulk operation: {Math.round(bulkOperationProgress)}% complete
              {/if}
            </p>
            
            {#if testingMode}
              <div class="flex justify-center gap-4 text-sm">
                <span class="text-green-300">✅ {Object.values(testResults).filter(r => r === 'success').length}</span>
                <span class="text-red-300">❌ {Object.values(testResults).filter(r => r === 'error').length}</span>
                <span class="text-yellow-300">⏱️ {Object.values(testResults).filter(r => r === 'timeout').length}</span>
              </div>
            {/if}
          </div>
        </div>
      </Card>
    {/if}

    <!-- Enhanced Controls with bits-ui v2 Components -->
    <Card class="p-6">
      <div class="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
        <!-- Enhanced Search with Combobox -->
        <div class="flex-1 max-w-md">
          <Label.Root class="block text-sm font-medium text-yorha-text-accent mb-2">
            🔍 Intelligent Route Search
          </Label.Root>
          <Combobox.Root bind:inputValue={searchComboValue} bind:value={searchValue}>
            <Combobox.Input
              placeholder="Search routes with auto-complete..."
              class="w-full bg-yorha-bg-secondary border border-yorha-text-muted p-3 rounded text-yorha-text-primary placeholder-yorha-text-muted focus:border-yorha-secondary focus:ring-2 focus:ring-yorha-secondary/20"
            />
            <Combobox.Content class="bg-gray-900 border border-amber-500/50 rounded-lg mt-1 max-h-48 overflow-y-auto">
              {#each filteredRoutes.slice(0, 8) as route}
                <Combobox.Item
                  value={route.route}
                  class="p-2 text-white hover:bg-amber-600/20 cursor-pointer flex items-center gap-2"
                >
                  <span>{route.icon}</span>
                  <span class="font-medium">{route.label}</span>
                  <code class="text-xs text-cyan-300 ml-auto">{route.route}</code>
                </Combobox.Item>
              {/each}
            </Combobox.Content>
          </Combobox.Root>
        </div>

        <!-- Category Filter with Select -->
        <div>
          <Label.Root class="block text-sm font-medium text-yorha-text-accent mb-2">
            📂 Category Filter
          </Label.Root>
          <Select.Root bind:selected={categoryValue}>
            <Select.Trigger class="min-w-[150px] bg-yorha-bg-secondary border border-yorha-text-muted p-3 rounded text-yorha-text-primary focus:border-yorha-secondary focus:ring-2 focus:ring-yorha-secondary/20">
              {categoryValue === 'all' ? 'All Categories' : 
               categoryValue === 'available' ? 'Available Only' :
               categoryValue === 'configured' ? 'Configured' :
               categoryValue === 'discovered' ? 'Discovered' :
               categoryValue.charAt(0).toUpperCase() + categoryValue.slice(1)}
            </Select.Trigger>
            <Select.Content class="bg-gray-900 border border-amber-500/50 rounded-lg">
              <Select.Item value="all" class="p-2 text-white hover:bg-amber-600/20">All Categories</Select.Item>
              <Select.Item value="available" class="p-2 text-white hover:bg-amber-600/20">Available Only</Select.Item>
              <Select.Item value="configured" class="p-2 text-white hover:bg-amber-600/20">Configured</Select.Item>
              <Select.Item value="discovered" class="p-2 text-white hover:bg-amber-600/20">Discovered</Select.Item>
              <Separator.Root class="my-1 h-px bg-gray-600" />
              <Select.Item value="main" class="p-2 text-white hover:bg-amber-600/20">🏠 Main</Select.Item>
              <Select.Item value="demo" class="p-2 text-white hover:bg-amber-600/20">🎯 Demo</Select.Item>
              <Select.Item value="ai" class="p-2 text-white hover:bg-amber-600/20">🤖 AI</Select.Item>
              <Select.Item value="legal" class="p-2 text-white hover:bg-amber-600/20">⚖️ Legal</Select.Item>
              <Select.Item value="dev" class="p-2 text-white hover:bg-amber-600/20">🔧 Development</Select.Item>
              <Select.Item value="admin" class="p-2 text-white hover:bg-amber-600/20">⚙️ Admin</Select.Item>
            </Select.Content>
          </Select.Root>
        </div>

        <!-- Sort with NavigationMenu -->
        <div>
          <Label.Root class="block text-sm font-medium text-yorha-text-accent mb-2">
            📊 Sort Options
          </Label.Root>
          <NavigationMenu.Root class="relative">
            <NavigationMenu.List class="flex">
              <NavigationMenu.Item>
                <NavigationMenu.Trigger class="px-4 py-2 bg-yorha-bg-secondary border border-yorha-text-muted rounded text-yorha-text-primary hover:bg-yorha-bg-tertiary focus:border-yorha-secondary">
                  {sortValue === 'name' ? '🔤 Name' : 
                   sortValue === 'category' ? '📁 Category' : 
                   sortValue === 'status' ? '📈 Status' : 'Sort By'} ▼
                </NavigationMenu.Trigger>
                <NavigationMenu.Content class="absolute top-full left-0 mt-2 bg-gray-900 border border-amber-500/50 rounded-lg shadow-xl min-w-[120px]">
                  <div class="p-2">
                    <Button.Root
                      on:on:on:click={() => sortValue = 'name'}
                      class="w-full text-left p-2 text-white hover:bg-amber-600/20 rounded {sortValue === 'name' ? 'bg-amber-600/30' : ''}"
                    >
                      🔤 Name
                    </Button.Root>
                    <Button.Root
                      on:on:on:click={() => sortValue = 'category'}
                      class="w-full text-left p-2 text-white hover:bg-amber-600/20 rounded {sortValue === 'category' ? 'bg-amber-600/30' : ''}"
                    >
                      📁 Category
                    </Button.Root>
                    <Button.Root
                      on:on:on:click={() => sortValue = 'status'}
                      class="w-full text-left p-2 text-white hover:bg-amber-600/20 rounded {sortValue === 'status' ? 'bg-amber-600/30' : ''}"
                    >
                      📈 Status
                    </Button.Root>
                  </div>
                </NavigationMenu.Content>
              </NavigationMenu.Item>
            </NavigationMenu.List>
          </NavigationMenu.Root>
        </div>
      </div>
    </Card>

    <!-- Statistics -->
    <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
      {#each Object.entries(routeCategories) as [category, info]}
        {@const count = allAvailableRoutes.filter(r => r.category === category).length}
        {#if count > 0}
          <Card class="p-4 text-center">
            <div class="text-2xl mb-2" style="color: {info.color}">{info.icon}</div>
            <div class="text-lg font-semibold text-yorha-text-primary">{count}</div>
            <div class="text-sm text-yorha-text-secondary">{info.label}</div>
          </Card>
        {/if}
      {/each}
    </div>

    <!-- Enhanced Routes Grid with N64 3D Gaming Components -->
    <div class={viewMode === 'grid' 
      ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4" 
      : "space-y-3"
    }>
      {#each filteredRoutes as route}
        {@const testResult = testResults[route.route]}
        
        <Card class="p-4 relative overflow-hidden transition-all duration-300 hover:scale-[1.02] 
                   {gamingMode ? 'bg-gradient-to-br from-gray-900 to-gray-800 border-2 border-amber-500/20' 
                                : 'bg-white dark:bg-gray-800'}
                   {testResult === 'success' ? 'border-green-500/50 bg-green-900/10' 
                   : testResult === 'error' ? 'border-red-500/50 bg-red-900/10'
                   : testResult === 'timeout' ? 'border-yellow-500/50 bg-yellow-900/10' : ''}">
          
          <!-- N64-style scanline effect -->
          {#if gamingMode}
            <div class="absolute inset-0 pointer-events-none opacity-10"
                 style="background: repeating-linear-gradient(
                   0deg,
                   transparent,
                   transparent 2px,
                   rgba(255, 255, 255, 0.1) 2px,
                   rgba(255, 255, 255, 0.1) 4px
                 )">
            </div>
          {/if}
          
          <div class="relative z-10">
            <div class="flex items-start justify-between mb-3">
              <div class="flex items-center gap-2">
                <span class="text-xl {gamingMode ? 'drop-shadow-glow' : ''}">{route.icon}</span>
                <span class="text-sm font-medium uppercase tracking-wide
                           {gamingMode ? 'text-amber-300' : 'text-gray-600 dark:text-gray-300'}">
                  {route.category}
                </span>
              </div>

              <div class="flex gap-1 flex-wrap">
                {#if testResult}
                  <Badge variant="outline" class="text-xs
                    {testResult === 'success' ? 'bg-green-500/20 text-green-300 border-green-500/30' :
                     testResult === 'error' ? 'bg-red-500/20 text-red-300 border-red-500/30' :
                     testResult === 'timeout' ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30' :
                     'bg-blue-500/20 text-blue-300 border-blue-500/30'}">
                    {testResult.toUpperCase()}
                  </Badge>
                {/if}
                
                {#if route.available}
                  <Badge variant="outline" class="text-xs {getRouteStatusColor(route.status)}">
                    {route.status}
                  </Badge>
                {:else}
                  <Badge variant="destructive" class="text-xs">
                    Missing
                  </Badge>
                {/if}

                {#if route.type === 'discovered'}
                  <Badge variant="secondary" class="text-xs">
                    Discovered
                  </Badge>
                {/if}
              </div>
            </div>

            <h3 class="text-lg font-semibold mb-2 leading-tight
                     {gamingMode ? 'text-amber-100' : 'text-gray-900 dark:text-gray-100'}">
              {route.label}
            </h3>

            <code class="block text-xs p-2 rounded mb-3 font-mono
                       {gamingMode ? 'text-cyan-300 bg-black/30 border border-cyan-500/20' 
                                  : 'text-blue-600 bg-gray-100 dark:bg-gray-700 dark:text-blue-400'}">
              {route.route}
            </code>

            <p class="text-sm mb-4 line-clamp-2
                    {gamingMode ? 'text-amber-200/80' : 'text-gray-600 dark:text-gray-300'}">
              {route.description}
            </p>

            {#if route.tags && route.tags.length > 0}
              <div class="flex flex-wrap gap-1 mb-4">
                {#each route.tags.slice(0, 3) as tag}
                  <span class="text-xs px-2 py-1 rounded
                             {gamingMode ? 'bg-purple-900/50 text-purple-200 border border-purple-500/30' 
                                        : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300'}">
                    {tag}
                  </span>
                {/each}
                {#if route.tags.length > 3}
                  <span class="text-xs {gamingMode ? 'text-amber-300/70' : 'text-gray-500'}">
                    +{route.tags.length - 3}
                  </span>
                {/if}
              </div>
            {/if}

            <!-- Enhanced Navigation with bits-ui v2 Components -->
            <div class="flex gap-2 items-center">
              <!-- Route Selection Checkbox -->
              <Tooltip.Root>
                <Tooltip.Trigger asChild let:builder>
                  <Button.Root
                    builders={[builder]}
                    on:on:on:click={() => toggleRouteSelection(route.route)}
                    class="p-2 {selectedRoutes.includes(route.route) ? 'bg-blue-600/30 border-blue-400' : 'bg-gray-600/20 border-gray-500'} border rounded"
                  >
                    {selectedRoutes.includes(route.route) ? '☑️' : '☐'}
                  </Button.Root>
                </Tooltip.Trigger>
                <Tooltip.Content class="bg-black/90 text-white p-2 rounded border border-amber-500/50">
                  {selectedRoutes.includes(route.route) ? 'Remove from selection' : 'Add to selection'}
                </Tooltip.Content>
              </Tooltip.Root>
              
              <!-- Favorite Toggle -->
              <Tooltip.Root>
                <Tooltip.Trigger asChild let:builder>
                  <Button.Root
                    builders={[builder]}
                    on:on:on:click={() => toggleFavorite(route.route)}
                    class="p-2 {favoriteRoutes.includes(route.route) ? 'text-yellow-400' : 'text-gray-400'} hover:text-yellow-300"
                  >
                    {favoriteRoutes.includes(route.route) ? '⭐' : '☆'}
                  </Button.Root>
                </Tooltip.Trigger>
                <Tooltip.Content class="bg-black/90 text-white p-2 rounded border border-amber-500/50">
                  {favoriteRoutes.includes(route.route) ? 'Remove from favorites' : 'Add to favorites'}
                </Tooltip.Content>
              </Tooltip.Root>
              
              {#if gamingMode}
                <N643DButton
                  on:on:on:click={() => navigateToRoute(route.route)}
                  disabled={!route.available}
                  variant={route.available ? 'primary' : 'secondary'}
                  size="small"
                  class="flex-1"
                  enableParticles={route.available}
                  enableLighting={true}
                  materialType="pbr"
                >
                  {route.available ? '🚀 NAVIGATE' : '❌ UNAVAILABLE'}
                </N643DButton>
                
                {#if route.available}
                  <NES8BitButton
                    on:on:on:click={() => testRoute(route.route).then(result => testResults[route.route] = result)}
                    disabled={testingMode}
                    variant="info"
                    size="small"
                    class="px-3"
                  >
                    TEST
                  </NES8BitButton>
                {/if}
              {:else}
                <!-- bits-ui Navigation Button -->
                <Button.Root
                  on:on:on:click={() => navigateToRoute(route.route)}
                  disabled={!route.available}
                  class="flex-1 px-4 py-2 rounded font-medium transition-colors
                         {route.available
                           ? 'bg-blue-600/80 text-white hover:bg-blue-700 border border-blue-500'
                           : 'bg-gray-600/30 text-gray-400 cursor-not-allowed border border-gray-600'}"
                >
                  {route.available ? '🚀 Navigate' : '❌ Unavailable'}
                </Button.Root>
                
                {#if route.available}
                  <Button.Root
                    on:on:on:click={async () => {
                      testResults[route.route] = 'pending';
                      const result = await testRoute(route.route);
                      testResults[route.route] = result;
                    }}
                    disabled={testingMode}
                    class="px-3 py-2 bg-green-600/80 text-white rounded hover:bg-green-700 border border-green-500 disabled:opacity-50"
                  >
                    🧪 Test
                  </Button.Root>
                {/if}
                
                {#if route.route.includes('/api/')}
                  <Button.Root
                    on:on:on:click={() => testApiEndpoint(route.route)}
                    class="px-3 py-2 bg-purple-600/80 text-white rounded hover:bg-purple-700 border border-purple-500"
                  >
                    🔌 API
                  </Button.Root>
                {/if}
              {/if}
            </div>
          </div>
        </Card>
      {/each}
    </div>

    {#if filteredRoutes.length === 0}
      <Card class="p-8 text-center">
        <div class="text-4xl mb-4">🔍</div>
        <h3 class="text-xl font-semibold text-yorha-text-primary mb-2">No Routes Found</h3>
        <p class="text-yorha-text-secondary">
          Try adjusting your search or filter criteria.
        </p>
      </Card>
    {/if}

    <!-- Footer Info -->
    <Card class="p-6">
      <h2 class="text-xl font-semibold text-yorha-accent mb-4">System Information</h2>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div>
          <h3 class="font-semibold text-yorha-text-primary mb-2">Route Discovery</h3>
          <ul class="text-sm text-yorha-text-secondary space-y-1">
            <li>• Configured routes: {allRoutes.length}</li>
            <li>• Discovered routes: {discoveredRoutes.length}</li>
            <li>• Available routes: {allAvailableRoutes.filter(r => r.available).length}</li>
            <li>• Missing routes: {allAvailableRoutes.filter(r => !r.available).length}</li>
          </ul>
        </div>

        <div>
          <h3 class="font-semibold text-yorha-text-primary mb-2">Navigation</h3>
          <ul class="text-sm text-yorha-text-secondary space-y-1">
            <li>• Current path: <code class="text-yorha-accent">{$page.url.pathname}</code></li>
            <li>• Route ID: <code class="text-yorha-accent">{$page.route?.id ?? 'unknown'}</code></li>
            <li>• Page params: <code class="text-yorha-accent">{JSON.stringify($page.params)}</code></li>
          </ul>
        </div>

        <div>
          <h3 class="font-semibold text-yorha-text-primary mb-2">🚀 Quick Actions & Recent API Tests</h3>
          <div class="space-y-2">
            {#if gamingMode}
              <N643DButton
                on:on:on:click={() => goto('/')}
                variant="info"
                size="small"
                class="w-full"
                materialType="phong"
              >
                🏠 HOME COMMAND CENTER
              </N643DButton>
              <N643DButton
                on:on:on:click={() => goto('/dev/dynamic-routing-test')}
                variant="warning"
                size="small"
                class="w-full"
                materialType="phong"
              >
                🛣️ ROUTING TEST SUITE
              </N643DButton>
            {:else}
              <Button.Root
                on:on:on:click={() => goto('/')}
                class="w-full px-4 py-2 bg-blue-600/80 text-white rounded hover:bg-blue-700 transition-colors border border-blue-500"
              >
                🏠 Go Home
              </Button.Root>
              <Button.Root
                on:on:on:click={() => goto('/dev/dynamic-routing-test')}
                class="w-full px-4 py-2 bg-green-600/80 text-white rounded hover:bg-green-700 transition-colors border border-green-500"
              >
                🛣️ Routing Test
              </Button.Root>
              
              <Button.Root
                on:on:on:click={() => fetchSystemHealth()}
                class="w-full px-4 py-2 bg-purple-600/80 text-white rounded hover:bg-purple-700 transition-colors border border-purple-500"
              >
                🔄 Refresh System Health
              </Button.Root>
            {/if}
            
            <!-- Recent API Test Result -->
            {#if apiEndpointTest}
              <div class="mt-4 p-3 bg-black/40 rounded border border-gray-600">
                <h4 class="text-sm font-semibold text-gray-300 mb-2">Last API Test</h4>
                <div class="text-xs space-y-1">
                  <div class="flex justify-between">
                    <span class="text-gray-400">Endpoint:</span>
                    <code class="text-cyan-300">{apiEndpointTest.endpoint}</code>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-gray-400">Status:</span>
                    <span class="{apiEndpointTest.ok ? 'text-green-300' : 'text-red-300'}">
                      {apiEndpointTest.status} {apiEndpointTest.ok ? '✅' : '❌'}
                    </span>
                  </div>
                  <div class="text-xs text-gray-500">
                    {new Date(apiEndpointTest.timestamp).toLocaleString()}
                  </div>
                </div>
              </div>
            {/if}
          </div>
        </div>
      </div>
    </Card>
  </div>
</NES8BitContainer>

<style>
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  
  /* N64 Gaming Enhancement Styles */
  .drop-shadow-glow {
    filter: drop-shadow(0 0 4px currentColor) drop-shadow(0 0 8px currentColor);
  }
  
  /* N64 Error Dialog Styles */
  :global(.n64-error-dialog) {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    animation: fadeIn 0.3s ease-in-out;
  }
  
  :global(.n64-dialog-content) {
    background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
    border: 2px solid #ff6b6b;
    border-radius: 8px;
    padding: 2rem;
    max-width: 400px;
    text-align: center;
    box-shadow: 
      0 10px 30px rgba(0, 0, 0, 0.5),
      inset 0 1px 0 rgba(255, 255, 255, 0.1);
    animation: slideIn 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
  }
  
  :global(.n64-dialog-header) {
    font-size: 1.2rem;
    font-weight: bold;
    color: #ff6b6b;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  
  :global(.n64-dialog-body) {
    color: #ffffff;
    margin-bottom: 1.5rem;
    line-height: 1.5;
  }
  
  :global(.n64-dialog-button) {
    background: linear-gradient(145deg, #ff6b6b, #ff5252);
    border: none;
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 4px;
    font-weight: bold;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 0 #cc4444;
  }
  
  :global(.n64-dialog-button:hover) {
    background: linear-gradient(145deg, #ff5252, #ff3333);
    transform: translateY(1px);
    box-shadow: 0 3px 0 #cc4444;
  }
  
  :global(.n64-dialog-button:active) {
    transform: translateY(2px);
    box-shadow: 0 2px 0 #cc4444;
  }
  
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  
  @keyframes slideIn {
    from { 
      transform: scale(0.8) translateY(-20px); 
      opacity: 0; 
    }
    to { 
      transform: scale(1) translateY(0); 
      opacity: 1; 
    }
  }
  
  /* Enhanced scanlines effect */
  @keyframes scanlines {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
  }
  
  /* Retro CRT glow effect */
  :global(.crt-glow) {
    text-shadow: 
      0 0 5px currentColor,
      0 0 10px currentColor,
      0 0 15px currentColor;
  }
  
  /* Terminal-style cursor animation */
  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }
  
  .terminal-cursor::after {
    content: '_';
    animation: blink 1s infinite;
  }
  
  /* N64 texture filtering simulation */
  :global(.n64-texture-filter) {
    image-rendering: -webkit-optimize-contrast;
    image-rendering: pixelated;
    filter: contrast(1.1) saturate(1.2);
  }
  
  /* Responsive design for mobile gaming */
  @media (max-width: 768px) {
    :global(.n64-dialog-content) {
      margin: 1rem;
      padding: 1.5rem;
    }
    
    .drop-shadow-glow {
      filter: drop-shadow(0 0 2px currentColor);
    }
  }
  
  /* High contrast mode support */
  @media (prefers-contrast: high) {
    :global(.n64-error-dialog) {
      background: rgba(0, 0, 0, 0.95);
    }
    
    :global(.n64-dialog-content) {
      border-width: 3px;
      box-shadow: none;
    }
  }
  
  /* Reduced motion support */
  @media (prefers-reduced-motion: reduce) {
    :global(.n64-dialog-content),
    :global(.n64-error-dialog) {
      animation: none;
    }
    
    .drop-shadow-glow {
      filter: none;
    }
    
    .terminal-cursor::after {
      animation: none;
    }
  }
</style>