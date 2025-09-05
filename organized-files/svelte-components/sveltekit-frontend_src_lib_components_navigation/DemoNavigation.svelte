<!-- Demo Navigation Component -->
<script lang="ts">
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  
  interface NavItem {
    label: string
    href: string
    description: string
    icon: string
    external?: boolean;
  }

  const navItems: NavItem[] = [
    {
      label: 'AI Document Demo',
      href: '/demo/document-ai',
      description: 'Document upload with AI processing',
      icon: '🤖'
    },
    {
      label: 'AI Service Test',
      href: 'http://localhost:8081/test',
      description: 'Go microservice test interface',
      icon: '🧪',
      external: true
    },
    {
      label: 'Service Health',
      href: 'http://localhost:8081/api/health',
      description: 'System health API endpoint',
      icon: '💚',
      external: true
    },
    {
      label: 'Dev Tools',
      href: '/dev/mcp-tools',
      description: 'MCP and development tools',
      icon: '🛠️'
    },
    {
      label: 'Legal AI Cases',
      href: '/cases',
      description: 'Case management interface',
      icon: '⚖️'
    }
  ];

  let isOpen = $state(false);
  
  function toggleNav() {
    isOpen = !isOpen;
  }

  function navigateTo(item: NavItem) {
    if (item.external) {
      window.open(item.href, '_blank');
    } else {
      goto(item.href);
      isOpen = false;
    }
  }

  function isCurrentPage(href: string): boolean {
    return $page.url.pathname === href;
  }
</script>

<div class="demo-navigation">
  <!-- Floating Navigation Button -->
  <button 
    class="nav-toggle fixed top-4 right-4 z-50 w-12 h-12 bg-green-600 hover:bg-green-700 rounded-full shadow-lg transition-all duration-300 flex items-center justify-center text-white font-bold text-lg"
    onclick={toggleNav}
    class:rotate-45={isOpen}
  >
    {isOpen ? '✕' : '🚀'}
  </button>

  <!-- Navigation Panel -->
  {#if isOpen}
    <div class="nav-panel fixed top-16 right-4 z-40 w-80 bg-gray-900 border border-gray-700 rounded-lg shadow-2xl p-6 animate-slideIn">
      <h3 class="text-xl font-bold text-green-400 mb-4 flex items-center gap-2">
        🧭 Demo Navigation
      </h3>
      
      <div class="space-y-3">
        {#each navItems as item}
          <button
            class="nav-item w-full text-left p-3 rounded-lg transition-all duration-200 border border-transparent hover:border-green-500 hover:bg-green-500/10"
            class:active={isCurrentPage(item.href)}
            onclick={() => navigateTo(item)}
          >
            <div class="flex items-start gap-3">
              <div class="text-2xl">{item.icon}</div>
              <div class="flex-1">
                <div class="font-semibold text-white flex items-center gap-2">
                  {item.label}
                  {#if item.external}
                    <span class="text-xs text-gray-400">↗</span>
                  {/if}
                </div>
                <div class="text-sm text-gray-400 mt-1">{item.description}</div>
              </div>
            </div>
          </button>
        {/each}
      </div>

      <!-- Quick Actions -->
      <div class="mt-6 pt-4 border-t border-gray-700">
        <h4 class="text-sm font-semibold text-blue-400 mb-3">🔧 Quick Actions</h4>
        <div class="grid grid-cols-2 gap-2">
          <button 
            class="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
            onclick={() => window.open('http://localhost:8081/api/health', '_blank')}
          >
            💚 Health
          </button>
          <button 
            class="px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm transition-colors"
            onclick={() => navigateTo(navItems[0])}
          >
            🤖 AI Demo
          </button>
          <button 
            class="px-3 py-2 bg-yellow-600 hover:bg-yellow-700 rounded text-sm transition-colors"
            onclick={() => window.open('http://localhost:8081/test', '_blank')}
          >
            🧪 Test UI
          </button>
          <button 
            class="px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded text-sm transition-colors"
            onclick={() => navigateTo({ href: '/dev/mcp-tools', external: false } as NavItem)}
          >
            🛠️ Tools
          </button>
        </div>
      </div>

      <!-- Status Indicators -->
      <div class="mt-4 pt-4 border-t border-gray-700">
        <h4 class="text-sm font-semibold text-green-400 mb-2">📊 Service Status</h4>
        <div class="grid grid-cols-2 gap-2 text-xs">
          <div class="bg-gray-800 rounded p-2 text-center">
            <div class="text-green-400">✅ Ollama</div>
            <div class="text-gray-400">:11434</div>
          </div>
          <div class="bg-gray-800 rounded p-2 text-center">
            <div class="text-green-400">✅ AI Service</div>
            <div class="text-gray-400">:8081</div>
          </div>
          <div class="bg-gray-800 rounded p-2 text-center">
            <div class="text-green-400">✅ SvelteKit</div>
            <div class="text-gray-400">:5175</div>
          </div>
          <div class="bg-gray-800 rounded p-2 text-center">
            <div class="text-blue-400">🔵 PostgreSQL</div>
            <div class="text-gray-400">:5432</div>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Backdrop -->
  {#if isOpen}
    <div 
      class="fixed inset-0 bg-black/50 z-30"
      onclick={toggleNav}
    ></div>
  {/if}
</div>

<style>
  .nav-toggle {
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  }
  
  .nav-panel {
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 64px rgba(0, 0, 0, 0.4);
  }
  
  .nav-item.active {
    background: rgba(34, 197, 94, 0.1);
    border-color: rgb(34, 197, 94);
  }
  
  .rotate-45 {
    transform: rotate(45deg);
  }
  
  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  .animate-slideIn {
    animation: slideIn 0.3s ease-out;
  }
</style>