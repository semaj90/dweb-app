<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { $derived } from 'svelte';
  import { 
    shortcuts, 
    shortcutCategories, 
    remoteCommands, 
    isRemoteConnected,
    formatShortcut,
    keyboardShortcutsService,
    setKeyboardContext,
    type KeyboardShortcut 
  } from '$lib/services/keyboard-shortcuts-service';
  import { Button } from '$lib/components/ui/button';
  import { Card } from "$lib/components/ui/card";
  import { Badge } from '$lib/components/ui/badge';
  import { Switch } from '$lib/components/ui/switch';
  
  let { visible = $bindable() } = $props(); // false;
  let { context = $bindable() } = $props(); // string[] = ['global'];
let searchQuery = $state('');
let selectedCategory = $state('all');
let showRemoteOnly = $state(false);
  
  let filteredShortcuts = $derived($shortcuts.filter(shortcut => {
    const matchesSearch = !searchQuery || 
      shortcut.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      shortcut.key.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesCategory = selectedCategory === 'all' || shortcut.category === selectedCategory;
    
    const matchesRemote = !showRemoteOnly || shortcut.remote;
    
    return matchesSearch && matchesCategory && matchesRemote;
  });
  
  let categories = $derived($shortcutCategories);
  let recentCommands = $derived($remoteCommands.slice(-10).reverse());
let helpModalVisible = $state(false);
let remoteStatusVisible = $state(false);
  
  onMount(() => {
    // Set context for shortcuts service
    setKeyboardContext(context);
    
    // Listen for custom events
    const handleToggleHelp = () => {
      helpModalVisible = !helpModalVisible;
    };
    
    const handleShowRemoteStatus = () => {
      remoteStatusVisible = !remoteStatusVisible;
    };
    
    document.addEventListener('show-keyboard-help', handleToggleHelp);
    document.addEventListener('show-remote-status', handleShowRemoteStatus);
    
    return () => {
      document.removeEventListener('show-keyboard-help', handleToggleHelp);
      document.removeEventListener('show-remote-status', handleShowRemoteStatus);
    };
  });
  
  function toggleShortcut(shortcut: KeyboardShortcut) {
    if (shortcut.enabled) {
      keyboardShortcutsService.disableShortcut(shortcut.id);
    } else {
      keyboardShortcutsService.enableShortcut(shortcut.id);
    }
  }
  
  function executeShortcut(shortcut: KeyboardShortcut) {
    keyboardShortcutsService.executeRemoteCommand({
      id: crypto.randomUUID(),
      command: shortcut.id,
      args: {},
      source: 'api',
      timestamp: Date.now()
    });
  }
  
  function getCategoryIcon(category: string): string {
    const icons = {
      navigation: '🧭',
      ai: '🤖',
      cases: '📁',
      evidence: '📋',
      system: '⚙️',
      remote: '🎮'
    };
    return icons[category] || '📌';
  }
  
  function getSourceIcon(source: string): string {
    const icons = {
      keyboard: '⌨️',
      api: '🔗',
      websocket: '📡',
      voice: '🎤'
    };
    return icons[source] || '❓';
  }
</script>

<!-- Help Modal -->
{#if helpModalVisible}
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
    <Card class="w-full max-w-4xl max-h-[80vh] overflow-hidden bg-gray-900 text-white border-gray-700">
      <div class="p-6">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-2xl font-bold text-green-400">⌨️ Keyboard Shortcuts</h2>
          <Button variant="ghost" on:on:click={() => helpModalVisible = false}>
            ✕
          </Button>
        </div>
        
        <!-- Search and filters -->
        <div class="flex gap-4 mb-4">
          <input
            type="text"
            placeholder="Search shortcuts..."
            bind:value={searchQuery}
            class="flex-1 px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white placeholder-gray-400"
          />
          
          <select 
            bind:value={selectedCategory}
            class="px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white"
          >
            <option value="all">All Categories</option>
            {#each categories as category}
              <option value={category.id}>{category.name}</option>
            {/each}
          </select>
          
          <label class="flex items-center gap-2">
            <Switch bind:checked={showRemoteOnly} />
            <span class="text-sm">Remote Only</span>
          </label>
        </div>
        
        <!-- Shortcuts list -->
        <div class="overflow-y-auto max-h-96">
          {#each categories as category}
            {#if selectedCategory === 'all' || selectedCategory === category.id}
              <div class="mb-6">
                <h3 class="text-lg font-semibold text-yellow-400 mb-3 flex items-center gap-2">
                  {getCategoryIcon(category.id)} {category.name}
                </h3>
                
                <div class="space-y-2">
                  {#each category.shortcuts.filter(s => filteredShortcuts.includes(s)) as shortcut}
                    <div class="flex items-center justify-between p-3 bg-gray-800 rounded-lg border border-gray-700">
                      <div class="flex-1">
                        <div class="flex items-center gap-3">
                          <span class="text-white">{shortcut.description}</span>
                          {#if shortcut.remote}
                            <Badge variant="outline" class="text-xs">Remote</Badge>
                          {/if}
                          {#if shortcut.context}
                            <Badge variant="secondary" class="text-xs">
                              {shortcut.context.join(', ')}
                            </Badge>
                          {/if}
                        </div>
                        <div class="text-sm text-gray-400 mt-1">
                          {formatShortcut(shortcut)}
                        </div>
                      </div>
                      
                      <div class="flex items-center gap-2">
                        <Switch 
                          checked={shortcut.enabled}
                          change={() => toggleShortcut(shortcut)}
                        />
                        <Button 
                          size="sm" 
                          variant="outline"
                          on:on:click={() => executeShortcut(shortcut)}
                          disabled={!shortcut.enabled}
                        >
                          Test
                        </Button>
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
          {/each}
        </div>
        
        <!-- Remote connection status -->
        {#if $isRemoteConnected}
          <div class="mt-4 p-3 bg-green-900 border border-green-700 rounded-lg">
            <div class="flex items-center gap-2">
              <span class="text-green-400">🔗 Connected to remote control</span>
              <Button size="sm" variant="outline" on:on:click={() => remoteStatusVisible = true}>
                View Status
              </Button>
            </div>
          </div>
        {:else}
          <div class="mt-4 p-3 bg-yellow-900 border border-yellow-700 rounded-lg">
            <div class="flex items-center gap-2">
              <span class="text-yellow-400">⚠️ Remote control disconnected</span>
              <Button size="sm" variant="outline" on:on:click={() => keyboardShortcutsService.connectRemote()}>
                Connect
              </Button>
            </div>
          </div>
        {/if}
      </div>
    </Card>
  </div>
{/if}

<!-- Remote Status Modal -->
{#if remoteStatusVisible}
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
    <Card class="w-full max-w-2xl max-h-[80vh] overflow-hidden bg-gray-900 text-white border-gray-700">
      <div class="p-6">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-xl font-bold text-green-400">📡 Remote Control Status</h2>
          <Button variant="ghost" on:on:click={() => remoteStatusVisible = false}>
            ✕
          </Button>
        </div>
        
        <!-- Connection status -->
        <div class="mb-6">
          <div class="flex items-center gap-3 mb-2">
            <div class="w-3 h-3 rounded-full {$isRemoteConnected ? 'bg-green-400' : 'bg-red-400'}"></div>
            <span class="font-medium">
              {$isRemoteConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div class="text-sm text-gray-400">
            Endpoint: ws://localhost:8085/keyboard-remote
          </div>
        </div>
        
        <!-- Recent commands -->
        <div class="mb-4">
          <h3 class="text-lg font-semibold text-yellow-400 mb-3">Recent Commands</h3>
          
          {#if recentCommands.length > 0}
            <div class="space-y-2 overflow-y-auto max-h-64">
              {#each recentCommands as command}
                <div class="flex items-center justify-between p-2 bg-gray-800 rounded border border-gray-700">
                  <div class="flex items-center gap-3">
                    <span>{getSourceIcon(command.source)}</span>
                    <span class="font-medium">{command.command}</span>
                    {#if command.args?.description}
                      <span class="text-gray-400 text-sm">- {command.args.description}</span>
                    {/if}
                  </div>
                  <div class="text-xs text-gray-400">
                    {new Date(command.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              {/each}
            </div>
          {:else}
            <div class="text-gray-400 text-center py-4">
              No recent commands
            </div>
          {/if}
        </div>
        
        <!-- Controls -->
        <div class="flex gap-2">
          {#if $isRemoteConnected}
            <Button variant="outline" on:on:click={() => keyboardShortcutsService.disconnectRemote()}>
              Disconnect
            </Button>
          {:else}
            <Button on:on:click={() => keyboardShortcutsService.connectRemote()}>
              Reconnect
            </Button>
          {/if}
          
          <Button variant="outline" on:on:click={() => $remoteCommands.length && remoteCommands.set([])}>
            Clear History
          </Button>
        </div>
      </div>
    </Card>
  </div>
{/if}

<!-- Floating shortcut indicator (when visible prop is true) -->
{#if visible}
  <div class="fixed bottom-4 right-4 z-40">
    <Card class="p-3 bg-gray-900 border-gray-700 text-white">
      <div class="flex items-center gap-2 text-sm">
        <span>🎹</span>
        <span>Press <kbd class="bg-gray-800 px-1 rounded">Shift</kbd> + <kbd class="bg-gray-800 px-1 rounded">H</kbd> for shortcuts</span>
        {#if $isRemoteConnected}
          <div class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
        {/if}
      </div>
    </Card>
  </div>
{/if}

<style>
  kbd {
    @apply inline-block px-1 py-0.5 text-xs font-mono bg-gray-800 border border-gray-600 rounded;
  }
</style>