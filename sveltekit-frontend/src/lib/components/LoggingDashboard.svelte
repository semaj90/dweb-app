<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  // Svelte runes are declared globally in src/types/svelte-helpers.d.ts
  import {
    logEntries,
    logStats,
    loggingService,
    type LogEntry,
    type LogLevel,
    type LogFilter
  } from '$lib/services/logging-aggregation-service';
  import { Button } from '$lib/components/ui/button';
  import { Badge } from '$lib/components/ui/badge';

  // Modern Svelte 5 props via $props rune
  let { visible = true, height = '600px' } = $props<{
    visible?: boolean;
    height?: string;
  }>();

  let selectedLevel = $state<LogLevel | 'all'>('all');
  let selectedCategory = $state('all');
  let searchQuery = $state('');
  let autoScroll = $state(true);
  let showDetails = $state(false);
  let selectedEntry = $state<LogEntry | null>(null);

  let filteredEntries = $derived(() => $logEntries.filter((entry: LogEntry) => {
    const matchesLevel = selectedLevel === 'all' || entry.level === selectedLevel;
    const matchesCategory = selectedCategory === 'all' || entry.category === selectedCategory;
    const matchesSearch = !searchQuery ||
      entry.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.category.toLowerCase().includes(searchQuery.toLowerCase());

    return matchesLevel && matchesCategory && matchesSearch;
  }));

  let categories = $derived(() => Array.from(new Set($logEntries.map((e: LogEntry) => e.category))).sort());
  let stats = $derived(() => $logStats);
  let logContainer = $state<HTMLDivElement | null>(null);
  let refreshInterval: ReturnType<typeof setInterval> | null = null;

  onMount(() => {
    // Auto-scroll to bottom when new entries arrive
    if (autoScroll) {
      scrollToBottom();
    }

    // Refresh dashboard periodically
  refreshInterval = setInterval(() => {
      if (autoScroll) {
        scrollToBottom();
      }
    }, 1000);
  });

  onDestroy(() => {
    if (refreshInterval) clearInterval(refreshInterval);
  });

  function scrollToBottom() {
    if (logContainer) {
      logContainer.scrollTop = logContainer.scrollHeight;
    }
  }

  function getLevelColor(level: LogLevel): string {
    const colors = {
      debug: 'bg-gray-100 text-gray-600',
      info: 'bg-blue-100 text-blue-600',
      warn: 'bg-yellow-100 text-yellow-600',
      error: 'bg-red-100 text-red-600',
      fatal: 'bg-red-200 text-red-800'
    };
    return colors[level] || 'bg-gray-100 text-gray-600';
  }

  function getLevelIcon(level: LogLevel): string {
    const icons = {
      debug: '🐛',
      info: 'ℹ️',
      warn: '⚠️',
      error: '❌',
      fatal: '💀'
    };
    return icons[level] || 'ℹ️';
  }

  function getCategoryIcon(category: string): string {
    const icons = {
      system: '⚙️',
      auth: '🔐',
      api: '🔗',
      ai: '🤖',
      database: '🗄️',
      frontend: '🎨',
      backend: '🖥️',
      security: '🛡️',
      console: '📟'
    };
    return icons[category] || '📋';
  }

  function formatTimestamp(timestamp: number): string {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3
    });
  }

  function formatData(data: any): string {
    if (!data) return '';
    try {
      return JSON.stringify(data, null, 2);
    } catch {
      return String(data);
    }
  }

  function clearLogs() {
    if (confirm('Are you sure you want to clear all logs?')) {
      // This would require a method in the logging service
      location.reload(); // Simple solution for now
    }
  }

  function exportLogs() {
    const filter: LogFilter = {
      category: selectedCategory !== 'all' ? [selectedCategory] : undefined,
      level: selectedLevel !== 'all' ? [selectedLevel] : undefined
    };

    const exportData = loggingService.exportLogs('json', filter);

    // Create download
    const blob = new Blob([exportData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `logs-${new Date().toISOString().slice(0, 19)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function selectEntry(entry: LogEntry) {
    selectedEntry = entry;
    showDetails = true;
  }
</script>

{#if visible}
  <div class="logging-dashboard bg-gray-900 text-white rounded-lg border border-gray-700" style="height: {height}">
    <!-- Header -->
    <div class="p-4 border-b border-gray-700">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-xl font-bold text-green-400 flex items-center gap-2">
          📊 Logging Dashboard
          <span class="text-sm text-gray-400">({stats.totalEntries} entries)</span>
        </h2>

        <div class="flex items-center gap-2">
          <Button size="sm" variant="outline" onclick={exportLogs}>
            📤 Export
          </Button>
          <Button size="sm" variant="outline" onclick={clearLogs}>
            🗑️ Clear
          </Button>
        </div>
      </div>

      <!-- Stats Bar -->
      <div class="grid grid-cols-6 gap-4 mb-4">
        <div class="text-center">
          <div class="text-gray-400 text-xs">Debug</div>
          <div class="text-gray-300 font-mono">{stats.entriesByLevel.debug}</div>
        </div>
        <div class="text-center">
          <div class="text-blue-400 text-xs">Info</div>
          <div class="text-blue-300 font-mono">{stats.entriesByLevel.info}</div>
        </div>
        <div class="text-center">
          <div class="text-yellow-400 text-xs">Warn</div>
          <div class="text-yellow-300 font-mono">{stats.entriesByLevel.warn}</div>
        </div>
        <div class="text-center">
          <div class="text-red-400 text-xs">Error</div>
          <div class="text-red-300 font-mono">{stats.entriesByLevel.error}</div>
        </div>
        <div class="text-center">
          <div class="text-red-400 text-xs">Fatal</div>
          <div class="text-red-300 font-mono">{stats.entriesByLevel.fatal}</div>
  <div class="w-full max-w-4xl max-h-[80vh] overflow-hidden bg-gray-900 text-white border border-gray-700 rounded-lg">
        <div class="text-center">
          <div class="text-green-400 text-xs">Rate</div>
          <div class="text-green-300 font-mono">{stats.avgLogsPerMinute}/min</div>
        </div>
      </div>

      <!-- Filters -->
      <div class="flex gap-4">
        <input
          type="text"
          placeholder="Search logs..."
          bind:value={searchQuery}
          class="flex-1 px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white placeholder-gray-400"
        />

        <select
          bind:value={selectedLevel}
          class="px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white"
        >
          <option value="all">All Levels</option>
          <option value="debug">Debug</option>
          <option value="info">Info</option>
          <option value="warn">Warn</option>
          <option value="error">Error</option>
          <option value="fatal">Fatal</option>
        </select>

        <select
          bind:value={selectedCategory}
          class="px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white"
        >
          <option value="all">All Categories</option>
          {#each categories as category}
            <option value={category}>{category}</option>
          {/each}
        </select>

        <label class="flex items-center gap-2">
          <input
            type="checkbox"
            bind:checked={autoScroll}
            class="w-4 h-4"
          />
          <span class="text-sm">Auto-scroll</span>
        </label>
      </div>
    </div>

    <!-- Log Entries -->
    <div
      class="flex-1 overflow-y-auto p-2"
      bind:this={logContainer}
    >
      {#if filteredEntries.length === 0}
        <div class="text-center text-gray-400 mt-8">
          <div class="text-4xl mb-2">📝</div>
          <div>No log entries match your filters</div>
        </div>
      {:else}
        <div class="space-y-1">
          {#each filteredEntries as entry (entry.id)}
            <button
              type="button"
              class="log-entry group w-full text-left hover:bg-gray-800 p-2 rounded cursor-pointer transition-colors duration-150"
              on:click={() => selectEntry(entry)}
            >
              <div class="flex items-start gap-3">
                <!-- Timestamp -->
                <div class="text-xs text-gray-400 font-mono min-w-[80px]">
                  {formatTimestamp(entry.timestamp)}
                </div>

                <!-- Level Badge -->
                <div class="min-w-[60px]">
                  <Badge class={getLevelColor(entry.level) + ' text-xs'}>
                    {getLevelIcon(entry.level)} {entry.level.toUpperCase()}
                  </Badge>
                </div>

                <!-- Category -->
                <div class="text-xs text-gray-300 min-w-[80px] flex items-center gap-1">
                  <span>{getCategoryIcon(entry.category)}</span>
                  <span>{entry.category}</span>
                </div>

                <!-- Message -->
                <div class="flex-1 text-sm">
                  <span class="text-white">{entry.message}</span>

                  {#if entry.service}
                    <span class="text-gray-400 text-xs ml-2">({entry.service})</span>
                  {/if}

                  {#if entry.data}
                    <div class="text-xs text-gray-400 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      📎 Has additional data
                    </div>
                  {/if}

                  {#if entry.error}
                    <div class="text-xs text-red-400 mt-1">
                      🐛 {entry.error.message}
                    </div>
                  {/if}
                </div>

                <!-- Actions -->
                <div class="opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button size="sm" variant="ghost" on:click={(e) => { e.stopPropagation(); selectEntry(entry); }}>
                    👁️
                  </Button>
                </div>
              </div>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </div>

  <!-- Details Modal -->
  {#if showDetails && selectedEntry}
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
    <div class="w-full max-w-4xl max-h-[80vh] overflow-hidden bg-gray-900 text-white border border-gray-700 rounded-lg">
      <div class="p-6">
      <div class="p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-xl font-bold text-green-400 flex items-center gap-2">
            {getLevelIcon(selectedEntry.level)} Log Entry Details
          </h3>
          <Button variant="ghost" on:click={() => showDetails = false}>
            ✕
          </Button>
        </div>

        <!-- Entry Details -->
        <div class="space-y-4 overflow-y-auto max-h-96">
  </div>
            <div role="group" aria-label="Timestamp">
              <div class="text-sm text-gray-400">Timestamp</div>
              <div class="font-mono text-white">
                {new Date(selectedEntry.timestamp).toISOString()}
              </div>
            </div>

            <div role="group" aria-label="Level">
              <div class="text-sm text-gray-400">Level</div>
              <div>
                <Badge class={getLevelColor(selectedEntry.level)}>
                  {selectedEntry.level.toUpperCase()}
                </Badge>
              </div>
            </div>

            <div role="group" aria-label="Category">
              <div class="text-sm text-gray-400">Category</div>
              <div class="text-white flex items-center gap-2">
                {getCategoryIcon(selectedEntry.category)} {selectedEntry.category}
              </div>
            </div>

            <div role="group" aria-label="Entry ID">
              <div class="text-sm text-gray-400">Entry ID</div>
              <div class="font-mono text-white text-sm">{selectedEntry.id}</div>
            </div>

            {#if selectedEntry.service}
              <div role="group" aria-label="Service">
                <div class="text-sm text-gray-400">Service</div>
                <div class="text-white">{selectedEntry.service}</div>
              </div>
            {/if}

            {#if selectedEntry.userId}
              <div role="group" aria-label="User ID">
                <div class="text-sm text-gray-400">User ID</div>
                <div class="font-mono text-white text-sm">{selectedEntry.userId}</div>
              </div>
            {/if}

            {#if selectedEntry.sessionId}
              <div role="group" aria-label="Session ID">
                <div class="text-sm text-gray-400">Session ID</div>
                <div class="font-mono text-white text-sm">{selectedEntry.sessionId}</div>
              </div>
            {/if}

            {#if selectedEntry.requestId}
              <div role="group" aria-label="Request ID">
                <div class="text-sm text-gray-400">Request ID</div>
                <div class="font-mono text-white text-sm">{selectedEntry.requestId}</div>
              </div>
            {/if}
          </div>

          <!-- Message -->
          <div>
            <div class="text-sm text-gray-400">Message</div>
            <div class="bg-gray-800 p-3 rounded border border-gray-600">
              <code class="text-white whitespace-pre-wrap">{selectedEntry.message}</code>
            </div>
          </div>

          <!-- Data -->
          {#if selectedEntry.data}
            <div>
              <div class="text-sm text-gray-400">Data</div>
              <div class="bg-gray-800 p-3 rounded border border-gray-600 overflow-x-auto">
                <pre class="text-green-300 text-sm"><code>{formatData(selectedEntry.data)}</code></pre>
              </div>
            </div>
          {/if}

          <!-- Error -->
          {#if selectedEntry.error}
            <div>
              <div class="text-sm text-gray-400">Error</div>
              <div class="bg-red-900 p-3 rounded border border-red-700">
                <div class="text-red-300 font-bold">{selectedEntry.error.message}</div>
                {#if selectedEntry.error.stack}
                  <pre class="text-red-200 text-xs mt-2 overflow-x-auto"><code>{selectedEntry.error.stack}</code></pre>
                {/if}
              </div>
            </div>
          {/if}

          <!-- Meta -->
          {#if selectedEntry.meta}
            <div>
              <div class="text-sm text-gray-400">Metadata</div>
              <div class="bg-gray-800 p-3 rounded border border-gray-600 overflow-x-auto">
                <pre class="text-blue-300 text-sm"><code>{formatData(selectedEntry.meta)}</code></pre>
              </div>
            </div>
          {/if}

          <!-- Tags -->
          {#if selectedEntry.tags && selectedEntry.tags.length > 0}
            <div>
              <div class="text-sm text-gray-400">Tags</div>
              <div class="flex flex-wrap gap-2 mt-2">
                {#each selectedEntry.tags as tag}
                  <Badge variant="outline" class="text-xs">{tag}</Badge>
                {/each}
              </div>
            </div>
          {/if}
        </div>

        <!-- Actions -->
        <div class="flex gap-2 mt-6">
          <Button
            variant="outline"
            on:click={() => navigator.clipboard.writeText(JSON.stringify(selectedEntry, null, 2))}
          >
            📋 Copy JSON
          </Button>
          <Button variant="outline" on:click={() => showDetails = false}>
            Close
          </Button>
        </div>
      </div>
    </div>
  {/if}

<style>
  .logging-dashboard {
    display: flex;
    flex-direction: column;
    font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
  }

  .log-entry {
    font-size: 0.875rem;
    line-height: 1.25rem;
  }

  pre {
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  :global(.logging-dashboard .log-entry:nth-child(even)) {
    background-color: rgba(255, 255, 255, 0.02);
  }

  /* Scrollbar styling */
  .overflow-y-auto::-webkit-scrollbar {
    width: 8px;
  }

  .overflow-y-auto::-webkit-scrollbar-track {
    background: #1f2937;
  }

  .overflow-y-auto::-webkit-scrollbar-thumb {
    background: #4b5563;
    border-radius: 4px;
  }

  .overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #6b7280;
  }
</style>