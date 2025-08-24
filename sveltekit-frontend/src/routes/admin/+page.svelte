<script lang="ts">
  import { onMount } from 'svelte';
  import { currentUser } from '$lib/auth/auth-store';
  import type { PageData } from './$types';
  
  export let data: PageData;
  
  // System metrics and status
  let systemMetrics = {
    totalUsers: 0,
    activeUsers: 0,
    totalCases: 0,
    activeCases: 0,
    totalEvidence: 0,
    storageUsed: '0 GB',
    systemUptime: '0 days',
    aiProcessingQueue: 0,
    lastBackup: 'Never',
    systemHealth: 'Unknown'
  };
  
  let recentActivity: Array<{
    id: string;
    type: string;
    user: string;
    action: string;
    timestamp: string;
    status: 'success' | 'warning' | 'error';
  }> = [];
  
  let isLoadingMetrics = true;
  
  // YoRHa styling classes
  const yorhaClasses = {
    card: 'bg-[#1a1a1a] border border-[#333333] p-4',
    cardHeader: 'text-[#00ff88] text-sm font-bold mb-4 tracking-wider',
    metric: 'text-2xl font-bold mb-1',
    metricLabel: 'text-xs opacity-60',
    button: 'px-4 py-2 border border-[#333333] bg-[#111111] hover:bg-[#2a2a2a] transition-colors text-sm',
    buttonPrimary: 'px-4 py-2 border border-[#00ff88] bg-[#002211] text-[#00ff88] hover:bg-[#003322] transition-colors text-sm',
    buttonDanger: 'px-4 py-2 border border-red-500 bg-red-900 text-red-100 hover:bg-red-800 transition-colors text-sm',
    table: 'w-full border-collapse',
    tableHeader: 'border-b border-[#333333] text-left p-2 text-xs opacity-60',
    tableCell: 'border-b border-[#222222] p-2 text-sm',
    statusSuccess: 'text-[#00ff88]',
    statusWarning: 'text-yellow-500',
    statusError: 'text-red-500'
  };
  
  onMount(async () => {
    await loadSystemMetrics();
    await loadRecentActivity();
  });
  
  async function loadSystemMetrics() {
    try {
      isLoadingMetrics = true;
      
      const response = await fetch('/api/admin/system/metrics', {
        credentials: 'include'
      });
      
      if (response.ok) {
        const data = await response.json();
        systemMetrics = { ...systemMetrics, ...data };
      }
    } catch (error) {
      console.error('Failed to load system metrics:', error);
    } finally {
      isLoadingMetrics = false;
    }
  }
  
  async function loadRecentActivity() {
    try {
      const response = await fetch('/api/admin/audit/recent?limit=10', {
        credentials: 'include'
      });
      
      if (response.ok) {
        const data = await response.json();
        recentActivity = data.activities || [];
      }
    } catch (error) {
      console.error('Failed to load recent activity:', error);
    }
  }
  
  async function performSystemAction(action: string) {
    try {
      const response = await fetch('/api/admin/system/action', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ action }),
        credentials: 'include'
      });
      
      if (response.ok) {
        // Refresh metrics after action
        await loadSystemMetrics();
        await loadRecentActivity();
      }
    } catch (error) {
      console.error('System action failed:', error);
    }
  }
  
  function getStatusIcon(status: string) {
    switch (status) {
      case 'success': return '◈';
      case 'warning': return '⚠';
      case 'error': return '✕';
      default: return '◯';
    }
  }
  
  function getHealthColor(health: string) {
    switch (health.toLowerCase()) {
      case 'excellent': return 'text-[#00ff88]';
      case 'good': return 'text-[#88ff00]';
      case 'warning': return 'text-yellow-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-400';
    }
  }
</script>

<svelte:head>
  <title>Admin Dashboard - Legal AI Platform</title>
</svelte:head>

<!-- Admin Dashboard -->
<div class="space-y-6">
  <!-- Dashboard Header -->
  <div class="flex items-center justify-between">
    <div>
      <h1 class="text-2xl font-bold tracking-wider">SYSTEM DASHBOARD</h1>
      <p class="text-sm opacity-60 mt-1">REAL-TIME SYSTEM MONITORING AND CONTROL</p>
    </div>
    
    <!-- Quick Actions -->
    <div class="flex space-x-2">
      <button 
        on:click={() => performSystemAction('refresh')}
        class={yorhaClasses.button}
      >
        ↻ REFRESH
      </button>
      <button 
        on:click={() => performSystemAction('backup')}
        class={yorhaClasses.buttonPrimary}
      >
        ◈ BACKUP
      </button>
    </div>
  </div>
  
  <!-- System Metrics Grid -->
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
    <!-- Users Metric -->
    <div class={yorhaClasses.card}>
      <div class={yorhaClasses.cardHeader}>USER STATISTICS</div>
      <div class="space-y-3">
        <div>
          <div class={yorhaClasses.metric}>
            {isLoadingMetrics ? '...' : systemMetrics.totalUsers}
          </div>
          <div class={yorhaClasses.metricLabel}>TOTAL USERS</div>
        </div>
        <div>
          <div class="text-lg font-bold text-[#00ff88]">
            {isLoadingMetrics ? '...' : systemMetrics.activeUsers}
          </div>
          <div class={yorhaClasses.metricLabel}>ACTIVE SESSIONS</div>
        </div>
      </div>
    </div>
    
    <!-- Cases Metric -->
    <div class={yorhaClasses.card}>
      <div class={yorhaClasses.cardHeader}>CASE STATISTICS</div>
      <div class="space-y-3">
        <div>
          <div class={yorhaClasses.metric}>
            {isLoadingMetrics ? '...' : systemMetrics.totalCases}
          </div>
          <div class={yorhaClasses.metricLabel}>TOTAL CASES</div>
        </div>
        <div>
          <div class="text-lg font-bold text-[#00ff88]">
            {isLoadingMetrics ? '...' : systemMetrics.activeCases}
          </div>
          <div class={yorhaClasses.metricLabel}>ACTIVE CASES</div>
        </div>
      </div>
    </div>
    
    <!-- Evidence Metric -->
    <div class={yorhaClasses.card}>
      <div class={yorhaClasses.cardHeader}>EVIDENCE STORAGE</div>
      <div class="space-y-3">
        <div>
          <div class={yorhaClasses.metric}>
            {isLoadingMetrics ? '...' : systemMetrics.totalEvidence}
          </div>
          <div class={yorhaClasses.metricLabel}>EVIDENCE FILES</div>
        </div>
        <div>
          <div class="text-lg font-bold text-yellow-500">
            {isLoadingMetrics ? '...' : systemMetrics.storageUsed}
          </div>
          <div class={yorhaClasses.metricLabel}>STORAGE USED</div>
        </div>
      </div>
    </div>
    
    <!-- System Health -->
    <div class={yorhaClasses.card}>
      <div class={yorhaClasses.cardHeader}>SYSTEM HEALTH</div>
      <div class="space-y-3">
        <div>
          <div class="text-2xl font-bold {getHealthColor(systemMetrics.systemHealth)}">
            {isLoadingMetrics ? '...' : systemMetrics.systemHealth.toUpperCase()}
          </div>
          <div class={yorhaClasses.metricLabel}>OVERALL STATUS</div>
        </div>
        <div>
          <div class="text-lg font-bold">
            {isLoadingMetrics ? '...' : systemMetrics.systemUptime}
          </div>
          <div class={yorhaClasses.metricLabel}>SYSTEM UPTIME</div>
        </div>
      </div>
    </div>
  </div>
  
  <!-- AI Processing Status -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <!-- AI Queue Status -->
    <div class={yorhaClasses.card}>
      <div class={yorhaClasses.cardHeader}>AI PROCESSING QUEUE</div>
      <div class="space-y-4">
        <div class="flex items-center justify-between">
          <span>Queue Size:</span>
          <span class="font-bold text-[#00ff88]">
            {isLoadingMetrics ? '...' : systemMetrics.aiProcessingQueue}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span>Status:</span>
          <span class="font-bold text-[#00ff88]">
            {systemMetrics.aiProcessingQueue > 0 ? 'PROCESSING' : 'IDLE'}
          </span>
        </div>
        <div class="flex space-x-2">
          <button 
            on:click={() => performSystemAction('clear_ai_queue')}
            class={yorhaClasses.buttonDanger}
          >
            CLEAR QUEUE
          </button>
          <button 
            on:click={() => performSystemAction('restart_ai')}
            class={yorhaClasses.button}
          >
            RESTART AI
          </button>
        </div>
      </div>
    </div>
    
    <!-- Backup Status -->
    <div class={yorhaClasses.card}>
      <div class={yorhaClasses.cardHeader}>BACKUP STATUS</div>
      <div class="space-y-4">
        <div class="flex items-center justify-between">
          <span>Last Backup:</span>
          <span class="font-bold">
            {isLoadingMetrics ? '...' : systemMetrics.lastBackup}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span>Auto Backup:</span>
          <span class="font-bold text-[#00ff88]">ENABLED</span>
        </div>
        <div class="flex space-x-2">
          <button 
            on:click={() => performSystemAction('backup_now')}
            class={yorhaClasses.buttonPrimary}
          >
            BACKUP NOW
          </button>
          <button 
            on:click={() => performSystemAction('restore')}
            class={yorhaClasses.button}
          >
            RESTORE
          </button>
        </div>
      </div>
    </div>
  </div>
  
  <!-- Recent Activity -->
  <div class={yorhaClasses.card}>
    <div class={yorhaClasses.cardHeader}>RECENT SYSTEM ACTIVITY</div>
    
    {#if recentActivity.length > 0}
      <div class="overflow-x-auto">
        <table class={yorhaClasses.table}>
          <thead>
            <tr>
              <th class={yorhaClasses.tableHeader}>STATUS</th>
              <th class={yorhaClasses.tableHeader}>TYPE</th>
              <th class={yorhaClasses.tableHeader}>USER</th>
              <th class={yorhaClasses.tableHeader}>ACTION</th>
              <th class={yorhaClasses.tableHeader}>TIMESTAMP</th>
            </tr>
          </thead>
          <tbody>
            {#each recentActivity as activity}
              <tr class="hover:bg-[#222222]">
                <td class={yorhaClasses.tableCell}>
                  <span class={activity.status === 'success' ? yorhaClasses.statusSuccess : 
                              activity.status === 'warning' ? yorhaClasses.statusWarning : yorhaClasses.statusError}>
                    {getStatusIcon(activity.status)}
                  </span>
                </td>
                <td class={yorhaClasses.tableCell}>{activity.type.toUpperCase()}</td>
                <td class={yorhaClasses.tableCell}>{activity.user.toUpperCase()}</td>
                <td class={yorhaClasses.tableCell}>{activity.action}</td>
                <td class={yorhaClasses.tableCell}>
                  {new Date(activity.timestamp).toLocaleString()}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else}
      <div class="text-center py-8 opacity-60">
        <div class="text-4xl mb-4">◯</div>
        <div>NO RECENT ACTIVITY</div>
      </div>
    {/if}
  </div>
  
  <!-- System Actions -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
    <!-- Maintenance -->
    <div class={yorhaClasses.card}>
      <div class={yorhaClasses.cardHeader}>MAINTENANCE</div>
      <div class="space-y-3">
        <button 
          on:click={() => performSystemAction('clear_cache')}
          class="{yorhaClasses.button} w-full"
        >
          CLEAR SYSTEM CACHE
        </button>
        <button 
          on:click={() => performSystemAction('optimize_db')}
          class="{yorhaClasses.button} w-full"
        >
          OPTIMIZE DATABASE
        </button>
        <button 
          on:click={() => performSystemAction('cleanup_logs')}
          class="{yorhaClasses.button} w-full"
        >
          CLEANUP LOG FILES
        </button>
      </div>
    </div>
    
    <!-- Security -->
    <div class={yorhaClasses.card}>
      <div class={yorhaClasses.cardHeader}>SECURITY</div>
      <div class="space-y-3">
        <button 
          on:click={() => performSystemAction('force_logout_all')}
          class="{yorhaClasses.buttonDanger} w-full"
        >
          FORCE LOGOUT ALL
        </button>
        <button 
          on:click={() => performSystemAction('reset_sessions')}
          class="{yorhaClasses.buttonDanger} w-full"
        >
          RESET ALL SESSIONS
        </button>
        <button 
          on:click={() => performSystemAction('security_scan')}
          class="{yorhaClasses.button} w-full"
        >
          SECURITY SCAN
        </button>
      </div>
    </div>
    
    <!-- System Control -->
    <div class={yorhaClasses.card}>
      <div class={yorhaClasses.cardHeader}>SYSTEM CONTROL</div>
      <div class="space-y-3">
        <button 
          on:click={() => performSystemAction('restart_services')}
          class="{yorhaClasses.buttonDanger} w-full"
        >
          RESTART SERVICES
        </button>
        <button 
          on:click={() => performSystemAction('enable_maintenance')}
          class="{yorhaClasses.button} w-full"
        >
          MAINTENANCE MODE
        </button>
        <button 
          on:click={() => performSystemAction('generate_report')}
          class="{yorhaClasses.buttonPrimary} w-full"
        >
          SYSTEM REPORT
        </button>
      </div>
    </div>
  </div>
</div>