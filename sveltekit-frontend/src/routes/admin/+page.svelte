<script lang="ts">
  import { onMount } from 'svelte';
  import { currentUser } from '$lib/auth/auth-store';
  import type { PageData } from './$types';

  let { data = $bindable() } = $props(); // PageData;

  // System metrics and status - Using $state for reactivity
  let systemMetrics = $state({
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
  });

  let recentActivity = $state<Array<{
    id: string;
    type: string;
    user: string;
    action: string;
    timestamp: string;
    status: 'success' | 'warning' | 'error';
  }>>([]);

  let isLoadingMetrics = $state(true);

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
        on:on:onclick={() => performSystemAction('refresh')}
        class={yorhaClasses.button}
      >
        ↻ REFRESH
      </button>
      <button
        on:on:onclick={() => performSystemAction('backup')}
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
    <div class={