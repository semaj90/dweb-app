<!-- YoRHa Detective Interface - Command Center Dashboard -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import DetectiveLayout from '$lib/components/DetectiveLayout.svelte';
  import StatsCard from '$lib/components/ui/StatsCard.svelte';
  import CaseItem from '$lib/components/ui/CaseItem.svelte';
  import SystemStatusCard from '$lib/components/ui/SystemStatusCard.svelte';
  import QuickActionButton from '$lib/components/ui/QuickActionButton.svelte';
  import AIAssistantModalSimple from '$lib/components/ai/AIAssistantModalSimple.svelte';

  // YoRHa Detective Interface State
  let currentTime = writable('');
  let systemStatus = writable('Operational');
  let showAIAssistant = false;

  // Dashboard Statistics
  let stats = {
    activeCases: 3,
    evidenceItems: 27,
    personsOfInterest: 8,
    recentActivity: 12
  };

  // Active Cases Data
  let activeCases = [
    {
      id: 'CORP-001',
      title: 'CORPORATE ESPIONAGE INVESTIGATION',
      items: 8,
      timeAgo: '2 hours ago',
      priority: 'high' as const,
      status: 'active' as const
    },
    {
      id: 'MISS-002', 
      title: 'MISSING PERSON: DR. SARAH CHEN',
      items: 15,
      timeAgo: '4 hours ago',
      priority: 'high' as const,
      status: 'active' as const
    },
    {
      id: 'FRAUD-003',
      title: 'FINANCIAL FRAUD ANALYSIS',
      items: 6,
      timeAgo: '1 day ago',
      priority: 'medium' as const,
      status: 'pending' as const
    }
  ];

  // System Status Alerts
  let systemAlerts = [
    {
      type: 'success' as const,
      message: 'System backup completed successfully',
      time: '10 minutes ago'
    },
    {
      type: 'warning' as const,
      message: 'Evidence analysis queue processing slowly',
      time: '1 hour ago'
    },
    {
      type: 'success' as const,
      message: 'New facial recognition matches found',
      time: '2 hours ago'
    }
  ];

  // Update current time
  onMount(() => {
    const updateTime = () => {
      const now = new Date();
      currentTime.set(`${now.toLocaleDateString()} ${now.toLocaleTimeString()}`);
    };
    
    updateTime();
    const interval = setInterval(updateTime, 1000);
    
    return () => clearInterval(interval);
  });

  // Quick Actions
  const quickActions = [
    { label: 'EVIDENCE BOARD', icon: 'folder', action: () => console.log('Evidence Board') },
    { label: 'TIMELINE ANALYSIS', icon: 'clock', action: () => console.log('Timeline') },
    { label: 'TERMINAL ACCESS', icon: 'terminal', action: () => console.log('Terminal') }
  ];

  function openAIAssistant() {
    console.log('AI Assistant button clicked - showAIAssistant was:', showAIAssistant);
    showAIAssistant = true;
    console.log('AI Assistant button clicked - showAIAssistant now:', showAIAssistant);
  }

  function closeAIAssistant() {
    showAIAssistant = false;
  }

  function createNewCase() {
    console.log('Creating new case...');
    // TODO: Navigate to case creation page
  }

  function openGlobalSearch() {
    console.log('Opening global search...');
    // TODO: Navigate to search page
  }
</script>

<svelte:head>
  <title>YoRHa Detective Interface - Command Center</title>
</svelte:head>

<DetectiveLayout>
  <!-- Command Center Header -->
  <div class="flex justify-between items-center mb-6 px-4 py-2 bg-stone-800/50 border border-stone-600">
    <div>
      <h1 class="text-2xl font-mono text-stone-100 tracking-wider">COMMAND CENTER</h1>
      <p class="text-sm text-stone-400 font-mono">YoRHa Detective Interface • {$currentTime}</p>
    </div>
    <div class="flex gap-4">
      <button
        on:click={openAIAssistant}
        class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-mono text-sm border border-blue-500 transition-colors"
      >
        + AI ASSISTANT
      </button>
      <button 
        on:click={createNewCase}
        class="px-4 py-2 bg-stone-700 hover:bg-stone-600 text-stone-100 font-mono text-sm border border-stone-600 transition-colors"
      >
        + NEW CASE
      </button>
      <button 
        on:click={openGlobalSearch}
        class="px-4 py-2 bg-stone-700 hover:bg-stone-600 text-stone-100 font-mono text-sm border border-stone-600 transition-colors"
      >
        🔍 GLOBAL SEARCH
      </button>
    </div>
  </div>

  <!-- Statistics Dashboard -->
  <div class="grid grid-cols-4 gap-4 mb-6">
    <StatsCard
      title="Active Cases"
      value={stats.activeCases}
      icon="folder"
      className="bg-stone-800/50 border-stone-600"
    />
    <StatsCard
      title="Evidence Items"
      value={stats.evidenceItems}
      icon="box"
      className="bg-stone-800/50 border-stone-600"
    />
    <StatsCard
      title="Persons of Interest"
      value={stats.personsOfInterest}
      icon="users"
      className="bg-stone-800/50 border-stone-600"
    />
    <StatsCard
      title="Recent Activity"
      value={stats.recentActivity}
      icon="activity"
      className="bg-stone-800/50 border-stone-600"
    />
  </div>

  <!-- Main Content Grid -->
  <div class="grid grid-cols-3 gap-6">
    <!-- Active Cases Section -->
    <div class="col-span-2 space-y-4">
      <div class="flex justify-between items-center">
        <h2 class="text-xl font-mono text-stone-100 tracking-wider">ACTIVE CASES</h2>
        <button class="text-sm font-mono text-stone-400 hover:text-stone-200 border border-stone-600 px-3 py-1">
          VIEW ALL →
        </button>
      </div>
      
      <div class="space-y-3">
        {#each activeCases as caseData}
          <CaseItem {caseData} />
        {/each}
      </div>
    </div>

    <!-- Right Sidebar -->
    <div class="space-y-6">
      <!-- System Status -->
      <div>
        <h3 class="text-lg font-mono text-stone-100 tracking-wider mb-4">SYSTEM STATUS</h3>
        <div class="space-y-3">
          {#each systemAlerts as alert}
            <SystemStatusCard {alert} />
          {/each}
        </div>
      </div>

      <!-- Quick Actions -->
      <div>
        <h3 class="text-lg font-mono text-stone-100 tracking-wider mb-4">QUICK ACTIONS</h3>
        <div class="space-y-2">
          {#each quickActions as action}
            <QuickActionButton {action} />
          {/each}
        </div>
      </div>
    </div>
  </div>

  <!-- Bottom Status Bar -->
  <div class="fixed bottom-0 left-0 right-0 bg-stone-900 border-t border-stone-600 px-4 py-2 flex justify-between items-center z-10">
    <div class="flex items-center gap-4 text-sm font-mono text-stone-400">
      <span class="flex items-center gap-2">
        <div class="w-2 h-2 bg-green-500 rounded-full"></div>
        Online
      </span>
      <span>System: {$systemStatus}</span>
      <span>{$currentTime}</span>
    </div>
    <div class="text-sm font-mono text-stone-400">
      Investigation Interface
    </div>
  </div>
</DetectiveLayout>

<!-- AI Assistant Modal -->
{#if showAIAssistant}
  <AIAssistantModalSimple on:close={closeAIAssistant} />
{/if}

<style>
  :global(body) {
    background-color: #1c1917;
    color: #e7e5e4;
    font-family: 'Courier New', monospace;
  }
</style>