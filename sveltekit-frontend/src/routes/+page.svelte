<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import Button from '$lib/components/ui/Button.svelte';
  // Simple modal implementation instead of problematic Dialog components

  // YoRHa Detective Interface State
  let currentTime = $state(new Date().toLocaleString());
  let activeSection = $state('command-center');
  let isNewCaseModalOpen = $state(false);
  let notification = $state({ show: false, message: '' });

  // Live statistics from database
  let stats = $state({
    activeCases: 0,
    evidenceItems: 0,
    personsOfInterest: 0,
    recentActivity: 0,
    loading: true
  });

  // Active cases from database
  let activeCases = $state([]);
  let loadingCases = $state(true);

  // System status
  let systemStatus = $state({
    uptime: '72h 14m',
    services: '8/9 ONLINE',
    lastSync: 'NOW',
    status: 'OPERATIONAL'
  });

  // Recent activity feed
  let recentActivity = $state([
    { time: '12:47:33', action: 'Evidence uploaded', details: 'Financial records - Case #2847' },
    { time: '12:45:10', action: 'Case updated', details: 'Corporate Espionage Investigation' },
    { time: '12:42:18', action: 'POI identified', details: 'Sarah Chen - Missing Person Case' },
    { time: '12:40:05', action: 'Analysis complete', details: 'Document classification completed' }
  ]);

  // New case form state
  let newCaseForm = $state({
    title: '',
    description: '',
    priority: 'medium',
    loading: false
  });

  // Priority options for select
  const priorityOptions = [
    { value: 'low', label: 'Low' },
    { value: 'medium', label: 'Medium' },
    { value: 'high', label: 'High' },
    { value: 'critical', label: 'Critical' }
  ];

  onMount(async () => {
    // Update current time every second
    const timeInterval = setInterval(() => {
      currentTime = new Date().toLocaleString();
    }, 1000);

    // Load dashboard data
    await loadDashboardData();

    return () => clearInterval(timeInterval);
  });

  async function loadDashboardData() {
    try {
      // Load statistics
      const statsResponse = await fetch('/api/dashboard/stats');
      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        stats = { ...statsData, loading: false };
      }

      // Load active cases
      const casesResponse = await fetch('/api/cases?status=active&limit=5');
      if (casesResponse.ok) {
        const casesData = await casesResponse.json();
        activeCases = casesData.cases || [];
      }
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      stats.loading = false;
      loadingCases = false;
    }
  }

  async function handleCreateCase(event) {
    event.preventDefault();
    
    if (!newCaseForm.title.trim() || !newCaseForm.description.trim()) {
      showNotification('Please fill in all required fields');
      return;
    }

    newCaseForm.loading = true;
    
    try {
      const response = await fetch('/api/cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: newCaseForm.title,
          description: newCaseForm.description,
          priority: newCaseForm.priority,
          status: 'active'
        })
      });

      if (response.ok) {
        const result = await response.json();
        showNotification('Case created successfully!');
        isNewCaseModalOpen = false;
        newCaseForm = { title: '', description: '', priority: 'medium', loading: false };
        await loadDashboardData(); // Refresh data
      } else {
        const error = await response.json();
        showNotification(`Error: ${error.message || 'Failed to create case'}`);
      }
    } catch (error) {
      console.error('Error creating case:', error);
      showNotification('Network error occurred');
    } finally {
      newCaseForm.loading = false;
    }
  }

  function showNotification(message: string) {
    notification = { show: true, message };
    setTimeout(() => {
      notification = { show: false, message: '' };
    }, 3000);
  }

  function closeModal() {
    isNewCaseModalOpen = false;
  }

  function handleNavigation(section: string) {
    activeSection = section;
    switch (section) {
      case 'evidence':
        goto('/evidence');
        break;
      case 'poi':
        goto('/poi');
        break;
      case 'analysis':
        goto('/analysis');
        break;
      case 'search':
        goto('/search');
        break;
      case 'terminal':
        goto('/terminal');
        break;
      default:
        // Stay on command center
        break;
    }
  }
</script>

<svelte:head>
  <title>YoRHa Detective - Command Center</title>
  <meta name="description" content="YoRHa Detective Command Center - Advanced case management and investigation platform" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap" rel="stylesheet">
</svelte:head>

<!-- YoRHa Detective Interface -->
<div class="min-h-screen bg-[#EAE8E1] font-mono text-[#3D3D3D]">
  <!-- Header -->
  <header class="flex justify-between items-center mb-6 p-4 lg:p-6">
    <div>
      <h1 class="text-2xl font-bold tracking-wider">COMMAND CENTER</h1>
      <p class="text-sm opacity-75">YoRHa Detective Interface - {currentTime}</p>
    </div>
    <div class="flex items-center gap-4">
      <Button 
        variant="default" 
        class="bg-[#F7F6F2] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-[#EAE8E1] font-bold px-4 py-2 flex items-center gap-2"
        onclick={() => isNewCaseModalOpen = true}
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
        </svg>
        NEW CASE
      </Button>
      <Button 
        variant="default"
        class="bg-[#F7F6F2] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-[#EAE8E1] font-bold px-4 py-2 flex items-center gap-2"
        onclick={() => handleNavigation('search')}
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
        </svg>
        GLOBAL SEARCH
      </Button>
    </div>
  </header>

  <!-- Main Content -->
  <main class="grid grid-cols-1 lg:grid-cols-4 gap-6 px-4 lg:px-6">
    <!-- Sidebar -->
    <aside class="lg:col-span-1 bg-[#F7F6F2] border border-[#D1CFC7] p-4">
      <h2 class="font-bold mb-4 text-center">YORHA DETECTIVE</h2>
      <nav class="space-y-2">
        <button 
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'command-center' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('command-center')}
        >
          COMMAND CENTER
        </button>
        <button 
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'evidence' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('evidence')}
        >
          EVIDENCE
        </button>
        <button 
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'poi' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('poi')}
        >
          PERSONS OF INTEREST
        </button>
        <button 
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'analysis' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('analysis')}
        >
          ANALYSIS
        </button>
        <button 
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'search' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('search')}
        >
          GLOBAL SEARCH
        </button>
        <button 
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'terminal' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('terminal')}
        >
          TERMINAL
        </button>
      </nav>
    </aside>

    <!-- Dashboard Content -->
    <div class="lg:col-span-3 space-y-6">
      <!-- Statistics Cards -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-4">
          <h3 class="font-bold text-sm mb-2">ACTIVE CASES</h3>
          <div class="text-2xl font-bold">
            {stats.loading ? '...' : stats.activeCases}
          </div>
        </div>
        <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-4">
          <h3 class="font-bold text-sm mb-2">EVIDENCE ITEMS</h3>
          <div class="text-2xl font-bold">
            {stats.loading ? '...' : stats.evidenceItems}
          </div>
        </div>
        <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-4">
          <h3 class="font-bold text-sm mb-2">PERSONS OF INTEREST</h3>
          <div class="text-2xl font-bold">
            {stats.loading ? '...' : stats.personsOfInterest}
          </div>
        </div>
        <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-4">
          <h3 class="font-bold text-sm mb-2">RECENT ACTIVITY</h3>
          <div class="text-2xl font-bold">
            {stats.loading ? '...' : stats.recentActivity}
          </div>
        </div>
      </div>

      <!-- Main Dashboard Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Active Cases Panel -->
        <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-6">
          <h2 class="text-xl font-bold mb-4">ACTIVE CASES</h2>
          <div class="space-y-3">
            {#if loadingCases}
              <div class="text-gray-500">Loading cases...</div>
            {:else if activeCases.length === 0}
              <div class="text-gray-500">No active cases found.</div>
            {:else}
              {#each activeCases.slice(0, 5) as caseItem}
                <div class="border-b border-[#D1CFC7] pb-2 last:border-b-0">
                  <div class="font-bold text-sm">{caseItem.title}</div>
                  <div class="text-xs opacity-75">{caseItem.status} • Priority: {caseItem.priority}</div>
                </div>
              {/each}
            {/if}
          </div>
        </div>

        <!-- System Status Panel -->
        <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-6">
          <h2 class="text-xl font-bold mb-4">SYSTEM STATUS</h2>
          <div class="space-y-3">
            <div class="flex justify-between">
              <span class="font-medium">UPTIME:</span>
              <span class="text-green-600 font-bold">{systemStatus.uptime}</span>
            </div>
            <div class="flex justify-between">
              <span class="font-medium">SERVICES:</span>
              <span class="text-blue-600 font-bold">{systemStatus.services}</span>
            </div>
            <div class="flex justify-between">
              <span class="font-medium">LAST SYNC:</span>
              <span class="font-bold">{systemStatus.lastSync}</span>
            </div>
            <div class="flex justify-between">
              <span class="font-medium">STATUS:</span>
              <span class="text-green-600 font-bold">{systemStatus.status}</span>
            </div>
          </div>
        </div>

        <!-- Recent Activity Panel -->
        <div class="lg:col-span-2 bg-[#F7F6F2] border border-[#D1CFC7] p-6">
          <h2 class="text-xl font-bold mb-4">RECENT ACTIVITY</h2>
          <div class="space-y-2">
            {#each recentActivity as activity}
              <div class="flex items-start gap-4 text-sm">
                <span class="font-mono text-xs bg-[#3D3D3D] text-[#F7F6F2] px-2 py-1 rounded">
                  {activity.time}
                </span>
                <div>
                  <span class="font-bold">{activity.action}:</span>
                  <span class="opacity-75">{activity.details}</span>
                </div>
              </div>
            {/each}
          </div>
        </div>
      </div>

      <!-- Quick Actions Panel -->
      <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-6">
        <h2 class="text-xl font-bold mb-4">QUICK ACTIONS</h2>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Button 
            variant="default"
            class="bg-[#EAE8E1] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-white font-bold p-4 h-auto flex flex-col items-center gap-2"
            onclick={() => isNewCaseModalOpen = true}
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
            </svg>
            CREATE CASE
          </Button>
          <Button 
            variant="default"
            class="bg-[#EAE8E1] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-white font-bold p-4 h-auto flex flex-col items-center gap-2"
            onclick={() => handleNavigation('evidence')}
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
            </svg>
            VIEW EVIDENCE
          </Button>
          <Button 
            variant="default"
            class="bg-[#EAE8E1] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-white font-bold p-4 h-auto flex flex-col items-center gap-2"
            onclick={() => handleNavigation('poi')}
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
            </svg>
            SEARCH POI
          </Button>
          <Button 
            variant="default"
            class="bg-[#EAE8E1] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-white font-bold p-4 h-auto flex flex-col items-center gap-2"
            onclick={() => handleNavigation('analysis')}
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
            </svg>
            RUN ANALYSIS
          </Button>
        </div>
      </div>
    </div>
  </main>
</div>

<!-- New Case Modal - Simple HTML Modal -->
{#if isNewCaseModalOpen}
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50" onclick={closeModal}>
    <div class="w-full max-w-2xl bg-[#F7F6F2] border border-[#D1CFC7] p-8 font-mono" onclick={(e) => e.stopPropagation()}>
      <div class="flex justify-between items-center mb-6">
        <h2 class="text-2xl font-bold">CREATE NEW CASE FILE</h2>
        <button onclick={() => isNewCaseModalOpen = false} class="text-gray-500 hover:text-black text-2xl">&times;</button>
      </div>
      
      <form onsubmit={handleCreateCase} class="space-y-6">
        <div>
          <label for="case-title" class="block text-sm font-bold mb-2">CASE TITLE</label>
          <input 
            type="text" 
            id="case-title" 
            bind:value={newCaseForm.title}
            required 
            class="w-full px-4 py-3 bg-white border border-[#D1CFC7] focus:outline-none focus:border-[#3D3D3D] focus:ring-2 focus:ring-[#3D3D3D]/20"
            placeholder="e.g., Corporate Espionage Investigation"
          />
        </div>
        
        <div>
          <label for="case-description" class="block text-sm font-bold mb-2">CASE DESCRIPTION / SYNOPSIS</label>
          <textarea 
            id="case-description" 
            bind:value={newCaseForm.description}
            rows="4" 
            required 
            class="w-full px-4 py-3 bg-white border border-[#D1CFC7] focus:outline-none focus:border-[#3D3D3D] focus:ring-2 focus:ring-[#3D3D3D]/20"
            placeholder="Initial details of the investigation..."
          ></textarea>
        </div>
        
        <div>
          <label class="block text-sm font-bold mb-2">PRIORITY LEVEL</label>
          <select 
            bind:value={newCaseForm.priority}
            class="w-full px-4 py-3 bg-white border border-[#D1CFC7] focus:outline-none focus:border-[#3D3D3D] focus:ring-2 focus:ring-[#3D3D3D]/20 font-bold"
          >
            <option value="low">Low Priority</option>
            <option value="medium" selected>Medium Priority</option>
            <option value="high">High Priority</option>
            <option value="critical">Critical Priority</option>
          </select>
        </div>
        
        <div class="flex justify-end">
          <Button 
            type="submit"
            disabled={newCaseForm.loading}
            class="bg-green-600/10 text-green-800 border border-green-700/50 hover:bg-green-600/20 font-bold px-6 py-3"
          >
            {newCaseForm.loading ? 'SAVING...' : 'SAVE TO DATABASE'}
          </Button>
        </div>
      </form>
    </div>
  </div>
{/if}

<!-- Notification -->
{#if notification.show}
  <div class="fixed bottom-5 right-5 bg-[#F7F6F2] border border-[#D1CFC7] p-4 max-w-sm font-mono shadow-lg transform transition-all duration-300">
    <p class="font-bold">{notification.message}</p>
  </div>
{/if}