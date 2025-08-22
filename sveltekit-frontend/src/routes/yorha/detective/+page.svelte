<!-- YoRHa Detective Command Center -->
<script lang="ts">
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { onMount } from 'svelte';
  import type { PageData } from './$types';
  import YoRHaCommandCenter from '$lib/components/yorha/YoRHaCommandCenter.svelte';
  import YoRHaModal from '$lib/components/yorha/YoRHaModal.svelte';
  import YoRHaNotificationManager from '$lib/components/yorha/YoRHaNotificationManager.svelte';

  // Props
  export let data: PageData;

  // State management
  let selectedSection = $state('command-center');
  let showNewCaseModal = $state(false);
  let newCaseData = $state({
    title: '',
    description: '',
    priority: 'medium' as 'low' | 'medium' | 'high' | 'critical'
  });

  // Notification state
  let notifications = $state<Array<{id: string, type: string, message: string, duration: number}>>([]);

  // Navigation sections
  const navigationSections = [
    { id: 'command-center', name: 'Command Center', icon: '🏢' },
    { id: 'evidence', name: 'Evidence', icon: '📁' },
    { id: 'persons', name: 'Persons of Interest', icon: '👤' },
    { id: 'analysis', name: 'Analysis', icon: '🔍' },
    { id: 'search', name: 'Global Search', icon: '🔎' },
    { id: 'terminal', name: 'Terminal', icon: '💻' }
  ];

  // Quick stats derived from data
  let quickStats = $derived(() => ({
    activeCases: data.systemData.activeCases,
    evidenceItems: data.systemData.evidenceItems,
    personsOfInterest: data.systemData.personsOfInterest,
    aiQueries: data.systemData.aiQueries
  }));

  // Handle section navigation
  function navigateToSection(sectionId: string) {
    selectedSection = sectionId;
    
    // Navigate to dedicated pages for complex sections
    switch (sectionId) {
      case 'evidence':
        goto('/evidence');
        break;
      case 'search':
        goto('/search');
        break;
      case 'terminal':
        goto('/yorha/terminal');
        break;
      default:
        selectedSection = sectionId;
    }
  }

  // Handle new case creation
  async function handleCreateCase() {
    try {
      const response = await fetch('/api/cases', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          title: newCaseData.title,
          description: newCaseData.description,
          priority: newCaseData.priority
        })
      });

      if (response.ok) {
        const result = await response.json();
        showNewCaseModal = false;
        
        // Reset form
        newCaseData = {
          title: '',
          description: '',
          priority: 'medium'
        };

        // Show success notification
        addNotification('success', `Case "${result.title}" created successfully`, 5000);

        // Refresh the page data
        goto($page.url, { invalidateAll: true });
      } else {
        throw new Error('Failed to create case');
      }
    } catch (error) {
      addNotification('error', 'Failed to create case. Please try again.', 5000);
    }
  }

  // Cancel new case modal
  function cancelNewCase() {
    showNewCaseModal = false;
    newCaseData = {
      title: '',
      description: '',
      priority: 'medium'
    };
  }

  // Helper function to add notifications
  function addNotification(type: string, message: string, duration: number = 5000) {
    const id = crypto.randomUUID();
    notifications.push({ id, type, message, duration });
    
    // Auto-remove notification after duration
    setTimeout(() => {
      notifications = notifications.filter(n => n.id !== id);
    }, duration);
  }
</script>

<!-- Main Detective Interface -->
<div class="yorha-detective min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-800 text-amber-300">
  
  <!-- Header Bar -->
  <div class="header-bar bg-black bg-opacity-70 border-b border-amber-400 border-opacity-30 p-4">
    <div class="flex justify-between items-center">
      <div class="flex items-center space-x-4">
        <div class="w-8 h-8 bg-amber-400 bg-opacity-20 border border-amber-400 border-opacity-50 flex items-center justify-center">
          <span class="text-amber-400 font-bold text-sm">YD</span>
        </div>
        <h1 class="text-xl font-bold text-amber-300">YoRHa Detective Command Center</h1>
      </div>
      
      <div class="flex items-center space-x-4">
        <button 
          class="px-4 py-2 bg-amber-600 bg-opacity-20 border border-amber-400 border-opacity-50 text-amber-300 hover:bg-opacity-30 transition-all duration-300"
          onclick={() => showNewCaseModal = true}
        >
          + New Case
        </button>
        
        <div class="flex items-center space-x-2 text-sm">
          <span>User:</span>
          <span class="text-amber-400">{data.user.firstName} {data.user.lastName}</span>
          <span class="px-2 py-1 bg-amber-600 bg-opacity-20 border border-amber-400 border-opacity-30 text-xs uppercase">
            {data.user.role}
          </span>
        </div>
      </div>
    </div>
  </div>

  <!-- Main Layout -->
  <div class="main-layout flex h-[calc(100vh-80px)]">
    
    <!-- Sidebar Navigation -->
    <div class="sidebar w-64 bg-black bg-opacity-50 border-r border-amber-400 border-opacity-30 p-4">
      <nav class="space-y-2">
        {#each navigationSections as section}
          <button
            class="nav-item w-full flex items-center space-x-3 p-3 text-left border border-transparent hover:border-amber-400 hover:border-opacity-30 hover:bg-amber-600 hover:bg-opacity-10 transition-all duration-300 {selectedSection === section.id ? 'border-amber-400 border-opacity-50 bg-amber-600 bg-opacity-20 text-amber-400' : 'text-amber-300'}"
            onclick={() => navigateToSection(section.id)}
          >
            <span class="text-lg">{section.icon}</span>
            <span class="font-medium">{section.name}</span>
          </button>
        {/each}
      </nav>
      
      <!-- Quick Stats in Sidebar -->
      <div class="mt-8 space-y-4">
        <h3 class="text-sm font-bold text-amber-400 uppercase tracking-wider">Quick Stats</h3>
        
        <div class="stat-item">
          <div class="text-xs text-amber-400 opacity-70">Active Cases</div>
          <div class="text-lg font-bold text-amber-300">{quickStats.activeCases}</div>
        </div>
        
        <div class="stat-item">
          <div class="text-xs text-amber-400 opacity-70">Evidence Items</div>
          <div class="text-lg font-bold text-amber-300">{quickStats.evidenceItems}</div>
        </div>
        
        <div class="stat-item">
          <div class="text-xs text-amber-400 opacity-70">Persons of Interest</div>
          <div class="text-lg font-bold text-amber-300">{quickStats.personsOfInterest}</div>
        </div>
        
        <div class="stat-item">
          <div class="text-xs text-amber-400 opacity-70">AI Queries</div>
          <div class="text-lg font-bold text-amber-300">{quickStats.aiQueries}</div>
        </div>
      </div>
    </div>

    <!-- Main Content Area -->
    <div class="content flex-1 p-6">
      {#if selectedSection === 'command-center'}
        <!-- Command Center Dashboard -->
        <YoRHaCommandCenter systemData={data.systemData} />
      {:else if selectedSection === 'evidence'}
        <!-- Evidence Section -->
        <div class="evidence-section">
          <h2 class="text-2xl font-bold text-amber-400 mb-6">Evidence Management</h2>
          
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Recent Evidence -->
            <div class="recent-evidence bg-black bg-opacity-30 border border-amber-400 border-opacity-30 p-6">
              <h3 class="text-lg font-bold text-amber-400 mb-4">Recent Evidence</h3>
              
              <div class="space-y-3">
                {#each data.recentEvidence.slice(0, 5) as evidence}
                  <div class="evidence-item p-3 border border-amber-400 border-opacity-20 hover:border-opacity-40 transition-all duration-300">
                    <div class="flex justify-between items-start">
                      <div>
                        <div class="font-medium text-amber-300">{evidence.title}</div>
                        <div class="text-xs text-amber-400 opacity-70">{evidence.evidenceType}</div>
                        {#if evidence.caseTitle}
                          <div class="text-xs text-amber-400 opacity-60">Case: {evidence.caseTitle}</div>
                        {/if}
                      </div>
                      <div class="text-xs text-amber-400 opacity-60">
                        {new Date(evidence.createdAt).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                {/each}
              </div>
            </div>
            
            <!-- Evidence Actions -->
            <div class="evidence-actions bg-black bg-opacity-30 border border-amber-400 border-opacity-30 p-6">
              <h3 class="text-lg font-bold text-amber-400 mb-4">Evidence Actions</h3>
              
              <div class="space-y-3">
                <button 
                  class="action-button w-full p-3 border border-amber-400 border-opacity-30 text-amber-300 hover:border-opacity-50 hover:bg-amber-600 hover:bg-opacity-10 transition-all duration-300"
                  onclick={() => goto('/evidence/upload')}
                >
                  📤 Upload Evidence
                </button>
                
                <button 
                  class="action-button w-full p-3 border border-amber-400 border-opacity-30 text-amber-300 hover:border-opacity-50 hover:bg-amber-600 hover:bg-opacity-10 transition-all duration-300"
                  onclick={() => goto('/evidence/analyze')}
                >
                  🔍 Analyze Evidence
                </button>
                
                <button 
                  class="action-button w-full p-3 border border-amber-400 border-opacity-30 text-amber-300 hover:border-opacity-50 hover:bg-amber-600 hover:bg-opacity-10 transition-all duration-300"
                  onclick={() => goto('/evidence/search')}
                >
                  🔎 Search Evidence
                </button>
              </div>
            </div>
          </div>
        </div>
      {:else if selectedSection === 'persons'}
        <!-- Persons of Interest Section -->
        <div class="persons-section">
          <h2 class="text-2xl font-bold text-amber-400 mb-6">Persons of Interest</h2>
          
          <div class="coming-soon bg-black bg-opacity-30 border border-amber-400 border-opacity-30 p-12 text-center">
            <div class="text-6xl mb-4">👤</div>
            <h3 class="text-xl font-bold text-amber-400 mb-2">Coming Soon</h3>
            <p class="text-amber-300 opacity-70">Person tracking and relationship mapping will be available in the next update.</p>
          </div>
        </div>
      {:else if selectedSection === 'analysis'}
        <!-- Analysis Section -->
        <div class="analysis-section">
          <h2 class="text-2xl font-bold text-amber-400 mb-6">Case Analysis</h2>
          
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Recent Cases -->
            <div class="recent-cases bg-black bg-opacity-30 border border-amber-400 border-opacity-30 p-6">
              <h3 class="text-lg font-bold text-amber-400 mb-4">Recent Cases</h3>
              
              <div class="space-y-3">
                {#each data.recentCases.slice(0, 5) as case_}
                  <div class="case-item p-3 border border-amber-400 border-opacity-20 hover:border-opacity-40 transition-all duration-300 cursor-pointer"
                       onclick={() => goto(`/cases/${case_.id}`)}>
                    <div class="flex justify-between items-start">
                      <div>
                        <div class="font-medium text-amber-300">{case_.title}</div>
                        <div class="text-xs text-amber-400 opacity-70">#{case_.caseNumber}</div>
                        {#if case_.createdBy}
                          <div class="text-xs text-amber-400 opacity-60">
                            By: {case_.createdBy} {case_.createdByLastName}
                          </div>
                        {/if}
                      </div>
                      <div class="flex flex-col items-end">
                        <span class="text-xs px-2 py-1 border border-amber-400 border-opacity-30 {case_.priority === 'critical' ? 'text-red-400' : case_.priority === 'high' ? 'text-orange-400' : 'text-amber-400'}">
                          {case_.priority}
                        </span>
                        <div class="text-xs text-amber-400 opacity-60 mt-1">
                          {new Date(case_.createdAt).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                  </div>
                {/each}
              </div>
            </div>
            
            <!-- Analysis Tools -->
            <div class="analysis-tools bg-black bg-opacity-30 border border-amber-400 border-opacity-30 p-6">
              <h3 class="text-lg font-bold text-amber-400 mb-4">Analysis Tools</h3>
              
              <div class="space-y-3">
                <button 
                  class="action-button w-full p-3 border border-amber-400 border-opacity-30 text-amber-300 hover:border-opacity-50 hover:bg-amber-600 hover:bg-opacity-10 transition-all duration-300"
                  onclick={() => goto('/ai-assistant')}
                >
                  🤖 AI Assistant
                </button>
                
                <button 
                  class="action-button w-full p-3 border border-amber-400 border-opacity-30 text-amber-300 hover:border-opacity-50 hover:bg-amber-600 hover:bg-opacity-10 transition-all duration-300"
                  onclick={() => goto('/detective/canvas')}
                >
                  🎨 Evidence Canvas
                </button>
                
                <button 
                  class="action-button w-full p-3 border border-amber-400 border-opacity-30 text-amber-300 hover:border-opacity-50 hover:bg-amber-600 hover:bg-opacity-10 transition-all duration-300"
                  onclick={() => goto('/reports')}
                >
                  📊 Generate Report
                </button>
              </div>
            </div>
          </div>
        </div>
      {/if}
    </div>
  </div>

  <!-- New Case Modal -->
  {#if showNewCaseModal}
    <YoRHaModal 
      title="Create New Case"
      open={showNewCaseModal}
      onclose={cancelNewCase}
    >
      <form class="space-y-4" onsubmit={handleCreateCase}>
        <!-- Case Title -->
        <div>
          <label for="case-title" class="block text-sm font-medium text-amber-400 mb-2">Case Title</label>
          <input
            id="case-title"
            type="text"
            bind:value={newCaseData.title}
            class="w-full p-3 bg-black bg-opacity-50 border border-amber-400 border-opacity-30 text-amber-300 placeholder-amber-400 placeholder-opacity-50 focus:border-opacity-60 focus:ring-2 focus:ring-amber-400 focus:ring-opacity-20"
            placeholder="Enter case title..."
            required
          />
        </div>

        <!-- Case Description -->
        <div>
          <label for="case-description" class="block text-sm font-medium text-amber-400 mb-2">Description</label>
          <textarea
            id="case-description"
            bind:value={newCaseData.description}
            rows="4"
            class="w-full p-3 bg-black bg-opacity-50 border border-amber-400 border-opacity-30 text-amber-300 placeholder-amber-400 placeholder-opacity-50 focus:border-opacity-60 focus:ring-2 focus:ring-amber-400 focus:ring-opacity-20"
            placeholder="Enter case description..."
          ></textarea>
        </div>

        <!-- Priority -->
        <div>
          <label for="case-priority" class="block text-sm font-medium text-amber-400 mb-2">Priority</label>
          <select
            id="case-priority"
            bind:value={newCaseData.priority}
            class="w-full p-3 bg-black bg-opacity-50 border border-amber-400 border-opacity-30 text-amber-300 focus:border-opacity-60 focus:ring-2 focus:ring-amber-400 focus:ring-opacity-20"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>

        <!-- Actions -->
        <div class="flex justify-end space-x-3 pt-4">
          <button
            type="button"
            onclick={cancelNewCase}
            class="px-6 py-2 border border-amber-400 border-opacity-30 text-amber-300 hover:border-opacity-50 hover:bg-amber-600 hover:bg-opacity-10 transition-all duration-300"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!newCaseData.title.trim()}
            class="px-6 py-2 bg-amber-600 bg-opacity-20 border border-amber-400 border-opacity-50 text-amber-300 hover:bg-opacity-30 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
          >
            Create Case
          </button>
        </div>
      </form>
    </YoRHaModal>
  {/if}

  <!-- Notification Manager -->
  {#if notifications.length > 0}
    <div class="notifications fixed top-4 right-4 space-y-2 z-50">
      {#each notifications as notification (notification.id)}
        <div class="notification bg-black bg-opacity-90 border border-amber-400 border-opacity-50 text-amber-300 p-3 rounded backdrop-blur-sm">
          <div class="flex items-center space-x-2">
            <span class="text-lg">
              {#if notification.type === 'success'}✅
              {:else if notification.type === 'error'}❌
              {:else if notification.type === 'warning'}⚠️
              {:else}ℹ️{/if}
            </span>
            <span class="text-sm">{notification.message}</span>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .yorha-detective {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
  }

  .nav-item {
    position: relative;
  }

  .nav-item::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 2px;
    background: rgba(251, 191, 36, 0.5);
    transform: scaleY(0);
    transition: transform 0.3s ease;
  }

  .nav-item:hover::before,
  .nav-item.active::before {
    transform: scaleY(1);
  }

  .stat-item {
    padding: 12px;
    border: 1px solid rgba(251, 191, 36, 0.2);
    background: rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
  }

  .stat-item:hover {
    border-color: rgba(251, 191, 36, 0.4);
    background: rgba(251, 191, 36, 0.05);
  }

  .evidence-item,
  .case-item {
    transition: all 0.3s ease;
  }

  .evidence-item:hover,
  .case-item:hover {
    background: rgba(251, 191, 36, 0.05);
    transform: translateX(4px);
  }

  .action-button {
    position: relative;
    overflow: hidden;
  }

  .action-button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(251, 191, 36, 0.1), transparent);
    transition: left 0.5s ease;
  }

  .action-button:hover::before {
    left: 100%;
  }

  .coming-soon {
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.8;
    }
  }

  /* Responsive design */
  @media (max-width: 1024px) {
    .main-layout {
      flex-direction: column;
    }
    
    .sidebar {
      width: 100%;
      height: auto;
    }
    
    .grid {
      grid-template-columns: 1fr;
    }
  }
</style>