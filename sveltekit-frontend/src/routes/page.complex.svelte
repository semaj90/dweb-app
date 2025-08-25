<!-- Modern Dark YoRHa Legal AI Platform -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import {
    FileText,
    Users,
    Activity,
    AlertTriangle,
    CheckCircle2,
    Search,
    Plus,
    BarChart3,
    Clock,
    Eye,
    Shield,
    Database,
    Settings,
    ChevronRight,
    Folder,
    UserCheck,
    TrendingUp,
    Terminal,
    Cpu,
    Monitor,
    Zap,
    Brain
  } from 'lucide-svelte';

  // Dashboard data
  let stats = $state({
    activeCases: 3,
    evidenceItems: 27,
    personsOfInterest: 8,
    recentActivity: 12
  });

  let activeCases = $state([
    {
      id: 1,
      title: "CORPORATE ESPIONAGE INVESTIGATION",
      items: 8,
      timeAgo: "2 hours ago",
      priority: "high",
      status: "active"
    },
    {
      id: 2,
      title: "MISSING PERSON: DR. SARAH CHEN",
      items: 15,
      timeAgo: "4 hours ago",
      priority: "high",
      status: "active"
    },
    {
      id: 3,
      title: "FINANCIAL FRAUD ANALYSIS",
      items: 4,
      timeAgo: "1 day ago",
      priority: "medium",
      status: "pending"
    }
  ]);

  let systemStatus = $state([
    {
      message: "System backup completed successfully",
      time: "10 minutes ago",
      type: "success"
    },
    {
      message: "Evidence analysis queue processing slowly",
      time: "1 hour ago",
      type: "warning"
    },
    {
      message: "New facial recognition matches found",
      time: "2 hours ago",
      type: "success"
    }
  ]);

  onMount(() => {
    console.log('YoRHa Legal AI Detective Interface initialized');
  });

  function handleNewCase() {
    goto('/cases/new');
  }

  function handleGlobalSearch() {
    goto('/search');
  }

  function handleViewAll() {
    goto('/cases');
  }
</script>

<svelte:head>
  <title>YoRHa Legal AI Detective Interface</title>
</svelte:head>

<div class="yorha-bg-primary min-h-screen">
  <!-- Quick Actions Header -->
  <div class="yorha-bg-secondary border-b border-gray-700 p-4">
    <div class="flex items-center justify-between">
      <div class="flex items-center space-x-4">
        <div class="text-sm yorha-text-muted">
          YoRHa Detective Interface • 8/20/2025 16:08:30
        </div>
      </div>
      <div class="flex items-center space-x-3">
        <button
          class="yorha-btn yorha-btn-primary"
          onclick={handleNewCase}
        >
          <Plus class="w-4 h-4 mr-2" />
          NEW CASE
        </button>
        <button
          class="yorha-btn yorha-btn-secondary"
          onclick={handleGlobalSearch}
        >
          <Search class="w-4 h-4 mr-2" />
          GLOBAL SEARCH
        </button>
      </div>
    </div>
  </div>

  <div class="flex">
    <!-- Sidebar -->
    <aside class="yorha-sidebar">
      <nav class="yorha-sidebar-nav space-y-1">
        <div class="yorha-nav-item active">
          <Database class="w-4 h-4" />
          <span>COMMAND CENTER</span>
        </div>

        <a href="/cases" class="yorha-nav-item">
          <Folder class="w-4 h-4" />
          <span>ACTIVE CASES</span>
          <ChevronRight class="w-4 h-4 ml-auto" />
        </a>

        <a href="/evidence" class="yorha-nav-item">
          <Eye class="w-4 h-4" />
          <span>EVIDENCE</span>
          <ChevronRight class="w-4 h-4 ml-auto" />
        </a>

        <a href="/persons" class="yorha-nav-item">
          <Users class="w-4 h-4" />
          <span>PERSONS OF INTEREST</span>
        </a>

        <a href="/analysis" class="yorha-nav-item">
          <BarChart3 class="w-4 h-4" />
          <span>ANALYSIS</span>
          <ChevronRight class="w-4 h-4 ml-auto" />
        </a>

        <a href="/search" class="yorha-nav-item">
          <Search class="w-4 h-4" />
          <span>GLOBAL SEARCH</span>
        </a>

        <a href="/terminal" class="yorha-nav-item">
          <Terminal class="w-4 h-4" />
          <span>TERMINAL</span>
        </a>

        <div class="pt-4 border-t border-gray-700 mt-4">
          <a href="/system" class="yorha-nav-item">
            <Settings class="w-4 h-4" />
            <span>SYSTEM CONFIG</span>
          </a>
        </div>
      </nav>

      <!-- Status footer -->
      <div class="absolute bottom-4 left-4 right-4 max-w-52">
        <div class="yorha-card p-3">
          <div class="flex items-center space-x-2">
            <div class="yorha-status-indicator yorha-status-online"></div>
            <span class="yorha-text-primary font-medium">Online</span>
          </div>
          <div class="yorha-text-muted text-xs mt-1">System: Operational</div>
          <div class="yorha-text-muted text-xs">16:08</div>
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 p-6">
      <!-- Stats Cards -->
      <div class="yorha-grid yorha-grid-cols-4 mb-6">
        <div class="yorha-card yorha-fade-in">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium yorha-text-secondary">Active Cases</p>
              <p class="text-2xl font-bold yorha-text-primary">{stats.activeCases}</p>
            </div>
            <Folder class="w-8 h-8 yorha-text-muted" />
          </div>
        </div>

        <div class="yorha-card yorha-fade-in" style="animation-delay: 0.1s">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium yorha-text-secondary">Evidence Items</p>
              <p class="text-2xl font-bold yorha-text-primary">{stats.evidenceItems}</p>
            </div>
            <Eye class="w-8 h-8 yorha-text-muted" />
          </div>
        </div>

        <div class="yorha-card yorha-fade-in" style="animation-delay: 0.2s">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium yorha-text-secondary">Persons of Interest</p>
              <p class="text-2xl font-bold yorha-text-primary">{stats.personsOfInterest}</p>
            </div>
            <UserCheck class="w-8 h-8 yorha-text-muted" />
          </div>
        </div>

        <div class="yorha-card yorha-fade-in" style="animation-delay: 0.3s">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-sm font-medium yorha-text-secondary">Recent Activity</p>
              <p class="text-2xl font-bold yorha-text-primary">{stats.recentActivity}</p>
            </div>
            <TrendingUp class="w-8 h-8 yorha-text-muted" />
          </div>
        </div>
      </div>

      <div class="yorha-grid yorha-grid-cols-3 gap-6">
        <!-- Active Cases -->
        <div class="col-span-2">
          <div class="yorha-card yorha-slide-in">
            <div class="yorha-card-header flex items-center justify-between">
              <h2 class="yorha-card-title">ACTIVE CASES</h2>
              <button
                class="yorha-btn yorha-btn-ghost text-sm"
                onclick={handleViewAll}
              >
                VIEW ALL
                <ChevronRight class="w-4 h-4 ml-1" />
              </button>
            </div>
            <div class="space-y-4">
              {#each activeCases as caseItem}
                <div class="border border-gray-700 rounded-lg p-4 hover:bg-gray-800 transition-colors cursor-pointer yorha-fade-in">
                  <div class="flex items-center justify-between mb-2">
                    <h3 class="font-medium yorha-text-primary">{caseItem.title}</h3>
                    <div class="flex space-x-2">
                      {#if caseItem.priority === 'high'}
                        <span class="yorha-badge yorha-badge-error">high</span>
                      {:else if caseItem.priority === 'medium'}
                        <span class="yorha-badge yorha-badge-warning">medium</span>
                      {/if}
                      {#if caseItem.status === 'active'}
                        <span class="yorha-badge yorha-badge-success">active</span>
                      {:else}
                        <span class="yorha-badge yorha-badge-info">pending</span>
                      {/if}
                    </div>
                  </div>
                  <div class="flex items-center justify-between text-sm yorha-text-secondary">
                    <span>{caseItem.items} items</span>
                    <span>{caseItem.timeAgo}</span>
                  </div>
                </div>
              {/each}
            </div>
          </div>
        </div>

        <!-- System Status & Quick Actions -->
        <div class="space-y-6">
          <!-- System Status -->
          <div class="yorha-card yorha-slide-in" style="animation-delay: 0.2s">
            <div class="yorha-card-header">
              <h2 class="yorha-card-title">SYSTEM STATUS</h2>
            </div>
            <div class="space-y-4">
              {#each systemStatus as status}
                <div class="flex items-start space-x-3 yorha-fade-in">
                  {#if status.type === 'success'}
                    <CheckCircle2 class="w-5 h-5 text-green-400 mt-0.5" />
                  {:else}
                    <AlertTriangle class="w-5 h-5 text-yellow-400 mt-0.5" />
                  {/if}
                  <div class="flex-1">
                    <p class="text-sm yorha-text-primary">{status.message}</p>
                    <p class="text-xs yorha-text-muted">{status.time}</p>
                  </div>
                </div>
              {/each}
            </div>
          </div>

          <!-- Quick Actions -->
          <div class="yorha-card yorha-slide-in" style="animation-delay: 0.4s">
            <div class="yorha-card-header">
              <h2 class="yorha-card-title">QUICK ACTIONS</h2>
            </div>
            <div class="space-y-3">
              <button class="w-full yorha-btn yorha-btn-secondary justify-start">
                <Database class="w-4 h-4 mr-3" />
                EVIDENCE BOARD
              </button>
              <button class="w-full yorha-btn yorha-btn-secondary justify-start">
                <Clock class="w-4 h-4 mr-3" />
                TIMELINE ANALYSIS
              </button>
              <button class="w-full yorha-btn yorha-btn-secondary justify-start">
                <Terminal class="w-4 h-4 mr-3" />
                TERMINAL ACCESS
              </button>
            </div>
          </div>

          <!-- AI Status -->
          <div class="yorha-card yorha-slide-in" style="animation-delay: 0.6s">
            <div class="yorha-card-header">
              <h2 class="yorha-card-title">AI SYSTEMS</h2>
            </div>
            <div class="space-y-3">
              <div class="flex items-center justify-between">
                <div class="flex items-center space-x-2">
                  <Brain class="w-4 h-4 yorha-text-accent" />
                  <span class="text-sm yorha-text-primary">Gemma3-Legal</span>
                </div>
                <span class="yorha-badge yorha-badge-success">ACTIVE</span>
              </div>
              <div class="flex items-center justify-between">
                <div class="flex items-center space-x-2">
                  <Cpu class="w-4 h-4 yorha-text-accent" />
                  <span class="text-sm yorha-text-primary">GPU Processing</span>
                </div>
                <span class="yorha-badge yorha-badge-success">78%</span>
              </div>
              <div class="flex items-center justify-between">
                <div class="flex items-center space-x-2">
                  <Monitor class="w-4 h-4 yorha-text-accent" />
                  <span class="text-sm yorha-text-primary">Vector Search</span>
                </div>
                <span class="yorha-badge yorha-badge-success">READY</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</div>

<style>
  /* Additional component-specific styles */
  .col-span-2 {
    grid-column: span 2;
  }

  .justify-start {
    justify-content: flex-start;
  }
</style>