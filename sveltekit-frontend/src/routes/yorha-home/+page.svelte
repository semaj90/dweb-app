<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from "$lib/components/ui/button";
  import { Badge } from "$lib/components/ui/badge";
  import { Card, CardHeader, CardTitle, CardContent } from "$lib/components/ui/card";
  import YorhaAIAssistant from "$lib/components/ai/YorhaAIAssistant.svelte";
  import {
    Search,
    FileText,
    Users,
    Activity,
    Shield,
    Database,
    Zap,
    Eye,
    Target,
    Brain,
    Lock,
    Monitor,
    Clock,
    CheckCircle,
    AlertTriangle
  } from 'lucide-svelte';

  // YoRHa Detective state
  let currentTime = $state('');
  let systemStatus = $state('OPERATIONAL');
  let activeConnectionCount = $state(0);
  let showAIAssistant = $state(false);

  // Dashboard stats
  let stats = $state({
    activeCases: 3,
    evidenceItems: 27,
    personsOfInterest: 8,
    recentActivity: 12
  });

  // Active cases data
  let activeCases = $state([
    {
      id: 'case-001',
      title: 'CORPORATE ESPIONAGE INVESTIGATION',
      status: 'high',
      priority: 'active',
      lastUpdated: '2 hours ago',
      progress: 75
    },
    {
      id: 'case-002',
      title: 'MISSING PERSON: DR. SARAH CHEN',
      status: 'high',
      priority: 'active',
      lastUpdated: '4 hours ago',
      progress: 60
    },
    {
      id: 'case-003',
      title: 'FINANCIAL FRAUD ANALYSIS',
      status: 'medium',
      priority: 'pending',
      lastUpdated: '1 day ago',
      progress: 40
    }
  ]);

  // System health monitoring
  let systemHealth = $state({
    database: { status: 'operational', uptime: '99.9%' },
    ai: { status: 'operational', uptime: '98.7%' },
    network: { status: 'operational', uptime: '99.5%' },
    security: { status: 'operational', uptime: '100%' }
  });

  onMount(() => {
    // Update time every second
    const timeInterval = setInterval(() => {
      const now = new Date();
      currentTime = now.toLocaleString('en-US', {
        weekday: 'short',
        year: 'numeric',
        month: 'short',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      });
    }, 1000);

    // Simulate active connections
    const connectionInterval = setInterval(() => {
      activeConnectionCount = Math.floor(Math.random() * 5) + 3;
    }, 5000);

    // System backup simulation
    setTimeout(() => {
      systemStatus = 'BACKUP COMPLETED';
      setTimeout(() => {
        systemStatus = 'OPERATIONAL';
      }, 5000);
    }, 10000);

    return () => {
      clearInterval(timeInterval);
      clearInterval(connectionInterval);
    };
  });

  function openGlobalSearch() {
    // Navigate to search page or open search modal
    window.location.href = '/search';
  }

  function createNewCase() {
    // Navigate to case creation page
    window.location.href = '/cases/new';
  }

  function navigateToSection(section: string) {
    window.location.href = `/${section}`;
  }
</script>

<svelte:head>
  <title>YoRHa Detective Interface - Command Center</title>
</svelte:head>

<!-- YoRHa Detective Command Center -->
<div class="min-h-screen bg-gray-900 text-gray-100 font-mono">

  <!-- Navigation Header -->
  <nav class="bg-gray-800 border-b border-amber-500/30 sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center h-16">

        <!-- Left Side - Logo and Navigation -->
        <div class="flex items-center space-x-8">
          <div class="flex items-center space-x-3">
            <div class="w-3 h-3 bg-amber-500 rounded-full animate-pulse"></div>
            <h1 class="text-xl font-bold text-amber-300">YORHA DETECTIVE</h1>
            <div class="text-xs text-gray-400 hidden md:block">Investigation Interface • v8.13.2025</div>
          </div>

          <!-- Quick Navigation -->
          <div class="hidden md:flex space-x-1">
            <Button
              variant="ghost"
              size="sm"
              onclick={() => navigateToSection('cases')}
              class="text-gray-300 hover:text-amber-300 hover:bg-gray-700"
            >
              ACTIVE CASES
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onclick={() => navigateToSection('evidence')}
              class="text-gray-300 hover:text-amber-300 hover:bg-gray-700"
            >
              EVIDENCE
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onclick={() => navigateToSection('persons')}
              class="text-gray-300 hover:text-amber-300 hover:bg-gray-700"
            >
              PERSONS OF INTEREST
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onclick={() => navigateToSection('analysis')}
              class="text-gray-300 hover:text-amber-300 hover:bg-gray-700"
            >
              ANALYSIS
            </Button>
          </div>
        </div>

        <!-- Right Side - Actions and Status -->
        <div class="flex items-center space-x-4">
          <!-- Global Search -->
          <Button
            onclick={openGlobalSearch}
            variant="outline"
            size="sm"
            class="border-amber-500/30 text-amber-300 hover:bg-amber-500/10"
          >
            <Search class="w-4 h-4 mr-2" />
            GLOBAL SEARCH
          </Button>

          <!-- New Case Button -->
          <Button
            onclick={createNewCase}
            class="bg-amber-600 hover:bg-amber-500 text-gray-900"
          >
            + NEW CASE
          </Button>

          <!-- System Status -->
          <div class="flex items-center space-x-2 text-xs">
            <div class="w-2 h-2 bg-green-500 rounded-full"></div>
            <span class="text-gray-400">Online</span>
          </div>
        </div>
      </div>
    </div>
  </nav>

  <!-- Main Dashboard Content -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

    <!-- Command Center Header -->
    <div class="mb-8">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-3xl font-bold text-amber-300 mb-2">COMMAND CENTER</h2>
          <p class="text-gray-400">YoRHa Detective Interface • {currentTime}</p>
        </div>
        <div class="text-right">
          <div class="text-amber-300 font-bold">SYSTEM STATUS</div>
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span class="text-green-400">{systemStatus}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Stats Overview -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">

      <!-- Active Cases -->
      <Card class="bg-gray-800 border-amber-500/30">
        <CardContent class="p-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-amber-300 text-sm font-medium">Active Cases</p>
              <p class="text-3xl font-bold text-white">{stats.activeCases}</p>
            </div>
            <div class="p-3 bg-amber-500/10 rounded-lg">
              <FileText class="w-6 h-6 text-amber-500" />
            </div>
          </div>
        </CardContent>
      </Card>

      <!-- Evidence Items -->
      <Card class="bg-gray-800 border-blue-500/30">
        <CardContent class="p-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-blue-300 text-sm font-medium">Evidence Items</p>
              <p class="text-3xl font-bold text-white">{stats.evidenceItems}</p>
            </div>
            <div class="p-3 bg-blue-500/10 rounded-lg">
              <Database class="w-6 h-6 text-blue-500" />
            </div>
          </div>
        </CardContent>
      </Card>

      <!-- Persons of Interest -->
      <Card class="bg-gray-800 border-purple-500/30">
        <CardContent class="p-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-purple-300 text-sm font-medium">Persons of Interest</p>
              <p class="text-3xl font-bold text-white">{stats.personsOfInterest}</p>
            </div>
            <div class="p-3 bg-purple-500/10 rounded-lg">
              <Users class="w-6 h-6 text-purple-500" />
            </div>
          </div>
        </CardContent>
      </Card>

      <!-- Recent Activity -->
      <Card class="bg-gray-800 border-green-500/30">
        <CardContent class="p-6">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-green-300 text-sm font-medium">Recent Activity</p>
              <p class="text-3xl font-bold text-white">{stats.recentActivity}</p>
            </div>
            <div class="p-3 bg-green-500/10 rounded-lg">
              <Activity class="w-6 h-6 text-green-500" />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>

    <!-- Main Content Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">

      <!-- Active Cases Section -->
      <div class="lg:col-span-2">
        <Card class="bg-gray-800 border-amber-500/30 h-full">
          <CardHeader class="border-b border-gray-700">
            <div class="flex items-center justify-between">
              <CardTitle class="text-amber-300 flex items-center">
                <Target class="w-5 h-5 mr-2" />
                ACTIVE CASES
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onclick={() => navigateToSection('cases')}
                class="text-amber-300 hover:text-amber-200"
              >
                VIEW ALL →
              </Button>
            </div>
          </CardHeader>
          <CardContent class="p-6">
            <div class="space-y-4">
              {#each activeCases as case}
                <div class="border border-gray-700 rounded-lg p-4 hover:border-amber-500/30 transition-colors cursor-pointer">
                  <div class="flex items-start justify-between mb-3">
                    <div class="flex-1">
                      <h3 class="font-semibold text-white text-sm mb-1">{case.title}</h3>
                      <div class="flex items-center space-x-4 text-xs text-gray-400">
                        <span class="flex items-center">
                          <Clock class="w-3 h-3 mr-1" />
                          {case.lastUpdated}
                        </span>
                        <span>Items: {case.id === 'case-001' ? '11' : case.id === 'case-002' ? '6' : '4'}</span>
                      </div>
                    </div>
                    <div class="flex flex-col items-end space-y-2">
                      <Badge
                        variant={case.status === 'high' ? 'destructive' : 'secondary'}
                        class="text-xs {case.status === 'high' ? 'bg-red-500/20 text-red-400 border-red-500/30' : 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'}"
                      >
                        {case.status.toUpperCase()}
                      </Badge>
                      <Badge
                        variant={case.priority === 'active' ? 'default' : 'outline'}
                        class="text-xs {case.priority === 'active' ? 'bg-green-500/20 text-green-400 border-green-500/30' : 'bg-gray-500/20 text-gray-400 border-gray-500/30'}"
                      >
                        {case.priority.toUpperCase()}
                      </Badge>
                    </div>
                  </div>

                  <!-- Progress Bar -->
                  <div class="w-full bg-gray-700 rounded-full h-2">
                    <div
                      class="bg-amber-500 h-2 rounded-full transition-all duration-500"
                      style="width: {case.progress}%"
                    ></div>
                  </div>
                  <div class="text-xs text-gray-400 mt-1 text-right">{case.progress}% Complete</div>
                </div>
              {/each}
            </div>
          </CardContent>
        </Card>
      </div>

      <!-- System Status Section -->
      <div class="space-y-6">

        <!-- System Health -->
        <Card class="bg-gray-800 border-green-500/30">
          <CardHeader class="border-b border-gray-700">
            <CardTitle class="text-green-300 flex items-center">
              <Monitor class="w-5 h-5 mr-2" />
              SYSTEM STATUS
            </CardTitle>
          </CardHeader>
          <CardContent class="p-4">
            <div class="space-y-3">
              {#each Object.entries(systemHealth) as [system, health]}
                <div class="flex items-center justify-between">
                  <div class="flex items-center space-x-2">
                    <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span class="text-gray-300 text-sm capitalize">{system}</span>
                  </div>
                  <div class="text-xs text-green-400">{health.uptime}</div>
                </div>
              {/each}
            </div>

            <div class="mt-4 pt-4 border-t border-gray-700">
              <div class="text-xs text-gray-400 mb-2">System backup completed successfully</div>
              <div class="text-xs text-gray-500">10 minutes ago</div>
            </div>
          </CardContent>
        </Card>

        <!-- Quick Actions -->
        <Card class="bg-gray-800 border-blue-500/30">
          <CardHeader class="border-b border-gray-700">
            <CardTitle class="text-blue-300 flex items-center">
              <Zap class="w-5 h-5 mr-2" />
              QUICK ACTIONS
            </CardTitle>
          </CardHeader>
          <CardContent class="p-4">
            <div class="space-y-3">
              <Button
                variant="outline"
                size="sm"
                class="w-full justify-start border-gray-600 text-gray-300 hover:border-blue-500/30 hover:text-blue-300"
                onclick={() => navigateToSection('evidence/board')}
              >
                <Eye class="w-4 h-4 mr-2" />
                EVIDENCE BOARD
              </Button>

              <Button
                variant="outline"
                size="sm"
                class="w-full justify-start border-gray-600 text-gray-300 hover:border-purple-500/30 hover:text-purple-300"
                onclick={() => navigateToSection('timeline')}
              >
                <Activity class="w-4 h-4 mr-2" />
                TIMELINE ANALYSIS
              </Button>

              <Button
                variant="outline"
                size="sm"
                class="w-full justify-start border-gray-600 text-gray-300 hover:border-amber-500/30 hover:text-amber-300"
                onclick={() => navigateToSection('terminal')}
              >
                <Monitor class="w-4 h-4 mr-2" />
                TERMINAL ACCESS
              </Button>
            </div>
          </CardContent>
        </Card>

        <!-- AI Assistant Trigger -->
        <Card class="bg-gray-800 border-cyan-500/30">
          <CardHeader class="border-b border-gray-700">
            <CardTitle class="text-cyan-300 flex items-center">
              <Brain class="w-5 h-5 mr-2" />
              AI ASSISTANT
            </CardTitle>
          </CardHeader>
          <CardContent class="p-4">
            <p class="text-gray-400 text-sm mb-4">Access YoRHa Detective AI for analysis, insights, and automated assistance.</p>
            <Button
              onclick={() => (showAIAssistant = true)}
              class="w-full bg-cyan-600 hover:bg-cyan-500 text-white"
            >
              <Brain class="w-4 h-4 mr-2" />
              ACTIVATE AI
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  </main>

  <!-- YoRHa AI Assistant Component -->
  <YorhaAIAssistant bind:isOpen={showAIAssistant} />
</div>

<style>
  /* @unocss-include */

  /* Additional YoRHa themed animations */
  @keyframes pulse-glow {
    0%, 100% {
      box-shadow: 0 0 5px rgba(217, 119, 6, 0.3);
    }
    50% {
      box-shadow: 0 0 20px rgba(217, 119, 6, 0.6);
    }
  }

  .animate-pulse-glow {
    animation: pulse-glow 2s ease-in-out infinite;
  }

  /* Custom hover effects */
  .hover\:glow:hover {
    box-shadow: 0 0 15px rgba(217, 119, 6, 0.4);
    transition: box-shadow 0.3s ease;
  }
</style>
