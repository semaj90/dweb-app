<!-- YoRHa Command Center Dashboard Component -->
<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import type { Props } from "$lib/types/global";
  import { onMount } from 'svelte';
  import { goto } from "$app/navigation";
  import YorhaAIAssistant from "$lib/components/ai/YorhaAIAssistant.svelte";
  
  // Props
  let { systemData } = $props<{
    systemData: {
      activeCases: number;
      evidenceItems: number;
      aiQueries: number;
      gpuUtilization: number;
      networkLatency: number;
    };
  }>();

  // Dashboard state
  let selectedCard = $state<string | null>(null);
  let animationPhase = $state(0);
  let recentActivity = $state([
    { id: 1, action: 'Case Analysis Completed', target: 'CASE-2024-087', time: '2 minutes ago', type: 'success' },
    { id: 2, action: 'Evidence Upload', target: 'Digital Forensics Report', time: '5 minutes ago', type: 'info' },
    { id: 3, action: 'AI Query Processed', target: 'Contract Liability Analysis', time: '8 minutes ago', type: 'ai' },
    { id: 4, action: 'System Alert', target: 'GPU Memory Optimization', time: '12 minutes ago', type: 'warning' },
    { id: 5, action: 'New Case Created', target: 'CASE-2024-088', time: '15 minutes ago', type: 'success' }
  ]);

  // Quick actions
  const quickActions = [
    { id: 'new-case', label: 'Create New Case', icon: '📁', route: '/cases', color: 'blue' },
    { id: 'upload-evidence', label: 'Upload Evidence', icon: '🔍', route: '/evidence/upload', color: 'green' },
    { id: 'ai-analysis', label: 'AI Analysis', icon: '🤖', route: '/ai-assistant', color: 'purple' },
    { id: 'ai-assistant-3d', label: '3D AI Assistant', icon: '🎮', route: '/ai-assistant-demo', color: 'pink' },
    { id: 'search-global', label: 'Global Search', icon: '🔎', route: '/search', color: 'orange' },
    { id: 'generate-report', label: 'Generate Report', icon: '📊', route: '/report-builder', color: 'teal' },
    { id: 'memory-dashboard', label: 'Memory Graph', icon: '🧠', route: '/memory-dashboard', color: 'cyan' }
  ];

  // System health indicators
  let systemHealth = $derived(() => {
    const avgLoad = (systemData.systemLoad + systemData.gpuUtilization + systemData.memoryUsage) / 3;
    if (avgLoad > 85) return { status: 'critical', color: 'red', message: 'System under heavy load' };
    if (avgLoad > 70) return { status: 'warning', color: 'yellow', message: 'Elevated resource usage' };
    return { status: 'optimal', color: 'green', message: 'All systems operational' };
  });

  // Animation cycle
  onMount(() => {
    const animationInterval = setInterval(() => {
      animationPhase = (animationPhase + 1) % 4;
    }, 2000);

    return () => clearInterval(animationInterval);
  });

  function handleQuickAction(action: any) {
    selectedCard = action.id;
    setTimeout(() => {
      goto(action.route);
    }, 300);
  }

  function getActivityIcon(type: string): string {
    switch (type) {
      case 'success': return '✅';
      case 'info': return 'ℹ️';
      case 'ai': return '🤖';
      case 'warning': return '⚠️';
    }
  }

  function getActivityColor(type: string): string {
    switch (type) {
      case 'success': return 'border-green-400 bg-green-400/10 text-green-300';
      case 'info': return 'border-blue-400 bg-blue-400/10 text-blue-300';
      case 'ai': return 'border-purple-400 bg-purple-400/10 text-purple-300';
      case 'warning': return 'border-yellow-400 bg-yellow-400/10 text-yellow-300';
    }
  }

  function getActionColor(color: string): string {
    switch (color) {
      case 'blue': return 'border-blue-400 bg-blue-400/10 hover:bg-blue-400/20 text-blue-300';
      case 'green': return 'border-green-400 bg-green-400/10 hover:bg-green-400/20 text-green-300';
      case 'purple': return 'border-purple-400 bg-purple-400/10 hover:bg-purple-400/20 text-purple-300';
      case 'orange': return 'border-orange-400 bg-orange-400/10 hover:bg-orange-400/20 text-orange-300';
      case 'teal': return 'border-teal-400 bg-teal-400/10 hover:bg-teal-400/20 text-teal-300';
      case 'pink': return 'border-pink-400 bg-pink-400/10 hover:bg-pink-400/20 text-pink-300';
      case 'cyan': return 'border-cyan-400 bg-cyan-400/10 hover:bg-cyan-400/20 text-cyan-300';
    }
  }
</script>

<!-- Command Center Dashboard -->
<div class="yorha-command-center min-h-full bg-yorha-dark text-yorha-light p-6">
  
  <!-- Header Section -->
  <div class="dashboard-header mb-8">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold text-yorha-accent-warm mb-2">
          YoRHa Command Center
        </h1>
        <p class="text-yorha-muted">
          Legal AI Operations Dashboard - {systemHealth.message}
        </p>
      </div>
      <div class="status-indicator flex items-center space-x-3">
        <div class="w-4 h-4 rounded-full bg-{systemHealth.color}-400 animate-pulse"></div>
        <span class="text-{systemHealth.color}-400 font-bold uppercase">
          {systemHealth.status}
        </span>
      </div>
    </div>
  </div>

  <!-- Main Dashboard Grid -->
  <div class="dashboard-grid grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
    
    <!-- System Overview Cards -->
    <div class="overview-cards lg:col-span-2 grid grid-cols-2 md:grid-cols-4 gap-4">
      
      <!-- Active Cases -->
      <div class="metric-card bg-yorha-darker border border-yorha-accent-warm/30 rounded-lg p-4 hover:border-yorha-accent-warm/50 transition-all duration-300">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs text-yorha-muted uppercase tracking-wider">Active Cases</span>
          <span class="text-2xl">📁</span>
        </div>
        <div class="text-2xl font-bold text-yorha-accent-warm mb-1">
          {systemData.activeCases}
        </div>
        <div class="text-xs text-green-400">+2 this week</div>
      </div>

      <!-- Evidence Items -->
      <div class="metric-card bg-yorha-darker border border-yorha-accent-warm/30 rounded-lg p-4 hover:border-yorha-accent-warm/50 transition-all duration-300">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs text-yorha-muted uppercase tracking-wider">Evidence</span>
          <span class="text-2xl">🔍</span>
        </div>
        <div class="text-2xl font-bold text-yorha-accent-warm mb-1">
          {systemData.evidenceItems}
        </div>
        <div class="text-xs text-blue-400">+15 today</div>
      </div>

      <!-- Persons of Interest -->
      <div class="metric-card bg-yorha-darker border border-yorha-accent-warm/30 rounded-lg p-4 hover:border-yorha-accent-warm/50 transition-all duration-300">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs text-yorha-muted uppercase tracking-wider">Persons</span>
          <span class="text-2xl">👤</span>
        </div>
        <div class="text-2xl font-bold text-yorha-accent-warm mb-1">
          {systemData.personsOfInterest}
        </div>
        <div class="text-xs text-yellow-400">3 flagged</div>
      </div>

      <!-- AI Queries -->
      <div class="metric-card bg-yorha-darker border border-yorha-accent-warm/30 rounded-lg p-4 hover:border-yorha-accent-warm/50 transition-all duration-300">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs text-yorha-muted uppercase tracking-wider">AI Queries</span>
          <span class="text-2xl">🤖</span>
        </div>
        <div class="text-2xl font-bold text-yorha-accent-warm mb-1">
          {systemData.aiQueries}
        </div>
        <div class="text-xs text-purple-400">94% accuracy</div>
      </div>
    </div>

    <!-- System Health Panel -->
    <div class="system-health bg-yorha-darker border border-yorha-accent-warm/30 rounded-lg p-6">
      <h3 class="text-lg font-bold text-yorha-accent-warm mb-4">System Health</h3>
      
      <div class="health-metrics space-y-4">
        <div class="metric">
          <div class="flex justify-between items-center mb-1">
            <span class="text-sm text-yorha-muted">CPU Load</span>
            <span class="text-sm font-mono">{systemData.systemLoad}%</span>
          </div>
          <div class="progress-bar w-full h-2 bg-yorha-dark rounded-full overflow-hidden">
            <div 
              class="h-full rounded-full transition-all duration-300 {systemData.systemLoad > 80 ? 'bg-red-500' : systemData.systemLoad > 60 ? 'bg-yellow-500' : 'bg-green-500'}"
              style="width: {systemData.systemLoad}%"
            ></div>
          </div>
        </div>

        <div class="metric">
          <div class="flex justify-between items-center mb-1">
            <span class="text-sm text-yorha-muted">GPU Usage</span>
            <span class="text-sm font-mono">{systemData.gpuUtilization}%</span>
          </div>
          <div class="progress-bar w-full h-2 bg-yorha-dark rounded-full overflow-hidden">
            <div 
              class="h-full rounded-full transition-all duration-300 {systemData.gpuUtilization > 80 ? 'bg-red-500' : systemData.gpuUtilization > 60 ? 'bg-yellow-500' : 'bg-green-500'}"
              style="width: {systemData.gpuUtilization}%"
            ></div>
          </div>
        </div>

        <div class="metric">
          <div class="flex justify-between items-center mb-1">
            <span class="text-sm text-yorha-muted">Memory</span>
            <span class="text-sm font-mono">{systemData.memoryUsage}%</span>
          </div>
          <div class="progress-bar w-full h-2 bg-yorha-dark rounded-full overflow-hidden">
            <div 
              class="h-full rounded-full transition-all duration-300 {systemData.memoryUsage > 80 ? 'bg-red-500' : systemData.memoryUsage > 60 ? 'bg-yellow-500' : 'bg-green-500'}"
              style="width: {systemData.memoryUsage}%"
            ></div>
          </div>
        </div>

        <div class="metric">
          <div class="flex justify-between items-center mb-1">
            <span class="text-sm text-yorha-muted">Network</span>
            <span class="text-sm font-mono">{systemData.networkLatency}ms</span>
          </div>
          <div class="network-status flex items-center space-x-2">
            <div class="w-2 h-2 rounded-full {systemData.networkLatency < 50 ? 'bg-green-400' : systemData.networkLatency < 100 ? 'bg-yellow-400' : 'bg-red-400'} animate-pulse"></div>
            <span class="text-xs text-yorha-muted">
              {systemData.networkLatency < 50 ? 'Excellent' : systemData.networkLatency < 100 ? 'Good' : 'Poor'}
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Quick Actions Section -->
  <div class="quick-actions mb-8">
    <h2 class="text-xl font-bold text-yorha-accent-warm mb-4">Quick Actions</h2>
    <div class="actions-grid grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {#each quickActions as action}
        <button
          class="action-card border rounded-lg p-4 text-center transition-all duration-300 hover:scale-105 hover:shadow-lg {getActionColor(action.color)} {selectedCard === action.id ? 'scale-95' : ''}"
          onclick={() => handleQuickAction(action)}
        >
          <div class="text-3xl mb-2">{action.icon}</div>
          <div class="text-sm font-medium">{action.label}</div>
        </button>
      {/each}
    </div>
  </div>

  <!-- Recent Activity Section -->
  <div class="recent-activity">
    <h2 class="text-xl font-bold text-yorha-accent-warm mb-4">Recent Activity</h2>
    <div class="activity-list space-y-3">
      {#each recentActivity as activity}
        <div class="activity-item border rounded-lg p-4 flex items-center justify-between {getActivityColor(activity.type)}">
          <div class="flex items-center space-x-3">
            <span class="text-lg">{getActivityIcon(activity.type)}</span>
            <div>
              <div class="font-medium">{activity.action}</div>
              <div class="text-sm opacity-75">{activity.target}</div>
            </div>
          </div>
          <div class="text-xs opacity-75">
            {activity.time}
          </div>
        </div>
      {/each}
    </div>
  </div>

  <!-- YoRHa AI Assistant - Integrated with Command Center -->
  <YorhaAIAssistant 
    theme="yorha"
    enableGPUAcceleration={true}
    enableMCPIntegration={true}
    initialOpen={false}
  />
</div>

<style>
  .yorha-command-center {
    --yorha-primary: #c4b49a;
    --yorha-secondary: #b5a48a;
    --yorha-accent-warm: #4a4a4a;
    --yorha-accent-cool: #6b6b6b;
    --yorha-light: #ffffff;
    --yorha-muted: #f0f0f0;
    --yorha-dark: #aca08a;
    --yorha-darker: #b8ad98;
    
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, var(--yorha-dark) 0%, var(--yorha-darker) 100%);
  }

  .metric-card {
    backdrop-filter: blur(10px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(212, 175, 55, 0.15);
  }

  .action-card {
    backdrop-filter: blur(10px);
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .activity-item {
    backdrop-filter: blur(10px);
    transition: all 0.2s ease;
  }

  .activity-item:hover {
    transform: translateX(4px);
  }

  .progress-bar {
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  /* Animation effects */
  @keyframes pulse-glow {
    0%, 100% { 
      box-shadow: 0 0 5px currentColor; 
    }
    50% { 
      box-shadow: 0 0 15px currentColor, 0 0 25px currentColor; 
    }
  }

  .animate-pulse {
    animation: pulse-glow 2s infinite;
  }

  /* Responsive adjustments */
  @media (max-width: 768px) {
    .yorha-command-center {
      padding: 16px;
    }
    
    .dashboard-header h1 {
      font-size: 24px;
    }
    
    .actions-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>


<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';
interface Props extends CommonProps {}
</script>
