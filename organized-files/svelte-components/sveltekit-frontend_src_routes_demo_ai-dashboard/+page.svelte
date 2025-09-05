<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { createActor } from 'xstate';
  import { evidenceProcessingMachine, type EvidenceProcessingContext } from '$lib/state/evidenceProcessingMachine';
  import { documentUploadMachine, type DocumentUploadContext } from '$lib/state/documentUploadMachine';
  import { performanceMonitor, formatMetricValue, getHealthStatusColor, getAlertSeverityColor } from '$lib/services/performanceMonitor';
  import type { AIMetrics, QueueMetrics, CacheMetrics, SystemMetrics } from '$lib/services/performanceMonitor';
  
  // State machine actors
  let evidenceActor: any = null;
  let uploadActor: any = null;
  let evidenceState = $state({ value: 'idle', context: {} as EvidenceProcessingContext });
  let uploadState = $state({ value: 'idle', context: {} as DocumentUploadContext });
  
  // Performance metrics stores
  let aiMetrics = $state<AIMetrics | null>(null);
  let queueMetrics = $state<QueueMetrics | null>(null);
  let cacheMetrics = $state<CacheMetrics | null>(null);
  let systemMetrics = $state<SystemMetrics | null>(null);
  let healthScore = $state(0);
  let alerts = $state<SystemMetrics['activeAlerts']>([]);
  
  // Demo state
  let selectedTab = $state<'overview' | 'processing' | 'queues' | 'cache' | 'system'>('overview');
  let isProcessingDemo = $state(false);
  let demoLog = $state<Array<{ timestamp: string level: 'info' | 'success' | 'warning' | 'error'; message: string }>>([]);
  
  // Demo document content
  const DEMO_DOCUMENT = `
CASE FILE: People v. Johnson
Case Number: CR-2024-005678

INCIDENT REPORT:
On March 15, 2024, at approximately 2:30 PM, officers responded to a report of theft at TechMart Electronics, 
located at 789 Commerce Street, Downtown District. Witness statements indicate that the suspect, 
identified as Marcus Johnson (DOB: 01/15/1985), allegedly concealed multiple electronic devices 
and attempted to leave the store without payment.

EVIDENCE COLLECTED:
1. Store surveillance footage showing suspect's actions
2. Recovered merchandise: iPad Pro (SKU: IPD-PRO-128), valued at $1,099.99
3. Recovered merchandise: Wireless headphones (SKU: WH-BT-500), valued at $349.99
4. Store receipt showing unpaid items
5. Suspect's statement recorded during interview

WITNESS STATEMENTS:
Store Manager Sarah Chen: "I observed the suspect placing items in a large bag and bypassing the checkout area."
Loss Prevention Officer David Wilson: "Suspect was cooperative during detention and admitted to taking the items."
Customer Jennifer Martinez: "I saw someone acting suspiciously near the electronics display."

BACKGROUND CHECK:
Suspect has prior convictions for petty theft (2019) and shoplifting (2021). Currently unemployed.
Address: 456 Elm Street, Apt 3B, Riverside District.

LEGAL ANALYSIS:
Under Penal Code Section 459.5, this constitutes shoplifting as the value exceeds $950.
Potential charges: Grand theft, burglary in the second degree.

NEXT STEPS:
Arraignment scheduled for April 2, 2024, at 10:00 AM, Superior Court Department 5.
Defendant released on $5,000 bail.
`;

  onMount(() => {
    // Initialize state machine actors
    evidenceActor = createActor(evidenceProcessingMachine);
    uploadActor = createActor(documentUploadMachine);
    
    evidenceActor.subscribe((state: any) => {
      evidenceState = state;
    });
    
    uploadActor.subscribe((state: any) => {
      uploadState = state;
    });
    
    evidenceActor.start();
    uploadActor.start();
    
    // Subscribe to performance metrics
    const unsubscribeAI = performanceMonitor.aiMetrics.subscribe(value => aiMetrics = value);
    const unsubscribeQueue = performanceMonitor.queueMetrics.subscribe(value => queueMetrics = value);
    const unsubscribeCache = performanceMonitor.cacheMetrics.subscribe(value => cacheMetrics = value);
    const unsubscribeSystem = performanceMonitor.systemMetrics.subscribe(value => systemMetrics = value);
    const unsubscribeHealth = performanceMonitor.healthScore.subscribe(value => healthScore = value);
    const unsubscribeAlerts = performanceMonitor.alerts.subscribe(value => alerts = value);
    
    // Start monitoring
    performanceMonitor.startMonitoring(10000); // Every 10 seconds for demo
    
    // Add demo start log
    addLog('info', 'AI Dashboard Demo initialized');
    
    return () => {
      unsubscribeAI();
      unsubscribeQueue();
      unsubscribeCache();
      unsubscribeSystem();
      unsubscribeHealth();
      unsubscribeAlerts();
      performanceMonitor.stopMonitoring();
    };
  });
  
  onDestroy(() => {
    evidenceActor?.stop();
    uploadActor?.stop();
    performanceMonitor.stopMonitoring();
  });
  
  // Demo functions
  function startProcessingDemo() {
    isProcessingDemo = true;
    addLog('info', 'Starting evidence processing demo...');
    
    // Start evidence processing
    evidenceActor.send({
      type: 'START_PROCESSING',
      evidenceId: `demo-evidence-${Date.now()}`,
      caseId: 'demo-case-001',
      userId: 'demo-user',
      filename: 'demo-legal-document.txt',
      content: DEMO_DOCUMENT,
      metadata: {
        title: 'People v. Johnson - Case File',
        type: 'legal_document',
        source: 'demo'
      }
    });
    
    addLog('success', 'Evidence processing started');
  }
  
  function startUploadDemo() {
    // Create a fake file for demo
    const blob = new Blob([DEMO_DOCUMENT], { type: 'text/plain' });
    const file = new File([blob], 'demo-case-file.txt', { type: 'text/plain' });
    
    addLog('info', 'Starting document upload demo...');
    
    uploadActor.send({
      type: 'SELECT_FILE',
      file,
      caseId: 'demo-case-001',
      userId: 'demo-user',
      title: 'Demo Case File',
      description: 'Demonstration of document upload and processing workflow',
      tags: ['demo', 'case-file', 'legal-document']
    });
    
    addLog('success', 'Document upload started');
  }
  
  function retryProcessing() {
    evidenceActor.send({ type: 'RETRY' });
    addLog('info', 'Retrying evidence processing...');
  }
  
  function cancelProcessing() {
    evidenceActor.send({ type: 'CANCEL' });
    uploadActor.send({ type: 'CANCEL_UPLOAD' });
    isProcessingDemo = false;
    addLog('warning', 'Processing cancelled');
  }
  
  function clearLogs() {
    demoLog = [];
  }
  
  function addLog(level: 'info' | 'success' | 'warning' | 'error', message: string) {
    demoLog = [
      ...demoLog,
      {
        timestamp: new Date().toLocaleTimeString(),
        level,
        message
      }
    ].slice(-50); // Keep last 50 logs
  }
  
  // Reactive effects for state changes
  $effect(() => {
    if (evidenceState.value === 'completed') {
      isProcessingDemo = false;
      addLog('success', 'Evidence processing completed successfully!');
    } else if (evidenceState.value === 'failed') {
      isProcessingDemo = false;
      addLog('error', `Evidence processing failed: ${evidenceState.context.error}`);
    } else if (evidenceState.value === 'cancelled') {
      isProcessingDemo = false;
      addLog('warning', 'Evidence processing cancelled');
    }
  });
  
  $effect(() => {
    if (uploadState.value === 'completed') {
      addLog('success', 'Document upload and processing completed!');
    } else if (uploadState.value === 'uploadFailed') {
      addLog('error', `Document upload failed: ${uploadState.context.error}`);
    }
  });
  
  // Helper functions
  function getStateColor(state: string): string {
    if (state.includes('error') || state.includes('failed')) return 'text-red-500';
    if (state === 'completed') return 'text-green-500';
    if (state.includes('processing') || state.includes('uploading')) return 'text-blue-500';
    return 'text-gray-500';
  }
  
  function getLogColor(level: string): string {
    switch (level) {
      case 'success': return 'text-green-500';
      case 'warning': return 'text-yellow-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-600';
    }
  }
</script>

<div class="min-h-screen bg-gray-50 p-6">
  <!-- Header -->
  <header class="mb-8">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold text-gray-900">AI Pipeline Demo Dashboard</h1>
        <p class="text-gray-600 mt-2">Comprehensive demonstration of legal AI system capabilities</p>
      </div>
      
      <!-- System Health Indicator -->
      <div class="flex items-center space-x-4">
        <div class="flex items-center space-x-2">
          <div class="w-3 h-3 rounded-full {healthScore >= 80 ? 'bg-green-500' : healthScore >= 60 ? 'bg-yellow-500' : 'bg-red-500'}"></div>
          <span class="text-sm font-medium">System Health: {Math.round(healthScore)}%</span>
        </div>
        
        {#if alerts.length > 0}
          <div class="flex items-center space-x-2 text-red-500">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
            </svg>
            <span class="text-sm font-medium">{alerts.length} Alert{alerts.length !== 1 ? 's' : ''}</span>
          </div>
        {/if}
      </div>
    </div>
  </header>

  <!-- Tab Navigation -->
  <nav class="mb-8">
    <div class="border-b border-gray-200">
      <div class="flex space-x-8">
        {#each [
          { id: 'overview', label: 'Overview' },
          { id: 'processing', label: 'Processing Demo' },
          { id: 'queues', label: 'Queue Metrics' },
          { id: 'cache', label: 'Cache Performance' },
          { id: 'system', label: 'System Health' }
        ] as tab}
          <button
            class="py-2 px-1 border-b-2 font-medium text-sm transition-colors {selectedTab === tab.id 
              ? 'border-blue-500 text-blue-600' 
              : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}"
            onclick={() => selectedTab = tab.id as 'overview' | 'processing' | 'queues' | 'cache' | 'system'}
          >
            {tab.label}
          </button>
        {/each}
      </div>
    </div>
  </nav>

  <!-- Tab Content -->
  {#if selectedTab === 'overview'}
    <!-- Overview Tab -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <!-- AI Processing Stats -->
      <div class="bg-white rounded-lg shadow p-6">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <div class="flex items-center justify-center w-8 h-8 bg-blue-100 rounded-md">
              <svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
              </svg>
            </div>
          </div>
          <div class="ml-5 w-0 flex-1">
            <dl>
              <dt class="text-sm font-medium text-gray-500 truncate">Documents Processed</dt>
              <dd class="text-lg font-medium text-gray-900">{aiMetrics?.documentsProcessed || 0}</dd>
            </dl>
          </div>
        </div>
      </div>

      <!-- Queue Status -->
      <div class="bg-white rounded-lg shadow p-6">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <div class="flex items-center justify-center w-8 h-8 bg-green-100 rounded-md">
              <svg class="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
              </svg>
            </div>
          </div>
          <div class="ml-5 w-0 flex-1">
            <dl>
              <dt class="text-sm font-medium text-gray-500 truncate">Jobs Completed</dt>
              <dd class="text-lg font-medium text-gray-900">{queueMetrics?.totalCompleted || 0}</dd>
            </dl>
          </div>
        </div>
      </div>

      <!-- Cache Performance -->
      <div class="bg-white rounded-lg shadow p-6">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <div class="flex items-center justify-center w-8 h-8 bg-purple-100 rounded-md">
              <svg class="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
              </svg>
            </div>
          </div>
          <div class="ml-5 w-0 flex-1">
            <dl>
              <dt class="text-sm font-medium text-gray-500 truncate">Cache Hit Rate</dt>
              <dd class="text-lg font-medium text-gray-900">{formatMetricValue(cacheMetrics?.hitRate || 0, 'percentage')}</dd>
            </dl>
          </div>
        </div>
      </div>

      <!-- System Health -->
      <div class="bg-white rounded-lg shadow p-6">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <div class="flex items-center justify-center w-8 h-8 bg-red-100 rounded-md">
              <svg class="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
              </svg>
            </div>
          </div>
          <div class="ml-5 w-0 flex-1">
            <dl>
              <dt class="text-sm font-medium text-gray-500 truncate">System Health</dt>
              <dd class="text-lg font-medium text-gray-900">{Math.round(healthScore)}%</dd>
            </dl>
          </div>
        </div>
      </div>
    </div>

    <!-- Active Alerts -->
    {#if alerts.length > 0}
      <div class="bg-white rounded-lg shadow p-6 mb-8">
        <h3 class="text-lg font-medium text-gray-900 mb-4">Active Alerts</h3>
        <div class="space-y-3">
          {#each alerts as alert}
            <div class="flex items-center p-3 bg-gray-50 rounded-md">
              <div class="flex-shrink-0">
                <div class="w-2 h-2 rounded-full {getAlertSeverityColor(alert.severity).replace('text-', 'bg-')}"></div>
              </div>
              <div class="ml-3 flex-1">
                <p class="text-sm font-medium text-gray-900">{alert.message}</p>
                <p class="text-xs text-gray-500">{alert.component} • {new Date(alert.timestamp).toLocaleTimeString()}</p>
              </div>
              <div class="ml-3">
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium {getAlertSeverityColor(alert.severity)} bg-opacity-10">
                  {alert.severity}
                </span>
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    <!-- Recent Activity -->
    <div class="bg-white rounded-lg shadow p-6">
      <h3 class="text-lg font-medium text-gray-900 mb-4">Activity Log</h3>
      <div class="max-h-64 overflow-y-auto">
        {#if demoLog.length === 0}
          <p class="text-gray-500 text-sm">No recent activity</p>
        {:else}
          <div class="space-y-2">
            {#each demoLog.slice().reverse() as log}
              <div class="flex items-center text-sm">
                <span class="text-gray-400 w-20">{log.timestamp}</span>
                <span class="w-2 h-2 rounded-full mx-3 {getLogColor(log.level).replace('text-', 'bg-')}"></span>
                <span class="{getLogColor(log.level)}">{log.message}</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

  {:else if selectedTab === 'processing'}
    <!-- Processing Demo Tab -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <!-- State Machine Visualization -->
      <div class="bg-white rounded-lg shadow p-6">
        <h3 class="text-lg font-medium text-gray-900 mb-4">Evidence Processing State Machine</h3>
        
        <div class="space-y-4">
          <!-- Current State -->
          <div class="p-4 bg-gray-50 rounded-md">
            <div class="flex items-center justify-between">
              <span class="text-sm font-medium text-gray-700">Current State:</span>
              <span class="px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800 {getStateColor(evidenceState.value)}">
                {evidenceState.value}
              </span>
            </div>
            {#if evidenceState.context.stage}
              <div class="mt-2 flex items-center justify-between">
                <span class="text-sm font-medium text-gray-700">Stage:</span>
                <span class="text-sm text-gray-600">{evidenceState.context.stage}</span>
              </div>
            {/if}
          </div>

          <!-- Progress Bar -->
          {#if evidenceState.context.progress > 0}
            <div class="space-y-2">
              <div class="flex items-center justify-between">
                <span class="text-sm font-medium text-gray-700">Progress:</span>
                <span class="text-sm text-gray-600">{evidenceState.context.progress}%</span>
              </div>
              <div class="w-full bg-gray-200 rounded-full h-2">
                <div class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: {evidenceState.context.progress}%"></div>
              </div>
            </div>
          {/if}

          <!-- Processing Times -->
          {#if Object.keys(evidenceState.context.processingTimes || {}).length > 0}
            <div class="space-y-2">
              <span class="text-sm font-medium text-gray-700">Processing Times:</span>
              <div class="space-y-1">
                {#each Object.entries(evidenceState.context.processingTimes) as [stage, time]}
                  <div class="flex items-center justify-between text-sm">
                    <span class="text-gray-600">{stage}:</span>
                    <span class="text-gray-900">{formatMetricValue(time, 'time')}</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}

          <!-- Error Display -->
          {#if evidenceState.context.error}
            <div class="p-3 bg-red-50 rounded-md">
              <p class="text-sm text-red-800">{evidenceState.context.error}</p>
            </div>
          {/if}
        </div>

        <!-- Actions -->
        <div class="mt-6 flex space-x-3">
          <button
            class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            onclick={startProcessingDemo}
            disabled={isProcessingDemo || evidenceState.value === 'processing'}
          >
            Start Processing Demo
          </button>
          
          {#if evidenceState.value === 'error'}
            <button
              class="px-4 py-2 bg-yellow-600 text-white rounded-md hover:bg-yellow-700"
              onclick={retryProcessing}
            >
              Retry
            </button>
          {/if}
          
          {#if isProcessingDemo}
            <button
              class="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
              onclick={cancelProcessing}
            >
              Cancel
            </button>
          {/if}
        </div>
      </div>

      <!-- Document Upload State Machine -->
      <div class="bg-white rounded-lg shadow p-6">
        <h3 class="text-lg font-medium text-gray-900 mb-4">Document Upload State Machine</h3>
        
        <div class="space-y-4">
          <!-- Upload State -->
          <div class="p-4 bg-gray-50 rounded-md">
            <div class="flex items-center justify-between">
              <span class="text-sm font-medium text-gray-700">Current State:</span>
              <span class="px-2 py-1 text-xs font-medium rounded-full bg-green-100 text-green-800 {getStateColor(uploadState.value)}">
                {uploadState.value}
              </span>
            </div>
          </div>

          <!-- Upload Progress -->
          {#if uploadState.context.uploadProgress > 0}
            <div class="space-y-2">
              <div class="flex items-center justify-between">
                <span class="text-sm font-medium text-gray-700">Upload Progress:</span>
                <span class="text-sm text-gray-600">{uploadState.context.uploadProgress}%</span>
              </div>
              <div class="w-full bg-gray-200 rounded-full h-2">
                <div class="bg-green-600 h-2 rounded-full transition-all duration-300" style="width: {uploadState.context.uploadProgress}%"></div>
              </div>
            </div>
          {/if}

          <!-- Validation Errors -->
          {#if uploadState.context.validationErrors?.length > 0}
            <div class="space-y-2">
              <span class="text-sm font-medium text-red-700">Validation Errors:</span>
              <div class="space-y-1">
                {#each uploadState.context.validationErrors as error}
                  <p class="text-sm text-red-600">• {error}</p>
                {/each}
              </div>
            </div>
          {/if}

          <!-- File Info -->
          {#if uploadState.context.filename}
            <div class="space-y-2">
              <span class="text-sm font-medium text-gray-700">File Info:</span>
              <div class="space-y-1 text-sm text-gray-600">
                <p>Name: {uploadState.context.filename}</p>
                <p>Size: {formatMetricValue(uploadState.context.fileSize, 'size')}</p>
                <p>Type: {uploadState.context.mimeType}</p>
              </div>
            </div>
          {/if}
        </div>

        <!-- Upload Actions -->
        <div class="mt-6">
          <button
            class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
            onclick={startUploadDemo}
          >
            Start Upload Demo
          </button>
        </div>
      </div>
    </div>

    <!-- Demo Document Preview -->
    <div class="mt-8 bg-white rounded-lg shadow p-6">
      <h3 class="text-lg font-medium text-gray-900 mb-4">Demo Document Content</h3>
      <div class="bg-gray-50 rounded-md p-4 max-h-64 overflow-y-auto">
        <pre class="text-sm text-gray-700 whitespace-pre-wrap">{DEMO_DOCUMENT}</pre>
      </div>
    </div>

  {:else if selectedTab === 'queues'}
    <!-- Queue Metrics Tab -->
    <div class="space-y-6">
      <!-- Queue Overview -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div class="bg-white rounded-lg shadow p-6">
          <div class="text-2xl font-bold text-gray-900">{queueMetrics?.totalJobs || 0}</div>
          <div class="text-sm text-gray-500">Total Jobs</div>
        </div>
        <div class="bg-white rounded-lg shadow p-6">
          <div class="text-2xl font-bold text-green-600">{queueMetrics?.totalCompleted || 0}</div>
          <div class="text-sm text-gray-500">Completed</div>
        </div>
        <div class="bg-white rounded-lg shadow p-6">
          <div class="text-2xl font-bold text-red-600">{queueMetrics?.totalFailed || 0}</div>
          <div class="text-sm text-gray-500">Failed</div>
        </div>
        <div class="bg-white rounded-lg shadow p-6">
          <div class="text-2xl font-bold text-blue-600">{queueMetrics?.overallThroughput.toFixed(1) || '0.0'}</div>
          <div class="text-sm text-gray-500">Jobs/min</div>
        </div>
      </div>

      <!-- Individual Queue Status -->
      <div class="bg-white rounded-lg shadow">
        <div class="px-6 py-4 border-b border-gray-200">
          <h3 class="text-lg font-medium text-gray-900">Queue Status</h3>
        </div>
        <div class="p-6">
          {#if queueMetrics?.queues}
            <div class="overflow-x-auto">
              <table class="min-w-full divide-y divide-gray-200">
                <thead>
                  <tr>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Queue</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Waiting</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Active</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Completed</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Failed</th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Throughput</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-gray-200">
                  {#each Object.entries(queueMetrics.queues) as [queueName, stats]}
                    <tr>
                      <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{queueName}</td>
                      <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{stats.waiting}</td>
                      <td class="px-6 py-4 whitespace-nowrap text-sm text-blue-600">{stats.active}</td>
                      <td class="px-6 py-4 whitespace-nowrap text-sm text-green-600">{stats.completed}</td>
                      <td class="px-6 py-4 whitespace-nowrap text-sm text-red-600">{stats.failed}</td>
                      <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{stats.throughput.toFixed(2)}/min</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {:else}
            <p class="text-gray-500 text-center py-8">No queue data available</p>
          {/if}
        </div>
      </div>
    </div>

  {:else if selectedTab === 'cache'}
    <!-- Cache Performance Tab -->
    <div class="space-y-6">
      <!-- Cache Overview -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div class="bg-white rounded-lg shadow p-6">
          <div class="text-2xl font-bold text-green-600">{formatMetricValue(cacheMetrics?.hitRate || 0, 'percentage')}</div>
          <div class="text-sm text-gray-500">Hit Rate</div>
        </div>
        <div class="bg-white rounded-lg shadow p-6">
          <div class="text-2xl font-bold text-gray-900">{cacheMetrics?.totalEntries || 0}</div>
          <div class="text-sm text-gray-500">Total Entries</div>
        </div>
        <div class="bg-white rounded-lg shadow p-6">
          <div class="text-2xl font-bold text-blue-600">{formatMetricValue(cacheMetrics?.totalSize || 0, 'size')}</div>
          <div class="text-sm text-gray-500">Total Size</div>
        </div>
        <div class="bg-white rounded-lg shadow p-6">
          <div class="text-2xl font-bold text-purple-600">{formatMetricValue(cacheMetrics?.averageAccessTime || 0, 'time')}</div>
          <div class="text-sm text-gray-500">Avg Access Time</div>
        </div>
      </div>

      <!-- Cache Layer Performance -->
      <div class="bg-white rounded-lg shadow">
        <div class="px-6 py-4 border-b border-gray-200">
          <h3 class="text-lg font-medium text-gray-900">Cache Layer Performance</h3>
        </div>
        <div class="p-6">
          {#if cacheMetrics?.layerStats}
            <div class="space-y-4">
              {#each Object.entries(cacheMetrics.layerStats) as [layer, stats]}
                <div class="flex items-center justify-between p-4 bg-gray-50 rounded-md">
                  <div class="flex-1">
                    <h4 class="text-sm font-medium text-gray-900 capitalize">{layer} Layer</h4>
                    <div class="mt-1 flex items-center space-x-4 text-xs text-gray-500">
                      <span>Entries: {stats.entries}</span>
                      {#if 'size' in stats}
                        <span>Size: {formatMetricValue(stats.size, 'size')}</span>
                      {/if}
                      {#if 'hitRate' in stats}
                        <span>Hit Rate: {formatMetricValue(stats.hitRate, 'percentage')}</span>
                      {/if}
                      {#if 'queries' in stats}
                        <span>Queries: {stats.queries}</span>
                      {/if}
                    </div>
                  </div>
                  {#if 'hitRate' in stats}
                    <div class="w-24">
                      <div class="w-full bg-gray-200 rounded-full h-2">
                        <div class="bg-green-600 h-2 rounded-full" style="width: {(stats.hitRate * 100)}%"></div>
                      </div>
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          {:else}
            <p class="text-gray-500 text-center py-8">No cache data available</p>
          {/if}
        </div>
      </div>
    </div>

  {:else if selectedTab === 'system'}
    <!-- System Health Tab -->
    <div class="space-y-6">
      <!-- Overall Health -->
      <div class="bg-white rounded-lg shadow p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-medium text-gray-900">System Health Overview</h3>
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 rounded-full {healthScore >= 80 ? 'bg-green-500' : healthScore >= 60 ? 'bg-yellow-500' : 'bg-red-500'}"></div>
            <span class="text-2xl font-bold text-gray-900">{Math.round(healthScore)}%</span>
          </div>
        </div>
        
        <div class="w-full bg-gray-200 rounded-full h-3">
          <div class="bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-3 rounded-full" style="width: {healthScore}%"></div>
        </div>
      </div>

      <!-- Component Status -->
      <div class="bg-white rounded-lg shadow">
        <div class="px-6 py-4 border-b border-gray-200">
          <h3 class="text-lg font-medium text-gray-900">Component Status</h3>
        </div>
        <div class="p-6">
          {#if systemMetrics?.components}
            <div class="space-y-4">
              {#each Object.entries(systemMetrics.components) as [component, status]}
                <div class="flex items-center justify-between p-4 bg-gray-50 rounded-md">
                  <div class="flex items-center space-x-3">
                    <div class="w-3 h-3 rounded-full {getHealthStatusColor(status.status).replace('text-', 'bg-')}"></div>
                    <div>
                      <h4 class="text-sm font-medium text-gray-900 capitalize">{component}</h4>
                      <p class="text-xs text-gray-500">Response: {formatMetricValue(status.responseTime, 'time')}</p>
                    </div>
                  </div>
                  <span class="px-2 py-1 text-xs font-medium rounded-full {getHealthStatusColor(status.status)} bg-opacity-10 capitalize">
                    {status.status}
                  </span>
                </div>
              {/each}
            </div>
          {:else}
            <p class="text-gray-500 text-center py-8">No component data available</p>
          {/if}
        </div>
      </div>

      <!-- System Metrics -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="bg-white rounded-lg shadow p-6">
          <h4 class="text-sm font-medium text-gray-900 mb-2">Memory Usage</h4>
          <div class="text-2xl font-bold text-gray-900">{formatMetricValue(systemMetrics?.memory.percentage || 0, 'percentage')}</div>
          <div class="text-xs text-gray-500">{formatMetricValue(systemMetrics?.memory.used || 0, 'size')} / {formatMetricValue(systemMetrics?.memory.total || 0, 'size')}</div>
        </div>
        
        <div class="bg-white rounded-lg shadow p-6">
          <h4 class="text-sm font-medium text-gray-900 mb-2">Response Time</h4>
          <div class="text-2xl font-bold text-gray-900">{formatMetricValue(systemMetrics?.responseTime || 0, 'time')}</div>
          <div class="text-xs text-gray-500">Average response time</div>
        </div>
        
        <div class="bg-white rounded-lg shadow p-6">
          <h4 class="text-sm font-medium text-gray-900 mb-2">Uptime</h4>
          <div class="text-2xl font-bold text-gray-900">{formatMetricValue(systemMetrics?.uptime || 0, 'time')}</div>
          <div class="text-xs text-gray-500">System uptime</div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Demo Controls -->
  <div class="fixed bottom-6 right-6 space-y-2">
    <button
      class="block w-full px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 text-sm"
      onclick={clearLogs}
    >
      Clear Logs
    </button>
  </div>
</div>