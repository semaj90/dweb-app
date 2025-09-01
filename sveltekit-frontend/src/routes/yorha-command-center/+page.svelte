<!-- YoRHa Command Center Page -->
<script lang="ts">
  import { onMount } from 'svelte';
  import YoRHaDetectiveCommandCenter from '$lib/components/yorha/YoRHaDetectiveCommandCenter.svelte';
  import { browser } from '$app/environment';
  
  // Page data and state
  let systemData = $state({
    activeCases: 12,
    evidenceItems: 247,
    personsOfInterest: 8,
    aiQueries: 1543,
    systemLoad: 34,
    gpuUtilization: 67,
    memoryUsage: 42,
    networkLatency: 23
  });
let currentUser = $state({
    id: 'user_001',
    name: 'Agent 2B',
    role: 'detective',
    clearanceLevel: 'high'
  });

  // Load system data on mount
  onMount(async () => {
    if (browser) {
      try {
        // Fetch real system data from your APIs
        const [systemStatusResponse, legalDataResponse] = await Promise.all([
          fetch('/api/yorha/system/status').catch(() => null),
          fetch('/api/yorha/legal-data').catch(() => null)
        ]);

        if (systemStatusResponse?.ok) {
          const status = await systemStatusResponse.json();
          systemData = {
            ...systemData,
            systemLoad: status.cpu || systemData.systemLoad,
            gpuUtilization: status.gpu || systemData.gpuUtilization,
            memoryUsage: status.memory || systemData.memoryUsage,
            networkLatency: status.network || systemData.networkLatency
          };
        }

        if (legalDataResponse?.ok) {
          const legalData = await legalDataResponse.json();
          systemData = {
            ...systemData,
            activeCases: legalData.cases?.total || systemData.activeCases,
            evidenceItems: legalData.evidence?.total || systemData.evidenceItems,
            personsOfInterest: legalData.persons?.total || systemData.personsOfInterest,
            aiQueries: legalData.queries?.total || systemData.aiQueries
          };
        }
      } catch (error) {
        console.error('Failed to load system data:', error);
        // Fall back to mock data if APIs are not available
      }
    }
  });
</script>

<svelte:head>
  <title>YoRHa Command Center | Legal AI Detective Interface</title>
  <meta name="description" content="YoRHa Legal AI Detective Command Center - Manage cases, evidence, and investigations" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap" rel="stylesheet">
</svelte:head>

<YoRHaDetectiveCommandCenter 
  {currentUser} 
  {systemData} 
/>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    font-family: 'Roboto Mono', monospace;
  }
</style>