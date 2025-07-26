&lt;script lang="ts"&gt;
  // ======================================================================
  // ENHANCED LEGAL AI DEMO COMPONENT
  // Demonstrating real-time AI processing with XState + Loki.js integration
  // ======================================================================
  
  import { onMount, onDestroy } from 'svelte';
  import { writable, derived } from 'svelte/store';
  
  // Enhanced stores and machines
  import { 
    evidenceProcessingStore,
    streamingStore,
    currentlyProcessingStore,
    processingResultsStore,
    aiRecommendationsStore,
    vectorSimilarityStore,
    graphRelationshipsStore,
    systemHealthStore,
    initializeEnhancedMachines
  } from '$lib/stores/enhancedStateMachines';
  
  import { 
    enhancedLoki,
    enhancedLokiStore,
    cacheStatsStore,
    cacheHealthStore
  } from '$lib/stores/enhancedLokiStore';
  
  // UI Components
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/Card';
  import { Badge } from '$lib/components/ui/Badge';
  import { Textarea } from '$lib/components/ui/textarea';
  
  // ======================================================================
  // COMPONENT STATE
  // ======================================================================
  
  let machines: any = null;
  let evidenceText = '';
  let selectedCaseId = 'demo-case-001';
  let processingActive = false;
  let realTimeUpdates: any[] = [];
  
  // Demo evidence samples
  const demoEvidences = [
    {
      id: 'evidence-001',
      fileName: 'witness-statement-1.txt',
      content: 'The defendant was seen leaving the building at approximately 11:30 PM on the night of the incident. The witness, Jane Doe, observed suspicious behavior including looking around nervously and carrying a large bag.',
      type: 'witness_statement',
      caseId: selectedCaseId
    },
    {
      id: 'evidence-002',
      fileName: 'security-footage-analysis.txt',
      content: 'Security camera footage shows an individual matching the defendant\'s description entering through the rear entrance at 11:15 PM. The timestamp corresponds with the security system breach recorded at 11:17 PM.',
      type: 'digital_evidence',
      caseId: selectedCaseId
    },
    {
      id: 'evidence-003',
      fileName: 'forensic-report.txt',
      content: 'DNA analysis of samples collected from the scene shows a 99.7% match with the defendant. Fingerprint analysis reveals partial prints on the door handle and window frame.',
      type: 'forensic_evidence',
      caseId: selectedCaseId
    }
  ];
  
  // ======================================================================
  // REACTIVE STATEMENTS
  // ======================================================================
  
  $: currentProcessing = $currentlyProcessingStore;
  $: processingResults = $processingResultsStore;
  $: aiRecommendations = $aiRecommendationsStore;
  $: vectorMatches = $vectorSimilarityStore;
  $: graphRelationships = $graphRelationshipsStore;
  $: systemHealth = $systemHealthStore;
  $: cacheStats = $cacheStatsStore;
  $: cacheHealth = $cacheHealthStore;
  $: streamingConnected = $streamingStore.connected;
  
  // ======================================================================
  // INITIALIZATION
  // ======================================================================
  
  onMount(async () =&gt; {
    try {
      // Initialize enhanced Loki database
      await enhancedLoki.init();
      
      // Initialize state machines
      machines = await initializeEnhancedMachines();
      
      // Subscribe to real-time updates
      if (machines?.streamingActor) {
        machines.streamingActor.subscribe((state: any) =&gt; {
          if (state.context.messageQueue.length &gt; realTimeUpdates.length) {
            realTimeUpdates = [...state.context.messageQueue];
          }
        });
      }
      
      console.log('Enhanced Legal AI system initialized successfully');
    } catch (error) {
      console.error('Failed to initialize enhanced system:', error);
    }
  });
  
  onDestroy(() =&gt; {
    if (machines) {
      machines.evidenceActor?.stop();
      machines.streamingActor?.stop();
    }
    enhancedLoki.destroy();
  });
  
  // ======================================================================
  // EVENT HANDLERS
  // ======================================================================
  
  async function addCustomEvidence() {
    if (!evidenceText.trim() || !machines?.evidenceActor) return;
    
    const evidence = {
      id: `evidence-${Date.now()}`,
      fileName: 'custom-evidence.txt',
      content: evidenceText.trim(),
      type: 'custom',
      caseId: selectedCaseId,
      confidence: 0,
      aiTags: [],
      relationships: []
    };
    
    // Add to state machine for processing
    machines.evidenceActor.send({
      type: 'ADD_EVIDENCE',
      evidence
    });
    
    // Cache in Loki
    await enhancedLoki.evidence.add(evidence);
    
    evidenceText = '';
    processingActive = true;
  }
  
  async function addDemoEvidence(demoEvidence: any) {
    if (!machines?.evidenceActor) return;
    
    machines.evidenceActor.send({
      type: 'ADD_EVIDENCE',
      evidence: demoEvidence
    });
    
    await enhancedLoki.evidence.add(demoEvidence);
    processingActive = true;
  }
  
  function checkSystemHealth() {
    if (machines?.evidenceActor) {
      machines.evidenceActor.send({ type: 'HEALTH_CHECK' });
    }
  }
  
  function syncCache() {
    if (machines?.evidenceActor) {
      machines.evidenceActor.send({ type: 'SYNC_CACHE' });
    }
  }
  
  function clearErrors() {
    if (machines?.evidenceActor) {
      machines.evidenceActor.send({ type: 'CLEAR_ERRORS' });
    }
  }
  
  function clearCache() {
    enhancedLoki.clearCache();
  }
  
  // ======================================================================
  // UTILITY FUNCTIONS
  // ======================================================================
  
  function formatTimestamp(date: Date | string) {
    return new Date(date).toLocaleTimeString();
  }
  
  function getHealthBadgeColor(health: string) {
    switch (health) {
      case 'healthy': return 'bg-green-500';
      case 'degraded': return 'bg-yellow-500';
      case 'critical': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  }
  
  function getCacheHealthColor(health: string) {
    switch (health) {
      case 'excellent': return 'text-green-600';
      case 'good': return 'text-blue-600';
      case 'fair': return 'text-yellow-600';
      case 'poor': return 'text-red-600';
      default: return 'text-gray-600';
    }
  }
&lt;/script&gt;

&lt;!-- ====================================================================== --&gt;
&lt;!-- ENHANCED LEGAL AI DEMO INTERFACE --&gt;
&lt;!-- ====================================================================== --&gt;

&lt;div class="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6"&gt;
  &lt;div class="max-w-7xl mx-auto space-y-6"&gt;
    
    &lt;!-- Header with System Status --&gt;
    &lt;div class="bg-white rounded-lg shadow-sm border p-6"&gt;
      &lt;div class="flex items-center justify-between"&gt;
        &lt;div&gt;
          &lt;h1 class="text-3xl font-bold text-gray-900"&gt;Enhanced Legal AI System&lt;/h1&gt;
          &lt;p class="text-gray-600 mt-1"&gt;Real-time AI processing with XState + Loki.js integration&lt;/p&gt;
        &lt;/div&gt;
        &lt;div class="flex items-center space-x-4"&gt;
          &lt;Badge class="{getHealthBadgeColor(systemHealth.health)} text-white"&gt;
            System: {systemHealth.health.toUpperCase()}
          &lt;/Badge&gt;
          &lt;Badge class="{streamingConnected ? 'bg-green-500' : 'bg-red-500'} text-white"&gt;
            Streaming: {streamingConnected ? 'Connected' : 'Disconnected'}
          &lt;/Badge&gt;
          &lt;Badge class="bg-blue-500 text-white"&gt;
            Cache Hits: {cacheStats.hits}
          &lt;/Badge&gt;
        &lt;/div&gt;
      &lt;/div&gt;
    &lt;/div&gt;

    &lt;div class="grid grid-cols-1 lg:grid-cols-3 gap-6"&gt;
      
      &lt;!-- Left Column: Evidence Input &amp; Processing --&gt;
      &lt;div class="space-y-6"&gt;
        
        &lt;!-- Evidence Input --&gt;
        &lt;Card&gt;
          &lt;CardHeader&gt;
            &lt;CardTitle&gt;Add Evidence&lt;/CardTitle&gt;
          &lt;/CardHeader&gt;
          &lt;CardContent class="space-y-4"&gt;
            &lt;Textarea
              bind:value={evidenceText}
              placeholder="Enter evidence content..."
              rows={4}
              class="w-full"
            /&gt;
            &lt;Button 
              on:click={addCustomEvidence}
              disabled={!evidenceText.trim() || processingActive}
              class="w-full"
            &gt;
              Process Evidence
            &lt;/Button&gt;
          &lt;/CardContent&gt;
        &lt;/Card&gt;

        &lt;!-- Demo Evidence --&gt;
        &lt;Card&gt;
          &lt;CardHeader&gt;
            &lt;CardTitle&gt;Demo Evidence&lt;/CardTitle&gt;
          &lt;/CardHeader&gt;
          &lt;CardContent class="space-y-3"&gt;
            {#each demoEvidences as demo}
              &lt;div class="border rounded-lg p-3"&gt;
                &lt;div class="flex items-center justify-between mb-2"&gt;
                  &lt;h4 class="font-medium text-sm"&gt;{demo.fileName}&lt;/h4&gt;
                  &lt;Badge variant="outline"&gt;{demo.type}&lt;/Badge&gt;
                &lt;/div&gt;
                &lt;p class="text-xs text-gray-600 mb-3"&gt;
                  {demo.content.slice(0, 100)}...
                &lt;/p&gt;
                &lt;Button 
                  size="sm" 
                  variant="outline"
                  on:click={() =&gt; addDemoEvidence(demo)}
                  disabled={processingActive}
                  class="w-full"
                &gt;
                  Process This Evidence
                &lt;/Button&gt;
              &lt;/div&gt;
            {/each}
          &lt;/CardContent&gt;
        &lt;/Card&gt;

        &lt;!-- System Controls --&gt;
        &lt;Card&gt;
          &lt;CardHeader&gt;
            &lt;CardTitle&gt;System Controls&lt;/CardTitle&gt;
          &lt;/CardHeader&gt;
          &lt;CardContent class="space-y-3"&gt;
            &lt;Button variant="outline" on:click={checkSystemHealth} class="w-full"&gt;
              Health Check
            &lt;/Button&gt;
            &lt;Button variant="outline" on:click={syncCache} class="w-full"&gt;
              Sync Cache
            &lt;/Button&gt;
            &lt;Button variant="outline" on:click={clearErrors} class="w-full"&gt;
              Clear Errors
            &lt;/Button&gt;
            &lt;Button variant="destructive" on:click={clearCache} class="w-full"&gt;
              Clear Cache
            &lt;/Button&gt;
          &lt;/CardContent&gt;
        &lt;/Card&gt;
      &lt;/div&gt;

      &lt;!-- Middle Column: Processing Results --&gt;
      &lt;div class="space-y-6"&gt;
        
        &lt;!-- Currently Processing --&gt;
        {#if currentProcessing}
          &lt;Card&gt;
            &lt;CardHeader&gt;
              &lt;CardTitle class="flex items-center space-x-2"&gt;
                &lt;div class="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"&gt;&lt;/div&gt;
                &lt;span&gt;Currently Processing&lt;/span&gt;
              &lt;/CardTitle&gt;
            &lt;/CardHeader&gt;
            &lt;CardContent&gt;
              &lt;div class="space-y-2"&gt;
                &lt;p class="font-medium"&gt;{currentProcessing.fileName}&lt;/p&gt;
                &lt;Badge&gt;{currentProcessing.type}&lt;/Badge&gt;
                &lt;p class="text-sm text-gray-600"&gt;
                  {currentProcessing.content.slice(0, 150)}...
                &lt;/p&gt;
              &lt;/div&gt;
            &lt;/CardContent&gt;
          &lt;/Card&gt;
        {/if}

        &lt;!-- Processing Results --&gt;
        &lt;Card&gt;
          &lt;CardHeader&gt;
            &lt;CardTitle&gt;Processing Results ({processingResults.length})&lt;/CardTitle&gt;
          &lt;/CardHeader&gt;
          &lt;CardContent&gt;
            {#if processingResults.length === 0}
              &lt;p class="text-gray-500 text-center py-4"&gt;No results yet&lt;/p&gt;
            {:else}
              &lt;div class="space-y-3 max-h-64 overflow-y-auto"&gt;
                {#each processingResults.slice(-5) as result}
                  &lt;div class="border rounded-lg p-3"&gt;
                    &lt;div class="flex items-center justify-between mb-2"&gt;
                      &lt;Badge class="{result.status === 'complete' ? 'bg-green-500' : 'bg-yellow-500'} text-white"&gt;
                        {result.status}
                      &lt;/Badge&gt;
                      &lt;span class="text-xs text-gray-500"&gt;
                        {formatTimestamp(result.timestamp)}
                      &lt;/span&gt;
                    &lt;/div&gt;
                    &lt;p class="text-sm"&gt;Evidence: {result.evidenceId}&lt;/p&gt;
                    &lt;p class="text-sm"&gt;Type: {result.type}&lt;/p&gt;
                    &lt;p class="text-sm"&gt;Confidence: {(result.confidence * 100).toFixed(1)}%&lt;/p&gt;
                    &lt;p class="text-sm"&gt;Time: {result.processingTime}ms&lt;/p&gt;
                  &lt;/div&gt;
                {/each}
              &lt;/div&gt;
            {/if}
          &lt;/CardContent&gt;
        &lt;/Card&gt;

        &lt;!-- AI Recommendations --&gt;
        &lt;Card&gt;
          &lt;CardHeader&gt;
            &lt;CardTitle&gt;AI Recommendations ({aiRecommendations.length})&lt;/CardTitle&gt;
          &lt;/CardHeader&gt;
          &lt;CardContent&gt;
            {#if aiRecommendations.length === 0}
              &lt;p class="text-gray-500 text-center py-4"&gt;No recommendations yet&lt;/p&gt;
            {:else}
              &lt;div class="space-y-3 max-h-48 overflow-y-auto"&gt;
                {#each aiRecommendations.slice(-3) as rec}
                  &lt;div class="border rounded-lg p-3"&gt;
                    &lt;div class="flex items-center justify-between mb-2"&gt;
                      &lt;Badge variant="outline"&gt;{rec.type}&lt;/Badge&gt;
                      &lt;span class="text-xs font-medium"&gt;
                        {(rec.confidence * 100).toFixed(0)}%
                      &lt;/span&gt;
                    &lt;/div&gt;
                    &lt;p class="text-sm"&gt;{rec.content}&lt;/p&gt;
                    &lt;p class="text-xs text-gray-500 mt-1"&gt;Source: {rec.source}&lt;/p&gt;
                  &lt;/div&gt;
                {/each}
              &lt;/div&gt;
            {/if}
          &lt;/CardContent&gt;
        &lt;/Card&gt;
      &lt;/div&gt;

      &lt;!-- Right Column: Vector Search &amp; Graph --&gt;
      &lt;div class="space-y-6"&gt;
        
        &lt;!-- Vector Similarity Matches --&gt;
        &lt;Card&gt;
          &lt;CardHeader&gt;
            &lt;CardTitle&gt;Vector Similarity Matches ({vectorMatches.length})&lt;/CardTitle&gt;
          &lt;/CardHeader&gt;
          &lt;CardContent&gt;
            {#if vectorMatches.length === 0}
              &lt;p class="text-gray-500 text-center py-4"&gt;No matches found&lt;/p&gt;
            {:else}
              &lt;div class="space-y-3 max-h-64 overflow-y-auto"&gt;
                {#each vectorMatches.slice(0, 5) as match}
                  &lt;div class="border rounded-lg p-3"&gt;
                    &lt;div class="flex items-center justify-between mb-2"&gt;
                      &lt;Badge&gt;Rank #{match.rank}&lt;/Badge&gt;
                      &lt;span class="text-sm font-medium text-green-600"&gt;
                        {(match.similarity * 100).toFixed(1)}%
                      &lt;/span&gt;
                    &lt;/div&gt;
                    &lt;p class="text-sm"&gt;{match.content.slice(0, 100)}...&lt;/p&gt;
                    &lt;p class="text-xs text-gray-500 mt-1"&gt;ID: {match.id}&lt;/p&gt;
                  &lt;/div&gt;
                {/each}
              &lt;/div&gt;
            {/if}
          &lt;/CardContent&gt;
        &lt;/Card&gt;

        &lt;!-- Graph Relationships --&gt;
        &lt;Card&gt;
          &lt;CardHeader&gt;
            &lt;CardTitle&gt;Graph Relationships ({graphRelationships.length})&lt;/CardTitle&gt;
          &lt;/CardHeader&gt;
          &lt;CardContent&gt;
            {#if graphRelationships.length === 0}
              &lt;p class="text-gray-500 text-center py-4"&gt;No relationships found&lt;/p&gt;
            {:else}
              &lt;div class="space-y-3 max-h-64 overflow-y-auto"&gt;
                {#each graphRelationships.slice(0, 5) as node}
                  &lt;div class="border rounded-lg p-3"&gt;
                    &lt;div class="flex items-center justify-between mb-2"&gt;
                      &lt;Badge variant="outline"&gt;{node.type}&lt;/Badge&gt;
                      &lt;span class="text-xs text-gray-500"&gt;
                        {node.connections?.length || 0} connections
                      &lt;/span&gt;
                    &lt;/div&gt;
                    &lt;p class="font-medium text-sm"&gt;{node.label}&lt;/p&gt;
                    {#if node.connections?.length}
                      &lt;div class="mt-2 space-y-1"&gt;
                        {#each node.connections.slice(0, 2) as conn}
                          &lt;div class="text-xs bg-gray-50 rounded p-1"&gt;
                            {conn.type} → {conn.to} (strength: {conn.strength.toFixed(2)})
                          &lt;/div&gt;
                        {/each}
                      &lt;/div&gt;
                    {/if}
                  &lt;/div&gt;
                {/each}
              &lt;/div&gt;
            {/if}
          &lt;/CardContent&gt;
        &lt;/Card&gt;

        &lt;!-- Cache Statistics --&gt;
        &lt;Card&gt;
          &lt;CardHeader&gt;
            &lt;CardTitle&gt;Cache Performance&lt;/CardTitle&gt;
          &lt;/CardHeader&gt;
          &lt;CardContent&gt;
            &lt;div class="space-y-3"&gt;
              &lt;div class="flex justify-between"&gt;
                &lt;span class="text-sm"&gt;Hit Rate:&lt;/span&gt;
                &lt;span class="text-sm font-medium {getCacheHealthColor(cacheHealth.health)}"&gt;
                  {(cacheHealth.hitRate * 100).toFixed(1)}% ({cacheHealth.health})
                &lt;/span&gt;
              &lt;/div&gt;
              &lt;div class="flex justify-between"&gt;
                &lt;span class="text-sm"&gt;Cache Hits:&lt;/span&gt;
                &lt;span class="text-sm font-medium"&gt;{cacheStats.hits}&lt;/span&gt;
              &lt;/div&gt;
              &lt;div class="flex justify-between"&gt;
                &lt;span class="text-sm"&gt;Cache Misses:&lt;/span&gt;
                &lt;span class="text-sm font-medium"&gt;{cacheStats.misses}&lt;/span&gt;
              &lt;/div&gt;
              &lt;div class="flex justify-between"&gt;
                &lt;span class="text-sm"&gt;Evictions:&lt;/span&gt;
                &lt;span class="text-sm font-medium"&gt;{cacheStats.evictions}&lt;/span&gt;
              &lt;/div&gt;
              &lt;div class="flex justify-between"&gt;
                &lt;span class="text-sm"&gt;Sync Ops:&lt;/span&gt;
                &lt;span class="text-sm font-medium"&gt;{cacheStats.syncOperations}&lt;/span&gt;
              &lt;/div&gt;
              {#if cacheStats.lastSync}
                &lt;div class="flex justify-between"&gt;
                  &lt;span class="text-sm"&gt;Last Sync:&lt;/span&gt;
                  &lt;span class="text-sm font-medium"&gt;
                    {formatTimestamp(cacheStats.lastSync)}
                  &lt;/span&gt;
                &lt;/div&gt;
              {/if}
            &lt;/div&gt;
          &lt;/CardContent&gt;
        &lt;/Card&gt;
      &lt;/div&gt;
    &lt;/div&gt;

    &lt;!-- Real-time Updates Footer --&gt;
    {#if realTimeUpdates.length &gt; 0}
      &lt;Card&gt;
        &lt;CardHeader&gt;
          &lt;CardTitle&gt;Real-time Updates ({realTimeUpdates.length})&lt;/CardTitle&gt;
        &lt;/CardHeader&gt;
        &lt;CardContent&gt;
          &lt;div class="space-y-2 max-h-32 overflow-y-auto"&gt;
            {#each realTimeUpdates.slice(-5) as update}
              &lt;div class="flex items-center justify-between text-sm bg-blue-50 rounded p-2"&gt;
                &lt;span&gt;{update.type || 'Update'}: {JSON.stringify(update).slice(0, 50)}...&lt;/span&gt;
                &lt;span class="text-xs text-gray-500"&gt;
                  {formatTimestamp(update.timestamp || new Date())}
                &lt;/span&gt;
              &lt;/div&gt;
            {/each}
          &lt;/div&gt;
        &lt;/CardContent&gt;
      &lt;/Card&gt;
    {/if}

  &lt;/div&gt;
&lt;/div&gt;

&lt;style&gt;
  /* Custom animations for processing indicators */
  @keyframes pulse-processing {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  .processing-indicator {
    animation: pulse-processing 2s infinite;
  }
  
  /* Smooth transitions for dynamic content */
  .transition-all {
    transition: all 0.3s ease-in-out;
  }
  
  /* Custom scrollbar for better UX */
  .overflow-y-auto::-webkit-scrollbar {
    width: 4px;
  }
  
  .overflow-y-auto::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 2px;
  }
  
  .overflow-y-auto::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 2px;
  }
  
  .overflow-y-auto::-webkit-scrollbar-thumb:hover {
    background: #555;
  }
&lt;/style&gt;
