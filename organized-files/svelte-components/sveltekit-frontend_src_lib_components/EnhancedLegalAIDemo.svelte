<script lang="ts">
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
    graphRelationshipsStore,
    systemHealthStore,
    initializeEnhancedMachines
  } from '$lib/stores/enhancedStateMachines';

  // Create local store for vector similarity since it's not exported
  const vectorSimilarityStore = writable([]);

  import {
    enhancedLoki,
    enhancedLokiStore,
    cacheStatsStore,
    cacheHealthStore
  } from '$lib/stores/enhancedLokiStore';

  // UI Components
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/badge';
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

  let currentProcessing = $derived($currentlyProcessingStore)
  let processingResults = $derived($processingResultsStore)
  let aiRecommendations = $derived($aiRecommendationsStore)
  let vectorMatches = $derived($vectorSimilarityStore)
  let graphRelationships = $derived($graphRelationshipsStore)
  let systemHealth = $derived($systemHealthStore)
  let cacheStats = $derived($cacheStatsStore)
  let cacheHealth = $derived($cacheHealthStore)
  let streamingConnected = $derived($streamingStore.isStreaming)

  // ======================================================================
  // INITIALIZATION
  // ======================================================================

  onMount(async () => {
    try {
      // Initialize enhanced Loki database
      await enhancedLoki.init();

      // Initialize state machines
      machines = await initializeEnhancedMachines();

      // Subscribe to real-time updates
      if (machines?.streamingActor) {
        machines.streamingActor.subscribe((state: any) => {
          if (state.context.messageQueue.length > realTimeUpdates.length) {
            realTimeUpdates = [...state.context.messageQueue];
          }
        });
      }

      console.log('Enhanced Legal AI system initialized successfully');
    } catch (error) {
      console.error('Failed to initialize enhanced system:', error);
    }
  });

  onDestroy(() => {
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
      updatedAt: new Date(),
      title: 'Custom Evidence',
      summary: evidenceText.trim().substring(0, 100) + '...',
      description: evidenceText.trim(),
      location: '',
      tags: [],
      aiSummary: '',
      aiTags: [],
      caseId: selectedCaseId,
      criminalId: '',
      evidenceType: 'document',
      fileType: 'text',
      subType: 'custom',
      fileUrl: null,
      fileName: 'custom-evidence.txt',
      fileSize: evidenceText.length,
      mimeType: 'text/plain',
      hash: null,
      chainOfCustody: [],
      collectedAt: new Date(),
      collectedBy: '',
      labAnalysis: {},
      aiAnalysis: {},
      isAdmissible: false,
      confidentialityLevel: 'internal',
      canvasPosition: {},
      uploadedBy: null,
      uploadedAt: new Date(),
      content: evidenceText.trim(),
      type: 'custom',
      confidence: 0,
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
</script>

<!-- ====================================================================== -->
<!-- ENHANCED LEGAL AI DEMO INTERFACE -->
<!-- ====================================================================== -->

<div class="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
  <div class="max-w-7xl mx-auto space-y-6">

    <!-- Header with System Status -->
    <div class="bg-white rounded-lg shadow-sm border p-6">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-3xl font-bold text-gray-900">Enhanced Legal AI System</h1>
          <p class="text-gray-600 mt-1">Real-time AI processing with XState + Loki.js integration</p>
        </div>
        <div class="flex items-center space-x-4">
          <Badge class="{getHealthBadgeColor(systemHealth.health)} text-white">
            System: {systemHealth.health.toUpperCase()}
          </Badge>
          <Badge class="{streamingConnected ? 'bg-green-500' : 'bg-red-500'} text-white">
            Streaming: {streamingConnected ? 'Connected' : 'Disconnected'}
          </Badge>
          <Badge class="bg-blue-500 text-white">
            Cache Hits: {cacheStats.hits}
          </Badge>
        </div>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

      <!-- Left Column: Evidence Input & Processing -->
      <div class="space-y-6">

        <!-- Evidence Input -->
        <Card>
          <CardHeader>
            <CardTitle>Add Evidence</CardTitle>
          </CardHeader>
          <CardContent class="space-y-4">
            <Textarea
              bind:value={evidenceText}
              placeholder="Enter evidence content..."
              rows={4}
              class="w-full"
            />
            <Button
              onclick={() => addCustomEvidence()}
              disabled={!evidenceText.trim() || processingActive}
              class="w-full"
            >
              Process Evidence
            </Button>
          </CardContent>
        </Card>

        <!-- Demo Evidence -->
        <Card>
          <CardHeader>
            <CardTitle>Demo Evidence</CardTitle>
          </CardHeader>
          <CardContent class="space-y-3">
            {#each demoEvidences as demo}
              <div class="border rounded-lg p-3">
                <div class="flex items-center justify-between mb-2">
                  <h4 class="font-medium text-sm">{demo.fileName}</h4>
                  <Badge variant="outline">{demo.type}</Badge>
                </div>
                <p class="text-xs text-gray-600 mb-3">
                  {demo.content.slice(0, 100)}...
                </p>
                <Button
                  size="sm"
                  variant="outline"
                  onclick={() => addDemoEvidence(demo)}
                  disabled={processingActive}
                  class="w-full"
                >
                  Process This Evidence
                </Button>
              </div>
            {/each}
          </CardContent>
        </Card>

        <!-- System Controls -->
        <Card>
          <CardHeader>
            <CardTitle>System Controls</CardTitle>
          </CardHeader>
          <CardContent class="space-y-3">
            <Button variant="outline" onclick={() => checkSystemHealth()} class="w-full">
              Health Check
            </Button>
            <Button variant="outline" onclick={() => syncCache()} class="w-full">
              Sync Cache
            </Button>
            <Button variant="outline" onclick={() => clearErrors()} class="w-full">
              Clear Errors
            </Button>
            <Button variant="destructive" onclick={() => clearCache()} class="w-full">
              Clear Cache
            </Button>
          </CardContent>
        </Card>
      </div>

      <!-- Middle Column: Processing Results -->
      <div class="space-y-6">

        <!-- Currently Processing -->
        {#if currentProcessing}
          <Card>
            <CardHeader>
              <CardTitle class="flex items-center space-x-2">
                <div class="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
                <span>Currently Processing</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div class="space-y-2">
                <p class="font-medium">{currentProcessing.fileName}</p>
                <Badge>{currentProcessing.type}</Badge>
                <p class="text-sm text-gray-600">
                  {currentProcessing.content.slice(0, 150)}...
                </p>
              </div>
            </CardContent>
          </Card>
        {/if}

        <!-- Processing Results -->
        <Card>
          <CardHeader>
            <CardTitle>Processing Results ({processingResults.length})</CardTitle>
          </CardHeader>
          <CardContent>
            {#if processingResults.length === 0}
              <p class="text-gray-500 text-center py-4">No results yet</p>
            {:else}
              <div class="space-y-3 max-h-64 overflow-y-auto">
                {#each processingResults.slice(-5) as result}
                  <div class="border rounded-lg p-3">
                    <div class="flex items-center justify-between mb-2">
                      <Badge class="{result.status === 'complete' ? 'bg-green-500' : 'bg-yellow-500'} text-white">
                        {result.status}
                      </Badge>
                      <span class="text-xs text-gray-500">
                        {formatTimestamp(result.timestamp)}
                      </span>
                    </div>
                    <p class="text-sm">Evidence: {result.evidenceId}</p>
                    <p class="text-sm">Type: {result.type}</p>
                    <p class="text-sm">Confidence: {(result.confidence * 100).toFixed(1)}%</p>
                    <p class="text-sm">Time: {result.processingTime}ms</p>
                  </div>
                {/each}
              </div>
            {/if}
          </CardContent>
        </Card>

        <!-- AI Recommendations -->
        <Card>
          <CardHeader>
            <CardTitle>AI Recommendations ({aiRecommendations.length})</CardTitle>
          </CardHeader>
          <CardContent>
            {#if aiRecommendations.length === 0}
              <p class="text-gray-500 text-center py-4">No recommendations yet</p>
            {:else}
              <div class="space-y-3 max-h-48 overflow-y-auto">
                {#each aiRecommendations.slice(-3) as rec}
                  <div class="border rounded-lg p-3">
                    <div class="flex items-center justify-between mb-2">
                      <Badge variant="outline">{rec.type}</Badge>
                      <span class="text-xs font-medium">
                        {(rec.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p class="text-sm">{rec.content}</p>
                    <p class="text-xs text-gray-500 mt-1">Source: {rec.source}</p>
                  </div>
                {/each}
              </div>
            {/if}
          </CardContent>
        </Card>
      </div>

      <!-- Right Column: Vector Search & Graph -->
      <div class="space-y-6">

        <!-- Vector Similarity Matches -->
        <Card>
          <CardHeader>
            <CardTitle>Vector Similarity Matches ({vectorMatches.length})</CardTitle>
          </CardHeader>
          <CardContent>
            {#if vectorMatches.length === 0}
              <p class="text-gray-500 text-center py-4">No matches found</p>
            {:else}
              <div class="space-y-3 max-h-64 overflow-y-auto">
                {#each vectorMatches.slice(0, 5) as match}
                  <div class="border rounded-lg p-3">
                    <div class="flex items-center justify-between mb-2">
                      <Badge>Rank #{match.rank}</Badge>
                      <span class="text-sm font-medium text-green-600">
                        {(match.similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p class="text-sm">{match.content.slice(0, 100)}...</p>
                    <p class="text-xs text-gray-500 mt-1">ID: {match.id}</p>
                  </div>
                {/each}
              </div>
            {/if}
          </CardContent>
        </Card>

        <!-- Graph Relationships -->
        <Card>
          <CardHeader>
            <CardTitle>Graph Relationships ({graphRelationships.length})</CardTitle>
          </CardHeader>
          <CardContent>
            {#if graphRelationships.length === 0}
              <p class="text-gray-500 text-center py-4">No relationships found</p>
            {:else}
              <div class="space-y-3 max-h-64 overflow-y-auto">
                {#each graphRelationships.slice(0, 5) as node}
                  <div class="border rounded-lg p-3">
                    <div class="flex items-center justify-between mb-2">
                      <Badge variant="outline">{node.type}</Badge>
                      <span class="text-xs text-gray-500">
                        {node.connections?.length || 0} connections
                      </span>
                    </div>
                    <p class="font-medium text-sm">{node.label}</p>
                    {#if node.connections?.length}
                      <div class="mt-2 space-y-1">
                        {#each node.connections.slice(0, 2) as conn}
                          <div class="text-xs bg-gray-50 rounded p-1">
                            {conn.type} → {conn.to} (strength: {conn.strength.toFixed(2)})
                          </div>
                        {/each}
                      </div>
                    {/if}
                  </div>
                {/each}
              </div>
            {/if}
          </CardContent>
        </Card>

        <!-- Cache Statistics -->
        <Card>
          <CardHeader>
            <CardTitle>Cache Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="space-y-3">
              <div class="flex justify-between">
                <span class="text-sm">Hit Rate:</span>
                <span class="text-sm font-medium {getCacheHealthColor(cacheHealth.health)}">
                  {(cacheHealth.hitRate * 100).toFixed(1)}% ({cacheHealth.health})
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-sm">Cache Hits:</span>
                <span class="text-sm font-medium">{cacheStats.hits}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-sm">Cache Misses:</span>
                <span class="text-sm font-medium">{cacheStats.misses}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-sm">Evictions:</span>
                <span class="text-sm font-medium">{cacheStats.evictions}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-sm">Sync Ops:</span>
                <span class="text-sm font-medium">{cacheStats.syncOperations}</span>
              </div>
              {#if cacheStats.lastSync}
                <div class="flex justify-between">
                  <span class="text-sm">Last Sync:</span>
                  <span class="text-sm font-medium">
                    {formatTimestamp(cacheStats.lastSync)}
                  </span>
                </div>
              {/if}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>

    <!-- Real-time Updates Footer -->
    {#if realTimeUpdates.length > 0}
      <Card>
        <CardHeader>
          <CardTitle>Real-time Updates ({realTimeUpdates.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <div class="space-y-2 max-h-32 overflow-y-auto">
            {#each realTimeUpdates.slice(-5) as update}
              <div class="flex items-center justify-between text-sm bg-blue-50 rounded p-2">
                <span>{update.type || 'Update'}: {JSON.stringify(update).slice(0, 50)}...</span>
                <span class="text-xs text-gray-500">
                  {formatTimestamp(update.timestamp || new Date())}
                </span>
              </div>
            {/each}
          </div>
        </CardContent>
      </Card>
    {/if}

  </div>
</div>

<style>
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
</style>
