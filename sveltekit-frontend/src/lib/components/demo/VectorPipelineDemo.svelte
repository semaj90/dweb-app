<!-- Vector Pipeline Demo Component -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { vectorPipelineState, vectorPipelineActions, type VectorPipelineJob } from '$lib/machines/vector-pipeline-machine';
  import Button from '$lib/components/ui/Button.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
let unsubscribe = $state<(() >(> void) | null = null);
  let state = $vectorPipelineState;

  // Sample job data
  const sampleJobs: Array<Omit<VectorPipelineJob, 'jobId' | 'status' | 'progress' | 'createdAt'>> = [
    { ownerType: 'evidence', ownerId: 'evidence-001', event: 'upsert' },
    { ownerType: 'document', ownerId: 'doc-legal-brief-2024', event: 'reembed' },
    { ownerType: 'case', ownerId: 'case-murder-investigation', event: 'upsert' },
    { ownerType: 'report', ownerId: 'forensic-report-dna-analysis', event: 'delete' }
  ];

  onMount(() => {
    unsubscribe = vectorPipelineState.subscribe(value => {
      state = value;
    });
  });

  onDestroy(() => {
    if (unsubscribe) {
      unsubscribe();
    }
  });

  function submitSingleJob() {
    const randomJob = sampleJobs[Math.floor(Math.random() * sampleJobs.length)];
    vectorPipelineActions.submitJob(randomJob);
  }

  function submitBatchJobs() {
    vectorPipelineActions.submitBatch(sampleJobs);
  }

  function runHealthCheck() {
    vectorPipelineActions.healthCheck();
  }

  function enableWebGPU() {
    vectorPipelineActions.enableWebGPU();
  }

  function disableWebGPU() {
    vectorPipelineActions.disableWebGPU();
  }

  function resetPipeline() {
    vectorPipelineActions.reset();
  }

  function retryFailedJobs() {
    vectorPipelineActions.retryFailedJobs();
  }

  // Get status indicators (runes-safe)
let pipelineStatus = $state({} as any);
let batchInfo = $state({} as any);
let metrics = $state({} as any);
let currentState = $state('unknown');
let errors = $state<string[] >([]);

  const pipelineStatusDerived = $derived(() => state.context?.pipeline);
  const batchInfoDerived = $derived(() => state.context?.batch);
  const metricsDerived = $derived(() => state.context?.metrics);
  const currentStateDerived = $derived(() => typeof state.value === 'string' ? state.value : 'unknown');
  const errorsDerived = $derived(() => state.context?.errors || []);

  $effect(() => {
    pipelineStatus = pipelineStatusDerived;
    batchInfo = batchInfoDerived;
    metrics = metricsDerived;
    currentState = currentStateDerived;
    errors = errorsDerived;
  });

  function getStatusColor(status: boolean): string {
    return status ? 'text-green-600' : 'text-red-600';
  }

  function getStatusIcon(status: boolean): string {
    return status ? '✅' : '❌';
  }
</script>

<div class="w-full max-w-6xl mx-auto p-4 space-y-6">
  <Card>
    <CardHeader>
      <CardTitle class="text-2xl font-bold">Vector Pipeline Demo</CardTitle>
      <p class="text-gray-600">
        XState machine orchestrating PostgreSQL → Redis Streams → Go microservice → CUDA worker → Qdrant
      </p>
    </CardHeader>
    <CardContent>
      <!-- Current State Display -->
      <div class="mb-6">
        <div class="flex items-center gap-2 mb-2">
          <span class="font-semibold">Current State:</span>
          <span class="px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
            {currentState}
          </span>
        </div>
        {#if errors.length > 0}
          <div class="text-red-600 text-sm">
            <strong>Errors:</strong> {errors.join(', ')}
          </div>
        {/if}
      </div>

      <!-- Control Buttons -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <Button on:on:click={submitSingleJob} disabled={currentState === 'processingJob'}>
          Submit Single Job
        </Button>
        <Button on:on:click={submitBatchJobs} disabled={currentState === 'processingBatch'}>
          Submit Batch
        </Button>
        <Button on:on:click={runHealthCheck} variant="outline">
          Health Check
        </Button>
        <Button on:on:click={resetPipeline} variant="destructive">
          Reset Pipeline
        </Button>
        <Button on:on:click={enableWebGPU} disabled={pipelineStatus?.webgpu}>
          Enable WebGPU
        </Button>
        <Button on:on:click={disableWebGPU} disabled={!pipelineStatus?.webgpu}>
          Disable WebGPU
        </Button>
        <Button on:on:click={retryFailedJobs} disabled={!batchInfo?.failedJobs || batchInfo.failedJobs === 0}>
          Retry Failed
        </Button>
      </div>

      <!-- Pipeline Status Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <!-- Service Status -->
        <Card>
          <CardHeader>
            <CardTitle class="text-lg">Service Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="space-y-2">
              <div class="flex justify-between">
                <span>PostgreSQL:</span>
                <span class={getStatusColor(pipelineStatus?.postgresql || false)}>
                  {getStatusIcon(pipelineStatus?.postgresql || false)}
                </span>
              </div>
              <div class="flex justify-between">
                <span>Redis:</span>
                <span class={getStatusColor(pipelineStatus?.redis || false)}>
                  {getStatusIcon(pipelineStatus?.redis || false)}
                </span>
              </div>
              <div class="flex justify-between">
                <span>Go Microservice:</span>
                <span class={getStatusColor(pipelineStatus?.goMicroservice || false)}>
                  {getStatusIcon(pipelineStatus?.goMicroservice || false)}
                </span>
              </div>
              <div class="flex justify-between">
                <span>CUDA Worker:</span>
                <span class={getStatusColor(pipelineStatus?.cudaWorker || false)}>
                  {getStatusIcon(pipelineStatus?.cudaWorker || false)}
                </span>
              </div>
              <div class="flex justify-between">
                <span>Qdrant:</span>
                <span class={getStatusColor(pipelineStatus?.qdrant || false)}>
                  {getStatusIcon(pipelineStatus?.qdrant || false)}
                </span>
              </div>
              <div class="flex justify-between">
                <span>WebGPU:</span>
                <span class={getStatusColor(pipelineStatus?.webgpu || false)}>
                  {getStatusIcon(pipelineStatus?.webgpu || false)}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        <!-- Batch Information -->
        <Card>
          <CardHeader>
            <CardTitle class="text-lg">Batch Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="space-y-2">
              <div class="flex justify-between">
                <span>Total Jobs:</span>
                <span class="font-mono">{batchInfo?.totalJobs || 0}</span>
              </div>
              <div class="flex justify-between">
                <span>Completed:</span>
                <span class="font-mono text-green-600">{batchInfo?.completedJobs || 0}</span>
              </div>
              <div class="flex justify-between">
                <span>Failed:</span>
                <span class="font-mono text-red-600">{batchInfo?.failedJobs || 0}</span>
              </div>
              <div class="flex justify-between">
                <span>Progress:</span>
                <span class="font-mono">{batchInfo?.progress || 0}%</span>
              </div>
              {#if batchInfo && batchInfo.progress > 0}
                <div class="w-full bg-gray-200 rounded-full h-2">
                  <div
                    class="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style="width: {batchInfo.progress}%"
                  ></div>
                </div>
              {/if}
            </div>
          </CardContent>
        </Card>

        <!-- Performance Metrics -->
        <Card>
          <CardHeader>
            <CardTitle class="text-lg">Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="space-y-2">
              <div class="flex justify-between">
                <span>Processed:</span>
                <span class="font-mono">{metrics?.totalJobsProcessed || 0}</span>
              </div>
              <div class="flex justify-between">
                <span>Avg Time:</span>
                <span class="font-mono">{metrics?.averageProcessingTime || 0}ms</span>
              </div>
              <div class="flex justify-between">
                <span>Throughput:</span>
                <span class="font-mono">{metrics?.throughputPerMinute || 0}/min</span>
              </div>
              <div class="flex justify-between">
                <span>Last Run:</span>
                <span class="text-sm">
                  {metrics?.lastProcessedAt ?
                    new Date(metrics.lastProcessedAt).toLocaleTimeString() :
                    'Never'
                  }
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <!-- Current Jobs Display -->
      {#if batchInfo && batchInfo.jobs.length > 0}
        <Card class="mt-6">
          <CardHeader>
            <CardTitle class="text-lg">Active Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="border-b">
                    <th class="text-left p-2">Job ID</th>
                    <th class="text-left p-2">Type</th>
                    <th class="text-left p-2">Owner ID</th>
                    <th class="text-left p-2">Event</th>
                    <th class="text-left p-2">Status</th>
                    <th class="text-left p-2">Progress</th>
                  </tr>
                </thead>
                <tbody>
                  {#each batchInfo.jobs as job}
                    <tr class="border-b hover:bg-gray-50">
                      <td class="p-2 font-mono text-xs">{job.jobId}</td>
                      <td class="p-2">{job.ownerType}</td>
                      <td class="p-2 font-mono text-xs">{job.ownerId}</td>
                      <td class="p-2">{job.event}</td>
                      <td class="p-2">
                        <span class="px-2 py-1 rounded text-xs font-medium
                          {job.status === 'succeeded' ? 'bg-green-100 text-green-800' :
                           job.status === 'failed' ? 'bg-red-100 text-red-800' :
                           job.status === 'processing' ? 'bg-blue-100 text-blue-800' :
                           'bg-gray-100 text-gray-800'}">
                          {job.status}
                        </span>
                      </td>
                      <td class="p-2">{job.progress}%</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      {/if}

      <!-- Integration Test Results -->
      <Card class="mt-6">
        <CardHeader>
          <CardTitle class="text-lg">Service Integration Test</CardTitle>
        </CardHeader>
        <CardContent>
          <p class="text-sm text-gray-600 mb-4">
            Test connectivity to backend services that the vector pipeline depends on:
          </p>
          <Button
            on:on:click={async () => {
              try {
                const response = await fetch('http://localhost:8094/api/health');
                const data = await response.json();
                alert(`Enhanced RAG Service: ${data.status} (${data.service})`);
              } catch (error) {
                alert(`Enhanced RAG Service: Error - ${error}`);
              }
            }}
            variant="outline"
            class="mr-2"
          >
            Test Enhanced RAG (8094)
          </Button>
          <Button
            on:on:click={async () => {
              try {
                const response = await fetch('http://localhost:6333/health');
                if (response.ok) {
                  alert('Qdrant Service: Healthy');
                } else {
                  alert(`Qdrant Service: Status ${response.status}`);
                }
              } catch (error) {
                alert(`Qdrant Service: Error - ${error}`);
              }
            }}
            variant="outline"
          >
            Test Qdrant (6333)
          </Button>
        </CardContent>
      </Card>
    </CardContent>
  </Card>
</div>