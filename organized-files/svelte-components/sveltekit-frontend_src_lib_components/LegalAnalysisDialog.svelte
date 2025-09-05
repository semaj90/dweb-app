<!-- Legal Case Analysis Dialog - Bits UI Component -->
<script lang="ts">
  import { Dialog } from 'bits-ui/components/dialog';
import { Select } from 'bits-ui/components/select';
import { Button } from 'bits-ui/components/button';
import { Badge } from 'bits-ui/components/badge';
import { Progress } from 'bits-ui/components/progress';
  import { legalCaseStore } from '$lib/stores/legal-case.store.svelte';
  import type { LegalCase } from '$lib/types/legal';

  interface Props {
    open: boolean
    onOpenChange: (open: boolean) => void;
  }

  let { open = $bindable(), onOpenChange } = $props();

  // Store access
  const { 
    filteredCases, 
    selectedCase, 
    aiInsights, 
    loading,
    selectCase, 
    analyzeCase 
  } = legalCaseStore;

  let selectedCaseForAnalysis = $state<string | null>(null);
  let analysisProgress = $state(0);
  let analysisStatus = $state<'idle' | 'analyzing' | 'complete' | 'error'>('idle');

  async function handleAnalysis() {
    if (!selectedCaseForAnalysis) return;

    analysisStatus = 'analyzing';
    analysisProgress = 0;

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        analysisProgress = Math.min(analysisProgress + 10, 90);
      }, 200);

      await analyzeCase(selectedCaseForAnalysis);
      
      clearInterval(progressInterval);
      analysisProgress = 100;
      analysisStatus = 'complete';

      // Auto-close after success
      setTimeout(() => {
        onOpenChange(false);
        analysisStatus = 'idle';
        analysisProgress = 0;
      }, 2000);

    } catch (error) {
      analysisStatus = 'error';
      console.error('Analysis failed:', error);
    }
  }

  function getRiskBadgeVariant(level: string) {
    switch (level) {
      case 'CRITICAL': return 'destructive';
      case 'HIGH': return 'secondary';
      case 'MEDIUM': return 'outline';
      case 'LOW': return 'default';
      default: return 'outline';
    }
  }
</script>

<Dialog.Root {open} {onOpenChange}>
  <Dialog.Trigger>
    <Button class="legal-action-btn bg-blue-600 hover:bg-blue-700 text-white">
      <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
      </svg>
      Analyze Case Documents
    </Button>
  </Dialog.Trigger>

  <Dialog.Content class="legal-dialog max-w-2xl w-full bg-white border border-gray-200 rounded-lg shadow-xl">
    <Dialog.Header class="border-b border-gray-100 p-6">
      <Dialog.Title class="text-xl font-semibold text-gray-900">
        Legal Document Analysis
      </Dialog.Title>
      <Dialog.Description class="text-gray-600 mt-2">
        Select a case to perform AI-powered legal analysis with compliance checking.
      </DialogDescription>
    </DialogHeader>

    <div class="p-6 space-y-6">
      <!-- Case Selection -->
      <div class="space-y-3">
        <label class="text-sm font-medium text-gray-700">Select Case for Analysis</label>
        <SelectRoot bind:value={selectedCaseForAnalysis} disabled={loading.analysis}>
          <SelectTrigger class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
            <SelectValue placeholder="Choose a case to analyze..." let:value>
              {value ? value.title : ''}
            </SelectValue>
          </SelectTrigger>
          <SelectContent class="bg-white border border-gray-200 rounded-md shadow-lg max-h-60 overflow-y-auto">
            {#each filteredCases() as legalCase}
              <SelectItem 
                value={legalCase.id} 
                class="px-3 py-2 hover:bg-gray-50 cursor-pointer border-b border-gray-100 last:border-b-0"
              >
                <div class="flex items-center justify-between w-full">
                  <div>
                    <div class="font-medium text-gray-900">{legalCase.title}</div>
                    <div class="text-sm text-gray-500">{legalCase.caseNumber}</div>
                  </div>
                  <div class="flex items-center space-x-2">
                    <Badge variant={legalCase.priority === 'high' ? 'destructive' : 'default'}>
                      {legalCase.priority}
                    </Badge>
                    <Badge variant="outline">{legalCase.status}</Badge>
                  </div>
                </div>
              </SelectItem>
            {/each}
          </SelectContent>
        </SelectRoot>
      </div>

      <!-- Analysis Progress -->
      {#if analysisStatus === 'analyzing'}
        <div class="space-y-3">
          <div class="flex items-center justify-between">
            <span class="text-sm font-medium text-gray-700">Analysis Progress</span>
            <span class="text-sm text-gray-500">{analysisProgress}%</span>
          </div>
          <Progress.Root value={analysisProgress} max={100} class="w-full">
            <Progress.Indicator 
              class="h-2 bg-blue-600 rounded-full transition-all duration-300"
              style="width: {analysisProgress}%"
            />
          </Progress.Root>
          <div class="text-sm text-gray-600">
            {#if analysisProgress < 30}
              Extracting document content...
            {:else if analysisProgress < 60}
              Performing AI analysis...
            {:else if analysisProgress < 90}
              Running compliance checks...
            {:else}
              Finalizing results...
            {/if}
          </div>
        </div>
      {/if}

      <!-- Analysis Results -->
      {#if selectedCaseForAnalysis && aiInsights[selectedCaseForAnalysis.id] && analysisStatus === 'complete'}
        {@const insights = aiInsights[selectedCaseForAnalysis.id]}
        <div class="space-y-4 border-t border-gray-100 pt-4">
          <h3 class="font-medium text-gray-900">Analysis Results</h3>
          
          <!-- Risk Assessment -->
          {#if insights.riskAssessment}
            <div class="flex items-center justify-between p-3 bg-gray-50 rounded-md">
              <span class="text-sm font-medium text-gray-700">Risk Level</span>
              <Badge variant={getRiskBadgeVariant(insights.riskAssessment.level)}>
                {insights.riskAssessment.level}
              </Badge>
            </div>
          {/if}

          <!-- Compliance Status -->
          {#if insights.complianceChecks}
            <div class="space-y-2">
              <span class="text-sm font-medium text-gray-700">Compliance Checks</span>
              <div class="grid grid-cols-2 gap-2">
                {#each insights.complianceChecks as check}
                  <div class="flex items-center space-x-2 p-2 bg-gray-50 rounded">
                    {#if check.passed}
                      <svg class="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                      </svg>
                    {:else}
                      <svg class="w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                      </svg>
                    {/if}
                    <span class="text-xs text-gray-600">{check.description}</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}

          <!-- Key Findings -->
          {#if insights.findings && insights.findings.length > 0}
            <div class="space-y-2">
              <span class="text-sm font-medium text-gray-700">Key Findings</span>
              <ul class="space-y-1">
                {#each insights.findings.slice(0, 3) as finding}
                  <li class="text-sm text-gray-600 flex items-start space-x-2">
                    <span class="w-1 h-1 bg-gray-400 rounded-full mt-2 flex-shrink-0"></span>
                    <span>{finding}</span>
                  </li>
                {/each}
              </ul>
            </div>
          {/if}
        </div>
      {/if}

      <!-- Error State -->
      {#if analysisStatus === 'error'}
        <div class="p-4 bg-red-50 border border-red-200 rounded-md">
          <div class="flex items-center space-x-2">
            <svg class="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
            </svg>
            <span class="text-sm font-medium text-red-800">Analysis Failed</span>
          </div>
          <p class="text-sm text-red-600 mt-1">
            Unable to complete the analysis. Please check the logs and try again.
          </p>
        </div>
      {/if}
    </div>

    <Dialog.Footer class="border-t border-gray-100 p-6 flex justify-end space-x-3">
      <Button 
        variant="outline" 
        onclick={() => onOpenChange(false)}
        disabled={loading.analysis}
      >
        Cancel
      </Button>
      <Button 
        onclick={handleAnalysis}
        disabled={!selectedCaseForAnalysis || loading.analysis || analysisStatus === 'analyzing'}
        class="bg-blue-600 hover:bg-blue-700 text-white"
      >
        {#if analysisStatus === 'analyzing'}
          <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          Analyzing...
        {:else if analysisStatus === 'complete'}
          <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
          </svg>
          Complete
        {:else}
          Start Analysis
        {/if}
      </Button>
    </DialogFooter>
  </Dialog.Content>
</Dialog.Root>

<style>
  .legal-dialog {
    animation: dialog-content-show 150ms cubic-bezier(0.16, 1, 0.3, 1);
  }

  @keyframes dialog-content-show {
    from {
      opacity: 0;
      transform: translate(-50%, -48%) scale(0.96);
    }
    to {
      opacity: 1;
      transform: translate(-50%, -50%) scale(1);
    }
  }

  .legal-action-btn {
    transition: all 0.2s ease-in-out;
  }

  .legal-action-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
  }
</style>
