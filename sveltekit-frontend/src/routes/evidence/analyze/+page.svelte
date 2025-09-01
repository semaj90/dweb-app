<script lang="ts">
  import { goto } from '$app/navigation';
  import Badge from '$lib/components/ui/Badge.svelte';
  import Button from '$lib/components/ui/button/Button.svelte';
  import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
  } from '$lib/components/ui/card';
  import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
  } from '$lib/components/ui/dialog';
  import Input from '$lib/components/ui/Input.svelte';
  import Label from '$lib/components/ui/Label.svelte';
  import { Progress } from '$lib/components/ui/progress';
  import {
    SelectContent,
    SelectItem,
    SelectRoot,
    SelectTrigger,
    SelectValue,
  } from '$lib/components/ui/select';
  import Textarea from '$lib/components/ui/textarea/Textarea.svelte';

  // Reactive state with Svelte 5 syntax
  let analyzing = $state(false);
  let results = $state(null);
  let error = $state('');
  let progress = $state(0);
  let showResults = $state(false);

  // Form data
  let caseId = $state('');
  let evidenceContent = $state('');
  let evidenceFile = $state(null);
  let evidenceType = $state('police_report');
  let priority = $state('medium');
  let sessionId = $state('');

  // Analysis pipeline steps with enhanced metadata
  const steps = [
    {
      name: 'Evidence Analysis',
      key: 'evidence_analysis',
      status: 'pending',
      description: 'Structuring document and extracting key facts',
      icon: '📋',
      duration: '30-45s',
    },
    {
      name: 'Person Extraction',
      key: 'persons_extracted',
      status: 'pending',
      description: 'Identifying persons of interest and roles',
      icon: '👥',
      duration: '20-30s',
    },
    {
      name: 'Relationship Mapping',
      key: 'neo4j_updates',
      status: 'pending',
      description: 'Building knowledge graph connections',
      icon: '🔗',
      duration: '15-25s',
    },
    {
      name: 'Case Synthesis',
      key: 'case_synthesis',
      status: 'pending',
      description: 'Generating prosecutorial analysis',
      icon: '⚖️',
      duration: '25-35s',
    },
  ];

  // Evidence type options
  const evidenceTypes = [
    { value: 'police_report', label: 'Police Report' },
    { value: 'witness_statement', label: 'Witness Statement' },
    { value: 'financial_records', label: 'Financial Records' },
    { value: 'digital_forensics', label: 'Digital Forensics' },
    { value: 'physical_evidence', label: 'Physical Evidence' },
    { value: 'expert_testimony', label: 'Expert Testimony' },
    { value: 'other', label: 'Other Document' },
  ];

  // Priority options
  const priorityOptions = [
    { value: 'low', label: 'Low Priority', color: 'bg-gray-100 text-gray-800' },
    {
      value: 'medium',
      label: 'Medium Priority',
      color: 'bg-blue-100 text-blue-800',
    },
    {
      value: 'high',
      label: 'High Priority',
      color: 'bg-orange-100 text-orange-800',
    },
    { value: 'urgent', label: 'Urgent', color: 'bg-red-100 text-red-800' },
  ];

  // Current step tracking
  let currentStep = $derived(
    steps.findIndex(
      (s) => progress > steps.indexOf(s) * 25 && progress <= (steps.indexOf(s) + 1) * 25
    )
  );

  // File upload handler
  function handleFileUpload(event) {
    const target = event.target;
    if (target.files && target.files.length > 0) {
      evidenceFile = target.files[0];

      // Read file content
      const reader = new FileReader();
      reader.onload = (e) => {
        evidenceContent = e.target?.result as string;
      };
      reader.readAsText(evidenceFile);
    }
  }

  // Start analysis
  async function startAnalysis() {
    if (!caseId || !evidenceContent) {
      error = 'Please provide a case ID and evidence content';
      return;
    }

    analyzing = true;
    error = '';
    results = null;
    progress = 0;

    try {
      const response = await fetch('/api/evidence/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          caseId,
          evidenceFile: evidenceFile?.name || 'uploaded_evidence.txt',
          evidenceContent,
          evidenceType,
          priority,
        }),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      sessionId = data.sessionId;

      // Start polling for results
      pollResults();
    } catch (err) {
      error = err instanceof Error ? err.message : 'Analysis failed';
      analyzing = false;
    }
  }

  // Poll for analysis results with enhanced progress tracking
  async function pollResults() {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/evidence/analyze/${sessionId}`);
        const data = await response.json();

        if (data.status === 'completed') {
          clearInterval(pollInterval);
          results = data;
          analyzing = false;
          progress = 100;
          showResults = true;
        } else if (data.status === 'failed') {
          clearInterval(pollInterval);
          error = data.error || 'Analysis failed';
          analyzing = false;
        } else {
          // Enhanced progress tracking
          const stepIndex = steps.findIndex((s) => s.key === data.step);
          if (stepIndex !== -1) {
            progress = Math.min((stepIndex + 0.5) * 25, 95);
            steps[stepIndex].status = 'processing';

            // Mark previous steps as completed
            for (let i = 0; i < stepIndex; i++) {
              steps[i].status = 'completed';
            }
          }
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 2000);
  }

  // Reset form
  function resetForm() {
    caseId = '';
    evidenceContent = '';
    evidenceFile = null;
    evidenceType = 'police_report';
    priority = 'medium';
    analyzing = false;
    results = null;
    error = '';
    progress = 0;
    showResults = false;
    sessionId = '';

    // Reset steps
    steps.forEach((step) => (step.status = 'pending'));
  }

  // View detailed results
  function viewDetailedResults(analysisData) {
    console.log('Opening detailed results:', analysisData);
    // Could open a modal or navigate to detailed view
  }
</script>

<div class="max-w-6xl mx-auto p-6 space-y-6">
  <div class="text-center space-y-2">
    <h1 class="text-4xl font-bold tracking-tight">Evidence Analysis Pipeline</h1>
    <p class="text-xl text-muted-foreground">AI-powered multi-agent legal document analysis</p>
  </div>

  <!-- Main Analysis Card -->
  <Card class="w-full">
    <CardHeader>
      <CardTitle class="flex items-center gap-2">📄 Evidence Upload & Configuration</CardTitle>
      <CardDescription>
        Configure your evidence analysis parameters and upload documents for processing
      </CardDescription>
    </CardHeader>

    <CardContent class="space-y-6">
      <!-- Form Configuration -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <!-- Case ID -->
        <div class="space-y-2">
          <Label for_="caseId">Case ID *</Label>
          <Input
            id="caseId"
            bind:value={caseId}
            placeholder="CASE-2024-001"
            disabled={analyzing}
            class="font-mono" />
        </div>

        <!-- Evidence Type -->
        <div class="space-y-2">
          <Label>Evidence Type</Label>
          <SelectRoot bind:value={evidenceType} disabled={analyzing}>
            <SelectTrigger>
              <SelectValue placeholder="Select evidence type" />
            </SelectTrigger>
            <SelectContent>
              {#each evidenceTypes as type}
                <SelectItem value={type.value}>{type.label}</SelectItem>
              {/each}
            </SelectContent>
          </SelectRoot>
        </div>

        <!-- Priority -->
        <div class="space-y-2">
          <Label>Priority Level</Label>
          <SelectRoot bind:value={priority} disabled={analyzing}>
            <SelectTrigger>
              <SelectValue placeholder="Select priority" />
            </SelectTrigger>
            <SelectContent>
              {#each priorityOptions as option}
                <SelectItem value={option.value}>{option.label}</SelectItem>
              {/each}
            </SelectContent>
          </SelectRoot>
        </div>
      </div>

      <!-- File Upload -->
      <div class="space-y-2">
        <Label for_="evidenceFile">Evidence File (Optional)</Label>
        <Input
          id="evidenceFile"
          type="file"
          accept=".txt,.pdf,.doc,.docx"
          change={handleFileUpload}
          disabled={analyzing}
          class="file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary hover:file:bg-opacity-80" />
        {#if evidenceFile}
          <div class="flex items-center gap-2 text-sm text-muted-foreground">
            <span>📎</span>
            <span>{evidenceFile.name}</span>
            <Badge variant="outline">{(evidenceFile.size / 1024).toFixed(1)} KB</Badge>
          </div>
        {/if}
      </div>

      <!-- Evidence Content -->
      <div class="space-y-2">
        <Label for_="evidenceContent">Evidence Content *</Label>
        <Textarea
          id="evidenceContent"
          bind:value={evidenceContent}
          placeholder="Paste evidence text here or upload a file above..."
          disabled={analyzing}
          rows={12}
          class="font-mono text-sm" />
        {#if evidenceContent}
          <div class="flex justify-between text-sm text-muted-foreground">
            <span>{evidenceContent.length} characters</span>
            <span>~{Math.ceil(evidenceContent.length / 4)} tokens</span>
          </div>
        {/if}
      </div>
    </CardContent>

    <CardFooter class="flex justify-between">
      <div class="flex items-center gap-2">
        {#if priority !== 'low'}
          <Badge class={priorityOptions.find((p) => p.value === priority)?.color}>
            {priorityOptions.find((p) => p.value === priority)?.label}
          </Badge>
        {/if}
        {#if evidenceType !== 'other'}
          <Badge variant="outline">
            {evidenceTypes.find((t) => t.value === evidenceType)?.label}
          </Badge>
        {/if}
      </div>

      <div class="flex gap-2">
        <Button variant="outline" on:on:click={resetForm} disabled={analyzing}>Reset</Button>
        <Button on:on:click={startAnalysis} disabled={analyzing || !caseId || !evidenceContent}>
          {analyzing ? 'Analyzing...' : 'Start Analysis'}
        </Button>
      </div>
    </CardFooter>
  </Card>

  <!-- Error Display -->
  {#if error}
    <Card class="border-destructive">
      <CardContent class="pt-6">
        <div class="flex items-center gap-2 text-destructive">
          <span>❌</span>
          <span class="font-semibold">Analysis Error:</span>
          <span>{error}</span>
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Progress Display -->
  {#if analyzing}
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center gap-2">🔄 Analysis in Progress</CardTitle>
        <CardDescription>Multi-agent pipeline processing your evidence document</CardDescription>
      </CardHeader>

      <CardContent class="space-y-6">
        <!-- Overall Progress -->
        <div class="space-y-2">
          <div class="flex justify-between text-sm">
            <span class="font-medium">Overall Progress</span>
            <span class="text-muted-foreground">{progress.toFixed(0)}%</span>
          </div>
          <Progress value={progress} class="h-3" />
        </div>

        <!-- Step-by-step Progress -->
        <div class="space-y-4">
          {#each steps as step, i}
            {@const isActive = currentStep === i}
            {@const isCompleted = step.status === 'completed'}
            {@const isProcessing = step.status === 'processing'}

            <Card
              class="transition-all duration-300 {isActive ? 'ring-2 ring-primary shadow-md' : ''}">
              <CardContent class="p-4">
                <div class="flex items-center gap-4">
                  <!-- Status Icon -->
                  <div class="flex-shrink-0">
                    {#if isCompleted}
                      <div
                        class="w-10 h-10 bg-green-100 text-green-600 rounded-full flex items-center justify-center">
                        ✓
                      </div>
                    {:else if isProcessing}
                      <div
                        class="w-10 h-10 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center animate-pulse">
                        {step.icon}
                      </div>
                    {:else}
                      <div
                        class="w-10 h-10 bg-gray-100 text-gray-400 rounded-full flex items-center justify-center">
                        {step.icon}
                      </div>
                    {/if}
                  </div>

                  <!-- Step Info -->
                  <div class="flex-grow">
                    <div class="flex items-center gap-2">
                      <h3 class="font-semibold {isActive ? 'text-primary' : ''}">
                        {step.name}
                      </h3>
                      {#if isProcessing}
                        <Badge variant="outline" class="animate-pulse">Processing</Badge>
                      {:else if isCompleted}
                        <Badge class="bg-green-100 text-green-800">Completed</Badge>
                      {:else}
                        <Badge variant="secondary">Pending</Badge>
                      {/if}
                    </div>
                    <p class="text-sm text-muted-foreground">
                      {step.description}
                    </p>
                    <p class="text-xs text-muted-foreground">
                      Est. {step.duration}
                    </p>
                  </div>

                  <!-- Mini Progress for Active Step -->
                  {#if isProcessing}
                    <div class="flex-shrink-0 w-20">
                      <Progress value={75} class="h-2" />
                    </div>
                  {/if}
                </div>
              </CardContent>
            </Card>
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Results Modal/Display -->
  {#if showResults && results}
    <Dialog bind:open={showResults}>
      <DialogContent class="max-w-4xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Analysis Results - {caseId}</DialogTitle>
          <DialogDescription>
            Multi-agent pipeline analysis completed successfully
          </DialogDescription>
        </DialogHeader>

        <div class="space-y-4">
          {#each Object.entries(results.outputs) as [key, data]}
            <Card>
              <CardHeader>
                <CardTitle class="text-lg">
                  {steps.find((s) => s.key === key)?.icon || '📄'}
                  {key.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div class="bg-muted p-4 rounded-lg">
                  <pre class="text-xs overflow-auto max-h-32 whitespace-pre-wrap">
										{JSON.stringify(data, null, 2)}
									</pre>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  class="mt-2"
                  on:on:click={() => viewDetailedResults(data)}>
                  View Details →
                </Button>
              </CardContent>
            </Card>
          {/each}
        </div>

        <DialogFooter>
          <Button variant="outline" on:on:click={() => (showResults = false)}>Close</Button>
          <Button on:on:click={() => goto(`/cases/${caseId}`)}>View Case Details</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  {/if}
</div>

<style>
  /* Custom animations for progress indicators */
  @keyframes pulse-glow {
    0%,
    100% {
      box-shadow: 0 0 5px rgba(59, 130, 246, 0.3);
    }
    50% {
      box-shadow: 0 0 20px rgba(59, 130, 246, 0.6);
    }
  }

  .animate-pulse-glow {
    animation: pulse-glow 2s infinite;
  }
</style>
