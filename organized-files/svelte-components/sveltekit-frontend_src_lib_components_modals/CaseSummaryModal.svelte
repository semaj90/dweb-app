<!-- Case Summary Modal with AI-generated insights -->
<script lang="ts">
  interface Props {
    onsummaryGenerated?: (event?: any) => void;
  }
  let {
    open = false,
    caseData,
    useDrawer = false
  } = $props();



  import { Button } from "$lib/components/ui/button";
    import Badge from '$lib/components/ui/Badge.svelte';
  import * as Dialog from '$lib/components/ui/dialog';
  import Drawer from '$lib/components/ui/drawer/Drawer.svelte';
  import Grid from '$lib/components/ui/grid/Grid.svelte';
  import GridItem from '$lib/components/ui/grid/GridItem.svelte';
// Icons
  import {
    AlertTriangle,
    Brain,
    Calendar,
    CheckCircle,
    Clock,
    FileText,
    Folder,
    Sparkles,
    Target,
    Users,
  } from "lucide-svelte";

    // SSR-compatible: all dates as strings
      id: string
    title: string
    description: string
    status: "active" | "pending" | "closed";
    priority: "low" | "medium" | "high" | "critical";
    createdAt: string // ISO string
    updatedAt: string // ISO string
    assignedTo?: string;
    tags?: string[];
    summary?: {
      aiGenerated: boolean
      overview: string
      keyFindings: string[];
      recommendations: string[];
      riskAssessment: {
        level: "low" | "medium" | "high";
        factors: string[];
      };
      timeline: Array<{
        date: string // ISO string
        event: string
        importance: "low" | "medium" | "high";
      }>;
      evidence: {
        total: number
        admissible: number
        questionable: number
        inadmissible: number
      };
      nextSteps: string[];
    };
    metrics?: {
      evidenceCount: number
      documentsReviewed: number
      witnessesInterviewed: number
      daysActive: number
      completionPercentage: number
    };
  } | null = null;

  
  
  let isGeneratingSummary = false;
  let activeTab: "overview" | "timeline" | "evidence" | "recommendations" =
    "overview";

  async function generateSummary() {
    if (!caseData) return;

    isGeneratingSummary = true;
    try {
      const response = await fetch("/api/cases/summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          caseId: caseData.id,
          includeEvidence: true,
          includeTimeline: true,
          analysisDepth: "comprehensive",
        }),
      });

      const result = await response.json();
      if (result.success) {
        caseData = { ...caseData, summary: result.summary };
        onsummaryGenerated?.();
}
    } catch (error) {
      console.error("Summary generation failed:", error);
    } finally {
      isGeneratingSummary = false;
}}
  function getStatusColor(status: string): string {
    switch (status) {
      case "active":
        return "bg-green-100 text-green-800 border-green-300";
      case "pending":
        return "bg-yellow-100 text-yellow-800 border-yellow-300";
      case "closed":
        return "bg-gray-100 text-gray-800 border-gray-300";
      default:
        return "bg-gray-100 text-gray-800 border-gray-300";
}}
  function getPriorityColor(priority: string): string {
    switch (priority) {
      case "critical":
        return "bg-red-100 text-red-800 border-red-300";
      case "high":
        return "bg-orange-100 text-orange-800 border-orange-300";
      case "medium":
        return "bg-yellow-100 text-yellow-800 border-yellow-300";
      case "low":
        return "bg-green-100 text-green-800 border-green-300";
      default:
        return "bg-gray-100 text-gray-800 border-gray-300";
}}
  function getRiskColor(level: string): string {
    switch (level) {
      case "high":
        return "text-red-600";
      case "medium":
        return "text-yellow-600";
      case "low":
        return "text-green-600";
      default:
        return "text-gray-600";
}}
  // SSR: parse date string only on client
  function formatDate(dateString: string): string {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    }).format(date);
}
</script>

{#if useDrawer}
  <Drawer
    bind:open
    title="Case Summary"
    description="Comprehensive AI-powered case analysis"
    side="right"
    size="xl"
  >
    <svelte:fragment slot="trigger">
      <slot name="trigger" />
    </svelte:fragment>

    <svelte:fragment slot="default">
      <slot name="content" />
    </svelte:fragment>
  </Drawer>
{:else}
  <Dialog.Root open={isOpen} onclose={closeModal}>
    <Dialog.Content size="lg">
      <Dialog.Header>
        <Dialog.Title>Case Summary</Dialog.Title>
        <Dialog.Description>AI-generated summary of your case</Dialog.Description>
      </Dialog.Header>
      <!-- Main content of the modal -->
      <div class="p-6 space-y-4">
        {#if isGeneratingSummary}
          <!-- Loading State -->
          <div class="flex flex-col items-center justify-center h-48">
            <Brain class="w-16 h-16 text-primary animate-pulse" />
            <span class="text-lg text-muted-foreground mt-4">Generating AI Summary...</span>
          </div>
        {:else if caseData?.summary}
          <!-- Summary Content -->
          <div class="space-y-4">
            <div class="flex justify-between items-center">
              <h3 class="text-lg font-semibold">Overview</h3>
              <Button onclick={generateSummary} disabled={isGeneratingSummary} size="sm" variant="outline">
                <Sparkles class="w-4 h-4 mr-2" /> Regenerate
              </Button>
            </div>
            <p class="text-muted-foreground">{caseData.summary.overview}</p>

            <h3 class="text-lg font-semibold">Key Findings</h3>
            <ul class="list-disc list-inside text-muted-foreground">
              {#each caseData.summary.keyFindings as finding}
                <li>{finding}</li>
              {/each}
            </ul>

            <h3 class="text-lg font-semibold">Recommendations</h3>
            <ul class="list-disc list-inside text-muted-foreground">
              {#each caseData.summary.recommendations as rec}
                <li>{rec}</li>
              {/each}
            </ul>

            <h3 class="text-lg font-semibold">Risk Assessment</h3>
            <div class="flex items-center gap-2">
              <Badge variant="outline" class="{getRiskColor(caseData.summary.riskAssessment.level)}">
                {caseData.summary.riskAssessment.level}
              </Badge>
              <span class="text-muted-foreground">({caseData.summary.riskAssessment.factors.join(', ')})</span>
            </div>

            <h3 class="text-lg font-semibold">Timeline</h3>
            <ol class="border-l border-gray-200 space-y-4 pl-4">
              {#each caseData.summary.timeline as event}
                <li>
                  <div class="flex items-center">
                    <div class="absolute w-3 h-3 bg-primary rounded-full mt-1.5 -left-1.5 border border-white"></div>
                    <time class="ml-2 text-sm font-semibold text-primary">{formatDate(event.date)}</time>
                  </div>
                  <p class="ml-2 text-muted-foreground">{event.event}</p>
                </li>
              {/each}
            </ol>

            <h3 class="text-lg font-semibold">Evidence Overview</h3>
            <div class="grid grid-cols-2 gap-4 text-muted-foreground">
              <div>Total Evidence: {caseData.summary.evidence.total}</div>
              <div>Admissible: {caseData.summary.evidence.admissible}</div>
              <div>Questionable: {caseData.summary.evidence.questionable}</div>
              <div>Inadmissible: {caseData.summary.evidence.inadmissible}</div>
            </div>

            <h3 class="text-lg font-semibold">Next Steps</h3>
            <ul class="list-disc list-inside text-muted-foreground">
              {#each caseData.summary.nextSteps as step}
                <li>{step}</li>
              {/each}
            </ul>
          </div>
        {:else}
          <div class="flex flex-col items-center justify-center h-48 text-muted-foreground">
            <Brain class="w-16 h-16 mb-4 opacity-50" />
            <p>No AI summary available for this case.</p>
            <Button onclick={generateSummary} disabled={isGeneratingSummary} class="mt-4">
              <Sparkles class="w-4 h-4 mr-2" /> Generate Summary
            </Button>
          </div>
        {/if}
      </div>
    </Dialog.Content>
  </Dialog.Root>
{/if}
