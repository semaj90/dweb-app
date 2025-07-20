<!-- Case Summary Modal with AI-generated insights -->
<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import { createEventDispatcher } from "svelte";
  import Badge from '$lib/components/ui/Badge.svelte';
  import Dialog from '$lib/components/ui/dialog/Dialog.svelte';
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

  export let open: boolean = false;
  // SSR-compatible: all dates as strings
  export let caseData: {
    id: string;
    title: string;
    description: string;
    status: "active" | "pending" | "closed";
    priority: "low" | "medium" | "high" | "critical";
    createdAt: string; // ISO string
    updatedAt: string; // ISO string
    assignedTo?: string;
    tags?: string[];
    summary?: {
      aiGenerated: boolean;
      overview: string;
      keyFindings: string[];
      recommendations: string[];
      riskAssessment: {
        level: "low" | "medium" | "high";
        factors: string[];
      };
      timeline: Array<{
        date: string; // ISO string
        event: string;
        importance: "low" | "medium" | "high";
      }>;
      evidence: {
        total: number;
        admissible: number;
        questionable: number;
        inadmissible: number;
      };
      nextSteps: string[];
    };
    metrics?: {
      evidenceCount: number;
      documentsReviewed: number;
      witnessesInterviewed: number;
      daysActive: number;
      completionPercentage: number;
    };
  } | null = null;

  export let useDrawer: boolean = false;

  const dispatch = createEventDispatcher();

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
        dispatch("summaryGenerated", caseData);
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
  <Dialog
    bind:open
    title="Case Summary"
    description="Comprehensive AI-powered case analysis"
    size="full"
  >
    <svelte:fragment slot="trigger">
      <slot name="trigger" />
    </svelte:fragment>

    {#if caseData}
      <div class="container mx-auto px-4">
        <!-- Case Header -->
        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4">
            <div class="container mx-auto px-4">
              <div class="container mx-auto px-4">
                <Folder class="container mx-auto px-4" />
              </div>
              <div>
                <h2 class="container mx-auto px-4">
                  {caseData.title}
                </h2>
                <p class="container mx-auto px-4">{caseData.description}</p>
                <div class="container mx-auto px-4">
                  <span
                    class="container mx-auto px-4"
                  >
                    {caseData.status}
                  </span>
                  <span
                    class="container mx-auto px-4"
                  >
                    {caseData.priority} priority
                  </span>
                </div>
              </div>
            </div>

            <div class="container mx-auto px-4">
              <Button
                variant="secondary"
                size="sm"
                on:click={() => generateSummary()}
                disabled={isGeneratingSummary}
              >
                {#if isGeneratingSummary}
                  <div
                    class="container mx-auto px-4"
                  ></div>
                  Generating...
                {:else}
                  <Sparkles class="container mx-auto px-4" />
                  Generate Summary
                {/if}
              </Button>
            </div>
          </div>

          <!-- Metrics -->
          {#if caseData.metrics}
            <Grid columns={5} gap="md">
              <GridItem colSpan={1}>
                <div class="container mx-auto px-4">
                  <FileText class="container mx-auto px-4" />
                  <div class="container mx-auto px-4">
                    {caseData.metrics.evidenceCount}
                  </div>
                  <div class="container mx-auto px-4">Evidence Items</div>
                </div>
              </GridItem>

              <GridItem colSpan={1}>
                <div class="container mx-auto px-4">
                  <Users class="container mx-auto px-4" />
                  <div class="container mx-auto px-4">
                    {caseData.metrics.witnessesInterviewed}
                  </div>
                  <div class="container mx-auto px-4">Witnesses</div>
                </div>
              </GridItem>

              <GridItem colSpan={1}>
                <div class="container mx-auto px-4">
                  <Calendar class="container mx-auto px-4" />
                  <div class="container mx-auto px-4">
                    {caseData.metrics.daysActive}
                  </div>
                  <div class="container mx-auto px-4">Days Active</div>
                </div>
              </GridItem>

              <GridItem colSpan={1}>
                <div class="container mx-auto px-4">
                  <Target class="container mx-auto px-4" />
                  <div class="container mx-auto px-4">
                    {caseData.metrics.completionPercentage}%
                  </div>
                  <div class="container mx-auto px-4">Complete</div>
                </div>
              </GridItem>

              <GridItem colSpan={1}>
                <div class="container mx-auto px-4">
                  <Clock class="container mx-auto px-4" />
                  <div class="container mx-auto px-4">
                    {caseData.metrics.documentsReviewed}
                  </div>
                  <div class="container mx-auto px-4">Docs Reviewed</div>
                </div>
              </GridItem>
            </Grid>
          {/if}
        </div>

        <!-- Tabs -->
        <div class="container mx-auto px-4">
          <nav class="container mx-auto px-4">
            {#each [{ id: "overview", label: "Overview", icon: Brain }, { id: "timeline", label: "Timeline", icon: Calendar }, { id: "evidence", label: "Evidence", icon: FileText }, { id: "recommendations", label: "Recommendations", icon: Target }] as tab}
              <button
                class="container mx-auto px-4"
                on:click={() => (activeTab = tab.id as typeof activeTab)}
              >
                <svelte:component this={tab.icon} class="container mx-auto px-4" />
                {tab.label}
              </button>
            {/each}
          </nav>
        </div>

        <!-- Tab Content -->
        {#if caseData.summary}
          <div class="container mx-auto px-4">
            {#if activeTab === "overview"}
              <Grid columns={12} gap="lg">
                <GridItem colSpan={8}>
                  <div class="container mx-auto px-4">
                    <h3 class="container mx-auto px-4">Case Overview</h3>
                    <p class="container mx-auto px-4">
                      {caseData.summary.overview}
                    </p>
                  </div>

                  <div class="container mx-auto px-4">
                    <h3 class="container mx-auto px-4">Key Findings</h3>
                    <ul class="container mx-auto px-4">
                      {#each caseData.summary.keyFindings as finding}
                        <li class="container mx-auto px-4">
                          <CheckCircle
                            class="container mx-auto px-4"
                          />
                          <span class="container mx-auto px-4">{finding}</span>
                        </li>
                      {/each}
                    </ul>
                  </div>
                </GridItem>

                <GridItem colSpan={4}>
                  <div class="container mx-auto px-4">
                    <h3 class="container mx-auto px-4">Risk Assessment</h3>
                    <div class="container mx-auto px-4">
                      <div class="container mx-auto px-4">
                        <span class="container mx-auto px-4">Risk Level</span>
                        <span
                          class="container mx-auto px-4"
                        >
                          {caseData.summary.riskAssessment.level.toUpperCase()}
                        </span>
                      </div>
                    </div>

                    <h4 class="container mx-auto px-4">Risk Factors</h4>
                    <ul class="container mx-auto px-4">
                      {#each caseData.summary.riskAssessment.factors as factor}
                        <li class="container mx-auto px-4">
                          <AlertTriangle
                            class="container mx-auto px-4"
                          />
                          <span class="container mx-auto px-4">{factor}</span>
                        </li>
                      {/each}
                    </ul>
                  </div>
                </GridItem>
              </Grid>
            {/if}

            {#if activeTab === "timeline"}
              <div class="container mx-auto px-4">
                <h3 class="container mx-auto px-4">Case Timeline</h3>
                <div class="container mx-auto px-4">
                  {#each caseData.summary.timeline as event}
                    <div class="container mx-auto px-4">
                      <div
                        class="container mx-auto px-4"
                      ></div>
                      <div class="container mx-auto px-4">
                        <div class="container mx-auto px-4">
                          <span class="container mx-auto px-4"
                            >{event.event}</span
                          >
                          <span class="container mx-auto px-4"
                            >{formatDate(event.date)}</span
                          >
                        </div>
                        <Badge
                          variant={event.importance === "high"
                            ? "destructive"
                            : event.importance === "medium"
                              ? "secondary"
                              : "outline"}
                          size="sm"
                        >
                          {event.importance} importance
                        </Badge>
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}

            {#if activeTab === "evidence"}
              <div class="container mx-auto px-4">
                <h3 class="container mx-auto px-4">Evidence Summary</h3>
                <Grid columns={4} gap="md">
                  <GridItem colSpan={1}>
                    <div class="container mx-auto px-4">
                      <div class="container mx-auto px-4">
                        {caseData.summary.evidence.total}
                      </div>
                      <div class="container mx-auto px-4">Total Items</div>
                    </div>
                  </GridItem>
                  <GridItem colSpan={1}>
                    <div class="container mx-auto px-4">
                      <div class="container mx-auto px-4">
                        {caseData.summary.evidence.admissible}
                      </div>
                      <div class="container mx-auto px-4">Admissible</div>
                    </div>
                  </GridItem>
                  <GridItem colSpan={1}>
                    <div class="container mx-auto px-4">
                      <div class="container mx-auto px-4">
                        {caseData.summary.evidence.questionable}
                      </div>
                      <div class="container mx-auto px-4">Questionable</div>
                    </div>
                  </GridItem>
                  <GridItem colSpan={1}>
                    <div class="container mx-auto px-4">
                      <div class="container mx-auto px-4">
                        {caseData.summary.evidence.inadmissible}
                      </div>
                      <div class="container mx-auto px-4">Inadmissible</div>
                    </div>
                  </GridItem>
                </Grid>
              </div>
            {/if}

            {#if activeTab === "recommendations"}
              <div class="container mx-auto px-4">
                <h3 class="container mx-auto px-4">AI Recommendations</h3>
                <div class="container mx-auto px-4">
                  {#each caseData.summary.recommendations as recommendation}
                    <div class="container mx-auto px-4">
                      <Target
                        class="container mx-auto px-4"
                      />
                      <span class="container mx-auto px-4">{recommendation}</span>
                    </div>
                  {/each}
                </div>

                <div class="container mx-auto px-4">
                  <h4 class="container mx-auto px-4">Next Steps</h4>
                  <ul class="container mx-auto px-4">
                    {#each caseData.summary.nextSteps as step}
                      <li class="container mx-auto px-4">
                        <div
                          class="container mx-auto px-4"
                        ></div>
                        <span class="container mx-auto px-4">{step}</span>
                      </li>
                    {/each}
                  </ul>
                </div>
              </div>
            {/if}
          </div>
        {:else}
          <div class="container mx-auto px-4">
            <Brain class="container mx-auto px-4" />
            <h3 class="container mx-auto px-4">
              No Summary Available
            </h3>
            <p class="container mx-auto px-4">
              Generate an AI-powered summary to see comprehensive case analysis.
            </p>
            <Button on:click={() => generateSummary()} disabled={isGeneratingSummary}>
              {#if isGeneratingSummary}
                <div
                  class="container mx-auto px-4"
                ></div>
                Generating Summary...
              {:else}
                <Sparkles class="container mx-auto px-4" />
                Generate Summary
              {/if}
            </Button>
          </div>
        {/if}
      </div>
    {/if}

    <svelte:fragment slot="footer" let:close>
      <Button variant="secondary" on:click={() => close()}>Close</Button>
      <Button
        variant="primary"
        on:click={() => dispatch("exportSummary", caseData)}
      >
        Export Summary
      </Button>
    </svelte:fragment>
  </Dialog>
{/if}
