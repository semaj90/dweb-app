<script lang="ts">
  import { Button } from "bits-ui";
  import { onMount } from "svelte";

  interface CaseScore {
    score: number
    breakdown: {
      admissibility: number
      relevance: number
      quality: number
      strategic: number
    };
    reasoning: string
    confidence: number
    lastUpdated: string
  }

  interface Props {
    caseId: string
    evidenceId?: string;
    content: string
    evidenceType: string
    autoScore?: boolean;
  }

  let {
    caseId,
    evidenceId,
    content,
    evidenceType,
    autoScore = true,
  } = $props();

  let scoring = $state<CaseScore | null>(null);
  let loading = $state(false);
  let error = $state("");

  async function calculateScore() {
    loading = true;
    error = "";

    try {
      const response = await fetch("/api/ai/case-scoring", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content,
          evidenceType,
          caseId,
          evidenceId,
        }),
      });

      if (!response.ok) throw new Error(`Scoring failed: ${response.status}`);

      scoring = await response.json();
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  function getScoreColor(score: number): string {
    if (score >= 80) return "bg-green-500";
    if (score >= 60) return "bg-blue-500";
    if (score >= 40) return "bg-yellow-500";
    return "bg-red-500";
  }

  function getScoreLabel(score: number): string {
    if (score >= 80) return "High Value";
    if (score >= 60) return "Medium Value";
    if (score >= 40) return "Low Value";
    return "Poor Quality";
  }

  onMount(() => {
    if (autoScore) calculateScore();
  });
</script>

<div
  class="case-scoring-card bg-slate-900 border border-slate-700 rounded-lg p-4"
>
  <div class="flex items-center justify-between mb-4">
    <h3 class="text-lg font-semibold text-white">Case AI Scoring</h3>
    <Button.Root
      size="sm"
      on:click={calculateScore}
      disabled={loading}
      class="text-xs border border-slate-600 hover:bg-slate-700"
    >
      {loading ? "Analyzing..." : "Rescore"}
    </Button.Root>
  </div>

  {#if loading}
    <div class="flex items-center gap-2 text-blue-400">
      <div
        class="animate-spin w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full"
      ></div>
      <span>AI analysis in progress...</span>
    </div>
  {:else if error}
    <div class="text-red-400 text-sm">
      Error: {error}
    </div>
  {:else if scoring}
    <!-- Main Score Display -->
    <div class="flex items-center gap-4 mb-6">
      <div class="relative">
        <div
          class="w-20 h-20 rounded-full bg-slate-800 flex items-center justify-center"
        >
          <span class="text-2xl font-bold text-white">{scoring.score}</span>
        </div>
        <div class="absolute -bottom-1 -right-1">
          <span
            class="{getScoreColor(
              scoring.score
            )} text-white text-xs px-2 py-1 rounded-full"
          >
            {getScoreLabel(scoring.score)}
          </span>
        </div>
      </div>

      <div class="flex-1">
        <div class="text-sm text-slate-400 mb-1">Overall Score</div>
        <div class="w-full h-2 bg-slate-800 rounded overflow-hidden">
          <div
            class="{getScoreColor(scoring.score)} h-full rounded transition-all"
            style="width: {scoring.score}%"
          ></div>
        </div>
        <div class="text-xs text-slate-500 mt-1">
          Confidence: {Math.round(scoring.confidence * 100)}%
        </div>
      </div>
    </div>

    <!-- Score Breakdown -->
    <div class="grid grid-cols-2 gap-3 mb-4">
      <div class="score-metric">
        <div class="flex justify-between items-center mb-1">
          <span class="text-sm text-slate-300">Admissibility</span>
          <span class="text-sm font-medium text-white"
            >{scoring.breakdown.admissibility}</span
          >
        </div>
        <div class="w-full h-1 bg-slate-800 rounded overflow-hidden">
          <div
            class="bg-blue-500 h-full rounded"
            style="width: {(scoring.breakdown.admissibility / 25) * 100}%"
          ></div>
        </div>
      </div>

      <div class="score-metric">
        <div class="flex justify-between items-center mb-1">
          <span class="text-sm text-slate-300">Relevance</span>
          <span class="text-sm font-medium text-white"
            >{scoring.breakdown.relevance}</span
          >
        </div>
        <div class="w-full h-1 bg-slate-800 rounded overflow-hidden">
          <div
            class="bg-green-500 h-full rounded"
            style="width: {(scoring.breakdown.relevance / 25) * 100}%"
          ></div>
        </div>
      </div>

      <div class="score-metric">
        <div class="flex justify-between items-center mb-1">
          <span class="text-sm text-slate-300">Quality</span>
          <span class="text-sm font-medium text-white"
            >{scoring.breakdown.quality}</span
          >
        </div>
        <div class="w-full h-1 bg-slate-800 rounded overflow-hidden">
          <div
            class="bg-yellow-500 h-full rounded"
            style="width: {(scoring.breakdown.quality / 25) * 100}%"
          ></div>
        </div>
      </div>

      <div class="score-metric">
        <div class="flex justify-between items-center mb-1">
          <span class="text-sm text-slate-300">Strategic</span>
          <span class="text-sm font-medium text-white"
            >{scoring.breakdown.strategic}</span
          >
        </div>
        <div class="w-full h-1 bg-slate-800 rounded overflow-hidden">
          <div
            class="bg-purple-500 h-full rounded"
            style="width: {(scoring.breakdown.strategic / 25) * 100}%"
          ></div>
        </div>
      </div>
    </div>

    <!-- AI Reasoning -->
    <div class="bg-slate-800 rounded-md p-3">
      <div class="text-sm font-medium text-slate-300 mb-1">AI Analysis</div>
      <div class="text-xs text-slate-400">
        {scoring.reasoning}
      </div>
    </div>

    <!-- Metadata -->
    <div class="mt-3 text-xs text-slate-500 flex justify-between">
      <span>Type: {evidenceType}</span>
      <span>Updated: {new Date(scoring.lastUpdated).toLocaleTimeString()}</span>
    </div>
  {:else}
    <div class="text-slate-400 text-center py-8">
      <Button.Root
        on:click={calculateScore}
        class="text-sm border border-slate-600 hover:bg-slate-700"
      >
        Calculate AI Score
      </Button.Root>
    </div>
  {/if}
</div>

<style>
  .case-scoring-card {
    min-height: 280px;
  }

  .score-metric {
    padding: 8px;
    background: rgba(30, 41, 59, 0.3);
    border-radius: 6px;
  }
</style>
