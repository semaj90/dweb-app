<script lang="ts">
  interface Props {
    documentId: string | null ;
    caseId: string | null ;
    initialContent: string
    documentType:  | "evidence";
    compact: boolean
  }
  let {
    documentId = null,
    caseId = null,
    initialContent = "",
    documentType,
    compact = false
  } = $props();



  import {
    aiSummaryMachine,
    type SummarySection,
  } from "$lib/machines/aiSummaryMachine";
  import { useMachine } from "@xstate/svelte";
  import {
    Brain,
    FileText,
    Pause,
    Play,
    Settings,
    SkipBack,
    SkipForward,
    Square,
    Zap,
  } from "lucide-svelte";
  import { onMount } from "svelte";
  import { fade, fly } from "svelte/transition";

             | "report"
     | "contract"
     | "case_law"
     | "general" = "evidence";

  const { state, send } = useMachine(aiSummaryMachine);

  // Reactive state helpers
  let isLoading = $derived($state.matches("loading") ||);
    $state.matches("generating") ||
    $state.matches("analyzing") ||
    $state.matches("synthesizing");
  let isReady = $derived($state.matches("ready"));
  let isReading = $derived($state.matches("ready.reading"));
  let isPlaying = $derived($state.context.isPlaying)
  let currentSection = $derived($state.context.sections[$state.context.currentSection])
  let progress = $derived($state.context.progress)
  let error = $derived($state.context.error)

  // Voice synthesis
  let speechSynthesis: SpeechSynthesis | null = null;
  let currentUtterance: SpeechSynthesisUtterance | null = null;

  onMount(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      speechSynthesis = window.speechSynthesis;
    }

    // Auto-load if content provided
    if (initialContent && documentType) {
      send({ type: "GENERATE_SUMMARY", content: initialContent, documentType });
    } else if (documentId) {
      send({ type: "LOAD_DOCUMENT", documentId, caseId });
    }
  });

  function toggleReading() {
    if (isPlaying) {
      send({ type: "PAUSE_READING" });
      if (currentUtterance && speechSynthesis) {
        speechSynthesis.pause();
      }
    } else {
      send({ type: "START_READING" });
      if ($state.context.voiceEnabled && currentSection) {
        speakSection(currentSection);
      }
    }
  }

  function stopReading() {
    send({ type: "STOP_READING" });
    if (speechSynthesis) {
      speechSynthesis.cancel();
    }
  }

  function nextSection() {
    send({ type: "NEXT_SECTION" });
    if ($state.context.voiceEnabled && isPlaying) {
      setTimeout(
        () =>
          speakSection($state.context.sections[$state.context.currentSection]),
        100
      );
    }
  }

  function previousSection() {
    send({ type: "PREVIOUS_SECTION" });
    if ($state.context.voiceEnabled && isPlaying) {
      setTimeout(
        () =>
          speakSection($state.context.sections[$state.context.currentSection]),
        100
      );
    }
  }

  function jumpToSection(index: number) {
    send({ type: "JUMP_TO_SECTION", sectionIndex: index });
    if ($state.context.voiceEnabled && isPlaying) {
      setTimeout(() => speakSection($state.context.sections[index]), 100);
    }
  }

  function speakSection(section: SummarySection) {
    if (!speechSynthesis || !$state.context.voiceEnabled) return;

    speechSynthesis.cancel();

    currentUtterance = new SpeechSynthesisUtterance(section.content);
    currentUtterance.rate = 0.9;
    currentUtterance.pitch = 1.0;
    currentUtterance.volume = 0.8;

    currentUtterance.onend = () => {
      if ($state.context.currentSection < $state.context.sections.length - 1) {
        nextSection();
      } else {
        stopReading();
      }
    };

    speechSynthesis.speak(currentUtterance);
  }

  function analyzeDocument() {
    send({ type: "ANALYZE_DOCUMENT" });
  }

  function synthesizeInsights() {
    send({ type: "SYNTHESIZE_INSIGHTS" });
  }

  function toggleVoice() {
    send({
      type: "UPDATE_PREFERENCES",
      preferences: { voiceEnabled: !$state.context.voiceEnabled },
    });
  }

  function getImportanceColor(importance: string) {
    switch (importance) {
      case "critical":
        return "text-red-600 border-red-200 bg-red-50";
      case "high":
        return "text-orange-600 border-orange-200 bg-orange-50";
      case "medium":
        return "text-yellow-600 border-yellow-200 bg-yellow-50";
      case "low":
        return "text-gray-600 border-gray-200 bg-gray-50";
      default:
        return "text-gray-600 border-gray-200 bg-gray-50";
    }
  }

  function getAnalysisScoreColor(score: number) {
    if (score >= 0.9) return "text-green-600 bg-green-100";
    if (score >= 0.7) return "text-yellow-600 bg-yellow-100";
    return "text-red-600 bg-red-100";
  }
</script>

<div class="ai-summary-reader" class:compact>
  <div class="bg-white border border-gray-200 rounded-lg shadow-sm">
    <!-- Header -->
    <div class="flex items-center justify-between p-4 border-b border-gray-200">
      <div class="flex items-center gap-3">
        <div class="p-2 bg-blue-100 rounded-lg">
          <Brain class="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h3 class="text-lg font-semibold text-gray-900">AI Summary Reader</h3>
          <p class="text-sm text-gray-600">
            {#if documentId}
              Document ID: {documentId}
            {:else if $state.context.documentType}
              {$state.context.documentType.charAt(0).toUpperCase() +
                $state.context.documentType.slice(1)} Analysis
            {/if}
          </p>
        </div>
      </div>

      <div class="flex items-center gap-2">
        <!-- Voice Toggle -->
        <button
          on:click={toggleVoice}
          class="p-2 rounded-md hover:bg-gray-100 transition-colors"
          class:text-blue-600={$state.context.voiceEnabled}
          class:text-gray-400={!$state.context.voiceEnabled}
          title={$state.context.voiceEnabled ? "Disable voice" : "Enable voice"}
        >
          <Settings class="w-4 h-4" />
        </button>

        <!-- Confidence Score -->
        {#if $state.context.confidence > 0}
          <div
            class="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium"
          >
            {Math.round($state.context.confidence * 100)}% confidence
          </div>
        {/if}
      </div>
    </div>

    <!-- Content -->
    <div class="p-4">
      {#if isLoading}
        <div class="flex items-center justify-center py-12">
          <div class="flex flex-col items-center gap-3">
            <div
              class="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"
            ></div>
            <p class="text-gray-600">
              {#if $state.matches("loading")}
                Loading document...
              {:else if $state.matches("generating")}
                Generating AI summary...
              {:else if $state.matches("analyzing")}
                Analyzing document...
              {:else if $state.matches("synthesizing")}
                Synthesizing insights...
              {/if}
            </p>
          </div>
        </div>
      {:else if error}
        <div
          class="bg-red-50 border border-red-200 rounded-lg p-4"
          transition:fade
        >
          <div class="flex items-center gap-2">
            <div class="text-red-600">⚠️</div>
            <div>
              <h4 class="font-medium text-red-800">Error</h4>
              <p class="text-red-700 text-sm">{error}</p>
            </div>
          </div>
          <button
            on:click={() => send({ type: "RETRY" })}
            class="mt-3 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors text-sm"
          >
            Retry
          </button>
        </div>
      {:else if isReady}
        <div class="space-y-6">
          <!-- Summary Overview -->
          {#if $state.context.summary}
            <div
              class="bg-blue-50 border border-blue-200 rounded-lg p-4"
              transition:fly={{ y: 20, duration: 300 }}
            >
              <h4 class="font-medium text-blue-900 mb-2">Executive Summary</h4>
              <p class="text-blue-800">{$state.context.summary}</p>
            </div>
          {/if}

          <!-- Key Insights -->
          {#if $state.context.keyInsights.length > 0}
            <div
              class="bg-green-50 border border-green-200 rounded-lg p-4"
              transition:fly={{ y: 20, duration: 300, delay: 100 }}
            >
              <h4 class="font-medium text-green-900 mb-3">Key Insights</h4>
              <ul class="space-y-2">
                {#each $state.context.keyInsights as insight}
                  <li class="flex items-start gap-2">
                    <Zap class="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                    <span class="text-green-800">{insight}</span>
                  </li>
                {/each}
              </ul>
            </div>
          {/if}

          <!-- Reading Controls -->
          <div
            class="flex items-center justify-between bg-gray-50 rounded-lg p-4"
          >
            <div class="flex items-center gap-3">
              <button
                on:click={toggleReading}
                class="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                disabled={!currentSection}
              >
                {#if isPlaying}
                  <Pause class="w-4 h-4" />
                  Pause
                {:else}
                  <Play class="w-4 h-4" />
                  {isReading ? "Resume" : "Start Reading"}
                {/if}
              </button>

              <button
                on:click={stopReading}
                class="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-200 rounded-md transition-colors"
                disabled={!isReading}
              >
                <Square class="w-4 h-4" />
              </button>

              <div class="flex items-center gap-1">
                <button
                  on:click={previousSection}
                  class="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-200 rounded-md transition-colors"
                  disabled={$state.context.currentSection === 0}
                >
                  <SkipBack class="w-4 h-4" />
                </button>

                <button
                  on:click={nextSection}
                  class="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-200 rounded-md transition-colors"
                  disabled={$state.context.currentSection >=
                    $state.context.sections.length - 1}
                >
                  <SkipForward class="w-4 h-4" />
                </button>
              </div>
            </div>

            <div class="text-sm text-gray-600">
              Section {$state.context.currentSection + 1} of {$state.context
                .sections.length}
              {#if $state.context.estimatedReadTime > 0}
                • ~{$state.context.estimatedReadTime} min read
              {/if}
            </div>
          </div>

          <!-- Progress Bar -->
          {#if isReading}
            <div class="bg-gray-200 rounded-full h-2" transition:fade>
              <div
                class="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style="width: {progress}%"
              ></div>
            </div>
          {/if}

          <!-- Section Navigation -->
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {#each $state.context.sections as section, index}
              <button
                on:click={() => jumpToSection(index)}
                class="text-left p-3 border rounded-lg transition-all hover:shadow-md"
                class:border-blue-500={index === $state.context.currentSection}
                class:bg-blue-50={index === $state.context.currentSection}
                class:shadow-sm={index === $state.context.currentSection}
                class:border-gray-200={index !== $state.context.currentSection}
              >
                <div class="flex items-center justify-between mb-2">
                  <span
                    class="text-sm font-medium {getImportanceColor(
                      section.importance
                    ).split(' ')[0]}">{section.title}</span
                  >
                  <span
                    class="text-xs px-2 py-1 rounded-full {getImportanceColor(
                      section.importance
                    )}"
                  >
                    {section.importance}
                  </span>
                </div>
                <p class="text-xs text-gray-600 line-clamp-2">
                  {section.content.substring(0, 100)}...
                </p>
                <div class="text-xs text-gray-500 mt-1">
                  {section.wordCount} words
                </div>
              </button>
            {/each}
          </div>

          <!-- Current Section Content -->
          {#if currentSection}
            <div
              class="bg-white border border-gray-200 rounded-lg p-6"
              transition:fly={{ y: 20, duration: 300 }}
            >
              <div class="flex items-center justify-between mb-4">
                <h4 class="text-xl font-semibold text-gray-900">
                  {currentSection.title}
                </h4>
                <span
                  class="text-sm px-3 py-1 rounded-full {getImportanceColor(
                    currentSection.importance
                  )}"
                >
                  {currentSection.importance.charAt(0).toUpperCase() +
                    currentSection.importance.slice(1)} Priority
                </span>
              </div>

              <div class="prose prose-gray max-w-none">
                <p class="text-gray-700 leading-relaxed">
                  {currentSection.content}
                </p>
              </div>

              <!-- Entities -->
              {#if currentSection.entities.length > 0}
                <div class="mt-6 pt-4 border-t border-gray-200">
                  <h5 class="text-sm font-medium text-gray-900 mb-3">
                    Key Entities
                  </h5>
                  <div class="flex flex-wrap gap-2">
                    {#each currentSection.entities as entity}
                      <span
                        class="px-2 py-1 text-xs rounded-md"
                        class:bg-blue-100={entity.type === "legal_term"}
                        class:text-blue-800={entity.type === "legal_term"}
                        class:bg-green-100={entity.type === "person"}
                        class:text-green-800={entity.type === "person"}
                        class:bg-purple-100={entity.type === "date"}
                        class:text-purple-800={entity.type === "date"}
                        class:bg-orange-100={entity.type === "organization"}
                        class:text-orange-800={entity.type === "organization"}
                        class:bg-gray-100={![
                          "legal_term",
                          "person",
                          "date",
                          "organization",
                        ].includes(entity.type)}
                        class:text-gray-800={![
                          "legal_term",
                          "person",
                          "date",
                          "organization",
                        ].includes(entity.type)}
                        title="Confidence: {Math.round(
                          entity.confidence * 100
                        )}%"
                      >
                        {entity.text}
                      </span>
                    {/each}
                  </div>
                </div>
              {/if}
            </div>
          {/if}

          <!-- Analysis Actions -->
          <div class="flex flex-wrap gap-3">
            <button
              on:click={analyzeDocument}
              class="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
              disabled={isLoading}
            >
              <FileText class="w-4 h-4" />
              Analyze Document
            </button>

            <button
              on:click={synthesizeInsights}
              class="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
              disabled={isLoading}
            >
              <Brain class="w-4 h-4" />
              Synthesize Insights
            </button>
          </div>

          <!-- Analysis Results -->
          {#if $state.context.analysisResults.length > 0}
            <div class="space-y-4" transition:fly={{ y: 20, duration: 300 }}>
              <h4 class="text-lg font-semibold text-gray-900">
                Analysis Results
              </h4>
              {#each $state.context.analysisResults as result}
                <div class="border border-gray-200 rounded-lg p-4">
                  <div class="flex items-center justify-between mb-2">
                    <h5 class="font-medium text-gray-900 capitalize">
                      {result.type.replace("_", " ")}
                    </h5>
                    <span
                      class="px-2 py-1 rounded-full text-sm font-medium {getAnalysisScoreColor(
                        result.score
                      )}"
                    >
                      {Math.round(result.score * 100)}%
                    </span>
                  </div>
                  <p class="text-gray-700 mb-3">{result.explanation}</p>
                  {#if result.recommendations.length > 0}
                    <div>
                      <h6 class="text-sm font-medium text-gray-900 mb-1">
                        Recommendations:
                      </h6>
                      <ul
                        class="text-sm text-gray-600 list-disc list-inside space-y-1"
                      >
                        {#each result.recommendations as recommendation}
                          <li>{recommendation}</li>
                        {/each}
                      </ul>
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          {/if}

          <!-- Synthesis Results -->
          {#if $state.context.synthesisData}
            <div class="space-y-6" transition:fly={{ y: 20, duration: 300 }}>
              <h4 class="text-lg font-semibold text-gray-900">
                Synthesis & Strategic Analysis
              </h4>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="space-y-4">
                  <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h5 class="font-medium text-blue-900 mb-3">Main Themes</h5>
                    <ul class="space-y-2">
                      {#each $state.context.synthesisData.mainThemes as theme}
                        <li class="flex items-start gap-2">
                          <div
                            class="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"
                          ></div>
                          <span class="text-blue-800">{theme}</span>
                        </li>
                      {/each}
                    </ul>
                  </div>

                  <div
                    class="bg-green-50 border border-green-200 rounded-lg p-4"
                  >
                    <h5 class="font-medium text-green-900 mb-3">
                      Supporting Evidence
                    </h5>
                    <ul class="space-y-2">
                      {#each $state.context.synthesisData.supportingEvidence as evidence}
                        <li class="flex items-start gap-2">
                          <div
                            class="w-2 h-2 bg-green-600 rounded-full mt-2 flex-shrink-0"
                          ></div>
                          <span class="text-green-800">{evidence}</span>
                        </li>
                      {/each}
                    </ul>
                  </div>
                </div>

                <div class="space-y-4">
                  <div
                    class="bg-yellow-50 border border-yellow-200 rounded-lg p-4"
                  >
                    <h5 class="font-medium text-yellow-900 mb-3">
                      Gaps & Contradictions
                    </h5>
                    <div class="space-y-3">
                      {#if $state.context.synthesisData.gaps.length > 0}
                        <div>
                          <h6 class="text-sm font-medium text-yellow-800">
                            Information Gaps:
                          </h6>
                          <ul class="mt-1 space-y-1">
                            {#each $state.context.synthesisData.gaps as gap}
                              <li class="text-sm text-yellow-700">• {gap}</li>
                            {/each}
                          </ul>
                        </div>
                      {/if}
                      {#if $state.context.synthesisData.contradictions.length > 0}
                        <div>
                          <h6 class="text-sm font-medium text-yellow-800">
                            Contradictions:
                          </h6>
                          <ul class="mt-1 space-y-1">
                            {#each $state.context.synthesisData.contradictions as contradiction}
                              <li class="text-sm text-yellow-700">
                                • {contradiction}
                              </li>
                            {/each}
                          </ul>
                        </div>
                      {/if}
                    </div>
                  </div>

                  <div
                    class="bg-purple-50 border border-purple-200 rounded-lg p-4"
                  >
                    <h5 class="font-medium text-purple-900 mb-3">
                      Legal Implications
                    </h5>
                    <ul class="space-y-2">
                      {#each $state.context.synthesisData.legalImplications as implication}
                        <li class="flex items-start gap-2">
                          <div
                            class="w-2 h-2 bg-purple-600 rounded-full mt-2 flex-shrink-0"
                          ></div>
                          <span class="text-purple-800">{implication}</span>
                        </li>
                      {/each}
                    </ul>
                  </div>
                </div>
              </div>

              <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h5 class="font-medium text-gray-900 mb-3">Next Steps</h5>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {#each $state.context.synthesisData.nextSteps as step, index}
                    <div class="flex items-start gap-3">
                      <span
                        class="flex items-center justify-center w-6 h-6 bg-gray-600 text-white text-sm rounded-full flex-shrink-0"
                      >
                        {index + 1}
                      </span>
                      <span class="text-gray-700">{step}</span>
                    </div>
                  {/each}
                </div>
              </div>
            </div>
          {/if}
        </div>
      {:else}
        <div class="text-center py-12">
          <div
            class="w-16 h-16 mx-auto bg-gray-100 rounded-lg flex items-center justify-center mb-4"
          >
            <FileText class="w-8 h-8 text-gray-400" />
          </div>
          <h4 class="text-lg font-medium text-gray-900 mb-2">
            No Document Loaded
          </h4>
          <p class="text-gray-600 mb-4">
            Load a document or provide content to generate an AI summary.
          </p>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .ai-summary-reader {
    width: 100%;
    max-width: 72rem;
    margin-left: auto
    margin-right: auto
  }

  .ai-summary-reader.compact {
    max-width: 32rem;
  }

  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical
    overflow: hidden
  }

  .prose p {
    margin-bottom: 1rem;
  }

  .prose p:last-child {
    margin-bottom: 0;
  }
</style>
