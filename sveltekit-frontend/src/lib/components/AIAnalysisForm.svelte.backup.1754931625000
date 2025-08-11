<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Button } from 'bits-ui';
  import { fade, slide } from 'svelte/transition';
  import { writable } from 'svelte/store';

  const dispatch = createEventDispatcher();

  export let formData: {
    case_strength_score: number;
    predicted_outcome: string;
    risk_factors: string[];
    recommendations: string[];
    similar_cases: Array<{ id: string; title: string; similarity: number }>;
  };

  export let evidenceData: {
    extracted_entities: Array<{ type: string; value: string; confidence: number }>;
    key_facts: string[];
    legal_issues: string[];
    precedents: Array<{ case_name: string; relevance: number; summary: string }>;
  };

  let isAnalyzing = false;
  let analysisProgress = writable(0);
  let currentAnalysisStep = writable('');
  let analysisResults = writable<any>(null);

  // Outcome options
  const possibleOutcomes = [
    'Favorable Settlement',
    'Favorable Court Decision',
    'Unfavorable Settlement',
    'Unfavorable Court Decision',
    'Dismissal',
    'Partial Victory',
    'Need More Evidence',
    'Mediation Recommended'
  ];

  async function runAIAnalysis() {
    if (evidenceData.key_facts.length === 0) {
      alert('No evidence data available for analysis. Please complete the evidence analysis first.');
      return;
    }

    isAnalyzing = true;
    analysisProgress.set(0);

    try {
      // Step 1: Case Strength Analysis
      currentAnalysisStep.set('Analyzing case strength...');
      await new Promise(resolve => setTimeout(resolve, 1500));

      const caseStrength = await analyzeCaseStrength();
      formData.case_strength_score = caseStrength;
      analysisProgress.set(20);

      // Step 2: Outcome Prediction
      currentAnalysisStep.set('Predicting case outcome...');
      await new Promise(resolve => setTimeout(resolve, 1200));

      const outcome = await predictOutcome();
      formData.predicted_outcome = outcome;
      analysisProgress.set(40);

      // Step 3: Risk Assessment
      currentAnalysisStep.set('Identifying risk factors...');
      await new Promise(resolve => setTimeout(resolve, 1000));

      const risks = await identifyRiskFactors();
      formData.risk_factors = risks;
      analysisProgress.set(60);

      // Step 4: Strategic Recommendations
      currentAnalysisStep.set('Generating recommendations...');
      await new Promise(resolve => setTimeout(resolve, 1300));

      const recommendations = await generateRecommendations();
      formData.recommendations = recommendations;
      analysisProgress.set(80);

      // Step 5: Similar Cases
      currentAnalysisStep.set('Finding similar cases...');
      await new Promise(resolve => setTimeout(resolve, 1000));

      const similarCases = await findSimilarCases();
      formData.similar_cases = similarCases;
      analysisProgress.set(100);

      currentAnalysisStep.set('AI analysis complete!');
      analysisResults.set(formData);

    } catch (error) {
      console.error('AI analysis failed:', error);
      alert('AI analysis failed. Please try again.');
    } finally {
      isAnalyzing = false;
    }
  }

  async function analyzeCaseStrength(): Promise<number> {
    // Mock case strength analysis based on evidence quality
    let strength = 50; // Base score

    // Factor in entity confidence
    const avgEntityConfidence = evidenceData.extracted_entities.length > 0
      ? evidenceData.extracted_entities.reduce((acc, e) => acc + e.confidence, 0) / evidenceData.extracted_entities.length
      : 0;
    strength += avgEntityConfidence * 20;

    // Factor in key facts quantity and quality
    strength += Math.min(evidenceData.key_facts.length * 5, 20);

    // Factor in legal issues identified
    strength += Math.min(evidenceData.legal_issues.length * 3, 15);

    // Factor in precedents found
    strength += Math.min(evidenceData.precedents.length * 5, 15);

    return Math.max(0, Math.min(100, strength));
  }

  async function predictOutcome(): Promise<string> {
    // Mock outcome prediction based on case strength
    const strength = formData.case_strength_score;

    if (strength >= 80) return 'Favorable Court Decision';
    if (strength >= 65) return 'Favorable Settlement';
    if (strength >= 50) return 'Partial Victory';
    if (strength >= 35) return 'Mediation Recommended';
    if (strength >= 20) return 'Unfavorable Settlement';
    return 'Dismissal';
  }

  async function identifyRiskFactors(): Promise<string[]> {
    const risks: string[] = [];

    // Analyze based on evidence quality
    if (formData.case_strength_score < 50) {
      risks.push('Weak evidence foundation');
    }

    if (evidenceData.key_facts.length < 3) {
      risks.push('Insufficient key facts documented');
    }

    if (evidenceData.extracted_entities.some(e => e.confidence < 0.7)) {
      risks.push('Low confidence in extracted information');
    }

    if (evidenceData.legal_issues.includes('Constitutional Rights')) {
      risks.push('Complex constitutional issues involved');
    }

    if (evidenceData.precedents.length === 0) {
      risks.push('No supporting precedents identified');
    }

    // Mock additional risks
    const commonRisks = [
      'Statute of limitations concerns',
      'Opposing party has strong legal representation',
      'Key witnesses may be unavailable',
      'Document authenticity challenges',
      'Jurisdictional complications'
    ];

    // Add 1-2 random common risks
    const additionalRisks = commonRisks.sort(() => 0.5 - Math.random()).slice(0, 2);
    risks.push(...additionalRisks);

    return risks;
  }

  async function generateRecommendations(): Promise<string[]> {
    const recommendations: string[] = [];

    // Strategic recommendations based on case strength
    if (formData.case_strength_score >= 70) {
      recommendations.push('Proceed with confidence to trial');
      recommendations.push('Consider demanding higher settlement amounts');
    } else if (formData.case_strength_score >= 40) {
      recommendations.push('Explore settlement opportunities');
      recommendations.push('Gather additional supporting evidence');
    } else {
      recommendations.push('Consider alternative dispute resolution');
      recommendations.push('Reassess case viability');
    }

    // Evidence-based recommendations
    if (evidenceData.key_facts.length < 5) {
      recommendations.push('Conduct additional fact-finding investigations');
    }

    if (evidenceData.precedents.length > 0) {
      recommendations.push('Leverage identified precedents in legal arguments');
    }

    // Mock additional strategic recommendations
    const strategicOptions = [
      'Request expert witness testimony',
      'File motion for summary judgment',
      'Pursue expedited discovery',
      'Consider class action status',
      'Negotiate fee arrangements with client'
    ];

    const additionalRecs = strategicOptions.sort(() => 0.5 - Math.random()).slice(0, 2);
    recommendations.push(...additionalRecs);

    return recommendations;
  }

  async function findSimilarCases(): Promise<Array<{ id: string; title: string; similarity: number }>> {
    // Mock similar cases based on legal issues
    const mockCases = [
      {
        id: 'case_001',
        title: 'Johnson v. ABC Corp - Contract Dispute',
        similarity: 0.87
      },
      {
        id: 'case_002',
        title: 'Smith & Associates vs. Downtown Properties',
        similarity: 0.74
      },
      {
        id: 'case_003',
        title: 'Estate of Williams - Property Rights',
        similarity: 0.68
      },
      {
        id: 'case_004',
        title: 'Metro LLC v. City Planning Commission',
        similarity: 0.61
      }
    ];

    return mockCases;
  }

  function getStrengthColor(score: number): string {
    if (score >= 70) return 'text-green-600 bg-green-100';
    if (score >= 50) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  }

  function getOutcomeColor(outcome: string): string {
    if (outcome.includes('Favorable')) return 'text-green-700 bg-green-50 border-green-200';
    if (outcome.includes('Partial') || outcome.includes('Settlement')) return 'text-yellow-700 bg-yellow-50 border-yellow-200';
    return 'text-red-700 bg-red-50 border-red-200';
  }

  function addCustomRecommendation() {
    formData.recommendations = [...formData.recommendations, ''];
  }

  function removeRecommendation(index: number) {
    formData.recommendations = formData.recommendations.filter((_, i) => i !== index);
  }

  function handleNext() {
    if (formData.case_strength_score === 0) {
      alert('Please run AI analysis before proceeding.');
      return;
    }

    dispatch('next', { step: 'ai_analysis', data: formData });
  }

  function handlePrevious() {
    dispatch('previous', { step: 'ai_analysis' });
  }

  function handleSaveDraft() {
    dispatch('saveDraft', { step: 'ai_analysis', data: formData });
  }
</script>

<div class="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg" transition:fade>
  <div class="mb-8">
    <h2 class="text-2xl font-bold text-gray-900 mb-2">AI Analysis</h2>
    <p class="text-gray-600">Generate AI-powered insights, predictions, and strategic recommendations</p>
  </div>

  <!-- AI Analysis Trigger -->
  {#if !isAnalyzing && formData.case_strength_score === 0}
    <div class="mb-8 bg-purple-50 border border-purple-200 rounded-lg p-6">
      <div class="text-center">
        <div class="text-4xl mb-4">🤖</div>
        <h3 class="text-xl font-medium text-purple-900 mb-2">AI-Powered Legal Analysis</h3>
        <p class="text-purple-700 mb-4">
          Advanced machine learning algorithms will analyze your case evidence and provide:
        </p>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div class="text-sm text-purple-600">
            • Case strength assessment
            • Outcome prediction
          </div>
          <div class="text-sm text-purple-600">
            • Risk factor identification
            • Strategic recommendations
          </div>
        </div>
        <Button.Root
          onclick={runAIAnalysis}
          class="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 text-lg font-medium"
        >
          🚀 Start AI Analysis
        </Button.Root>
      </div>
    </div>
  {/if}

  <!-- Analysis Progress -->
  {#if isAnalyzing}
    <div class="mb-8 bg-purple-50 border border-purple-200 rounded-lg p-6" transition:slide>
      <div class="space-y-4">
        <div class="flex items-center justify-between">
          <h3 class="text-lg font-medium text-purple-900">Running AI Analysis...</h3>
          <span class="text-sm text-purple-700">{$analysisProgress}%</span>
        </div>

        <div class="bg-purple-200 rounded-full h-3">
          <div
            class="bg-purple-600 h-3 rounded-full transition-all duration-500"
            style="width: {$analysisProgress}%"
          ></div>
        </div>

        <div class="flex items-center space-x-3">
          <div class="animate-spin rounded-full h-5 w-5 border-b-2 border-purple-600"></div>
          <p class="text-sm text-purple-700">{$currentAnalysisStep}</p>
        </div>
      </div>
    </div>
  {/if}

  <!-- Case Strength Score -->
  {#if formData.case_strength_score > 0}
    <div class="mb-8" transition:slide>
      <h3 class="text-lg font-medium text-gray-900 mb-4">Case Strength Assessment</h3>

      <div class="bg-gray-50 border border-gray-200 rounded-lg p-6">
        <div class="flex items-center justify-between mb-4">
          <span class="text-sm font-medium text-gray-700">Overall Case Strength</span>
          <span class="text-2xl font-bold {getStrengthColor(formData.case_strength_score)} px-3 py-1 rounded-lg">
            {formData.case_strength_score}/100
          </span>
        </div>

        <div class="bg-gray-200 rounded-full h-4 mb-2">
          <div
            class="h-4 rounded-full transition-all duration-1000 {formData.case_strength_score >= 70 ? 'bg-green-500' : formData.case_strength_score >= 50 ? 'bg-yellow-500' : 'bg-red-500'}"
            style="width: {formData.case_strength_score}%"
          ></div>
        </div>

        <p class="text-sm text-gray-600">
          {#if formData.case_strength_score >= 70}
            Strong case with good prospects for success
          {:else if formData.case_strength_score >= 50}
            Moderate case strength, consider strategic options
          {:else}
            Weak case, significant challenges ahead
          {/if}
        </p>
      </div>
    </div>
  {/if}

  <!-- Predicted Outcome -->
  {#if formData.predicted_outcome}
    <div class="mb-8" transition:slide>
      <h3 class="text-lg font-medium text-gray-900 mb-4">Predicted Outcome</h3>

      <div class="border rounded-lg p-4 {getOutcomeColor(formData.predicted_outcome)}">
        <div class="flex items-center">
          <span class="text-2xl mr-3">🎯</span>
          <div>
            <h4 class="font-medium text-lg">{formData.predicted_outcome}</h4>
            <p class="text-sm opacity-75">
              Based on case strength, evidence quality, and historical data analysis
            </p>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Risk Factors -->
  {#if formData.risk_factors.length > 0}
    <div class="mb-8" transition:slide>
      <h3 class="text-lg font-medium text-gray-900 mb-4">Identified Risk Factors</h3>

      <div class="space-y-2">
        {#each formData.risk_factors as risk}
          <div class="flex items-center p-3 bg-red-50 border border-red-200 rounded-lg" transition:fade>
            <span class="text-red-600 mr-3">⚠️</span>
            <span class="text-sm text-red-800">{risk}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Strategic Recommendations -->
  {#if formData.recommendations.length > 0}
    <div class="mb-8" transition:slide>
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-medium text-gray-900">Strategic Recommendations</h3>
        <Button.Root
          onclick={addCustomRecommendation}
          class="px-3 py-1 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500"
        >
          + Add Custom
        </Button.Root>
      </div>

      <div class="space-y-3">
        {#each formData.recommendations as recommendation, index}
          <div class="flex gap-3" transition:fade>
            {#if recommendation.trim() === ''}
              <div class="flex-1">
                <textarea
                  bind:value={formData.recommendations[index]}
                  rows="2"
                  placeholder="Enter your custom recommendation..."
                  class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                ></textarea>
              </div>
              <Button.Root
                onclick={() => removeRecommendation(index)}
                class="px-3 py-2 text-red-600 hover:text-red-800 focus:outline-none"
              >
                Remove
              </Button.Root>
            {:else}
              <div class="flex items-center p-3 bg-green-50 border border-green-200 rounded-lg flex-1">
                <span class="text-green-600 mr-3">💡</span>
                <span class="text-sm text-green-800">{recommendation}</span>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Similar Cases -->
  {#if formData.similar_cases.length > 0}
    <div class="mb-8" transition:slide>
      <h3 class="text-lg font-medium text-gray-900 mb-4">Similar Cases</h3>

      <div class="space-y-3">
        {#each formData.similar_cases as similarCase}
          <div class="bg-blue-50 border border-blue-200 rounded-lg p-4" transition:fade>
            <div class="flex items-center justify-between">
              <div class="flex-1">
                <h4 class="text-sm font-medium text-blue-900">{similarCase.title}</h4>
                <p class="text-xs text-blue-700 mt-1">Case ID: {similarCase.id}</p>
              </div>
              <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                {Math.round(similarCase.similarity * 100)}% similar
              </span>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Form Actions -->
  <div class="flex justify-between pt-6 border-t border-gray-200">
    <Button.Root
      onclick={handlePrevious}
      class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      ← Previous
    </Button.Root>

    <div class="flex space-x-3">
      <Button.Root
        onclick={handleSaveDraft}
        class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        Save Draft
      </Button.Root>

      <Button.Root
        onclick={handleNext}
        disabled={formData.case_strength_score === 0}
        class="px-6 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Next: Review & Submit →
      </Button.Root>
    </div>
  </div>
</div>
