<script lang="ts">
  import { onMount } from 'svelte';
  import { useMachine } from '@xstate/svelte';
  
  import { 
    legalFormMachine, 
    getStateDescription, 
    getAISuggestions, 
    calculateProgressPercentage,
    getNextPossibleActions,
    type LegalFormContext 
  } from '$lib/state/legalFormMachine';
  import { MatrixUICompiler, type MatrixUINode } from '$lib/ui/matrix-compiler';
  import { LegalAIReranker, enhancedSearch, type UserContext } from '$lib/ai/custom-reranker';
  import { PredictivePrefetcher } from '$lib/workers/predictive-prefetch';

  interface Props {
    className?: string;
  }

  let { className = '' } = $props();

  // XState machine integration
  const { state, send, context } = useMachine(legalFormMachine);
  
  // Phase 8 system components
  let matrixCompiler: MatrixUICompiler
  let reranker: LegalAIReranker
  let prefetcher: PredictivePrefetcher
  
  // Melt UI Accordion for multi-step form
  const accordion = Accordion({ 
    multiple: false,
    value: $derived(getAccordionValue($state.value))
  });
  
  // Reactive state calculations
  let currentStateDescription = $derived(getStateDescription($state.value));
  let aiSuggestions = $derived(getAISuggestions($context, $state.value));
  let progressPercentage = $derived(calculateProgressPercentage($context));
  let possibleActions = $derived(getNextPossibleActions($state.value));
  let aiConfidence = $derived($context.confidence)
  
  // Form data
  let fileInput: HTMLInputElement
  let caseTitle = $state('');
  let caseDescription = $state('');
  let selectedPriority = $state<'low' | 'medium' | 'high' | 'critical'>('medium');
  let selectedEvidenceType = $state<'digital' | 'physical' | 'testimony' | 'forensic'>('digital');

  // AI-aware UI state
  let aiRecommendations = $derived($context.aiRecommendations)
  let showAIPanel = $state(false);
  let matrixUINodes: MatrixUINode[] = $state([]);

  function getAccordionValue(stateValue: any): string {
    const stateMapping = {
      evidenceUpload: 'step-1',
      caseDetails: 'step-2', 
      review: 'step-3',
      submitting: 'step-4',
      success: 'step-4',
      error: 'step-4'
    };
    return stateMapping[stateValue as keyof typeof stateMapping] || 'step-1';
  }

  onMount(async () => {
    // Initialize Phase 8 components
    matrixCompiler = new MatrixUICompiler();
    reranker = new LegalAIReranker();
    prefetcher = new PredictivePrefetcher();
    await prefetcher.initialize();
    
    // Generate initial matrix UI nodes
    updateMatrixUINodes();
    
    // Set up AI-aware prefetching
    setupPredictivePrefetching();
  });

  function updateMatrixUINodes(): void {
    const currentState = $state.value as string;
    
    matrixUINodes = [
      {
        type: 'card',
        id: `state-card-${currentState}`,
        matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        styles: {
          base: `yorha-card p-6 ${getStateCardClass(currentState)}`,
          hover: 'transform scale-105 transition-transform',
          active: 'ring-2 ring-yellow-400'
        },
        events: ['click', 'mouseover'],
        metadata: {
          priority: 'high',
          confidence: aiConfidence,
          aiGenerated: true,
          workflowState: currentState
        }
      },
      {
        type: 'button',
        id: 'ai-help-btn',
        matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 300, 50, 0, 1],
        styles: {
          base: 'yorha-button px-4 py-2 bg-blue-600 text-white',
          hover: 'bg-blue-700 transform scale-105',
          disabled: 'opacity-50 cursor-not-allowed'
        },
        events: ['click'],
        metadata: {
          priority: 'medium',
          confidence: 90,
          aiGenerated: false
        }
      }
    ];
  }

  function getStateCardClass(state: string): string {
    const classes = {
      evidenceUpload: 'border-blue-400 bg-blue-900/20',
      caseDetails: 'border-yellow-400 bg-yellow-900/20',
      review: 'border-purple-400 bg-purple-900/20',
      submitting: 'border-orange-400 bg-orange-900/20',
      success: 'border-green-400 bg-green-900/20',
      error: 'border-red-400 bg-red-900/20'
    };
    return classes[state as keyof typeof classes] || 'border-gray-400 bg-gray-900/20';
  }

  async function setupPredictivePrefetching(): Promise<void> {
    const userContext: UserContext = {
      intent: 'create',
      timeOfDay: getTimeOfDay(),
      currentCase: 'NEW_CASE',
      recentActions: ['open_form', 'start_evidence_upload'],
      userRole: 'prosecutor',
      workflowState: 'draft'
    };

    // Predict and prefetch based on current state
    const intentPrediction = await prefetcher.predictIntent({
      currentPage: '/cases/new',
      focusedElement: `step-${$context.currentStep}`,
      recentActions: userContext.recentActions,
      caseId: userContext.currentCase
    });

    if (intentPrediction) {
      await prefetcher.executePrefetch(intentPrediction);
    }
  }

  function getTimeOfDay(): 'morning' | 'afternoon' | 'evening' | 'night' {
    const hour = new Date().getHours();
    if (hour < 12) return 'morning';
    if (hour < 17) return 'afternoon';
    if (hour < 21) return 'evening';
    return 'night';
  }

  // Event handlers with AI awareness
  function handleFileUpload(event: Event): void {
    const target = event.target as HTMLInputElement;
    const files = Array.from(target.files || []);
    
    send({ type: 'UPLOAD_EVIDENCE', files });
    
    // Update matrix UI based on file types
    updateMatrixUINodes();
    
    // Trigger AI reranking for file suggestions
    performAIReranking('file_upload', files.map(f => f.name));
  }

  function handleCaseDetailsUpdate(): void {
    send({ 
      type: 'UPDATE_CASE_DETAILS', 
      title: caseTitle, 
      description: caseDescription 
    });
    
    // Update matrix UI
    updateMatrixUINodes();
    
    // Trigger AI reranking for case suggestions
    performAIReranking('case_details', [caseTitle, caseDescription]);
  }

  async function performAIReranking(action: string, context: string[]): Promise<void> {
    try {
      const userContext: UserContext = {
        intent: 'create',
        timeOfDay: getTimeOfDay(),
        focusedElement: `step-${$context.currentStep}`,
        currentCase: 'NEW_CASE',
        recentActions: ['form_interaction', action],
        userRole: 'prosecutor',
        workflowState: 'draft'
      };

      // Use enhanced search with custom reranker
      const query = context.join(' ');
      const results = await enhancedSearch(query, userContext, 5);
      
      // Update AI suggestions based on reranked results
      const suggestions = results.map(r => r.content).slice(0, 3);
      send({ type: 'AI_SUGGESTION', suggestions });
      
    } catch (error) {
      console.warn('AI reranking failed:', error);
    }
  }

  function handleNextStep(): void {
    send({ type: 'NEXT' });
    updateMatrixUINodes();
    setupPredictivePrefetching();
  }

  function handleBackStep(): void {
    send({ type: 'BACK' });
    updateMatrixUINodes();
  }

  function handleSubmit(): void {
    send({ type: 'SUBMIT' });
    updateMatrixUINodes();
  }

  function requestAIHelp(): void {
    send({ type: 'REQUEST_AI_HELP' });
    showAIPanel = true;
  }

  function applyAIRecommendation(recommendation: string): void {
    send({ type: 'APPLY_AI_RECOMMENDATION', recommendation });
    
    // Apply the recommendation based on its content
    if (recommendation.includes('priority to HIGH')) {
      selectedPriority = 'high';
      send({ type: 'SET_PRIORITY', priority: 'high' });
    }
    
    updateMatrixUINodes();
  }

  // Watch for state changes and update UI accordingly
  $effect(() => {
    updateMatrixUINodes();
    
    // Update form fields from context
    caseTitle = $context.caseTitle;
    caseDescription = $context.caseDescription;
    selectedPriority = $context.priority;
    selectedEvidenceType = $context.evidenceType;
  });
</script>

<div class="xstate-phase8-integration {className}">
  <!-- Progress Header -->
  <div class="progress-header yorha-panel p-6 mb-6">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-2xl font-bold text-yellow-400">Legal Case Form</h2>
      <button 
        class="yorha-button px-3 py-1 text-sm bg-blue-600 text-white"
        onclick={requestAIHelp}
      >
        AI Assistant
      </button>
    </div>
    
    <!-- Progress Bar -->
    <div class="progress-bar bg-gray-700 rounded-full h-2 mb-2">
      <div 
        class="progress-fill bg-yellow-400 h-2 rounded-full transition-all duration-500"
        style="width: {progressPercentage}%"
      ></div>
    </div>
    
    <div class="flex justify-between text-sm">
      <span class="text-gray-300">
        State: {currentStateDescription}
      </span>
      <span class="text-gray-300">
        AI Confidence: {aiConfidence}%
      </span>
      <span class="text-gray-300">
        Step {$context.currentStep} of {$context.totalSteps}
      </span>
    </div>
  </div>

  <!-- Multi-Step Form with Accordion -->
  <div class="form-content grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Main Form -->
    <div class="lg:col-span-2">
      <div use:accordion.root class="accordion-root space-y-4">
        
        <!-- Step 1: Evidence Upload -->
        <div use:accordion.item value="step-1" class="accordion-item">
          <h3 use:accordion.header class="accordion-header">
            <button 
              use:accordion.trigger
              class="accordion-trigger yorha-button w-full text-left p-4 data-[state=open]:bg-blue-900/30"
            >
              <span class="flex items-center gap-3">
                <span class="step-number">1</span>
                Evidence Upload
                {#if $context.evidenceFiles.length > 0}
                  <span class="text-green-400 text-sm">({$context.evidenceFiles.length} files)</span>
                {/if}
              </span>
            </button>
          </h3>
          
          <div use:accordion.content class="accordion-content p-4 border-l-4 border-blue-400">
            <div class="space-y-4">
              <div class="file-upload-zone border-2 border-dashed border-gray-600 rounded-lg p-8 text-center">
                <input 
                  bind:this={fileInput}
                  type="file" 
                  multiple 
                  accept=".pdf,.jpg,.jpeg,.png,.doc,.docx"
                  onchange={handleFileUpload}
                  class="hidden"
                />
                <button 
                  onclick={() => fileInput?.click()}
                  class="yorha-button px-6 py-3 bg-blue-600 text-white"
                >
                  Select Evidence Files
                </button>
                <p class="text-gray-400 text-sm mt-2">
                  Supported: PDF, Images, Documents
                </p>
              </div>
              
              <div class="evidence-type-selector">
                <label class="block text-sm font-medium text-gray-300 mb-2">
                  Evidence Type
                </label>
                <select 
                  bind:value={selectedEvidenceType}
                  onchange={() => send({ type: 'SET_EVIDENCE_TYPE', evidenceType: selectedEvidenceType })}
                  class="yorha-select w-full p-2 bg-gray-800 border border-gray-600 rounded"
                >
                  <option value="digital">Digital Evidence</option>
                  <option value="physical">Physical Evidence</option>
                  <option value="testimony">Witness Testimony</option>
                  <option value="forensic">Forensic Analysis</option>
                </select>
              </div>
              
              {#if $state.matches('evidenceUpload')}
                <button 
                  onclick={handleNextStep}
                  disabled={$context.evidenceFiles.length === 0}
                  class="yorha-button px-6 py-2 bg-yellow-400 text-black disabled:opacity-50"
                >
                  Next: Case Details
                </button>
              {/if}
            </div>
          </div>
        </div>

        <!-- Step 2: Case Details -->
        <div use:accordion.item value="step-2" class="accordion-item">
          <h3 use:accordion.header class="accordion-header">
            <button 
              use:accordion.trigger
              class="accordion-trigger yorha-button w-full text-left p-4 data-[state=open]:bg-yellow-900/30"
            >
              <span class="flex items-center gap-3">
                <span class="step-number">2</span>
                Case Details
                {#if $context.caseTitle}
                  <span class="text-green-400 text-sm">✓</span>
                {/if}
              </span>
            </button>
          </h3>
          
          <div use:accordion.content class="accordion-content p-4 border-l-4 border-yellow-400">
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium text-gray-300 mb-2">
                  Case Title
                </label>
                <input 
                  bind:value={caseTitle}
                  onblur={handleCaseDetailsUpdate}
                  type="text" 
                  placeholder="Enter case title..."
                  class="yorha-input w-full p-3 bg-gray-800 border border-gray-600 rounded"
                />
                {#if $context.validationErrors.caseTitle}
                  <p class="text-red-400 text-sm mt-1">{$context.validationErrors.caseTitle}</p>
                {/if}
              </div>
              
              <div>
                <label class="block text-sm font-medium text-gray-300 mb-2">
                  Case Description
                </label>
                <textarea 
                  bind:value={caseDescription}
                  onblur={handleCaseDetailsUpdate}
                  rows="4"
                  placeholder="Detailed case description..."
                  class="yorha-input w-full p-3 bg-gray-800 border border-gray-600 rounded"
                ></textarea>
                {#if $context.validationErrors.caseDescription}
                  <p class="text-red-400 text-sm mt-1">{$context.validationErrors.caseDescription}</p>
                {/if}
              </div>
              
              <div>
                <label class="block text-sm font-medium text-gray-300 mb-2">
                  Priority Level
                </label>
                <select 
                  bind:value={selectedPriority}
                  onchange={() => send({ type: 'SET_PRIORITY', priority: selectedPriority })}
                  class="yorha-select w-full p-2 bg-gray-800 border border-gray-600 rounded"
                >
                  <option value="low">Low Priority</option>
                  <option value="medium">Medium Priority</option>
                  <option value="high">High Priority</option>
                  <option value="critical">Critical Priority</option>
                </select>
              </div>
              
              {#if $state.matches('caseDetails')}
                <div class="flex gap-3">
                  <button 
                    onclick={handleBackStep}
                    class="yorha-button px-4 py-2 bg-gray-600 text-white"
                  >
                    Back
                  </button>
                  <button 
                    onclick={handleNextStep}
                    disabled={!caseTitle.trim() || !caseDescription.trim()}
                    class="yorha-button px-6 py-2 bg-yellow-400 text-black disabled:opacity-50"
                  >
                    Next: Review
                  </button>
                </div>
              {/if}
            </div>
          </div>
        </div>

        <!-- Step 3: Review -->
        <div use:accordion.item value="step-3" class="accordion-item">
          <h3 use:accordion.header class="accordion-header">
            <button 
              use:accordion.trigger
              class="accordion-trigger yorha-button w-full text-left p-4 data-[state=open]:bg-purple-900/30"
            >
              <span class="flex items-center gap-3">
                <span class="step-number">3</span>
                Review & Submit
              </span>
            </button>
          </h3>
          
          <div use:accordion.content class="accordion-content p-4 border-l-4 border-purple-400">
            <div class="space-y-4">
              <div class="review-summary yorha-panel p-4">
                <h4 class="text-lg font-semibold text-purple-400 mb-3">Case Summary</h4>
                <div class="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span class="text-gray-400">Title:</span>
                    <span class="text-white ml-2">{$context.caseTitle}</span>
                  </div>
                  <div>
                    <span class="text-gray-400">Priority:</span>
                    <span class="text-white ml-2 capitalize">{$context.priority}</span>
                  </div>
                  <div>
                    <span class="text-gray-400">Evidence Type:</span>
                    <span class="text-white ml-2 capitalize">{$context.evidenceType}</span>
                  </div>
                  <div>
                    <span class="text-gray-400">Files:</span>
                    <span class="text-white ml-2">{$context.evidenceFiles.length} uploaded</span>
                  </div>
                </div>
                <div class="mt-3">
                  <span class="text-gray-400">Description:</span>
                  <p class="text-white mt-1">{$context.caseDescription}</p>
                </div>
              </div>
              
              {#if $state.matches('review')}
                <div class="flex gap-3">
                  <button 
                    onclick={handleBackStep}
                    class="yorha-button px-4 py-2 bg-gray-600 text-white"
                  >
                    Back
                  </button>
                  <button 
                    onclick={handleSubmit}
                    class="yorha-button px-6 py-2 bg-green-600 text-white"
                  >
                    Submit Case
                  </button>
                </div>
              {/if}
              
              {#if $state.matches('submitting')}
                <div class="submitting-state text-center p-6">
                  <div class="loading-spinner mx-auto mb-4"></div>
                  <p class="text-yellow-400">Submitting case...</p>
                </div>
              {/if}
              
              {#if $state.matches('success')}
                <div class="success-state text-center p-6">
                  <div class="text-green-400 text-4xl mb-4">✓</div>
                  <p class="text-green-400 text-lg">Case submitted successfully!</p>
                  <button 
                    onclick={() => send({ type: 'RESET_FORM' })}
                    class="yorha-button px-6 py-2 bg-blue-600 text-white mt-4"
                  >
                    Create New Case
                  </button>
                </div>
              {/if}
              
              {#if $state.matches('error')}
                <div class="error-state text-center p-6">
                  <div class="text-red-400 text-4xl mb-4">✗</div>
                  <p class="text-red-400 text-lg">Submission failed</p>
                  <p class="text-gray-400 text-sm">{$context.validationErrors.submit}</p>
                  <button 
                    onclick={handleBackStep}
                    class="yorha-button px-6 py-2 bg-yellow-400 text-black mt-4"
                  >
                    Try Again
                  </button>
                </div>
              {/if}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- AI Panel -->
    <div class="ai-panel">
      <div class="yorha-panel p-4">
        <h3 class="text-lg font-semibold text-yellow-400 mb-4">AI Assistant</h3>
        
        <!-- AI Suggestions -->
        <div class="ai-suggestions mb-4">
          <h4 class="text-sm font-medium text-gray-300 mb-2">Suggestions</h4>
          <div class="space-y-2">
            {#each aiSuggestions as suggestion}
              <div class="suggestion-item text-sm text-blue-400 bg-blue-900/20 p-2 rounded">
                {suggestion}
              </div>
            {/each}
          </div>
        </div>
        
        <!-- AI Recommendations -->
        {#if aiRecommendations.length > 0}
          <div class="ai-recommendations mb-4">
            <h4 class="text-sm font-medium text-gray-300 mb-2">Recommendations</h4>
            <div class="space-y-2">
              {#each aiRecommendations as rec}
                <div class="recommendation-item bg-yellow-900/20 p-3 rounded">
                  <div class="flex justify-between items-start mb-2">
                    <span class="text-yellow-400 text-sm font-medium">{rec.nextAction}</span>
                    <span class="text-gray-400 text-xs">{rec.confidence}%</span>
                  </div>
                  <p class="text-gray-300 text-xs mb-2">{rec.reasoning}</p>
                  <button 
                    onclick={() => applyAIRecommendation(rec.nextAction)}
                    class="yorha-button px-3 py-1 text-xs bg-yellow-400 text-black"
                  >
                    Apply
                  </button>
                </div>
              {/each}
            </div>
          </div>
        {/if}
        
        <!-- Possible Actions -->
        <div class="possible-actions">
          <h4 class="text-sm font-medium text-gray-300 mb-2">Available Actions</h4>
          <div class="flex flex-wrap gap-1">
            {#each possibleActions as action}
              <span class="action-tag text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded">
                {action}
              </span>
            {/each}
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  /* @unocss-include */
  .xstate-phase8-integration {
    @apply max-w-7xl mx-auto p-6;
  }

  .step-number {
    @apply w-6 h-6 bg-yellow-400 text-black rounded-full flex items-center justify-center text-sm font-bold;
  }

  .progress-fill {
    transition: width 0.5s ease-in-out;
  }

  .loading-spinner {
    @apply w-8 h-8 border-2 border-yellow-400 border-t-transparent rounded-full animate-spin;
  }

  .accordion-trigger[data-state="open"] {
    @apply bg-opacity-30;
  }

  .accordion-content {
    animation: slideDown 0.3s ease-out;
  }

  @keyframes slideDown {
    from {
      height: 0;
      opacity: 0;
    }
    to {
      height: auto
      opacity: 1;
    }
  }

  .suggestion-item {
    border-left: 3px solid rgb(59 130 246);
  }

  .recommendation-item {
    border-left: 3px solid rgb(251 191 36);
  }

  .action-tag {
    font-family: 'Courier New', monospace;
  }
</style>