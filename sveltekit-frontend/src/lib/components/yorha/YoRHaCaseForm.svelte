<!-- Enhanced YoRHa Case Creation Form with Superforms + XState Integration -->
<script lang="ts">
  import { createEventDispatcher, onMount, onDestroy } from 'svelte';
  import { createCaseCreationForm, FormStatePersistence, FORM_STORAGE_KEYS } from '$lib/forms/superforms-xstate-integration';
  import { enhancedCaseAPI } from '$lib/api/enhanced-case-api';
  import { goto } from '$app/navigation';
  import { z } from 'zod';
  
  const dispatch = createEventDispatcher<{
    success: { case: any };
    error: { message: string };
    close: void;
    stateChange: { state: string; context: any };
  }>();

  // Enhanced Zod schema for case creation with legal AI context
  const CaseCreationSchema = z.object({
    title: z.string().min(3, 'Title must be at least 3 characters'),
    description: z.string().min(10, 'Description must be at least 10 characters'),
    priority: z.enum(['low', 'medium', 'high', 'critical']).default('medium'),
    status: z.enum(['open', 'investigating', 'pending', 'closed', 'archived']).default('open'),
    location: z.string().optional(),
    jurisdiction: z.string().optional(),
    caseType: z.enum(['civil', 'criminal', 'corporate', 'family', 'intellectual_property']).default('civil'),
    assignedTo: z.string().optional(),
    clientName: z.string().optional(),
    tags: z.array(z.string()).default([]),
    notes: z.string().optional()
  });

  // Initialize form state persistence
  const formStatePersistence = new FormStatePersistence(FORM_STORAGE_KEYS.CASE_CREATION);
  
  // Form integration state
  let formIntegration: any = $state(null);
  let currentStep = $state(0);
let totalSteps = $state(3); // Basic Info, Legal Details, Review
let unsubscribe = $state<(() >(> void) | null = null);
  
  // Reactive derived state from form integration
  let isSubmitting = $derived(formIntegration?.isSubmitting?.get() || false);
  let isValid = $derived(formIntegration?.isValid?.get() || false);
  let progress = $derived(formIntegration?.progress?.get() || 0);
  let errors = $derived(formIntegration?.errors?.get() || {});
  let formState = $derived(formIntegration?.state?.get() || 'idle');
  let formContext = $derived(formIntegration?.context?.get() || {});

  // Initialize form integration on mount
  onMount(async () => {
    // Load saved form data if available
    const savedData = formStatePersistence.load();
    const initialData = savedData || {
      title: '',
      description: '',
      priority: 'medium',
      status: 'open',
      location: '',
      jurisdiction: '',
      caseType: 'civil',
      assignedTo: '',
      clientName: '',
      tags: [],
      notes: ''
    };

    // Create form integration with enhanced options
    formIntegration = createCaseCreationForm(initialData, {
      autoSave: true,
      autoSaveDelay: 2000,
      resetOnSuccess: false,
      onSuccess: handleFormSuccess,
      onError: handleFormError,
      onSubmit: handleEnhancedSubmit
    });

    // Subscribe to state changes for debugging and events
    unsubscribe = formIntegration.state.subscribe((state: string) => {
      dispatch('stateChange', { 
        state, 
        context: formIntegration.context.get() 
      });
      console.log('📊 Case Form State:', state, formIntegration.context.get());
    });
  });

  onDestroy(() => {
    if (unsubscribe) {
      unsubscribe();
    }
    // Save form state on unmount
    if (formIntegration) {
      const formData = formIntegration.form.get();
      formStatePersistence.save(formData);
    }
  });

  // Enhanced form submission with PostgreSQL-first worker integration
  async function handleEnhancedSubmit(formData: any) {
    console.log('🚀 Enhanced case creation starting:', formData);
    
    try {
      // Use the enhanced case API for full workflow integration
      const caseResponse = await enhancedCaseAPI.createCase({
        ...formData,
        metadata: {
          createdVia: 'yorha-command-center',
          formVersion: '2.0',
          workflowStep: 'case-creation',
          timestamp: new Date().toISOString()
        }
      });

      if (!caseResponse.success) {
        throw new Error(caseResponse.error || 'Failed to create case');
      }

      const createdCase = caseResponse.data;
      console.log('✅ Case created successfully with enhanced API:', createdCase);

      return createdCase;

    } catch (error) {
      console.error('❌ Enhanced case creation failed:', error);
      throw error;
    }
  }

  // Success handler
  function handleFormSuccess(result: any) {
    console.log('🎉 Form submission successful:', result);
    dispatch('success', { case: result });
    
    // Clear saved draft
    formStatePersistence.clear();
    
    // Navigate to the new case
    if (result.id) {
      goto(`/cases/${result.id}`);
    }
  }

  // Error handler
  function handleFormError(error: any) {
    console.error('❌ Form submission error:', error);
    dispatch('error', { message: error.message || error || 'Case creation failed' });
  }

  // Step navigation
  function nextStep() {
    if (currentStep < totalSteps - 1) {
      currentStep += 1;
    }
  }

  function previousStep() {
    if (currentStep > 0) {
      currentStep -= 1;
    }
  }

  // Get step progress percentage
  function getStepProgress() {
    return ((currentStep + 1) / totalSteps) * 100;
  }
</script>

<!-- Enhanced Multi-Step YoRHa Styled Form -->
{#if formIntegration}
<div class="yorha-case-form bg-yorha-dark border border-yorha-accent-warm/30 rounded-lg p-6">
  <!-- Enhanced Form Header with Progress -->
  <div class="form-header mb-6">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-xl font-bold text-yorha-accent-warm">
        📁 CREATE NEW CASE FILE
      </h2>
      <div class="form-state-indicator">
        <span class="text-xs text-yorha-muted uppercase tracking-wide">
          State: {formState}
        </span>
        {#if progress > 0}
          <div class="w-16 h-1 bg-yorha-darker rounded-full mt-1 overflow-hidden">
            <div 
              class="h-full bg-yorha-accent-warm transition-all duration-300" 
              style="width: {progress}%"
            ></div>
          </div>
        {/if}
      </div>
    </div>
    
    <!-- Multi-step Progress Indicator -->
    <div class="step-indicator flex items-center justify-between mb-4">
      <div class="step-progress flex items-center space-x-2">
        {#each Array(totalSteps) as _, index}
          <div class="flex items-center">
            <div 
              class="step-circle w-8 h-8 rounded-full border-2 flex items-center justify-center text-xs font-bold transition-colors"
              class:active={index === currentStep}
              class:completed={index < currentStep}
              class:pending={index > currentStep}
            >
              {index + 1}
            </div>
            {#if index < totalSteps - 1}
              <div class="step-line w-12 h-0.5 bg-yorha-accent-warm/30 mx-2"></div>
            {/if}
          </div>
        {/each}
      </div>
      <div class="step-label text-sm text-yorha-muted">
        Step {currentStep + 1} of {totalSteps}: 
        {#if currentStep === 0}Basic Information{:else if currentStep === 1}Legal Details{:else}Review & Submit{/if}
      </div>
    </div>
    
    <p class="text-yorha-muted text-sm">
      Initialize new investigation case in the YoRHa Legal AI System
    </p>
  </div>

  <!-- Enhanced Form with XState Integration -->
  <form use:formIntegration.form.enhance class="space-y-6">
    
    {#if currentStep === 0}
      <!-- Step 1: Basic Information -->
      <div class="form-step" data-step="basic-info">
        <!-- Case Title -->
        <div class="form-group">
          <label for="case-title" class="form-label block text-sm font-bold text-yorha-light mb-2">
            CASE TITLE *
          </label>
          <input
            id="case-title"
            name="title"
            type="text"
            value={formIntegration.form.get().title || ''}
            input={(e) => formIntegration.form.update(data => ({ ...data, title: e.target.value }))}
            placeholder="e.g., Corporate Fraud Investigation - TechCorp"
            required
            class="form-input w-full p-3 bg-yorha-darker border border-yorha-accent-warm/30 rounded text-yorha-light placeholder-yorha-muted focus:border-yorha-accent-warm focus:outline-none transition-colors"
            class:border-red-500={errors.title}
          />
          {#if errors.title}
            <p class="text-red-400 text-xs mt-1">{errors.title.join(', ')}</p>
          {/if}
        </div>

        <!-- Case Description -->
        <div class="form-group">
          <label for="case-description" class="form-label block text-sm font-bold text-yorha-light mb-2">
            CASE DESCRIPTION / SYNOPSIS *
          </label>
          <textarea
            id="case-description"
            name="description"
            value={formIntegration.form.get().description || ''}
            input={(e) => formIntegration.form.update(data => ({ ...data, description: e.target.value }))}
            rows="4"
            placeholder="Initial details and background of the investigation..."
            class="form-input w-full p-3 bg-yorha-darker border border-yorha-accent-warm/30 rounded text-yorha-light placeholder-yorha-muted focus:border-yorha-accent-warm focus:outline-none transition-colors resize-none"
            class:border-red-500={errors.description}
          ></textarea>
          {#if errors.description}
            <p class="text-red-400 text-xs mt-1">{errors.description.join(', ')}</p>
          {/if}
        </div>

        <!-- Priority and Status Row -->
        <div class="form-row grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Priority Level -->
          <div class="form-group">
            <label for="case-priority" class="form-label block text-sm font-bold text-yorha-light mb-2">
              PRIORITY LEVEL
            </label>
            <select
              id="case-priority"
              name="priority"
              value={formIntegration.form.get().priority || 'medium'}
              change={(e) => formIntegration.form.update(data => ({ ...data, priority: e.target.value }))}
              class="form-input w-full p-3 bg-yorha-darker border border-yorha-accent-warm/30 rounded text-yorha-light focus:border-yorha-accent-warm focus:outline-none transition-colors"
            >
              <option value="low">🟢 Low Priority</option>
              <option value="medium">🟡 Medium Priority</option>
              <option value="high">🟠 High Priority</option>
              <option value="critical">🔴 Critical Priority</option>
            </select>
          </div>

          <!-- Case Type -->
          <div class="form-group">
            <label for="case-type" class="form-label block text-sm font-bold text-yorha-light mb-2">
              CASE TYPE
            </label>
            <select
              id="case-type"
              name="caseType"
              value={formIntegration.form.get().caseType || 'civil'}
              change={(e) => formIntegration.form.update(data => ({ ...data, caseType: e.target.value }))}
              class="form-input w-full p-3 bg-yorha-darker border border-yorha-accent-warm/30 rounded text-yorha-light focus:border-yorha-accent-warm focus:outline-none transition-colors"
            >
              <option value="civil">⚖️ Civil</option>
              <option value="criminal">🔒 Criminal</option>
              <option value="corporate">🏢 Corporate</option>
              <option value="family">👨‍👩‍👧‍👦 Family</option>
              <option value="intellectual_property">🧠 Intellectual Property</option>
            </select>
          </div>
        </div>
      </div>
      
      <!-- Step Navigation -->
      <div class="step-navigation flex justify-end pt-4">
        <button
          type="button"
          on:onclick={nextStep}
          disabled={!formIntegration.form.get().title || !formIntegration.form.get().description}
          class="next-btn px-6 py-3 bg-yorha-accent-warm text-yorha-dark rounded font-bold hover:bg-yorha-accent-warm/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Next: Legal Details →
        </button>
      </div>
    
    {:else if currentStep === 1}
      <!-- Step 2: Legal Details -->
      <div class="form-step" data-step="legal-details">

        <!-- Location and Jurisdiction Row -->
        <div class="form-row grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Location -->
          <div class="form-group">
            <label for="case-location" class="form-label block text-sm font-bold text-yorha-light mb-2">
              LOCATION
            </label>
            <input
              id="case-location"
              name="location"
              type="text"
              value={formIntegration.form.get().location || ''}
              input={(e) => formIntegration.form.update(data => ({ ...data, location: e.target.value }))}
              placeholder="e.g., Downtown Financial District"
              class="form-input w-full p-3 bg-yorha-darker border border-yorha-accent-warm/30 rounded text-yorha-light placeholder-yorha-muted focus:border-yorha-accent-warm focus:outline-none transition-colors"
            />
          </div>

          <!-- Jurisdiction -->
          <div class="form-group">
            <label for="case-jurisdiction" class="form-label block text-sm font-bold text-yorha-light mb-2">
              JURISDICTION
            </label>
            <input
              id="case-jurisdiction"
              name="jurisdiction"
              type="text"
              value={formIntegration.form.get().jurisdiction || ''}
              input={(e) => formIntegration.form.update(data => ({ ...data, jurisdiction: e.target.value }))}
              placeholder="e.g., Federal, State, Local"
              class="form-input w-full p-3 bg-yorha-darker border border-yorha-accent-warm/30 rounded text-yorha-light placeholder-yorha-muted focus:border-yorha-accent-warm focus:outline-none transition-colors"
            />
          </div>
        </div>
        
        <!-- Assignment and Client Row -->
        <div class="form-row grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Assigned To -->
          <div class="form-group">
            <label for="assigned-to" class="form-label block text-sm font-bold text-yorha-light mb-2">
              ASSIGNED TO
            </label>
            <input
              id="assigned-to"
              name="assignedTo"
              type="text"
              value={formIntegration.form.get().assignedTo || ''}
              input={(e) => formIntegration.form.update(data => ({ ...data, assignedTo: e.target.value }))}
              placeholder="e.g., Agent Smith, Detective Jones"
              class="form-input w-full p-3 bg-yorha-darker border border-yorha-accent-warm/30 rounded text-yorha-light placeholder-yorha-muted focus:border-yorha-accent-warm focus:outline-none transition-colors"
            />
          </div>

          <!-- Client Name -->
          <div class="form-group">
            <label for="client-name" class="form-label block text-sm font-bold text-yorha-light mb-2">
              CLIENT NAME
            </label>
            <input
              id="client-name"
              name="clientName"
              type="text"
              value={formIntegration.form.get().clientName || ''}
              input={(e) => formIntegration.form.update(data => ({ ...data, clientName: e.target.value }))}
              placeholder="e.g., TechCorp Industries"
              class="form-input w-full p-3 bg-yorha-darker border border-yorha-accent-warm/30 rounded text-yorha-light placeholder-yorha-muted focus:border-yorha-accent-warm focus:outline-none transition-colors"
            />
          </div>
        </div>
        
        <!-- Case Notes -->
        <div class="form-group">
          <label for="case-notes" class="form-label block text-sm font-bold text-yorha-light mb-2">
            INITIAL NOTES
          </label>
          <textarea
            id="case-notes"
            name="notes"
            value={formIntegration.form.get().notes || ''}
            input={(e) => formIntegration.form.update(data => ({ ...data, notes: e.target.value }))}
            rows="3"
            placeholder="Additional case notes, preliminary observations, or special instructions..."
            class="form-input w-full p-3 bg-yorha-darker border border-yorha-accent-warm/30 rounded text-yorha-light placeholder-yorha-muted focus:border-yorha-accent-warm focus:outline-none transition-colors resize-none"
          ></textarea>
        </div>
      </div>
      
      <!-- Step Navigation -->
      <div class="step-navigation flex justify-between pt-4">
        <button
          type="button"
          on:onclick={previousStep}
          class="prev-btn px-6 py-3 border border-yorha-accent-warm/50 text-yorha-light rounded hover:bg-yorha-accent-warm/10 transition-colors"
        >
          ← Back: Basic Info
        </button>
        <button
          type="button"
          on:onclick={nextStep}
          class="next-btn px-6 py-3 bg-yorha-accent-warm text-yorha-dark rounded font-bold hover:bg-yorha-accent-warm/90 transition-colors"
        >
          Next: Review →
        </button>
      </div>
    
    {:else if currentStep === 2}
      <!-- Step 3: Review & Submit -->
      <div class="form-step" data-step="review">
        <div class="review-section space-y-4">
          <h3 class="text-lg font-bold text-yorha-accent-warm mb-4">📋 Review Case Details</h3>
          
          <!-- Case Summary -->
          <div class="review-item p-4 bg-yorha-darker rounded border border-yorha-accent-warm/20">
            <h4 class="font-bold text-yorha-light mb-2">Case Title</h4>
            <p class="text-yorha-muted">{formIntegration.form.get().title || 'Not specified'}</p>
          </div>
          
          <div class="review-item p-4 bg-yorha-darker rounded border border-yorha-accent-warm/20">
            <h4 class="font-bold text-yorha-light mb-2">Description</h4>
            <p class="text-yorha-muted text-sm">{formIntegration.form.get().description || 'Not specified'}</p>
          </div>
          
          <div class="review-grid grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="review-item p-4 bg-yorha-darker rounded border border-yorha-accent-warm/20">
              <h4 class="font-bold text-yorha-light mb-2">Priority</h4>
              <p class="text-yorha-muted capitalize">{formIntegration.form.get().priority || 'medium'}</p>
            </div>
            
            <div class="review-item p-4 bg-yorha-darker rounded border border-yorha-accent-warm/20">
              <h4 class="font-bold text-yorha-light mb-2">Case Type</h4>
              <p class="text-yorha-muted capitalize">{formIntegration.form.get().caseType || 'civil'}</p>
            </div>
          </div>
          
          <!-- AI Processing Indicator -->
          {#if formState === 'submitting' || formState === 'validating'}
            <div class="ai-processing-indicator p-4 bg-yorha-accent-warm/10 rounded border border-yorha-accent-warm/50">
              <div class="flex items-center space-x-3">
                <div class="spinner w-5 h-5 border-2 border-yorha-accent-warm/30 border-t-yorha-accent-warm rounded-full animate-spin"></div>
                <div>
                  <h4 class="font-bold text-yorha-accent-warm">🤖 AI Processing Active</h4>
                  <p class="text-yorha-muted text-sm">
                    {#if formState === 'validating'}
                      Validating case data and checking for duplicates...
                    {:else if formState === 'submitting'}
                      Creating case and triggering PostgreSQL-first worker...
                    {/if}
                  </p>
                  {#if progress > 0}
                    <div class="progress-bar w-full h-1 bg-yorha-darker rounded-full mt-2 overflow-hidden">
                      <div 
                        class="h-full bg-yorha-accent-warm transition-all duration-300" 
                        style="width: {progress}%"
                      ></div>
                    </div>
                  {/if}
                </div>
              </div>
            </div>
          {/if}
        </div>
      </div>
      
      <!-- Final Step Navigation -->
      <div class="step-navigation flex justify-between pt-4">
        <button
          type="button"
          on:onclick={previousStep}
          disabled={isSubmitting}
          class="prev-btn px-6 py-3 border border-yorha-accent-warm/50 text-yorha-light rounded hover:bg-yorha-accent-warm/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          ← Back: Legal Details
        </button>
        <div class="final-actions flex space-x-4">
          <button
            type="button"
            on:onclick={() => dispatch('close')}
            disabled={isSubmitting}
            class="cancel-btn px-6 py-3 border border-yorha-accent-warm/50 text-yorha-light rounded hover:bg-yorha-accent-warm/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={isSubmitting || !isValid}
            class="submit-btn px-6 py-3 bg-yorha-accent-warm text-yorha-dark rounded font-bold hover:bg-yorha-accent-warm/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {#if isSubmitting}
              <div class="spinner w-4 h-4 border-2 border-yorha-dark/30 border-t-yorha-dark rounded-full animate-spin"></div>
              <span>Creating Case...</span>
            {:else}
              <span>💾 CREATE CASE</span>
            {/if}
          </button>
        </div>
      </div>
    {/if}

    
  </form>
  
  <!-- Debug Panel (development only) -->
  {#if process.env.NODE_ENV === 'development'}
    <div class="debug-panel mt-6 p-4 bg-yorha-darker/50 border border-yorha-accent-warm/20 rounded text-xs">
      <details>
        <summary class="cursor-pointer text-yorha-muted hover:text-yorha-light">🐛 Debug Info</summary>
        <div class="mt-2 space-y-1 text-yorha-muted">
          <p><strong>Form State:</strong> {formState}</p>
          <p><strong>Current Step:</strong> {currentStep + 1}/{totalSteps}</p>
          <p><strong>Is Valid:</strong> {isValid}</p>
          <p><strong>Is Submitting:</strong> {isSubmitting}</p>
          <p><strong>Progress:</strong> {progress}%</p>
          <p><strong>Errors:</strong> {Object.keys(errors).length > 0 ? JSON.stringify(errors, null, 2) : 'None'}</p>
        </div>
      </details>
    </div>
  {/if}
</div>
{:else}
  <!-- Loading State -->
  <div class="yorha-case-form bg-yorha-dark border border-yorha-accent-warm/30 rounded-lg p-6">
    <div class="flex items-center justify-center space-x-3">
      <div class="spinner w-6 h-6 border-2 border-yorha-accent-warm/30 border-t-yorha-accent-warm rounded-full animate-spin"></div>
      <span class="text-yorha-light">Initializing form...</span>
    </div>
  </div>
{/if}

<style>
  .yorha-case-form {
    --yorha-primary: #c4b49a;
    --yorha-secondary: #b5a48a;
    --yorha-accent-warm: #d4af37;
    --yorha-accent-cool: #6b6b6b;
    --yorha-light: #ffffff;
    --yorha-muted: #a0a0a0;
    --yorha-dark: #2a2a2a;
    --yorha-darker: #1a1a1a;

    font-family: 'JetBrains Mono', monospace;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
  }

  /* Multi-step progress indicators */
  .step-circle {
    transition: all 0.3s ease;
  }
  
  .step-circle.active {
    background-color: var(--yorha-accent-warm);
    color: var(--yorha-dark);
    border-color: var(--yorha-accent-warm);
    box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.3);
  }
  
  .step-circle.completed {
    background-color: rgba(212, 175, 55, 0.8);
    color: var(--yorha-dark);
    border-color: var(--yorha-accent-warm);
  }
  
  .step-circle.pending {
    background-color: transparent;
    color: var(--yorha-muted);
    border-color: rgba(212, 175, 55, 0.3);
  }
  
  .step-line {
    background: linear-gradient(90deg, var(--yorha-accent-warm) 0%, rgba(212, 175, 55, 0.3) 100%);
    transition: all 0.3s ease;
  }

  /* Form animations */
  .form-step {
    animation: fadeInUp 0.4s ease-out;
  }
  
  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  /* Enhanced form inputs */
  .form-input {
    transition: all 0.2s ease;
    position: relative;
  }
  
  .form-input:focus {
    box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.2);
    transform: translateY(-1px);
  }
  
  .form-input:hover:not(:focus) {
    border-color: rgba(212, 175, 55, 0.5);
  }

  /* Button enhancements */
  .next-btn, .prev-btn, .submit-btn {
    position: relative;
    transition: all 0.2s ease;
    overflow: hidden;
  }
  
  .next-btn:hover, .submit-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(212, 175, 55, 0.3);
  }
  
  .prev-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(212, 175, 55, 0.1);
  }
  
  .next-btn::before, .submit-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transition: left 0.5s;
  }
  
  .next-btn:hover::before, .submit-btn:hover::before {
    left: 100%;
  }

  /* Progress animations */
  .progress-bar, .step-progress {
    animation: slideIn 0.3s ease-out;
  }
  
  @keyframes slideIn {
    from {
      width: 0;
      opacity: 0;
    }
    to {
      width: 100%;
      opacity: 1;
    }
  }

  /* AI Processing indicator */
  .ai-processing-indicator {
    animation: pulseGlow 2s infinite;
  }
  
  @keyframes pulseGlow {
    0%, 100% {
      box-shadow: 0 0 5px rgba(212, 175, 55, 0.3);
    }
    50% {
      box-shadow: 0 0 20px rgba(212, 175, 55, 0.6);
    }
  }

  /* Review section styling */
  .review-section {
    animation: fadeIn 0.5s ease-out;
  }
  
  .review-item {
    transition: all 0.2s ease;
  }
  
  .review-item:hover {
    background-color: rgba(212, 175, 55, 0.05);
    transform: translateY(-1px);
  }

  /* Loading spinner */
  .spinner {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  /* Debug panel */
  .debug-panel {
    font-family: 'Courier New', monospace;
    font-size: 10px;
  }
  
  .debug-panel details[open] {
    background-color: rgba(212, 175, 55, 0.05);
  }

  /* State indicators */
  .form-state-indicator {
    animation: slideInRight 0.3s ease-out;
  }
  
  @keyframes slideInRight {
    from {
      transform: translateX(20px);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .form-row, .review-grid {
      grid-template-columns: 1fr;
    }
    
    .step-navigation, .form-actions, .final-actions {
      flex-direction: column;
      gap: 1rem;
    }
    
    .step-indicator {
      flex-direction: column;
      align-items: flex-start;
      gap: 1rem;
    }
    
    .step-progress {
      width: 100%;
      justify-content: space-between;
    }
    
    .step-line {
      width: 100%;
      height: 2px;
      transform: rotate(90deg);
      margin: 0.5rem 0;
    }
  }
  
  @media (max-width: 480px) {
    .yorha-case-form {
      padding: 1rem;
    }
    
    .step-circle {
      width: 2rem;
      height: 2rem;
    }
  }
</style>