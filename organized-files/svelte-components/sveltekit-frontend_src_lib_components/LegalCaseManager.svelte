<script lang="ts">
  interface Props {
    caseId: string | null ;
  }
  let {
    caseId = null
  } = $props();



  import { onMount } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';
  import { writable, derived } from 'svelte/store';
  import type { OCRResult } from '$lib/services/ocr-processor';
  import { ocrProcessor } from '$lib/services/ocr-processor';
  import CaseInfoForm from './CaseInfoForm.svelte';
  import DocumentUploadForm from './DocumentUploadForm.svelte';
  import EvidenceAnalysisForm from './EvidenceAnalysisForm.svelte';
  import AIAnalysisForm from './AIAnalysisForm.svelte';
  import ReviewSubmitForm from './ReviewSubmitForm.svelte';
  import ProgressIndicator from './ProgressIndicator.svelte';
  import LoadingSpinner from './LoadingSpinner.svelte';

  // Form state management
   // string | null = null;

  interface FormData {
    caseInfo: {
      title: string
      client_name: string
      case_type: string
      jurisdiction: string
      priority: 'low' | 'medium' | 'high' | 'urgent';
      description: string
      key_dates: Array<{ date: string description: string }>;
    };
    documents: {
      uploaded_files: File[];
      ocr_results: OCRResult[];
      processing_status: 'pending' | 'processing' | 'completed' | 'error';
    };
    evidence: {
      extracted_entities: Array<{ type: string value: string confidence: number }>;
      key_facts: string[];
      legal_issues: string[];
      precedents: Array<{ case_name: string relevance: number summary: string }>;
    };
    ai_analysis: {
      case_strength_score: number
      predicted_outcome: string
      risk_factors: string[];
      recommendations: string[];
      similar_cases: Array<{ id: string title: string similarity: number }>;
    };
    review: {
      final_review: string
      quality_score: number
      completeness_check: boolean
      ready_for_submission: boolean
    };
  }

  // Initialize form data
  const formData = writable<FormData>({
    caseInfo: {
      title: '',
      client_name: '',
      case_type: '',
      jurisdiction: '',
      priority: 'medium',
      description: '',
      key_dates: []
    },
    documents: {
      uploaded_files: [],
      ocr_results: [],
      processing_status: 'pending'
    },
    evidence: {
      extracted_entities: [],
      key_facts: [],
      legal_issues: [],
      precedents: []
    },
    ai_analysis: {
      case_strength_score: 0,
      predicted_outcome: '',
      risk_factors: [],
      recommendations: [],
      similar_cases: []
    },
    review: {
      final_review: '',
      quality_score: 0,
      completeness_check: false,
      ready_for_submission: false
    }
  });

  // Form step management
  const currentStep = writable(1);
  const totalSteps = 5;
  const isLoading = writable(false);
  const processingMessage = writable('');

  // Form validation state
  const stepValidation = derived([formData, currentStep], ([$formData, $currentStep]) => {
    const validations = {
      1: $formData.caseInfo.title && $formData.caseInfo.client_name && $formData.caseInfo.case_type,
      2: $formData.documents.uploaded_files.length > 0 && $formData.documents.processing_status === 'completed',
      3: $formData.evidence.key_facts.length > 0 && $formData.evidence.legal_issues.length > 0,
      4: $formData.ai_analysis.case_strength_score > 0,
      5: $formData.review.completeness_check && $formData.review.ready_for_submission
    };
    return validations[$currentStep as keyof typeof validations] || false;
  });

  // Auto-save functionality
  let autoSaveTimeout: NodeJS.Timeout;

  $effect(() => { if ($formData) {
    clearTimeout(autoSaveTimeout);
    autoSaveTimeout = setTimeout(() => {
      saveFormData();
    }, 2000);
  }

  async function saveFormData() {
    try {
      localStorage.setItem(`legal-case-form-${caseId || 'new'}`, JSON.stringify($formData));
      console.log('✅ Form data auto-saved');
    } catch (error) {
      console.error('❌ Failed to auto-save form data:', error);
    }
  }

  async function loadFormData() {
    try {
      const saved = localStorage.getItem(`legal-case-form-${caseId || 'new'}`);
      if (saved) {
        const parsedData = JSON.parse(saved);
        formData.set(parsedData);
        console.log('✅ Form data loaded from local storage');
      }
    } catch (error) {
      console.error('❌ Failed to load form data:', error);
    }
  }

  // Document processing
  async function processDocuments(files: File[]) {
    isLoading.set(true);
    processingMessage.set('Processing uploaded documents...');

    try {
      formData.update(data => ({
        ...data,
        documents: {
          ...data.documents,
          uploaded_files: files,
          processing_status: 'processing'
        }
      }));

      const ocrResults: OCRResult[] = [];

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        processingMessage.set(`Processing document ${i + 1}/${files.length}: ${file.name}`);

        // Save file temporarily for processing
        const formData_upload = new FormData();
        formData_upload.append('file', file);

        const uploadResponse = await fetch('/api/upload-temp', {
          method: 'POST',
          body: formData_upload
        });

        if (!uploadResponse.ok) {
          throw new Error(`Failed to upload ${file.name}`);
        }

        const { filePath } = await uploadResponse.json();

        // Process with OCR
        const ocrResult = await ocrProcessor.processDocument(filePath);
        ocrResults.push(ocrResult);

        // Clean up temp file
        await fetch('/api/cleanup-temp', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filePath })
        });
      }

      formData.update(data => ({
        ...data,
        documents: {
          ...data.documents,
          ocr_results: ocrResults,
          processing_status: 'completed'
        }
      }));

      console.log('✅ Document processing completed');

      // Auto-advance to next step
      setTimeout(() => {
        nextStep();
      }, 1000);

    } catch (error) {
      console.error('❌ Document processing failed:', error);
      formData.update(data => ({
        ...data,
        documents: {
          ...data.documents,
          processing_status: 'error'
        }
      }));
    } finally {
      isLoading.set(false);
      processingMessage.set('');
    }
  }

  // Evidence extraction
  async function extractEvidence() {
    isLoading.set(true);
    processingMessage.set('Extracting legal entities and evidence...');

    try {
      const response = await fetch('/api/evidence/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ocr_results: $formData.documents.ocr_results,
          case_context: $formData.caseInfo
        })
      });

      if (!response.ok) {
        throw new Error('Evidence extraction failed');
      }

      const evidenceData = await response.json();

      formData.update(data => ({
        ...data,
        evidence: evidenceData
      }));

      console.log('✅ Evidence extraction completed');

    } catch (error) {
      console.error('❌ Evidence extraction failed:', error);
    } finally {
      isLoading.set(false);
      processingMessage.set('');
    }
  }

  // AI analysis
  async function performAIAnalysis() {
    isLoading.set(true);
    processingMessage.set('Performing AI case analysis...');

    try {
      const response = await fetch('/api/ai/analyze-case', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_info: $formData.caseInfo,
          evidence: $formData.evidence,
          documents: $formData.documents.ocr_results
        })
      });

      if (!response.ok) {
        throw new Error('AI analysis failed');
      }

      const analysisData = await response.json();

      formData.update(data => ({
        ...data,
        ai_analysis: analysisData
      }));

      console.log('✅ AI analysis completed');

    } catch (error) {
      console.error('❌ AI analysis failed:', error);
    } finally {
      isLoading.set(false);
      processingMessage.set('');
    }
  }

  // Navigation functions
  function nextStep() {
    if ($currentStep < totalSteps) {
      currentStep.update(step => step + 1);
    }
  }

  function prevStep() {
    if ($currentStep > 1) {
      currentStep.update(step => step - 1);
    }
  }

  function goToStep(step: number) {
    if (step >= 1 && step <= totalSteps) {
      currentStep.set(step);
    }
  }

  // Form submission
  async function submitForm() {
    isLoading.set(true);
    processingMessage.set('Submitting case for review...');

    try {
      const response = await fetch('/api/cases/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify($formData)
      });

      if (!response.ok) {
        throw new Error('Case submission failed');
      }

      const result = await response.json();

      // Clear form data
      localStorage.removeItem(`legal-case-form-${caseId || 'new'}`);

      // Redirect to case details
      window.location.href = `/cases/${result.case_id}`;

    } catch (error) {
      console.error('❌ Case submission failed:', error);
    } finally {
      isLoading.set(false);
      processingMessage.set('');
    }
  }

  onMount(() => {
    loadFormData();
  });
</script>

<div class="legal-case-manager">
  <!-- Progress Header -->
  <div class="progress-header">
    <h1 class="text-3xl font-bold text-gray-900 mb-4">
      {caseId ? 'Edit Case' : 'Create New Legal Case'}
    </h1>

    <ProgressIndicator
      currentStep={$currentStep}
      totalSteps={totalSteps}
      stepTitles={[
        'Case Information',
        'Document Upload',
        'Evidence Analysis',
        'AI Analysis',
        'Review & Submit'
      ]}
      on:stepClick={(e) => goToStep(e.detail)}
    />
  </div>

  <!-- Loading Overlay -->
  {#if $isLoading}
    <div class="loading-overlay" transitionfade={{ duration: 300 }}>
      <LoadingSpinner />
      <p class="loading-message">{$processingMessage}</p>
    </div>
  {/if}

  <!-- Form Steps -->
  <div class="form-container" class:loading={$isLoading}>
    {#if $currentStep === 1}
      <div transitionslide={{ duration: 300, easing: cubicOut }}>
        <CaseInfoForm
          bind:data={$formData.caseInfo}
          onnext={nextStep}
          isValid={$stepValidation}
        />
      </div>
    {:else if $currentStep === 2}
      <div transitionslide={{ duration: 300, easing: cubicOut }}>
        <DocumentUploadForm
          bind:data={$formData.documents}
          onprocess={(e) => processDocuments(e.detail)}
          onnext={nextStep}
          onprev={prevStep}
          isValid={$stepValidation}
        />
      </div>
    {:else if $currentStep === 3}
      <div transitionslide={{ duration: 300, easing: cubicOut }}>
        <EvidenceAnalysisForm
          bind:data={$formData.evidence}
          ocrResults={$formData.documents.ocr_results}
          onextract={extractEvidence}
          onnext={nextStep}
          onprev={prevStep}
          isValid={$stepValidation}
        />
      </div>
    {:else if $currentStep === 4}
      <div transitionslide={{ duration: 300, easing: cubicOut }}>
        <AIAnalysisForm
          bind:data={$formData.ai_analysis}
          caseData={$formData}
          onanalyze={performAIAnalysis}
          onnext={nextStep}
          onprev={prevStep}
          isValid={$stepValidation}
        />
      </div>
    {:else if $currentStep === 5}
      <div transitionslide={{ duration: 300, easing: cubicOut }}>
        <ReviewSubmitForm
          bind:data={$formData.review}
          fullCaseData={$formData}
          onsubmit={submitForm}
          onprev={prevStep}
          isValid={$stepValidation}
        />
      </div>
    {/if}
  </div>

  <!-- Debug Panel (Development only) -->
  {#if import.meta.env.DEV}
    <div class="debug-panel">
      <details>
        <summary>Debug Info</summary>
        <pre>{JSON.stringify($formData, null, 2)}</pre>
      </details>
    </div>
  {/if}
</div>

<style>
  .legal-case-manager {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    position: relative
  }

  .progress-header {
    margin-bottom: 3rem;
  }

  .loading-overlay {
    position: fixed
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex
    flex-direction: column
    align-items: center
    justify-content: center
    z-index: 1000;
  }

  .loading-message {
    color: white
    font-size: 1.1rem;
    margin-top: 1rem;
    text-align: center
  }

  .form-container {
    transition: opacity 0.3s ease;
  }

  .form-container.loading {
    opacity: 0.3;
    pointer-events: none
  }

  .debug-panel {
    margin-top: 2rem;
    padding: 1rem;
    background: #f5f5f5;
    border-radius: 8px;
    font-size: 0.8rem;
  }

  .debug-panel pre {
    max-height: 300px;
    overflow: auto
    background: white
    padding: 1rem;
    border-radius: 4px;
  }

  @media (max-width: 768px) {
    .legal-case-manager {
      padding: 1rem;
    }

    .progress-header h1 {
      font-size: 1.8rem;
    }
  }
</style>
