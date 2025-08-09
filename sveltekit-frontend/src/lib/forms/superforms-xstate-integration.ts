// @ts-nocheck
// Superforms + XState Integration for Legal AI Forms
// Advanced form management with state machines and validation

import { superForm, type SuperValidated, type Infer } from 'sveltekit-superforms';
import { zod } from 'sveltekit-superforms/adapters';
import { writable, derived, type Writable, type Readable } from 'svelte/store';
import { createActor, type ActorRefFrom } from 'xstate';
import {
  DocumentUploadSchema,
  CaseCreationSchema,
  SearchQuerySchema,
  AIAnalysisSchema,
  documentUploadMachine,
  caseCreationMachine,
  searchMachine,
  aiAnalysisMachine,
  type DocumentUploadActor,
  type CaseCreationActor,
  type SearchActor,
  type AIAnalysisActor
} from '$lib/state/legal-form-machines';
import type { z } from 'zod';

// ============================================================================
// FORM STATE INTEGRATION TYPES
// ============================================================================

export interface FormMachineIntegration<T extends Record<string, unknown>, M> {
  form: ReturnType<typeof superForm<T>>;
  actor: M;
  state: Writable<any>;
  context: Writable<any>;
  isValid: Readable<boolean>;
  isSubmitting: Readable<boolean>;
  errors: Readable<Record<string, string[]>>;
  progress: Readable<number>;
}

export interface FormOptions {
  onSubmit?: (data: any) => void | Promise<void>;
  onSuccess?: (result: any) => void;
  onError?: (error: string) => void;
  autoSave?: boolean;
  autoSaveDelay?: number;
  resetOnSuccess?: boolean;
}

// ============================================================================
// DOCUMENT UPLOAD FORM INTEGRATION
// ============================================================================

export function createDocumentUploadForm(
  data: SuperValidated<Infer<typeof DocumentUploadSchema>>,
  options: FormOptions = {}
): FormMachineIntegration<Infer<typeof DocumentUploadSchema>, DocumentUploadActor> {
  
  // Create XState actor
  const actor = createActor(documentUploadMachine);
  actor.start();

  // Create Superform
  const form = superForm(data, {
    validators: zod(DocumentUploadSchema),
    resetForm: options.resetOnSuccess ?? true,
    delayMs: 300,
    timeoutMs: 8000,
    invalidateAll: false,
    onUpdated: ({ form }) => {
      if ((form as any).valid) {
        actor.send({
          type: 'VALIDATE_FORM',
          data: (form as any).data || form
        });
      }
    },
    onSubmit: async ({ formData, cancel }) => {
      if (options.onSubmit) {
        cancel();
        await options.onSubmit(formData);
      } else {
        actor.send({ type: 'UPLOAD' });
      }
    },
    onResult: ({ result }) => {
      if (result.type === 'success' && options.onSuccess) {
        options.onSuccess(result.data);
      } else if (result.type === 'error' && options.onError) {
        options.onError(result.error.message);
      }
    }
  });

  // Reactive state stores
  const state = writable(actor.getSnapshot().value);
  const context = writable(actor.getSnapshot().context);
  const isValid = derived([form.form], ([$form]) => !!($form as any).valid);
  const isSubmitting = derived(state, ($state) => 
    $state === 'uploading' || $state === 'processing' || $state === 'validating'
  );
  const errors = derived([form.errors, context], ([$errors, $context]) => {
    // Flatten the complex superforms error structure to match interface
    const flattened: Record<string, string[]> = {};
    
    // Handle superforms errors (which can be nested objects)
    const flattenErrors = (obj: any, prefix = ''): void: any => {
      for (const [key, value] of Object.entries(obj || {})) {
        const fullKey = prefix ? `${prefix}.${key}` : key;
        if (Array.isArray(value)) {
          flattened[fullKey] = value as string[];
        } else if (typeof value === 'object' && value !== null) {
          flattenErrors(value, fullKey);
        }
      }
    };
    
    flattenErrors($errors);
    
    // Add context validation errors
    if ($context.validationErrors) {
      Object.assign(flattened, $context.validationErrors);
    }
    
    return flattened;
  });
  const progress = derived(context, ($context) => 
    Math.max($context.uploadProgress, $context.processingProgress)
  );

  // Subscribe to actor changes
  actor.subscribe((snapshot) => {
    state.set(snapshot.value);
    context.set(snapshot.context);
    
    // Handle state-specific actions
    if (snapshot.value === 'completed' && options.onSuccess) {
      options.onSuccess(snapshot.context.aiResults);
    } else if (snapshot.value === 'failed' && options.onError) {
      options.onError(snapshot.context.error || 'Upload failed');
    }
  });

  // Auto-save functionality
  if (options.autoSave) {
    const autoSaveDelay = options.autoSaveDelay ?? 2000;
    let autoSaveTimeout: NodeJS.Timeout;

    form.form.subscribe(($form) => {
      if (($form as any).valid) {
        clearTimeout(autoSaveTimeout);
        autoSaveTimeout = setTimeout(() => {
          localStorage.setItem('document-upload-draft', JSON.stringify(($form as any).data || $form));
        }, autoSaveDelay);
      }
    });
  }

  return {
    form,
    actor,
    state,
    context,
    isValid,
    isSubmitting,
    errors,
    progress
  };
}

// ============================================================================
// CASE CREATION FORM INTEGRATION
// ============================================================================

export function createCaseCreationForm(
  data: SuperValidated<Infer<typeof CaseCreationSchema>>,
  options: FormOptions = {}
): FormMachineIntegration<Infer<typeof CaseCreationSchema>, CaseCreationActor> {
  
  const actor = createActor(caseCreationMachine);
  actor.start();

  const form = superForm(data, {
    validators: zod(CaseCreationSchema),
    resetForm: options.resetOnSuccess ?? false, // Don't reset case creation forms
    delayMs: 500, // Longer delay for case creation
    timeoutMs: 10000,
    invalidateAll: true,
    onUpdated: ({ form }) => {
      actor.send({
        type: 'UPDATE_FORM',
        data: form.data
      });
    },
    onSubmit: async ({ formData, cancel }) => {
      if (options.onSubmit) {
        cancel();
        await options.onSubmit(formData);
      } else {
        actor.send({ type: 'SUBMIT' });
      }
    }
  });

  const state = writable(actor.getSnapshot().value);
  const context = writable(actor.getSnapshot().context);
  const isValid = derived([form.form], ([$form]) => !!($form as any).valid);
  const isSubmitting = derived(state, ($state) => 
    $state === 'submitting' || $state === 'validating'
  );
  const errors = derived([form.errors, context], ([$errors, $context]) => {
    // Flatten the complex superforms error structure to match interface
    const flattened: Record<string, string[]> = {};
    
    // Handle superforms errors (which can be nested objects)
    const flattenErrors = (obj: any, prefix = ''): void: any => {
      for (const [key, value] of Object.entries(obj || {})) {
        const fullKey = prefix ? `${prefix}.${key}` : key;
        if (Array.isArray(value)) {
          flattened[fullKey] = value as string[];
        } else if (typeof value === 'object' && value !== null) {
          flattenErrors(value, fullKey);
        }
      }
    };
    
    flattenErrors($errors);
    
    // Add context validation errors
    if ($context.validationErrors) {
      Object.assign(flattened, $context.validationErrors);
    }
    
    return flattened;
  });
  const progress = derived([state, context], ([$state, $context]) => {
    if ($state === 'completed') return 100;
    if ($state === 'submitting') return 80;
    if ($state === 'validating') return 60;
    if ($state === 'editing' && $context.isAutoSaving) return 30;
    return 0;
  });

  actor.subscribe((snapshot) => {
    state.set(snapshot.value);
    context.set(snapshot.context);
    
    if (snapshot.value === 'completed' && options.onSuccess) {
      options.onSuccess(snapshot.context.createdCase);
    } else if (snapshot.context.error && options.onError) {
      options.onError(snapshot.context.error);
    }
  });

  // Auto-save is built into the case creation machine
  actor.send({ type: 'START_CREATION' });

  return {
    form,
    actor,
    state,
    context,
    isValid,
    isSubmitting,
    errors,
    progress
  };
}

// ============================================================================
// SEARCH FORM INTEGRATION
// ============================================================================

export function createSearchForm(
  data: SuperValidated<Infer<typeof SearchQuerySchema>>,
  options: FormOptions = {}
): FormMachineIntegration<Infer<typeof SearchQuerySchema>, SearchActor> {
  
  const actor = createActor(searchMachine);
  actor.start();

  const form = superForm(data, {
    validators: zod(SearchQuerySchema),
    resetForm: false, // Keep search forms populated
    delayMs: 300,
    timeoutMs: 15000, // Longer timeout for searches
    invalidateAll: false,
    onUpdated: ({ form }) => {
      // Don't auto-submit on every change for search
      if (form.data.query && form.data.query.length > 2) {
        // Optional: Trigger search suggestions
      }
    },
    onSubmit: async ({ formData, cancel }) => {
      if (options.onSubmit) {
        cancel();
        await options.onSubmit(formData);
      } else {
        actor.send({
          type: 'SEARCH',
          data: formData
        });
      }
    }
  });

  const state = writable(actor.getSnapshot().value);
  const context = writable(actor.getSnapshot().context);
  const isValid = derived([form.form], ([$form]) => !!($form as any).valid);
  const isSubmitting = derived(state, ($state) => 
    $state === 'searching' || $state === 'validating' || $state === 'loadingMore'
  );
  const errors = derived([form.errors, context], ([$errors, $context]) => {
    // Flatten the complex superforms error structure to match interface
    const flattened: Record<string, string[]> = {};
    
    // Handle superforms errors (which can be nested objects)
    const flattenErrors = (obj: any, prefix = ''): void: any => {
      for (const [key, value] of Object.entries(obj || {})) {
        const fullKey = prefix ? `${prefix}.${key}` : key;
        if (Array.isArray(value)) {
          flattened[fullKey] = value as string[];
        } else if (typeof value === 'object' && value !== null) {
          flattenErrors(value, fullKey);
        }
      }
    };
    
    flattenErrors($errors);
    
    // Add context validation errors
    if ($context.validationErrors) {
      Object.assign(flattened, $context.validationErrors);
    }
    
    return flattened;
  });
  const progress = derived([state], ([$state]) => {
    if ($state === 'results') return 100;
    if ($state === 'searching') return 60;
    if ($state === 'validating') return 20;
    return 0;
  });

  actor.subscribe((snapshot) => {
    state.set(snapshot.value);
    context.set(snapshot.context);
    
    if (snapshot.value === 'results' && options.onSuccess) {
      options.onSuccess({
        results: snapshot.context.results,
        analytics: snapshot.context.analytics
      });
    } else if (snapshot.value === 'error' && options.onError) {
      options.onError(snapshot.context.error || 'Search failed');
    }
  });

  // Load search history on initialization
  actor.send({ type: 'LOAD_HISTORY' });

  return {
    form,
    actor,
    state,
    context,
    isValid,
    isSubmitting,
    errors,
    progress
  };
}

// ============================================================================
// AI ANALYSIS FORM INTEGRATION
// ============================================================================

export function createAIAnalysisForm(
  data: SuperValidated<Infer<typeof AIAnalysisSchema>>,
  options: FormOptions = {}
): FormMachineIntegration<Infer<typeof AIAnalysisSchema>, AIAnalysisActor> {
  
  const actor = createActor(aiAnalysisMachine);
  actor.start();

  const form = superForm(data, {
    validators: zod(AIAnalysisSchema),
    resetForm: options.resetOnSuccess ?? false,
    delayMs: 200,
    timeoutMs: 30000, // Very long timeout for AI analysis
    invalidateAll: false,
    onSubmit: async ({ formData, cancel }) => {
      if (options.onSubmit) {
        cancel();
        await options.onSubmit(formData);
      } else {
        actor.send({
          type: 'START_ANALYSIS',
          data: formData
        });
      }
    }
  });

  const state = writable(actor.getSnapshot().value);
  const context = writable(actor.getSnapshot().context);
  const isValid = derived([form.form], ([$form]) => !!($form as any).valid);
  const isSubmitting = derived(state, ($state) => 
    $state === 'analyzing' || $state === 'validating'
  );
  const errors = derived([form.errors, context], ([$errors, $context]) => {
    // Flatten the complex superforms error structure to match interface
    const flattened: Record<string, string[]> = {};
    
    // Handle superforms errors (which can be nested objects)
    const flattenErrors = (obj: any, prefix = ''): void: any => {
      for (const [key, value] of Object.entries(obj || {})) {
        const fullKey = prefix ? `${prefix}.${key}` : key;
        if (Array.isArray(value)) {
          flattened[fullKey] = value as string[];
        } else if (typeof value === 'object' && value !== null) {
          flattenErrors(value, fullKey);
        }
      }
    };
    
    flattenErrors($errors);
    
    // Add context validation errors
    if ($context.validationErrors) {
      Object.assign(flattened, $context.validationErrors);
    }
    
    return flattened;
  });
  const progress = derived([state, context], ([$state, $context]) => {
    if ($state === 'completed') return 100;
    if ($state === 'analyzing') return $context.isStreaming ? 70 : 50;
    if ($state === 'validating') return 10;
    return 0;
  });

  actor.subscribe((snapshot) => {
    state.set(snapshot.value);
    context.set(snapshot.context);
    
    if (snapshot.value === 'completed' && options.onSuccess) {
      options.onSuccess({
        results: snapshot.context.analysisResults,
        confidence: snapshot.context.confidence,
        processingTime: snapshot.context.processingTime,
        tokensUsed: snapshot.context.tokensUsed
      });
    } else if (snapshot.value === 'error' && options.onError) {
      options.onError(snapshot.context.error || 'Analysis failed');
    }
  });

  return {
    form,
    actor,
    state,
    context,
    isValid,
    isSubmitting,
    errors,
    progress
  };
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

export function createFormValidator<T extends z.ZodType>(schema: T) {
  return {
    validate: (data: unknown): data is z.infer<T> => {
      return schema.safeParse(data).success;
    },
    getErrors: (data: unknown): Record<string, string[]> => {
      const result = schema.safeParse(data);
      if (result.success) return {};
      return result.error.flatten().fieldErrors;
    },
    validateAsync: async (data: unknown): Promise<z.infer<T>> => {
      return schema.parseAsync(data);
    }
  };
}

export function createMultiStepForm<T extends z.ZodType[]>(...schemas: T) {
  const currentStep = writable(0);
  const isLastStep = derived(currentStep, ($step) => $step === schemas.length - 1);
  const isFirstStep = derived(currentStep, ($step) => $step === 0);
  const progress = derived(currentStep, ($step) => (($step + 1) / schemas.length) * 100);

  return {
    currentStep,
    isLastStep,
    isFirstStep,
    progress,
    totalSteps: schemas.length,
    nextStep: () => currentStep.update(n: any => Math.min(n + 1, schemas.length - 1)),
    previousStep: () => currentStep.update(n: any => Math.max(n - 1, 0)),
    goToStep: (step: number) => {
      if (step >= 0 && step < schemas.length) {
        currentStep.set(step);
      }
    },
    validateStep: (step: number, data: unknown) => {
      if (step >= 0 && step < schemas.length) {
        return createFormValidator(schemas[step]).validate(data);
      }
      return false;
    },
    getStepErrors: (step: number, data: unknown) => {
      if (step >= 0 && step < schemas.length) {
        return createFormValidator(schemas[step]).getErrors(data);
      }
      return {};
    }
  };
}

// ============================================================================
// FORM STATE PERSISTENCE
// ============================================================================

export class FormStatePersistence {
  private readonly storageKey: string;
  
  constructor(storageKey: string) {
    this.storageKey = storageKey;
  }

  save(data: any): void {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify({
        data,
        timestamp: Date.now()
      }));
    } catch (error) {
      console.warn('Failed to save form state:', error);
    }
  }

  load(): any | null {
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        const parsed = JSON.parse(stored);
        // Check if data is not too old (24 hours)
        if (Date.now() - parsed.timestamp < 24 * 60 * 60 * 1000) {
          return parsed.data;
        }
      }
    } catch (error) {
      console.warn('Failed to load form state:', error);
    }
    return null;
  }

  clear(): void {
    try {
      localStorage.removeItem(this.storageKey);
    } catch (error) {
      console.warn('Failed to clear form state:', error);
    }
  }

  createAutoSave(store: Writable<any>, delayMs = 1000) {
    let timeout: NodeJS.Timeout;
    
    return store.subscribe((value) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        this.save(value);
      }, delayMs);
    });
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export const formValidators = {
  documentUpload: createFormValidator(DocumentUploadSchema),
  caseCreation: createFormValidator(CaseCreationSchema),
  searchQuery: createFormValidator(SearchQuerySchema),
  aiAnalysis: createFormValidator(AIAnalysisSchema)
};

export const FORM_STORAGE_KEYS = {
  DOCUMENT_UPLOAD: 'legal-ai:document-upload',
  CASE_CREATION: 'legal-ai:case-creation',
  SEARCH_QUERY: 'legal-ai:search-query',
  AI_ANALYSIS: 'legal-ai:ai-analysis'
} as const;