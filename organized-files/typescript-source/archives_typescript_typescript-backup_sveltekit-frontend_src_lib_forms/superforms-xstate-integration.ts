// Superforms + XState Integration for Legal AI Forms
// Advanced form management with state machines and validation

import { superForm } from "sveltekit-superforms";
import { zod } from "sveltekit-superforms/adapters";
import { writable, derived, type Writable, type Readable } from "svelte/store";
import { createActor } from "xstate";
import { z } from "zod";
import {
  DocumentUploadSchema,
  CaseCreationSchema,
  SearchQuerySchema,
  AIAnalysisSchema
} from "$lib/schemas/forms";
import {
  documentUploadMachine,
  caseCreationMachine,
  searchMachine,
  aiAnalysisMachine
} from "$lib/machines";

// ============================================================================
// FORM STATE INTEGRATION TYPES
// ============================================================================

export type SnapshotOf<M> = M extends { getSnapshot: () => infer S } ? S : { status: any; context: any };

export interface FormMachineIntegration<M extends { getSnapshot: () => any }> {
  form: ReturnType<typeof superForm>;
  actor: M;
  state: Writable<any>;
  context: Writable<any>;
  isValid: Readable<boolean>;
  isSubmitting: Readable<boolean>;
  errors: Readable<Record<string, string[]>>;
  progress: Readable<number>;
  autoSaveDelay?: number;
  resetOnSuccess?: boolean;
}

export interface FormOptions {
  autoSave?: boolean;
  autoSaveDelay?: number;
  resetOnSuccess?: boolean;
  onSubmit?: (formData: any) => Promise<any> | void;
  onSuccess?: (data: any) => void;
  onError?: (error: any) => void;
}

// ============================================================================
// DOCUMENT UPLOAD FORM INTEGRATION
// ============================================================================

export function createDocumentUploadForm(
  data: any, // SuperValidated<Infer<typeof DocumentUploadSchema>>,
  options: FormOptions = {}
): unknown { // FormMachineIntegration<Infer<typeof DocumentUploadSchema>, DocumentUploadActor> {

  // Create XState actor
  const actor = createActor(documentUploadMachine as any);
  actor.start();

  // Create Superform
  const form = superForm(data, {
    validators: zod(DocumentUploadSchema as any),
    resetForm: options.resetOnSuccess ?? true,
    delayMs: 300,
    timeoutMs: 8000,
    invalidateAll: false,
    onUpdated: ({ form }) => {
      // When the form becomes valid, ask the actor to validate the form data;
      // otherwise inform the machine of the updated form data.
      if ((form as any).valid) {
        actor.send({
          type: 'VALIDATE_FORM',
          data: (form as any).data || form
        } as any);
      } else {
        actor.send({
          type: 'UPDATE_FORM',
          data: (form as any).data || form
        } as any);
      }
    },
    onSubmit: async ({ formData, cancel }) => {
      if (options.onSubmit) {
        // allow the caller to handle submit if provided
        cancel();
        await options.onSubmit(formData);
      } else {
        actor.send({
          type: 'SUBMIT',
          data: formData
        } as any);
      }
    }
  });

  // Reactive state stores (XState v5 compatibility)
  const snapshot = actor.getSnapshot() as any;
  const state = writable(snapshot.status || snapshot.value || 'idle');
  const context = writable(snapshot.context || {});
  const isValid = derived([form.form], ([$form]) => !!($form as any).valid);
  const isSubmitting = derived(state, ($state) =>
    $state === 'uploading' || $state === 'processing' || $state === 'validating'
  );
  const errors = derived([form.errors, context], ([$errors, $context]) => {
    // Flatten the complex superforms error structure to match interface
    const flattened: Record<string, string[]> = {};

    // Handle superforms errors (which can be nested objects)
    const flattenErrors = (obj: any, prefix = ''): void => {
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
    if ($context?.validationErrors) {
      Object.assign(flattened, $context.validationErrors);
    }

    return flattened;
  });
  const progress = derived(context, ($context) =>
    Math.max($context?.uploadProgress ?? 0, $context?.processingProgress ?? 0)
  );

  // Subscribe to actor changes (XState v5 compatibility)
  actor.subscribe((snapshot) => {
    const stateValue = (snapshot as any).status || (snapshot as any).value || 'idle';
    const contextValue = (snapshot as any).context || {};
    state.set(stateValue);
    context.set(contextValue);

    // Handle state-specific actions
    if (stateValue === 'completed' && options.onSuccess) {
      options.onSuccess(contextValue?.aiResults);
    } else if (stateValue === 'failed' && options.onError) {
      options.onError(contextValue?.error || 'Upload failed');
    }
  });

  // Auto-save functionality
  if (options.autoSave) {
    const autoSaveDelay = options.autoSaveDelay ?? 2000;
    let autoSaveTimeout: ReturnType<typeof setTimeout>;

    form.form.subscribe(($form) => {
      if (($form as any).valid) {
        clearTimeout(autoSaveTimeout);
        autoSaveTimeout = setTimeout(() => {
          try {
            localStorage.setItem('document-upload-draft', JSON.stringify(($form as any).data || $form));
          } catch (e: any) {
            console.warn('Failed to write draft to localStorage', e);
          }
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
  data: any, // SuperValidated<Infer<typeof CaseCreationSchema>>,
  options: FormOptions = {}
): unknown { // FormMachineIntegration<Infer<typeof CaseCreationSchema>, CaseCreationActor> {

  const actor = createActor(caseCreationMachine as any);
  actor.start();

  const form = superForm(data, {
    validators: zod(CaseCreationSchema as any),
    resetForm: options.resetOnSuccess ?? false, // Don't reset case creation forms
    delayMs: 500, // Longer delay for case creation
    timeoutMs: 10000,
    invalidateAll: true,
    onUpdated: ({ form }) => {
      actor.send({
        type: 'UPDATE_FORM',
        data: (form as any).data || form
      } as any);
    },
    onSubmit: async ({ formData, cancel }) => {
      if (options.onSubmit) {
        cancel();
        await options.onSubmit(formData);
      } else {
        actor.send({
          type: 'SUBMIT_CASE',
          data: formData
        } as any);
      }
    }
  });

  const snapshot = actor.getSnapshot() as any;
  const state = writable(snapshot.status || snapshot.value || 'idle');
  const context = writable(snapshot.context || {});
  const isValid = derived([form.form], ([$form]) => !!($form as any).valid);
  const isSubmitting = derived(state, ($state) =>
    $state === 'submitting' || $state === 'validating'
  );
  const errors = derived([form.errors, context], ([$errors, $context]) => {
    // Flatten the complex superforms error structure to match interface
    const flattened: Record<string, string[]> = {};

    // Handle superforms errors (which can be nested objects)
    const flattenErrors = (obj: any, prefix = ''): void => {
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
    if ($context?.validationErrors) {
      Object.assign(flattened, $context.validationErrors);
    }

    return flattened;
  });
  const progress = derived([state, context], ([$state, $context]) => {
    if ($state === 'completed') return 100;
    if ($state === 'submitting') return 80;
    if ($state === 'validating') return 60;
    if ($state === 'editing' && $context?.isAutoSaving) return 30;
    return 0;
  });

  actor.subscribe((snapshot) => {
    const stateValue = (snapshot as any).status || (snapshot as any).value || 'idle';
    const contextValue = (snapshot as any).context || {};
    state.set(stateValue);
    context.set(contextValue);

    if (stateValue === 'completed' && options.onSuccess) {
      options.onSuccess(contextValue?.createdCase);
    } else if (contextValue?.error && options.onError) {
      options.onError(contextValue.error);
    }
  });

  // Auto-save is built into the case creation machine
  actor.send({ type: 'START_CREATION' } as any);

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
  data: any, // SuperValidated<Infer<typeof SearchQuerySchema>>,
  options: FormOptions = {}
): unknown { // FormMachineIntegration<Infer<typeof SearchQuerySchema>, SearchActor> {

  const actor = createActor(searchMachine as any);
  actor.start();

  const form = superForm(data, {
    validators: zod(SearchQuerySchema as any),
    resetForm: false, // Keep search forms populated
    delayMs: 300,
    timeoutMs: 15000, // Longer timeout for searches
    invalidateAll: false,
    onUpdated: ({ form }) => {
      // Don't auto-submit on every change for search
      if ((form as any).data?.query && (form as any).data.query.length > 2) {
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
        } as any);
      }
    }
  });

  const snapshot = actor.getSnapshot() as any;
  const state = writable(snapshot.status || snapshot.value || 'idle');
  const context = writable(snapshot.context || {});
  const isValid = derived([form.form], ([$form]) => !!($form as any).valid);
  const isSubmitting = derived(state, ($state) =>
    $state === 'searching' || $state === 'validating'
  );

  const errors = derived([form.errors, context], ([$errors, $context]) => {
    // Flatten the complex superforms error structure to match interface
    const flattened: Record<string, string[]> = {};

    // Handle superforms errors (which can be nested objects)
    const flattenErrors = (obj: any, prefix = ''): void => {
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
    if ($context?.validationErrors) {
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
    const stateValue = (snapshot as any).status || (snapshot as any).value || 'idle';
    const contextValue = (snapshot as any).context || {};
    state.set(stateValue);
    context.set(contextValue);

    if (stateValue === 'results' && options.onSuccess) {
      options.onSuccess({
        results: contextValue?.results,
        analytics: contextValue?.analytics
      });
    } else if (stateValue === 'error' && options.onError) {
      options.onError(contextValue?.error || 'Search failed');
    }
  });

  // Load search history on initialization
  actor.send({ type: 'LOAD_HISTORY' } as any);

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
  data: any, // SuperValidated<Infer<typeof AIAnalysisSchema>>,
  options: FormOptions = {}
): unknown { // FormMachineIntegration<Infer<typeof AIAnalysisSchema>, AIAnalysisActor> {

  const actor = createActor(aiAnalysisMachine as any);
  actor.start();

  const form = superForm(data, {
    validators: zod(AIAnalysisSchema as any),
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
        } as any);
      }
    }
  });

  const snapshot = actor.getSnapshot() as any;
  const state = writable(snapshot.status || snapshot.value || 'idle');
  const context = writable(snapshot.context || {});
  const isValid = derived([form.form], ([$form]) => !!($form as any).valid);
  const isSubmitting = derived(state, ($state) =>
    $state === 'analyzing' || $state === 'validating'
  );

  const errors = derived([form.errors, context], ([$errors, $context]) => {
    // Flatten the complex superforms error structure to match interface
    const flattened: Record<string, string[]> = {};

    // Handle superforms errors (which can be nested objects)
    const flattenErrors = (obj: any, prefix = ''): void => {
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
    if ($context?.validationErrors) {
      Object.assign(flattened, $context.validationErrors);
    }

    return flattened;
  });
  const progress = derived([state, context], ([$state, $context]) => {
    if ($state === 'completed') return 100;
    if ($state === 'analyzing') return $context?.isStreaming ? 70 : 50;
    if ($state === 'validating') return 10;
    return 0;
  });

  actor.subscribe((snapshot) => {
    const stateValue = (snapshot as any).status || (snapshot as any).value || 'idle';
    const contextValue = (snapshot as any).context || {};
    state.set(stateValue);
    context.set(contextValue);

    if (stateValue === 'completed' && options.onSuccess) {
      options.onSuccess({
        results: contextValue?.analysisResults,
        confidence: contextValue?.confidence,
        processingTime: contextValue?.processingTime,
        tokensUsed: contextValue?.tokensUsed
      });
    } else if (stateValue === 'error' && options.onError) {
      options.onError(contextValue?.error || 'Analysis failed');
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
    validate: (data: any): data is z.infer<T> => {
      return schema.safeParse(data).success;
    },
    getErrors: (data: any): Record<string, string[]> => {
      const result = schema.safeParse(data);
      if (result.success) return {};
      return result.error.flatten().fieldErrors;
    },
    validateAsync: async (data: any): Promise<z.infer<T>> => {
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
    nextStep: () => currentStep.update((n: any) => Math.min(n + 1, schemas.length - 1)),
    previousStep: () => currentStep.update((n: any) => Math.max(n - 1, 0)),
    goToStep: (step: number) => {
      if (step >= 0 && step < schemas.length) {
        currentStep.set(step);
      }
    },
    validateStep: (step: number, data: any) => {
      if (step >= 0 && step < schemas.length) {
        return createFormValidator(schemas[step]).validate(data);
      }
      return false;
    },
    getStepErrors: (step: number, data: any) => {
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
    } catch (error: any) {
      console.warn('Failed to save form state:', error);
    }
  }

  load(): unknown | null {
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        const parsed = JSON.parse(stored);
        // Check if data is not too old (24 hours)
        if (Date.now() - parsed.timestamp < 24 * 60 * 60 * 1000) {
          return parsed.data;
        }
      }
    } catch (error: any) {
      console.warn('Failed to load form state:', error);
    }
    return null;
  }

  clear(): void {
    try {
      localStorage.removeItem(this.storageKey);
    } catch (error: any) {
      console.warn('Failed to clear form state:', error);
    }
  }

  createAutoSave(store: Writable<any>, delayMs = 1000) {
    let timeout: ReturnType<typeof setTimeout>;

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