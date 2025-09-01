// Enhanced Cache-First Forms Integration
// Superforms + Zod + LokiJS for Legal AI Platform

import { z } from 'zod';
import { superForm, type SuperValidated, type Infer } from 'sveltekit-superforms';
import { zod } from 'sveltekit-superforms/adapters';
import { writable, derived, type Writable } from 'svelte/store';
import { cacheFirstService, CaseSchema, EvidenceSchema } from './cache-first-architecture.js';

// ===== ENHANCED CASE FORM SCHEMA =====

export const EnhancedCaseFormSchema = CaseSchema.extend({
  // Additional form-specific fields
  attachments: z.array(z.instanceof(File)).default([]),
  legalCategory: z.enum(['civil', 'criminal', 'corporate', 'family', 'immigration']).optional(),
  jurisdiction: z.string().min(1, 'Jurisdiction required'),
  estimatedDuration: z.number().min(1).max(365).optional(),
  budget: z.number().min(0).optional(),
  clientContact: z.object({
    name: z.string().min(1, 'Contact name required'),
    email: z.string().email('Valid email required'),
    phone: z.string().optional()
  }),
  // Form state
  step: z.number().min(1).max(4).default(1),
  isDraft: z.boolean().default(true)
}).omit({ 
  id: true, 
  createdAt: true, 
  updatedAt: true, 
  _cached: true, 
  _lastSync: true, 
  _dirty: true 
});

export type EnhancedCaseForm = z.infer<typeof EnhancedCaseFormSchema>;

// ===== EVIDENCE UPLOAD FORM SCHEMA =====

export const EvidenceUploadFormSchema = EvidenceSchema.extend({
  file: z.instanceof(File),
  description: z.string().min(10, 'Description must be at least 10 characters'),
  relevanceScore: z.number().min(1).max(10).default(5),
  confidentialityLevel: z.enum(['public', 'restricted', 'confidential', 'secret']).default('restricted'),
  chain_of_custody: z.array(z.object({
    timestamp: z.date(),
    handler: z.string(),
    action: z.string(),
    notes: z.string().optional()
  })).default([])
}).omit({ 
  id: true, 
  createdAt: true, 
  fileUrl: true, 
  fileSize: true,
  _cached: true, 
  _lastSync: true, 
  _dirty: true 
});

export type EvidenceUploadForm = z.infer<typeof EvidenceUploadFormSchema>;

// ===== CACHE-FIRST FORM MANAGER =====

export class CacheFirstFormManager {
  private formCache = new Map<string, any>();
  private autosaveTimers = new Map<string, NodeJS.Timeout>();
  
  // Form state stores
  public activeForms = writable<string[]>([]);
  public formErrors = writable<Record<string, any>>({});
  public formProgress = writable<Record<string, number>>({});

  // ===== ENHANCED CASE FORM =====

  createEnhancedCaseForm(initialData?: Partial<EnhancedCaseForm>) {
    const formId = `case-form-${Date.now()}`;
    
    // Initialize with cache-first data
    const defaultData: EnhancedCaseForm = {
      title: '',
      description: '',
      status: 'open',
      priority: 'medium',
      userId: '', // Will be filled from session
      metadata: {},
      attachments: [],
      jurisdiction: '',
      clientContact: {
        name: '',
        email: '',
        phone: ''
      },
      step: 1,
      isDraft: true,
      ...initialData
    };

    // Store in cache
    this.formCache.set(formId, defaultData);
    this.updateActiveForm(formId);

    const form = superForm(
      { data: defaultData } as SuperValidated<Infer<typeof EnhancedCaseFormSchema>>,
      {
        SPA: true,
        validators: zod(EnhancedCaseFormSchema),
        resetForm: false,
        invalidateAll: false,
        
        // Cache-first validation
        onUpdate: ({ form }) => {
          this.handleFormUpdate(formId, form.data);
          this.startAutosave(formId, form.data);
        },

        onSubmit: ({ formData, cancel }) => {
          // Prevent submission if not on final step
          const currentData = this.formCache.get(formId);
          if (currentData?.step < 4) {
            cancel();
            this.nextStep(formId);
            return;
          }
        },

        onResult: async ({ result, formEl, cancel }) => {
          if (result.type === 'success') {
            // Cache the created case immediately
            const caseData = result.data as any;
            await cacheFirstService.createCase(caseData);
            
            // Clean up form cache
            this.cleanupForm(formId);
          }
        },

        onError: ({ result }) => {
          this.formErrors.update(errors => ({
            ...errors,
            [formId]: result.error
          }));
        }
      }
    );

    return {
      form,
      formId,
      // Step management
      nextStep: () => this.nextStep(formId),
      prevStep: () => this.prevStep(formId),
      goToStep: (step: number) => this.goToStep(formId, step),
      // Cache management
      saveAsDraft: () => this.saveFormDraft(formId),
      loadDraft: (draftId: string) => this.loadFormDraft(formId, draftId),
      // Progress tracking
      progress: derived([this.formProgress], ([$progress]) => $progress[formId] || 0)
    };
  }

  // ===== EVIDENCE UPLOAD FORM =====

  createEvidenceUploadForm(caseId: string, initialData?: Partial<EvidenceUploadForm>) {
    const formId = `evidence-form-${Date.now()}`;
    
    const defaultData: EvidenceUploadForm = {
      caseId,
      title: '',
      evidenceType: 'document',
      description: '',
      relevanceScore: 5,
      confidentialityLevel: 'restricted',
      analysisResults: {},
      tags: [],
      chain_of_custody: [],
      file: null as any, // Will be set by file input
      ...initialData
    };

    this.formCache.set(formId, defaultData);
    this.updateActiveForm(formId);

    const form = superForm(
      { data: defaultData } as SuperValidated<Infer<typeof EvidenceUploadFormSchema>>,
      {
        SPA: true,
        validators: zod(EvidenceUploadFormSchema),
        
        onUpdate: ({ form }) => {
          this.handleFormUpdate(formId, form.data);
          // Auto-generate title from file name
          if (form.data.file && !form.data.title) {
            form.data.title = form.data.file.name.replace(/\.[^/.]+$/, '');
          }
        },

        onSubmit: async ({ formData, cancel }) => {
          // Handle file upload first
          const file = formData.get('file') as File;
          if (file) {
            cancel();
            await this.uploadFileWithProgress(formId, file);
          }
        },

        onResult: async ({ result }) => {
          if (result.type === 'success') {
            const evidenceData = result.data as any;
            await cacheFirstService.createEvidence(evidenceData);
            this.cleanupForm(formId);
          }
        }
      }
    );

    return {
      form,
      formId,
      uploadProgress: derived([this.formProgress], ([$progress]) => $progress[formId] || 0),
      addToChainOfCustody: (action: string, handler: string, notes?: string) =>
        this.addChainOfCustodyEntry(formId, action, handler, notes)
    };
  }

  // ===== FORM STEP MANAGEMENT =====

  private nextStep(formId: string) {
    const formData = this.formCache.get(formId);
    if (formData && formData.step < 4) {
      formData.step += 1;
      this.formCache.set(formId, formData);
      this.updateProgress(formId, (formData.step / 4) * 100);
    }
  }

  private prevStep(formId: string) {
    const formData = this.formCache.get(formId);
    if (formData && formData.step > 1) {
      formData.step -= 1;
      this.formCache.set(formId, formData);
      this.updateProgress(formId, (formData.step / 4) * 100);
    }
  }

  private goToStep(formId: string, step: number) {
    const formData = this.formCache.get(formId);
    if (formData && step >= 1 && step <= 4) {
      formData.step = step;
      this.formCache.set(formId, formData);
      this.updateProgress(formId, (step / 4) * 100);
    }
  }

  // ===== CACHE MANAGEMENT =====

  private handleFormUpdate(formId: string, data: any) {
    this.formCache.set(formId, data);
    
    // Calculate progress based on filled fields
    const progress = this.calculateFormProgress(data);
    this.updateProgress(formId, progress);
  }

  private calculateFormProgress(data: any): number {
    const requiredFields = ['title', 'description', 'clientContact.name', 'clientContact.email'];
    const filledFields = requiredFields.filter(field => {
      if (field.includes('.')) {
        const [parent, child] = field.split('.');
        return data[parent]?.[child];
      }
      return data[field];
    });
    
    return (filledFields.length / requiredFields.length) * 100;
  }

  private updateProgress(formId: string, progress: number) {
    this.formProgress.update(current => ({
      ...current,
      [formId]: progress
    }));
  }

  private updateActiveForm(formId: string) {
    this.activeForms.update(forms => [...forms, formId]);
  }

  private cleanupForm(formId: string) {
    this.formCache.delete(formId);
    this.clearAutosave(formId);
    
    this.activeForms.update(forms => forms.filter(id => id !== formId));
    this.formErrors.update(errors => {
      delete errors[formId];
      return errors;
    });
    this.formProgress.update(progress => {
      delete progress[formId];
      return progress;
    });
  }

  // ===== AUTOSAVE FUNCTIONALITY =====

  private startAutosave(formId: string, data: any) {
    this.clearAutosave(formId);
    
    const timer = setTimeout(async () => {
      await this.saveFormDraft(formId);
    }, 2000); // Autosave after 2 seconds of inactivity
    
    this.autosaveTimers.set(formId, timer);
  }

  private clearAutosave(formId: string) {
    const timer = this.autosaveTimers.get(formId);
    if (timer) {
      clearTimeout(timer);
      this.autosaveTimers.delete(formId);
    }
  }

  private async saveFormDraft(formId: string) {
    const formData = this.formCache.get(formId);
    if (formData) {
      try {
        // Save to localStorage as backup
        localStorage.setItem(`draft-${formId}`, JSON.stringify({
          data: formData,
          timestamp: new Date().toISOString()
        }));
        
        console.log(`Draft saved for form ${formId}`);
      } catch (error) {
        console.error('Failed to save draft:', error);
      }
    }
  }

  private loadFormDraft(formId: string, draftId: string) {
    try {
      const draftData = localStorage.getItem(`draft-${draftId}`);
      if (draftData) {
        const { data, timestamp } = JSON.parse(draftData);
        this.formCache.set(formId, data);
        console.log(`Draft loaded from ${timestamp}`);
        return data;
      }
    } catch (error) {
      console.error('Failed to load draft:', error);
    }
    return null;
  }

  // ===== FILE UPLOAD WITH PROGRESS =====

  private async uploadFileWithProgress(formId: string, file: File) {
    const formData = new FormData();
    formData.append('file', file);
    
    return new Promise<string>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const progress = (e.loaded / e.total) * 100;
          this.updateProgress(formId, progress);
        }
      });
      
      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          const response = JSON.parse(xhr.responseText);
          resolve(response.fileUrl);
        } else {
          reject(new Error(`Upload failed: ${xhr.statusText}`));
        }
      });
      
      xhr.addEventListener('error', () => {
        reject(new Error('Upload failed'));
      });
      
      xhr.open('POST', '/api/upload');
      xhr.send(formData);
    });
  }

  // ===== CHAIN OF CUSTODY =====

  private addChainOfCustodyEntry(formId: string, action: string, handler: string, notes?: string) {
    const formData = this.formCache.get(formId);
    if (formData) {
      formData.chain_of_custody = formData.chain_of_custody || [];
      formData.chain_of_custody.push({
        timestamp: new Date(),
        handler,
        action,
        notes
      });
      this.formCache.set(formId, formData);
    }
  }

  // ===== FORM RECOVERY =====

  async recoverForms(): Promise<string[]> {
    const recoveredForms: string[] = [];
    
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith('draft-')) {
        try {
          const draftData = localStorage.getItem(key);
          if (draftData) {
            const { data, timestamp } = JSON.parse(draftData);
            const formId = key.replace('draft-', '');
            this.formCache.set(formId, data);
            recoveredForms.push(formId);
            console.log(`Recovered form ${formId} from ${timestamp}`);
          }
        } catch (error) {
          console.error(`Failed to recover form ${key}:`, error);
        }
      }
    }
    
    return recoveredForms;
  }
}

// ===== GLOBAL FORM MANAGER =====

export const formManager = new CacheFirstFormManager();