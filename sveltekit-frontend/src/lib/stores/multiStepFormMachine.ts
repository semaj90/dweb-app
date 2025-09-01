import * as path from 'path';
import * as crypto from 'crypto';
import { z } from 'zod';

// src/lib/stores/multiStepFormMachine.ts - XState v5 Multi-step Forms with Superforms & Zod
import { setup, createActor, assign, fromPromise } from 'xstate';
import { cases, evidence } from "../server/db/schema-unified";
import { db } from "../server/db/drizzle";

// Zod Validation Schemas
export const CaseFormSchema = z.object({
  // Step 1: Basic Information
  title: z.string().min(3, "Title must be at least 3 characters"),
  caseNumber: z.string().min(1, "Case number is required"),
  description: z.string().min(10, "Description must be at least 10 characters"),
  incidentDate: z.string().optional(),
  location: z.string().optional(),

  // Step 2: Classification
  priority: z.enum(["low", "medium", "high", "urgent"]),
  status: z.enum(["open", "closed", "pending", "archived", "under_review"]),
  category: z.string().min(1, "Category is required"),
  dangerScore: z.number().min(0).max(10).default(0),
  estimatedValue: z.number().optional(),
  jurisdiction: z.string().optional(),

  // Step 3: Assignment
  leadProsecutor: z.string().optional(),
  assignedTeam: z.array(z.string()).default([]),
  tags: z.array(z.string()).default([]),

  // Step 4: Additional Details
  aiSummary: z.string().optional(),
  metadata: z.record(z.any()).default({}),
});

export const EvidenceFormSchema = z.object({
  // Step 1: Evidence Details
  title: z.string().min(3, "Title must be at least 3 characters"),
  description: z.string().optional(),
  evidenceType: z.enum([
    "document",
    "image",
    "video",
    "audio",
    "physical",
    "digital",
    "testimony",
  ]),
  subType: z.string().optional(),

  // Step 2: File Information
  fileName: z.string().optional(),
  fileSize: z.number().optional(),
  mimeType: z.string().optional(),
  hash: z.string().optional(),

  // Step 3: Chain of Custody
  collectedAt: z.string().optional(),
  collectedBy: z.string().optional(),
  location: z.string().optional(),
  chainOfCustody: z.array(z.any()).default([]),

  // Step 4: Classification
  tags: z.array(z.string()).default([]),
  isAdmissible: z.boolean().default(true),
  confidentialityLevel: z
    .enum(["public", "standard", "confidential", "classified"])
    .default("standard"),

  // Step 5: Analysis
  aiAnalysis: z.record(z.any()).default({}),
  aiTags: z.array(z.string()).default([]),
  aiSummary: z.string().optional(),
  summary: z.string().optional(),
});

export const CriminalFormSchema = z.object({
  // Step 1: Personal Information
  firstName: z.string().min(1, "First name is required"),
  lastName: z.string().min(1, "Last name is required"),
  middleName: z.string().optional(),
  aliases: z.array(z.string()).default([]),
  dateOfBirth: z.string().optional(),
  placeOfBirth: z.string().optional(),

  // Step 2: Contact Information
  address: z.string().optional(),
  phone: z.string().optional(),
  email: z.string().email().optional().or(z.literal("")),

  // Step 3: Identification
  socialSecurityNumber: z.string().optional(),
  driversLicense: z.string().optional(),

  // Step 4: Physical Description
  height: z.number().optional(),
  weight: z.number().optional(),
  eyeColor: z.string().optional(),
  hairColor: z.string().optional(),
  distinguishingMarks: z.string().optional(),

  // Step 5: Status & Classification
  status: z.enum([
    "suspect",
    "person_of_interest",
    "witness",
    "victim",
    "defendant",
  ]),
  dangerLevel: z.enum(["low", "medium", "high", "extreme"]).default("low"),
  currentLocation: z.string().optional(),
  knownAssociates: z.array(z.string()).default([]),
  criminalHistory: z.array(z.any()).default([]),

  // Step 6: Case Association
  associatedCases: z.array(z.string()).default([]),
  notes: z.string().optional(),
  metadata: z.record(z.any()).default({}),
});

// Form step validation schemas
export const CaseFormSteps = {
  step1: CaseFormSchema.pick({
    title: true,
    caseNumber: true,
    description: true,
    incidentDate: true,
    location: true,
  }),
  step2: CaseFormSchema.pick({
    priority: true,
    status: true,
    category: true,
    dangerScore: true,
    estimatedValue: true,
    jurisdiction: true,
  }),
  step3: CaseFormSchema.pick({
    leadProsecutor: true,
    assignedTeam: true,
    tags: true,
  }),
  step4: CaseFormSchema.pick({
    aiSummary: true,
    metadata: true,
  }),
};

export const EvidenceFormSteps = {
  step1: EvidenceFormSchema.pick({
    title: true,
    description: true,
    evidenceType: true,
    subType: true,
  }),
  step2: EvidenceFormSchema.pick({
    fileName: true,
    fileSize: true,
    mimeType: true,
    hash: true,
  }),
  step3: EvidenceFormSchema.pick({
    collectedAt: true,
    collectedBy: true,
    location: true,
    chainOfCustody: true,
  }),
  step4: EvidenceFormSchema.pick({
    tags: true,
    isAdmissible: true,
    confidentialityLevel: true,
  }),
  step5: EvidenceFormSchema.pick({
    aiAnalysis: true,
    aiTags: true,
    aiSummary: true,
    summary: true,
  }),
};

export const CriminalFormSteps = {
  step1: CriminalFormSchema.pick({
    firstName: true,
    lastName: true,
    middleName: true,
    aliases: true,
    dateOfBirth: true,
    placeOfBirth: true,
  }),
  step2: CriminalFormSchema.pick({
    address: true,
    phone: true,
    email: true,
  }),
  step3: CriminalFormSchema.pick({
    socialSecurityNumber: true,
    driversLicense: true,
  }),
  step4: CriminalFormSchema.pick({
    height: true,
    weight: true,
    eyeColor: true,
    hairColor: true,
    distinguishingMarks: true,
  }),
  step5: CriminalFormSchema.pick({
    status: true,
    dangerLevel: true,
    currentLocation: true,
    knownAssociates: true,
    criminalHistory: true,
  }),
  step6: CriminalFormSchema.pick({
    associatedCases: true,
    notes: true,
    metadata: true,
  }),
};

// Database operations
const saveToDatabase = fromPromise(
  async ({ input }: { input: { formType: string; data: any } }) => {
    const { formType, data } = input;

    try {
      switch (formType) {
        case "case":
          const [newCase] = await db
            .insert(cases)
            .values({
              ...data,
              id: crypto.randomUUID(),
              createdAt: new Date(),
              updatedAt: new Date(),
            })
            .returning();
          return { success: true, id: newCase.id, data: newCase };

        case "evidence":
          const [newEvidence] = await db
            .insert(evidence)
            .values({
              ...data,
              id: crypto.randomUUID(),
              createdAt: new Date(),
              updatedAt: new Date(),
            })
            .returning();
          return { success: true, id: newEvidence.id, data: newEvidence };

        case "criminal":
          const [newCriminal] = await db
            // TODO: verify correct table name; placeholder symbol 'criminals' was likely corrupted
            // .insert(criminals)
            .insert([] as any)
            .values({
              ...data,
              id: crypto.randomUUID(),
              createdAt: new Date(),
              updatedAt: new Date(),
            })
            .returning();
          return { success: true, id: newCriminal.id, data: newCriminal };

        default:
          throw new Error(`Unknown form type: ${formType}`);
      }
    } catch (error: any) {
      console.error("Database save error:", error);
      throw error;
    }
  },
);

// Vector embedding integration
const generateEmbeddings = fromPromise(
  async ({ input }: { input: { formType: string; data: any; id: string } }) => {
    const { formType, data, id } = input;

    try {
      // Create searchable text content
      let searchableContent = "";

      switch (formType) {
        case "case":
          searchableContent = `${data.title} ${data.description} ${data.category} ${data.tags?.join(" ") || ""}`;
          break;
        case "evidence":
          searchableContent = `${data.title} ${data.description} ${data.evidenceType} ${data.summary || ""} ${data.tags?.join(" ") || ""}`;
          break;
        case "criminal":
          searchableContent = `${data.firstName} ${data.lastName} ${data.aliases?.join(" ") || ""} ${data.notes || ""}`;
          break;
      }

      // Generate embeddings using Ollama or other embedding service
      const response = await fetch("/api/embeddings/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: searchableContent,
          metadata: {
            id,
            type: formType,
            timestamp: new Date().toISOString(),
          },
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate embeddings");
      }

      const embeddings = await response.json();
      return { success: true, embeddings };
    } catch (error: any) {
      console.error("Embedding generation error:", error);
      return { success: false, error: error.message };
    }
  },
);

// Multi-step form machine setup
export const multiStepFormMachine = setup({
  types: {
    context: {} as {
      formType: "case" | "evidence" | "criminal";
      currentStep: number;
      totalSteps: number;
      formData: Record<string, any>;
      stepData: Record<string, any>;
      errors: Record<string, string[]>;
      isValid: boolean;
      isSubmitting: boolean;
      submitResult: any;
      userId: string;
    },
    events: {} as
      | { type: "NEXT"; stepData: Record<string, any> }
      | { type: "PREVIOUS" }
      | { type: "GOTO_STEP"; step: number }
      | { type: "UPDATE_STEP_DATA"; data: Record<string, any> }
      | { type: "VALIDATE_STEP" }
      | { type: "SUBMIT" }
      | { type: "RESET" }
      | { type: "SET_FORM_TYPE"; formType: "case" | "evidence" | "criminal" },
  },
  actors: {
    saveToDatabase,
    generateEmbeddings,
  },
}).createMachine({
  id: "multiStepForm",
  initial: "editing",
  context: {
    formType: "case",
    currentStep: 1,
    totalSteps: 4,
    formData: {},
    stepData: {},
    errors: {},
    isValid: false,
    isSubmitting: false,
    submitResult: null,
    userId: "",
  },
  states: {
    editing: {
      on: {
        NEXT: {
          target: "validating",
          actions: assign({
            stepData: ({ event }) => event.stepData,
          }),
        },
        PREVIOUS: {
          actions: assign({
            currentStep: ({ context }) => Math.max(1, context.currentStep - 1),
            errors: {},
          }),
        },
        GOTO_STEP: {
          actions: assign({
            currentStep: ({ event }) => event.step,
            errors: {},
          }),
        },
        UPDATE_STEP_DATA: {
          actions: assign({
            stepData: ({ context, event }) => ({
              ...context.stepData,
              ...event.data,
            }),
          }),
        },
        SET_FORM_TYPE: {
          actions: assign({
            formType: ({ event }) => event.formType,
            currentStep: 1,
            totalSteps: ({ event }) => {
              switch (event.formType) {
                case "case":
                  return 4;
                case "evidence":
                  return 5;
                case "criminal":
                  return 6;
                default:
                  return 4;
              }
            },
            formData: {},
            stepData: {},
            errors: {},
          }),
        },
        SUBMIT: {
          target: "submitting",
          guard: ({ context }) => context.currentStep === context.totalSteps,
        },
        RESET: {
          actions: assign({
            currentStep: 1,
            formData: {},
            stepData: {},
            errors: {},
            isValid: false,
            submitResult: null,
          }),
        },
      },
    },
    validating: {
      entry: assign({
        isValid: false,
        errors: {},
      }),
      always: [
        {
          target: "editing",
          guard: ({ context }) => {
            const { formType, currentStep, stepData } = context;

            try {
              // Validate current step data
              let stepSchema;
              switch (formType) {
                case "case":
                  stepSchema = CaseFormSteps[`step${currentStep}`];
                  break;
                case "evidence":
                  stepSchema = EvidenceFormSteps[`step${currentStep}`];
                  break;
                case "criminal":
                  stepSchema = CriminalFormSteps[`step${currentStep}`];
                  break;
                default:
                  return false;
              }

              stepSchema.parse(stepData);
              return true;
            } catch (error: any) {
              if (error instanceof z.ZodError) {
                const fieldErrors: Record<string, string[]> = {};
                error.errors.forEach((err) => {
                  const field = err.path.join(".");
                  if (!fieldErrors[field]) {
                    fieldErrors[field] = [];
                  }
                  fieldErrors[field].push(err.message);
                });
                // Set errors in context
                return false;
              }
              return false;
            }
          },
          actions: assign({
            isValid: true,
            formData: ({ context }) => ({
              ...context.formData,
              ...context.stepData,
            }),
            currentStep: ({ context }) =>
              Math.min(context.totalSteps, context.currentStep + 1),
            stepData: {},
            errors: {},
          }),
        },
        {
          target: "editing",
          actions: assign(({ context }) => {
            const { formType, currentStep, stepData } = context;

            try {
              let stepSchema;
              switch (formType) {
                case "case":
                  stepSchema = CaseFormSteps[`step${currentStep}`];
                  break;
                case "evidence":
                  stepSchema = EvidenceFormSteps[`step${currentStep}`];
                  break;
                case "criminal":
                  stepSchema = CriminalFormSteps[`step${currentStep}`];
                  break;
              }

              stepSchema.parse(stepData);
              return { isValid: true, errors: {} };
            } catch (error: any) {
              if (error instanceof z.ZodError) {
                const fieldErrors: Record<string, string[]> = {};
                error.errors.forEach((err) => {
                  const field = err.path.join(".");
                  if (!fieldErrors[field]) {
                    fieldErrors[field] = [];
                  }
                  fieldErrors[field].push(err.message);
                });
                return { isValid: false, errors: fieldErrors };
              }
              return {
                isValid: false,
                errors: { general: ["Validation failed"] },
              };
            }
          }),
        },
      ],
    },
    submitting: {
      entry: assign({
        isSubmitting: true,
      }),
      invoke: {
        src: "saveToDatabase",
        input: ({ context }) => ({
          formType: context.formType,
          data: {
            ...context.formData,
            ...context.stepData,
            userId: context.userId,
          },
        }),
        onDone: {
          target: "generating_embeddings",
          actions: assign({
            submitResult: ({ event }) => event.output,
          }),
        },
        onError: {
          target: "submit_error",
          actions: assign({
            submitResult: ({ event }) => ({
              success: false,
              error: event.error,
            }),
            isSubmitting: false,
          }),
        },
      },
    },
    generating_embeddings: {
      invoke: {
        src: "generateEmbeddings",
        input: ({ context }) => ({
          formType: context.formType,
          data: {
            ...context.formData,
            ...context.stepData,
          },
          id: context.submitResult?.id,
        }),
        onDone: {
          target: "success",
          actions: assign({
            isSubmitting: false,
            submitResult: ({ context, event }) => ({
              ...context.submitResult,
              embeddings: event.output,
            }),
          }),
        },
        onError: {
          target: "success", // Continue even if embeddings fail
          actions: assign({
            isSubmitting: false,
            submitResult: ({ context, event }) => ({
              ...context.submitResult,
              embeddingsError: event.error,
            }),
          }),
        },
      },
    },
    success: {
      on: {
        RESET: {
          target: "editing",
          actions: assign({
            currentStep: 1,
            formData: {},
            stepData: {},
            errors: {},
            isValid: false,
            submitResult: null,
          }),
        },
      },
    },
    submit_error: {
      on: {
        SUBMIT: {
          target: "submitting",
        },
        RESET: {
          target: "editing",
          actions: assign({
            currentStep: 1,
            formData: {},
            stepData: {},
            errors: {},
            isValid: false,
            submitResult: null,
            isSubmitting: false,
          }),
        },
      },
    },
  },
});

// Helper functions for Svelte components
export function createMultiStepFormActor(
  userId: string,
  formType: "case" | "evidence" | "criminal" = "case"
) {
  const actor = createActor(multiStepFormMachine, {
    input: {
      userId,
      formType,
      totalSteps: formType === "case" ? 4 : formType === "evidence" ? 5 : 6,
    },
  });

  actor.start();
  return actor;
}

export function getStepSchema(formType: string, step: number) {
  switch (formType) {
    case "case":
      return CaseFormSteps[`step${step}`];
    case "evidence":
      return EvidenceFormSteps[`step${step}`];
    case "criminal":
      return CriminalFormSteps[`step${step}`];
    default:
      throw new Error(`Unknown form type: ${formType}`);
  }
}

export function getFullSchema(formType: string) {
  switch (formType) {
    case "case":
      return CaseFormSchema;
    case "evidence":
      return EvidenceFormSchema;
    case "criminal":
      return CriminalFormSchema;
    default:
      throw new Error(`Unknown form type: ${formType}`);
  }
}

// Export types for TypeScript
export type MultiStepFormContext = {
  formType: "case" | "evidence" | "criminal";
  currentStep: number;
  totalSteps: number;
  formData: Record<string, any>;
  stepData: Record<string, any>;
  errors: Record<string, string[]>;
  isValid: boolean;
  isSubmitting: boolean;
  submitResult: any;
  userId: string;
};

export type CaseFormData = z.infer<typeof CaseFormSchema>;
export type EvidenceFormData = z.infer<typeof EvidenceFormSchema>;
export type CriminalFormData = z.infer<typeof CriminalFormSchema>;
