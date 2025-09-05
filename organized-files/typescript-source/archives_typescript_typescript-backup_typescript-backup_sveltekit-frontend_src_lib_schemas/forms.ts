import { z } from "zod";


// Case creation form schema
export const caseFormSchema = z.object({
  caseNumber: z
    .string()
    .min(1, "Case number is required")
    .max(50, "Case number too long")
    .regex(
      /^[A-Z]{2,4}-\d{4}-\d{6}$/,
      "Case number must follow format: ABC-2024-123456",
    ),

  title: z.string().min(1, "Title is required").max(200, "Title too long"),

  description: z
    .string()
    .min(10, "Description must be at least 10 characters")
    .max(1000, "Description too long")
    .optional(),

  priority: z.enum(["low", "medium", "high"], {
    errorMap: () => ({ message: "Please select a priority level" })
  }),

  status: z
    .enum(["draft", "active", "pending", "closed"], {
      errorMap: () => ({ message: "Please select a status" })
    })
    .default("draft"),

  assignedTo: z.string().uuid("Please select a valid user").optional(),

  dueDate: z
    .string()
    .refine((date) => {
      const parsed = new Date(date);
      return parsed > new Date();
    }, "Due date must be in the future")
    .optional(),

  tags: z
    .array(z.string().min(1).max(30))
    .max(10, "Maximum 10 tags allowed")
    .optional(),

  isConfidential: z.boolean().default(false),

  notifyAssignee: z.boolean().default(true),
});

// Evidence upload form schema
export const evidenceFormSchema = z.object({
  title: z.string().min(1, "Title is required").max(100, "Title too long"),

  description: z.string().max(500, "Description too long").optional(),

  evidenceType: z.enum(["document", "image", "video", "audio", "digital"], {
    errorMap: () => ({ message: "Please select evidence type" })
  }),

  tags: z
    .array(z.string().min(1).max(30))
    .max(15, "Maximum 15 tags allowed")
    .optional(),

  isChainOfCustodyRequired: z.boolean().default(false),

  custodyNotes: z.string().max(300, "Custody notes too long").optional(),

  collectedBy: z
    .string()
    .min(1, "Collector name is required")
    .max(100, "Name too long"),

  collectedAt: z.string().refine((date) => {
    const parsed = new Date(date);
    const now = new Date();
    return parsed <= now;
  }, "Collection date cannot be in the future"),

  location: z
    .string()
    .min(1, "Collection location is required")
    .max(200, "Location too long"),
});

// User authentication schema
export const authFormSchema = z.object({
  email: z
    .string()
    .email("Please enter a valid email address")
    .min(1, "Email is required"),

  password: z
    .string()
    .min(8, "Password must be at least 8 characters")
    .max(128, "Password too long")
    .regex(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
      "Password must contain uppercase, lowercase, and number",
    ),

  rememberMe: z.boolean().default(false),
});

// Registration schema
export const registerFormSchema = z
  .object({
    firstName: z
      .string()
      .min(1, "First name is required")
      .max(50, "First name too long")
      .regex(/^[a-zA-Z\s-']+$/, "First name contains invalid characters"),

    lastName: z
      .string()
      .min(1, "Last name is required")
      .max(50, "Last name too long")
      .regex(/^[a-zA-Z\s-']+$/, "Last name contains invalid characters"),

    email: z
      .string()
      .email("Please enter a valid email address")
      .min(1, "Email is required"),

    password: z
      .string()
      .min(8, "Password must be at least 8 characters")
      .max(128, "Password too long")
      .regex(
        /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])/,
        "Password must contain uppercase, lowercase, number, and special character",
      ),

    confirmPassword: z.string(),

    role: z.enum(["prosecutor", "investigator", "legal_assistant", "admin"], {
      errorMap: () => ({ message: "Please select a role" })
    }),

    agreeToTerms: z
      .boolean()
      .refine(
        (val) => val === true,
        "You must agree to the terms and conditions",
      ),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ["confirmPassword"],
  });

// Search and filter schema
export const searchFormSchema = z.object({
  query: z.string().max(200, "Search query too long").optional(),

  caseStatus: z
    .array(z.enum(["draft", "active", "pending", "closed"]))
    .optional(),

  evidenceType: z
    .array(z.enum(["document", "image", "video", "audio", "digital"]))
    .optional(),

  dateRange: z
    .object({
      from: z.string().optional(),
      to: z.string().optional(),
    })
    .refine((data) => {
      if (data.from && data.to) {
        return new Date(data.from) <= new Date(data.to);
      }
      return true;
    }, "Start date must be before end date")
    .optional(),

  priority: z.array(z.enum(["low", "medium", "high"])).optional(),

  tags: z.array(z.string()).optional(),

  sortBy: z
    .enum(["relevance", "date", "priority", "status"])
    .default("relevance"),

  sortOrder: z.enum(["asc", "desc"]).default("desc"),
});

// Additional schemas for missing imports
export const DocumentUploadSchema = evidenceFormSchema;
export const CaseCreationSchema = caseFormSchema;
export const SearchQuerySchema = searchFormSchema;
export const AIAnalysisSchema = z.object({
  prompt: z.string().min(1, 'Analysis prompt is required'),
  context: z.object({
    caseId: z.string().optional(),
    documentIds: z.array(z.string()).optional(),
    analysisType: z.enum(['summary', 'recommendation', 'risk-assessment', 'precedent-analysis'])
  }),
  options: z.object({
    includeReferences: z.boolean().default(true),
    maxTokens: z.number().min(100).max(4000).default(1000),
    temperature: z.number().min(0).max(1).default(0.7),
    model: z.string().optional()
  }).optional()
});

// Export type definitions
export type CaseForm = z.infer<typeof caseFormSchema>;
export type EvidenceForm = z.infer<typeof evidenceFormSchema>;
export type AuthForm = z.infer<typeof authFormSchema>;
export type RegisterForm = z.infer<typeof registerFormSchema>;
export type SearchForm = z.infer<typeof searchFormSchema>;
export type DocumentUploadData = z.infer<typeof DocumentUploadSchema>;
export type CaseCreationData = z.infer<typeof CaseCreationSchema>;
export type SearchQueryData = z.infer<typeof SearchQuerySchema>;
export type AIAnalysisData = z.infer<typeof AIAnalysisSchema>;
