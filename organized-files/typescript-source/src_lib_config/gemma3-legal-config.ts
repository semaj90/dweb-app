// Gemma3 Legal AI Configuration
// Complete configuration for the enhanced Legal AI system

import type { LegalAIConfig, AIModelConfig } from "$lib/types/ai";

// Model configurations
export const GEMMA3_MODELS: AIModelConfig[] = [
  {
    name: "gemma3-legal-ai",
    displayName: "Gemma3 Legal AI (Custom)",
    description:
      "Fine-tuned Gemma3 model specialized for legal analysis and document review",
    capabilities: [
      "legal_analysis",
      "document_review",
      "case_research",
      "contract_analysis",
      "legal_writing",
      "citation_generation",
    ],
    parameters: {
      maxTokens: 8192,
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      repeatPenalty: 1.1,
    },
  },
  {
    name: "gemma2:9b",
    displayName: "Gemma2 9B (Fallback)",
    description:
      "General-purpose Gemma2 model used as fallback for legal AI tasks",
    capabilities: [
      "general_chat",
      "document_analysis",
      "research_assistance",
      "writing_support",
    ],
    parameters: {
      maxTokens: 8192,
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      repeatPenalty: 1.1,
    },
  },
  {
    name: "llama3.1:8b",
    displayName: "Llama 3.1 8B",
    description:
      "Alternative model for legal AI tasks with strong reasoning capabilities",
    capabilities: [
      "legal_reasoning",
      "document_analysis",
      "research_assistance",
      "writing_support",
    ],
    parameters: {
      maxTokens: 8192,
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      repeatPenalty: 1.1,
    },
  },
];

// Main configuration
export const GEMMA3_CONFIG: LegalAIConfig = {
  models: GEMMA3_MODELS,
  defaultModel: "gemma3-legal-ai",
  fallbackModel: "gemma2:9b",
  features: {
    streaming: true,
    citations: true,
    ragIntegration: true,
    documentAnalysis: true,
    contextInjection: true,
  },
  limits: {
    maxMessageLength: 32000,
    maxContextLength: 8192,
    maxSessionDuration: 3600000, // 1 hour in ms
    maxConcurrentRequests: 5,
  },
};

// Ollama-specific configuration
export const OLLAMA_CONFIG = {
  endpoint: "http://localhost:11434",
  models: {
    primary: "gemma3-legal-ai",
    fallback: "gemma2:9b",
    alternatives: ["llama3.1:8b", "mistral:7b"],
  },
  parameters: {
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    repeat_penalty: 1.1,
    num_ctx: 8192,
    num_predict: 2048,
    stream: true,
  },
  timeout: 30000, // 30 seconds
  retries: 3,
  health_check_interval: 30000, // 30 seconds
};

// Legal AI prompt templates
export const LEGAL_AI_PROMPTS = {
  caseAnalysis:
    "Analyze the following legal case and provide key insights, including relevant precedents, legal principles, and potential outcomes:",
  documentReview:
    "Review this legal document and identify important clauses, potential risks, and recommendations for revision:",
  contractAnalysis:
    "Analyze this contract for key terms, potential risks, compliance issues, and negotiation points:",
  legalResearch:
    "Research the following legal topic and provide comprehensive information including relevant statutes, case law, and current interpretations:",
  briefWriting:
    "Help draft a legal brief addressing the following issue, including relevant authorities and persuasive arguments:",
  complianceCheck:
    "Review the following for compliance with applicable laws and regulations, highlighting any potential issues:",
  riskAssessment:
    "Conduct a risk assessment of the following legal matter, identifying potential liabilities and mitigation strategies:",
  precedentSearch:
    "Find relevant case law and legal precedents for the following legal issue:",
  statutoryAnalysis:
    "Analyze the relevant statutes and regulations pertaining to the following legal matter:",
  legalMemo:
    "Draft a legal memorandum addressing the following question, including analysis and recommendations:",
};

// Document types and their analysis templates
export const DOCUMENT_ANALYSIS_TEMPLATES = {
  contract: {
    prompt:
      "Analyze this contract focusing on key terms, obligations, risks, and compliance requirements:",
    sections: [
      "parties",
      "terms",
      "obligations",
      "risks",
      "termination",
      "dispute_resolution",
    ],
  },
  brief: {
    prompt:
      "Review this legal brief for argument structure, citation accuracy, and persuasive strength:",
    sections: ["issue", "facts", "argument", "conclusion", "citations"],
  },
  motion: {
    prompt:
      "Analyze this motion for procedural compliance, legal basis, and likelihood of success:",
    sections: [
      "jurisdiction",
      "standing",
      "legal_basis",
      "relief_sought",
      "supporting_evidence",
    ],
  },
  pleading: {
    prompt:
      "Review this pleading for legal sufficiency, factual allegations, and procedural compliance:",
    sections: ["jurisdiction", "parties", "facts", "claims", "relief"],
  },
  opinion: {
    prompt:
      "Analyze this legal opinion for reasoning, precedent application, and broader implications:",
    sections: ["facts", "issue", "holding", "reasoning", "implications"],
  },
  statute: {
    prompt:
      "Analyze this statute for scope, requirements, exceptions, and practical applications:",
    sections: [
      "scope",
      "definitions",
      "requirements",
      "exceptions",
      "penalties",
      "effective_date",
    ],
  },
};

// Context injection settings
export const CONTEXT_SETTINGS = {
  maxContextMessages: 20,
  contextWindowSize: 4096,
  relevanceThreshold: 0.7,
  contextTypes: {
    conversation: { weight: 1.0, maxMessages: 10 },
    documents: { weight: 0.8, maxTokens: 2048 },
    citations: { weight: 0.6, maxItems: 5 },
    precedents: { weight: 0.9, maxItems: 3 },
  },
};

// RAG (Retrieval-Augmented Generation) configuration
export const RAG_CONFIG = {
  enabled: true,
  vectorDatabase: {
    type: "qdrant" as const,
    endpoint: "http://localhost:6333",
    collection: "legal_documents",
    dimension: 384,
  },
  embedding: {
    model: "all-MiniLM-L6-v2",
    endpoint: "http://localhost:11434/api/embeddings",
  },
  retrieval: {
    topK: 5,
    scoreThreshold: 0.7,
    maxTokens: 2048,
    rerank: true,
  },
  documentTypes: [
    "case_law",
    "statutes",
    "regulations",
    "legal_articles",
    "contracts",
    "briefs",
  ],
};

// Error handling configuration
export const ERROR_CONFIG = {
  retryAttempts: 3,
  retryDelay: 1000,
  fallbackEnabled: true,
  gracefulDegradation: true,
  errorMessages: {
    modelNotFound:
      "The requested AI model is not available. Switching to fallback model.",
    connectionError:
      "Unable to connect to the AI service. Please check your connection.",
    rateLimited:
      "Rate limit exceeded. Please wait a moment before trying again.",
    invalidRequest:
      "Invalid request format. Please check your input and try again.",
    unknown:
      "An unexpected error occurred. Please try again or contact support.",
  },
};

// Performance monitoring
export const PERFORMANCE_CONFIG = {
  metrics: {
    enabled: true,
    endpoint: "/api/metrics",
    interval: 60000, // 1 minute
  },
  logging: {
    level: "info" as const,
    includeUserData: false,
    maxLogSize: 1000000, // 1MB
  },
  caching: {
    enabled: true,
    ttl: 300000, // 5 minutes
    maxSize: 100, // Max cached responses
  },
};

// Export combined configuration
export const LEGAL_AI_CONFIG = {
  ...GEMMA3_CONFIG,
  ollama: OLLAMA_CONFIG,
  prompts: LEGAL_AI_PROMPTS,
  documents: DOCUMENT_ANALYSIS_TEMPLATES,
  context: CONTEXT_SETTINGS,
  rag: RAG_CONFIG,
  errors: ERROR_CONFIG,
  performance: PERFORMANCE_CONFIG,
} as const;

// Type-safe config access
export function getModelConfig(modelName: string): AIModelConfig | undefined {
  return GEMMA3_MODELS.find((model) => model.name === modelName);
}

export function getPromptTemplate(
  templateName: keyof typeof LEGAL_AI_PROMPTS
): string {
  return LEGAL_AI_PROMPTS[templateName];
}

export function getDocumentTemplate(
  docType: keyof typeof DOCUMENT_ANALYSIS_TEMPLATES
) {
  return DOCUMENT_ANALYSIS_TEMPLATES[docType];
}
