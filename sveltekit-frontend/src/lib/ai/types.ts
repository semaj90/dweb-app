// Type definitions for multi-LLM synthesis and legal AI pipeline

export interface AIModelOutput {
  content: string;
  suggestedFixes?: string[];
  codeReview?: string;
  analysis?: string;
  summary?: string;
  nextSteps?: string[];
}

export interface UserHistory {
  actions: string[];
  feedback?: string[];
}

export interface UploadedFile {
  name: string;
  textContent?: string;
  metadata?: Record<string, unknown>;
}

export interface MCPServerData {
  serverId: string;
  dataSummary: string;
  status?: string;
}

export interface SynthesisOptions {
  cacheEnabled?: boolean;
  autoEncode?: boolean;
  trainOnFeedback?: boolean;
  [key: string]: unknown;
}

// TODO: Wire up actual implementations for:
// - LLM output aggregation
// - User history feedback loop
// - Uploaded file parsing
// - MCP server data integration
// - Cache and auto-encoding logic
// - Training on user feedback
// - Best practices enforcement
// - Generative autocomplete and self-prompting
//
// Stub mocks for development/testing:
export const mockAIModelOutput: AIModelOutput = {
  content: "Sample LLM output.",
  suggestedFixes: ["Fix typo in section 2", "Clarify legal precedent"],
  codeReview: "No major issues found.",
  analysis: "Document is relevant to case.",
  summary: "Summary of document.",
  nextSteps: ["Review evidence", "Contact witness"],
};

export const mockUserHistory: UserHistory = {
  actions: ["uploaded document", "requested summary", "added note"],
  feedback: ["summary was helpful", "fixes were accurate"],
};

export const mockUploadedFile: UploadedFile = {
  name: "evidence.pdf",
  textContent: "This is the content of the uploaded file.",
};

export const mockMCPServerData: MCPServerData = {
  serverId: "mcp-001",
  dataSummary: "Server processed 10 documents.",
};

export const mockSynthesisOptions: SynthesisOptions = {
  cacheEnabled: true,
  autoEncode: true,
  trainOnFeedback: false,
};

// --- Phase 10: Context7 Semantic Search, Logging, Agent Integration Types ---

// Semantic search audit result structure (for /api/audit/semantic and UI)
export interface SemanticAuditResult {
  step: string; // Pipeline step or feature
  status: "ok" | "missing" | "error" | "improvement";
  message: string;
  suggestedFix?: string;
  todoId?: string;
  agentTriggered?: boolean;
}

// Log entry for audit results (for phase10-todo.log or DB)
export interface AuditLogEntry {
  timestamp: string;
  step: string;
  status: string;
  message: string;
  suggestedFix?: string;
  agentTriggered?: boolean;
}

// Agent action trigger structure
export interface AgentTrigger {
  todoId: string;
  action: "code_review" | "fix" | "analyze" | "summarize";
  status: "pending" | "in_progress" | "done";
  result?: string;
}

// TODO: After initial test, wire up real Context7 semantic search, logging, and agent triggers using mcp_memory_create_relations and mcp_context7_resolve-library-id
// These types are used by the backend endpoint, UI, and agent orchestration for Phase 10 full-stack integration.
