// Enhanced XState Machine for AI Agent Shell with MCP Tools Integration
import { createMachine, assign } from "xstate";
import { goServiceClient, type RAGResponse, type UploadResponse } from '../services/goServiceClient';
import { productionServiceClient, services } from '../services/productionServiceClient';
import { mcpTools, type MCPToolResponse } from '../../mcp/index';
import type { Case, Evidence, User } from "$lib/types/index.js";

// Enhanced context with MCP tool integration
export interface AgentShellContext {
  input: string;
  response: string;
  jobId?: string;
  rating?: number;
  searchQuery?: string;
  searchResults?: RAGResponse;
  uploadResults?: UploadResponse;
  userId?: string;
  caseId?: string;
  currentCase?: Case;
  currentEvidence?: Evidence[];
  currentUser?: User;
  serviceHealth?: {
    enhancedRAG: boolean;
    uploadService: boolean;
    kratosServer: boolean;
    mcpDatabase: boolean;
  };
  mcpResults?: {
    cases?: MCPToolResponse<Case[]>;
    evidence?: MCPToolResponse<Evidence[]>;
    users?: MCPToolResponse<User[]>;
    analytics?: MCPToolResponse<any>;
  };
  error?: string;
}

// Enhanced event types with MCP operations
type AgentShellEvent =
  | { type: "PROMPT"; input: string; userId?: string; caseId?: string }
  | { type: "xstate.done.actor.callAgent"; data: string }
  | { type: "ACCEPT_PATCH"; jobId: string; userId: string; patchContent: string }
  | { type: "RATE_SUGGESTION"; jobId: string; rating: number; userId: string; feedback?: string }
  | { type: "SEMANTIC_SEARCH"; query: string; userId: string; caseId?: string }
  | { type: "FILE_UPLOAD"; file: File; userId: string; caseId?: string }
  | { type: "CHECK_HEALTH" }
  // MCP operations
  | { type: "MCP_LOAD_CASE"; caseId: string }
  | { type: "MCP_LOAD_CASES"; userId: string; filters?: any }
  | { type: "MCP_CREATE_CASE"; caseData: any; userId: string }
  | { type: "MCP_LOAD_EVIDENCE"; caseId: string }
  | { type: "MCP_CREATE_EVIDENCE"; evidenceData: any; caseId: string }
  | { type: "MCP_FIND_SIMILAR_CASES"; embedding: number[]; caseId?: string }
  | { type: "MCP_FIND_SIMILAR_EVIDENCE"; embedding: number[]; caseId?: string }
  | { type: "MCP_GET_ANALYTICS"; userId?: string; caseId?: string };

export const agentShellMachineMCP = createMachine({
  id: "agentShellMCP",
  initial: "idle",
  context: {
    input: "",
    response: "",
    mcpResults: {},
    serviceHealth: {
      enhancedRAG: false,
      uploadService: false,
      kratosServer: false,
      mcpDatabase: false
    }
  },
  types: {} as {
    context: AgentShellContext;
    events: AgentShellEvent;
  },
  states: {
    idle: {
      on: {
        PROMPT: {
          target: "processing",
          actions: assign({
            input: ({ event }) => (event as any).input || "",
            userId: ({ event }) => (event as any).userId,
            caseId: ({ event }) => (event as any).caseId,
          }),
        },
        SEMANTIC_SEARCH: {
          target: "searching",
          actions: assign({
            searchQuery: ({ event }) => (event as any).query,
            userId: ({ event }) => (event as any).userId,
            caseId: ({ event }) => (event as any).caseId,
          }),
        },
        FILE_UPLOAD: {
          target: "uploading",
          actions: assign({
            userId: ({ event }) => (event as any).userId,
            caseId: ({ event }) => (event as any).caseId,
          }),
        },
        CHECK_HEALTH: {
          target: "checkingHealth",
        },
        // MCP operations
        MCP_LOAD_CASE: {
          target: "mcpLoadingCase",
          actions: assign({
            caseId: ({ event }) => (event as any).caseId,
          }),
        },
        MCP_LOAD_CASES: {
          target: "mcpLoadingCases",
          actions: assign({
            userId: ({ event }) => (event as any).userId,
          }),
        },
        MCP_CREATE_CASE: {
          target: "mcpCreatingCase",
          actions: assign({
            userId: ({ event }) => (event as any).userId,
          }),
        },
        MCP_LOAD_EVIDENCE: {
          target: "mcpLoadingEvidence",
          actions: assign({
            caseId: ({ event }) => (event as any).caseId,
          }),
        },
        MCP_CREATE_EVIDENCE: {
          target: "mcpCreatingEvidence",
          actions: assign({
            caseId: ({ event }) => (event as any).caseId,
          }),
        },
        MCP_FIND_SIMILAR_CASES: {
          target: "mcpFindingSimilarCases",
          actions: assign({
            caseId: ({ event }) => (event as any).caseId,
          }),
        },
        MCP_FIND_SIMILAR_EVIDENCE: {
          target: "mcpFindingSimilarEvidence",
          actions: assign({
            caseId: ({ event }) => (event as any).caseId,
          }),
        },
        MCP_GET_ANALYTICS: {
          target: "mcpGettingAnalytics",
          actions: assign({
            userId: ({ event }) => (event as any).userId,
            caseId: ({ event }) => (event as any).caseId,
          }),
        },
      },
    },
    processing: {
      invoke: {
        src: "callAgentWithMCP",
        input: ({ context }) => ({
          input: context.input,
          userId: context.userId,
          caseId: context.caseId,
        }),
        onDone: {
          target: "idle",
          actions: assign({
            response: (_, e) => (e && "data" in e ? (e as any).data : ""),
          }),
        },
        onError: {
          target: "idle",
          actions: assign({
            error: (_, e) => (e && "data" in e ? (e as any).data?.message || "Processing error" : "Unknown error"),
          }),
        },
      },
      on: {
        ACCEPT_PATCH: {
          actions: "acceptPatchAction",
        },
        RATE_SUGGESTION: {
          actions: "rateSuggestionAction",
        },
      },
    },
    searching: {
      invoke: {
        src: "performSemanticSearchWithMCP",
        input: ({ context }) => ({
          query: context.searchQuery,
          userId: context.userId,
          caseId: context.caseId,
        }),
        onDone: {
          target: "idle",
          actions: assign({
            searchResults: (_, e) => (e && "data" in e ? (e as any).data : null),
          }),
        },
        onError: {
          target: "idle",
          actions: assign({
            error: (_, e) => (e && "data" in e ? (e as any).data?.message || "Search error" : "Unknown error"),
          }),
        },
      },
    },
    uploading: {
      invoke: {
        src: "performFileUploadWithMCP",
        input: ({ context, event }) => ({
          file: (event as any).file,
          userId: context.userId,
          caseId: context.caseId,
        }),
        onDone: {
          target: "idle",
          actions: assign({
            uploadResults: (_, e) => (e && "data" in e ? (e as any).data : null),
          }),
        },
        onError: {
          target: "idle",
          actions: assign({
            error: (_, e) => (e && "data" in e ? (e as any).data?.message || "Upload error" : "Unknown error"),
          }),
        },
      },
    },
    checkingHealth: {
      invoke: {
        src: "checkServiceHealthWithMCP",
        onDone: {
          target: "idle",
          actions: assign({
            serviceHealth: (_, e) => (e && "data" in e ? (e as any).data : null),
          }),
        },
        onError: {
          target: "idle",
          actions: assign({
            error: (_, e) => (e && "data" in e ? (e as any).data?.message || "Health check error" : "Unknown error"),
          }),
        },
      },
    },
    // MCP states
    mcpLoadingCase: {
      invoke: {
        src: "mcpLoadCase",
        input: ({ context }) => ({ caseId: context.caseId }),
        onDone: {
          target: "idle",
          actions: assign({
            currentCase: (_, e) => (e && "data" in e ? (e as any).data : null),
            mcpResults: ({ context, event }) => ({
              ...(context.mcpResults || {}),
              cases: event && "data" in event ? (event as any).data : null,
            }),
          }),
        },
        onError: {
          target: "idle",
          actions: assign({
            error: (_, e) => (e && "data" in e ? (e as any).data?.message || "MCP case load error" : "Unknown error"),
          }),
        },
      },
    },
    mcpLoadingCases: {
      invoke: {
        src: "mcpLoadCases",
        input: ({ context }) => ({ userId: context.userId }),
        onDone: {
          target: "idle",
          actions: assign({
            mcpResults: ({ context, event }) => ({
              ...(context.mcpResults || {}),
              cases: event && "data" in event ? (event as any).data : null,
            }),
          }),
        },
        onError: {
          target: "idle",
          actions: assign({
            error: (_, e) => (e && "data" in e ? (e as any).data?.message || "MCP cases load error" : "Unknown error"),
          }),
        },
      },
    },
    mcpCreatingCase: {
      invoke: {
        src: "mcpCreateCase",
        input: ({ context, event }) => ({
          caseData: (event as any).caseData,
          userId: context.userId,
        }),
        onDone: {
          target: "idle",
          actions: assign({
            currentCase: (_, e) => (e && "data" in e ? (e as any).data : null),
            mcpResults: ({ context, event }) => ({
              ...(context.mcpResults || {}),
              cases: event && "data" in event ? (event as any).data : null,
            }),
          }),
        },
        onError: {
          target: "idle",
          actions: assign({
            error: (_, e) => (e && "data" in e ? (e as any).data?.message || "MCP case creation error" : "Unknown error"),
          }),
        },
      },
    },
    mcpLoadingEvidence: {
      invoke: {
        src: "mcpLoadEvidence",
        input: ({ context }) => ({ caseId: context.caseId }),
        onDone: {
          target: "idle",
          actions: assign({
            currentEvidence: (_, e) => (e && "data" in e ? (e as any).data : []),
            mcpResults: ({ context }, e) => ({
              ...context.mcpResults,
              evidence: e && "data" in e ? (e as any).data : null,
            }),
          }),
        },
        onError: {
          target: "idle",
          actions: assign({
            error: (_, e) => (e && "data" in e ? (e as any).data?.message || "MCP evidence load error" : "Unknown error"),
          }),
        },
      },
    },
    mcpCreatingEvidence: {
      invoke: {
        src: "mcpCreateEvidence",
        input: ({ context, event }) => ({
          evidenceData: (event as any).evidenceData,
          caseId: context.caseId,
        }),
        onDone: {
          target: "idle",
          actions: assign({
            mcpResults: ({ context }, e) => ({
              ...context.mcpResults,
              evidence: e && "data" in e ? (e as any).data : null,
            }),
          }),
        },
        onError: {
          target: "idle",
          actions: assign({
            error: (_, e) => (e && "data" in e ? (e as any).data?.message || "MCP evidence creation error" : "Unknown error"),
          }),
        },
      },
    },
    mcpFindingSimilarCases: {
      invoke: {
        src: "mcpFindSimilarCases",
        input: ({ context, event }) => ({
          embedding: (event as any).embedding,
          caseId: context.caseId,
        }),
        onDone: {
          target: "idle",
          actions: assign({
            mcpResults: ({ context, event }) => ({
              ...(context.mcpResults || {}),
              cases: event && "data" in event ? (event as any).data : null,
            }),
          }),
        },
        onError: {
          target: "idle",
          actions: assign({
            error: (_, e) => (e && "data" in e ? (e as any).data?.message || "MCP similar cases error" : "Unknown error"),
          }),
        },
      },
    },
    mcpFindingSimilarEvidence: {
      invoke: {
        src: "mcpFindSimilarEvidence",
        input: ({ context, event }) => ({
          embedding: (event as any).embedding,
          caseId: context.caseId,
        }),
        onDone: {
          target: "idle",
          actions: assign({
            mcpResults: ({ context }, e) => ({
              ...context.mcpResults,
              evidence: e && "data" in e ? (e as any).data : null,
            }),
          }),
        },
        onError: {
          target: "idle",
          actions: assign({
            error: (_, e) => (e && "data" in e ? (e as any).data?.message || "MCP similar evidence error" : "Unknown error"),
          }),
        },
      },
    },
    mcpGettingAnalytics: {
      invoke: {
        src: "mcpGetAnalytics",
        input: ({ context }) => ({
          userId: context.userId,
          caseId: context.caseId,
        }),
        onDone: {
          target: "idle",
          actions: assign({
            mcpResults: ({ context }, e) => ({
              ...context.mcpResults,
              analytics: e && "data" in e ? (e as any).data : null,
            }),
          }),
        },
        onError: {
          target: "idle",
          actions: assign({
            error: (_, e) => (e && "data" in e ? (e as any).data?.message || "MCP analytics error" : "Unknown error"),
          }),
        },
      },
    },
  },
});

// Enhanced service implementations with MCP integration
export const agentShellServicesMCP = {
  // Enhanced agent call with MCP context
  callAgentWithMCP: async ({ input, userId, caseId }: { input: string; userId?: string; caseId?: string }) => {
    try {
      // First, gather relevant context using MCP tools
      let contextData = {};

      if (caseId) {
        // Load case data and related evidence
        const caseResult = await mcpTools.cases.loadCases({ userId, limit: 1, offset: 0 });
        if (caseResult.success && caseResult.data) {
          contextData = { ...contextData, currentCase: caseResult.data[0] };
        }

        const evidenceResult = await mcpTools.evidence.loadEvidence({ caseId, limit: 10 });
        if (evidenceResult.success && evidenceResult.data) {
          contextData = { ...contextData, evidence: evidenceResult.data };
        }
      }

      if (userId) {
        // Load user profile
        const userResult = await mcpTools.users.getUserById(userId);
        if (userResult.success && userResult.data) {
          contextData = { ...contextData, userProfile: userResult.data };
        }
      }

      // Enhance prompt with MCP context
      const enhancedInput = `${input}\n\nContext: ${JSON.stringify(contextData)}`;

      // Use production service client with enhanced context
      const response = await services.queryRAG(enhancedInput, { userId, caseId });
      return response.response || response.data?.response || 'No response';
    } catch (error: any) {
      console.error("Enhanced agent call failed:", error);
      throw error;
    }
  },

  // Enhanced semantic search with MCP integration
  performSemanticSearchWithMCP: async ({ query, userId, caseId }: { query: string; userId: string; caseId?: string }) => {
    try {
      // Use MCP tools for vector similarity search
      const promises = [];

      if (caseId) {
        // Search for similar evidence in current case
        promises.push(mcpTools.evidence.loadEvidence({
          caseId,
          query,
          limit: 5
        }));
      } else {
        // Search across all user cases
        promises.push(mcpTools.cases.loadCases({
          userId,
          query,
          limit: 5
        }));
      }

      const [searchResults] = await Promise.all(promises);

      // Combine with traditional RAG search
      const ragResponse = await services.queryRAG(`semantic_search: ${query}`, { userId, caseId });

      return {
        ...ragResponse,
        mcpResults: searchResults,
        enhancedContext: true
      };
    } catch (error: any) {
      console.error("Enhanced semantic search failed:", error);
      throw error;
    }
  },

  // Enhanced file upload with MCP integration
  performFileUploadWithMCP: async ({ file, userId, caseId }: { file: File; userId: string; caseId?: string }) => {
    try {
      // Upload file using production service
      const uploadResponse = await services.uploadFile(file, { userId, caseId });

      // If successful and we have a case ID, create evidence record
      if (uploadResponse.success && caseId) {
        const evidenceResult = await mcpTools.evidence.createEvidence({
          caseId,
          title: file.name,
          description: `Uploaded file: ${file.name}`,
          evidenceType: 'document',
          tags: ['uploaded']
        });

        return {
          ...uploadResponse,
          evidenceCreated: evidenceResult.success,
          evidenceId: evidenceResult.data?.id
        };
      }

      return uploadResponse;
    } catch (error: any) {
      console.error("Enhanced file upload failed:", error);
      throw error;
    }
  },

  // Enhanced health check with MCP integration
  checkServiceHealthWithMCP: async () => {
    try {
      const healthChecks = await Promise.allSettled([
        productionServiceClient.checkAllServicesHealth(),
        goServiceClient.checkServiceHealth(),
        // Test MCP database connectivity
        mcpTools.users.getUserAnalytics(),
      ]);

      return {
        production: healthChecks[0].status === 'fulfilled' ? healthChecks[0].value : null,
        legacy: healthChecks[1].status === 'fulfilled' ? healthChecks[1].value : null,
        mcpDatabase: healthChecks[2].status === 'fulfilled' ? true : false,
      };
    } catch (error: any) {
      console.error("Enhanced health check failed:", error);
      throw error;
    }
  },

  // MCP service implementations
  mcpLoadCase: async ({ caseId }: { caseId: string }) => {
    const result = await mcpTools.cases.loadCases({ limit: 1, offset: 0 });
    return result;
  },

  mcpLoadCases: async ({ userId }: { userId: string }) => {
    const result = await mcpTools.cases.loadCases({ userId, limit: 20 });
    return result;
  },

  mcpCreateCase: async ({ caseData, userId }: { caseData: any; userId: string }) => {
    const result = await mcpTools.cases.createCase({
      ...caseData,
      userId,
    });
    return result;
  },

  mcpLoadEvidence: async ({ caseId }: { caseId: string }) => {
    const result = await mcpTools.evidence.loadEvidence({ caseId, limit: 50 });
    return result;
  },

  mcpCreateEvidence: async ({ evidenceData, caseId }: { evidenceData: any; caseId: string }) => {
    const result = await mcpTools.evidence.createEvidence({
      ...evidenceData,
      caseId,
    });
    return result;
  },

  mcpFindSimilarCases: async ({ embedding, caseId }: { embedding: number[]; caseId?: string }) => {
    const result = await mcpTools.cases.findSimilarCases(embedding, 10);
    return result;
  },

  mcpFindSimilarEvidence: async ({ embedding, caseId }: { embedding: number[]; caseId?: string }) => {
    const result = await mcpTools.evidence.findSimilarEvidence({
      embedding,
      caseId,
      limit: 10,
      threshold: 0.7
    });
    return result;
  },

  mcpGetAnalytics: async ({ userId, caseId }: { userId?: string; caseId?: string }) => {
    const promises = [
      mcpTools.cases.getCaseAnalytics(userId),
      mcpTools.evidence.getEvidenceAnalytics(caseId),
      mcpTools.users.getUserAnalytics(),
    ];

    const [caseAnalytics, evidenceAnalytics, userAnalytics] = await Promise.allSettled(promises);

    return {
      cases: caseAnalytics.status === 'fulfilled' ? caseAnalytics.value : null,
      evidence: evidenceAnalytics.status === 'fulfilled' ? evidenceAnalytics.value : null,
      users: userAnalytics.status === 'fulfilled' ? userAnalytics.value : null,
    };
  },

  // Legacy action implementations
  acceptPatchAction: ({ context, event }: { context: AgentShellContext; event: any }) => {
    console.log("Accepting patch:", event.patchContent, "for job:", event.jobId);
  },

  rateSuggestionAction: ({ context, event }: { context: AgentShellContext; event: any }) => {
    console.log("Rating suggestion:", event.rating, "for job:", event.jobId, "feedback:", event.feedback);
  },
};