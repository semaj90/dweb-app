/**
 * MCP Tools Index
 * Barrel export for all MCP database operation tools
 * Provides clean import pattern for the Legal AI Platform
 */

// Export MCP Tools
export { CasesMCPTool, casesMCPTool } from './cases.mcp.js';
export { EvidenceMCPTool, evidenceMCPTool } from './evidence.mcp.js';
export { UsersMCPTool, usersMCPTool } from './users.mcp.js';
export { AIAnalysisMCPTool, aiAnalysisMCPTool } from './ai-analysis.mcp.js';

// Export shared interfaces
export type {
  MCPToolResponse
} from './cases.mcp.js';

// Export specific parameter interfaces for each tool
export type {
  CaseCreateParams,
  CaseUpdateParams,
  CaseSearchParams,
  EvidenceAddParams
} from './cases.mcp.js';

export type {
  EvidenceCreateParams,
  EvidenceUpdateParams,
  EvidenceSearchParams,
  EvidenceVectorSearchParams
} from './evidence.mcp.js';

export type {
  UserCreateParams,
  UserUpdateParams,
  UserSearchParams,
  UserProfileMatchParams
} from './users.mcp.js';

export type {
  DocumentAnalysisParams,
  LegalAnalysisParams,
  SimilaritySearchParams,
  BatchAnalysisParams,
  RiskAssessmentParams
} from './ai-analysis.mcp.js';

// Utility type for all MCP tools
export interface MCPToolsCollection {
  cases: CasesMCPTool;
  evidence: EvidenceMCPTool;
  users: UsersMCPTool;
  aiAnalysis: AIAnalysisMCPTool;
}

// Singleton collection of all MCP tools
export const mcpTools: MCPToolsCollection = {
  cases: casesMCPTool,
  evidence: evidenceMCPTool,
  users: usersMCPTool,
  aiAnalysis: aiAnalysisMCPTool
};

/**
 * Helper function to get standardized metadata for MCP operations
 */
export function createMCPMetadata(tool: string, operation: string): Record<string, any> {
  return {
    tool: `${tool}.${operation}`,
    timestamp: Date.now(),
    version: '1.0',
    platform: 'legal-ai-sveltekit'
  };
}

/**
 * Helper function to standardize MCP error responses
 */
export function createMCPError(
  tool: string, 
  operation: string, 
  error: string | Error,
  additionalMetadata?: Record<string, any>
): MCPToolResponse<never> {
  return {
    success: false,
    error: error instanceof Error ? error.message : error,
    metadata: {
      ...createMCPMetadata(tool, operation),
      ...additionalMetadata
    }
  };
}

/**
 * Helper function to standardize MCP success responses
 */
export function createMCPSuccess<T>(
  tool: string,
  operation: string,
  data: T,
  additionalMetadata?: Record<string, any>
): MCPToolResponse<T> {
  return {
    success: true,
    data,
    metadata: {
      ...createMCPMetadata(tool, operation),
      ...additionalMetadata
    }
  };
}