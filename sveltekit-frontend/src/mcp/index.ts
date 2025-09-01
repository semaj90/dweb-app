/**
 * MCP Tools Index
 * Barrel export for all MCP database operation tools
 * Provides clean import pattern for the Legal AI Platform
 */

// Export MCP Tools
import { CasesMCPTool } from './cases.mcp';
import { EvidenceMCPTool } from './evidence.mcp';
import { UsersMCPTool } from './users.mcp';
import { AIAnalysisMCPTool } from './ai-analysis.mcp';
import type { MCPToolResponse } from './cases.mcp';

// Expect per-tool files to export singleton instances; if not, create them here
import { CasesMCPTool as _CasesMCPTool } from './cases.mcp';
export const casesMCPTool = new _CasesMCPTool();
import { EvidenceMCPTool as _EvidenceMCPTool } from './evidence.mcp';
export const evidenceMCPTool = new _EvidenceMCPTool();
import { UsersMCPTool as _UsersMCPTool } from './users.mcp';
export const usersMCPTool = new _UsersMCPTool();
import { AIAnalysisMCPTool as _AIAnalysisMCPTool } from './ai-analysis.mcp';
export const aiAnalysisMCPTool = new _AIAnalysisMCPTool();

// Export shared interfaces
export type {
  MCPToolResponse
} from './cases.mcp';

// Export specific parameter interfaces for each tool
export type {
  CaseCreateParams,
  CaseUpdateParams,
  CaseSearchParams,
  EvidenceAddParams
} from './cases.mcp';

export type {
  EvidenceCreateParams,
  EvidenceUpdateParams,
  EvidenceSearchParams,
  EvidenceVectorSearchParams
} from './evidence.mcp';

export type {
  UserCreateParams,
  UserUpdateParams,
  UserSearchParams,
  UserProfileMatchParams
} from './users.mcp';

export type {
  DocumentAnalysisParams,
  LegalAnalysisParams,
  SimilaritySearchParams,
  BatchAnalysisParams,
  RiskAssessmentParams
} from './ai-analysis.mcp';

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