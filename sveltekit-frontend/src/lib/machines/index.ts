// XState Machine Exports
// Centralized export for all state machines

export { default as documentUploadMachine } from './document-upload-machine';
export { default as caseCreationMachine } from './case-creation-machine';
export { default as searchMachine } from './search-machine';
export { default as aiAnalysisMachine } from './ai-analysis-machine';

// Re-export existing machines
export { agentShellMachine } from './agentShellMachine';
export { legalAIMachine } from './legalAIMachine';
export { authMachine } from './auth-machine';
export { uploadMachine } from './uploadMachine';
export { sessionMachine } from './sessionMachine';

// Export types
export type { DocumentUploadContext } from './document-upload-machine';
export type { CaseCreationContext } from './case-creation-machine';
export type { SearchContext } from './search-machine';
export type { AIAnalysisContext } from './ai-analysis-machine';