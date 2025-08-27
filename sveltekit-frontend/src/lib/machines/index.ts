// XState Machine Exports
// Centralized export for all state machines

export { default as documentUploadMachine } from './document-upload-machine';
export { default as caseCreationMachine } from './case-creation-machine';
export { default as searchMachine } from './search-machine';
export { default as aiAnalysisMachine } from './ai-analysis-machine';

// Re-export existing machines
export { default as agentShellMachine } from './agentShellMachine';
export { default as legalAIMachine } from './legalAIMachine';
export { default as authMachine } from './auth-machine';
export { default as uploadMachine } from './uploadMachine';
export { default as sessionMachine } from './sessionMachine';

// Export types
export type { DocumentUploadContext } from './document-upload-machine';
export type { CaseCreationContext } from './case-creation-machine';
export type { SearchContext } from './search-machine';
export type { AIAnalysisContext } from './ai-analysis-machine';