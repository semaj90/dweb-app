// Compatibility layer: provide both camelCase and snake_case exports for schema modules
export * from './legal-documents';

// Named aliases for snake_case imports used across the codebase
import * as docs from './legal-documents';
const _docs: any = docs;

export const legal_documents = _docs.legalDocuments;
export const legal_cases = _docs.legalCases;
export const case_documents = _docs.caseDocuments;
export const legal_entities = _docs.legalEntities;
export const agent_analysis_cache = _docs.agentAnalysisCache;
export const embedding_cache = _docs.embeddingCache;

// Default export for modules that import default
export default {
  legalDocuments: _docs.legalDocuments,
  legalCases: _docs.legalCases,
};
