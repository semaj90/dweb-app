// Re-export schema
export * from '../database/schema/legal-documents';

// Also export common aliases for backward compatibility
export { 
  legalDocuments as documents,
  legalCases as cases,
  legalEntities as entities,
  caseDocuments,
  agentAnalysisCache
} from '../database/schema/legal-documents';