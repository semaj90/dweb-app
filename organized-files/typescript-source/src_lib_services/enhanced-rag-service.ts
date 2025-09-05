// Enhanced RAG Service - Main Export
// Re-exports the enhanced RAG semantic analyzer with backwards compatibility

export { 
  EnhancedRAGSemanticAnalyzer as EnhancedRAGService,
  enhancedRAGService 
} from './enhanced-rag-semantic-analyzer';

export type {
  RAGQuery,
  RAGResponse,
  SearchResult,
  LegalAnalysisRequest,
  LegalAnalysisResponse
} from './enhanced-rag-semantic-analyzer';