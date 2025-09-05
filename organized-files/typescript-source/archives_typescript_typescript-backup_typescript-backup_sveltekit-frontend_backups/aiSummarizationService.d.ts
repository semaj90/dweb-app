export interface AIAnalysis {
  extractedTags: string[];
  summary: string;
}

export interface CaseSummaryReport {
  content: string;
  richTextContent: string;
  metadata: Record<string, any>;
  canvasElements: unknown[];
}

export interface ProsecutionStrategyReport {
  content: string;
  richTextContent: string;
  metadata: Record<string, any>;
  canvasElements: unknown[];
}

export interface AISummarizationService {
  analyzeEvidence: (evidence: unknown) => Promise<AIAnalysis>;
  generateCaseSummary: (data: unknown) => Promise<CaseSummaryReport>;
  generateProsecutionStrategy: (
    data: unknown
  ) => Promise<ProsecutionStrategyReport>;
}
