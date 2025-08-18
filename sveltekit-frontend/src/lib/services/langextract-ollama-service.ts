// Note: langextract package may not be available, using mock implementation
// import langextract from 'langextract';

// Mock langextract interface for development
interface LangExtract {
  extract(options: {
    text_or_documents: string;
    prompt_description: string;
    examples: Array<{ input: string; output: any }>;
    model_id: string;
    model_url: string;
  }): Promise<{
    extracted_data: any;
    confidence: number;
    processing_time: number;
  }>;
}

// Mock implementation that falls back to Ollama direct API
const langextract: LangExtract = {
  async extract(options) {
    // Direct Ollama API call as fallback
    const response = await fetch(`${options.model_url}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: options.model_id,
        prompt: `${options.prompt_description}\n\nText to extract from:\n${options.text_or_documents}`,
        stream: false
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }

    const data = await response.json();

    return {
      extracted_data: data.response,
      confidence: 0.8, // Default confidence
      processing_time: 1000 // Mock processing time
    };
  }
};

/**
 * LangExtract + Ollama Integration Service
 * Provides local LLM processing for legal document extraction
 */

export interface LegalExtractionRequest {
  text: string;
  documentType: 'contract' | 'case_law' | 'statute' | 'evidence' | 'motion' | 'brief';
  extractionType: 'entities' | 'summary' | 'key_terms' | 'obligations' | 'risks' | 'dates';
  model?: string;
  examples?: Array<{ input: string; output: any }>;
}

export interface LegalExtractionResult {
  extracted_data: any;
  confidence: number;
  processing_time: number;
  model_used: string;
  document_type: string;
  extraction_type: string;
}

export class LangExtractOllamaService {
  private ollamaUrl: string;
  private defaultModel: string;

  constructor(ollamaUrl = 'http://localhost:11434', defaultModel = 'gemma2:2b') {
    this.ollamaUrl = ollamaUrl;
    this.defaultModel = defaultModel;
  }

  /**
   * Extract legal entities from document text
   */
  async extractLegalEntities(request: LegalExtractionRequest): Promise<LegalExtractionResult> {
    const startTime = Date.now();

    const prompt = this.buildExtractionPrompt(request);
    const examples = this.getLegalExamples(request.documentType, request.extractionType);

    try {
      const result = await langextract.extract({
        text_or_documents: request.text,
        prompt_description: prompt,
        examples: examples,
        model_id: request.model || this.defaultModel,
        model_url: this.ollamaUrl
      });

      return {
        extracted_data: result,
        confidence: this.calculateConfidence(result, request),
        processing_time: Date.now() - startTime,
        model_used: request.model || this.defaultModel,
        document_type: request.documentType,
        extraction_type: request.extractionType
      };
    } catch (error) {
      console.error('LangExtract processing error:', error);
      throw new Error(`LangExtract processing failed: ${error}`);
    }
  }

  /**
   * Extract contract terms and obligations
   */
  async extractContractTerms(text: string, model?: string): Promise<LegalExtractionResult> {
    return this.extractLegalEntities({
      text,
      documentType: 'contract',
      extractionType: 'obligations',
      model
    });
  }

  /**
   * Extract case law citations and holdings
   */
  async extractCaseLawCitations(text: string, model?: string): Promise<LegalExtractionResult> {
    return this.extractLegalEntities({
      text,
      documentType: 'case_law',
      extractionType: 'entities',
      model
    });
  }

  /**
   * Extract key dates from legal documents
   */
  async extractLegalDates(text: string, documentType: LegalExtractionRequest['documentType'] = 'contract', model?: string): Promise<LegalExtractionResult> {
    return this.extractLegalEntities({
      text,
      documentType,
      extractionType: 'dates',
      model
    });
  }

  /**
   * Generate legal document summary
   */
  async generateLegalSummary(text: string, documentType: LegalExtractionRequest['documentType'], model?: string): Promise<LegalExtractionResult> {
    return this.extractLegalEntities({
      text,
      documentType,
      extractionType: 'summary',
      model
    });
  }

  /**
   * Extract risk factors from legal documents
   */
  async extractRiskFactors(text: string, documentType: LegalExtractionRequest['documentType'] = 'contract', model?: string): Promise<LegalExtractionResult> {
    return this.extractLegalEntities({
      text,
      documentType,
      extractionType: 'risks',
      model
    });
  }

  /**
   * Batch process multiple documents
   */
  async batchExtract(requests: LegalExtractionRequest[]): Promise<LegalExtractionResult[]> {
    const results: LegalExtractionResult[] = [];

    for (const request of requests) {
      try {
        const result = await this.extractLegalEntities(request);
        results.push(result);
      } catch (error) {
        console.error(`Batch extraction failed for ${request.documentType}:`, error);
        results.push({
          extracted_data: null,
          confidence: 0,
          processing_time: 0,
          model_used: request.model || this.defaultModel,
          document_type: request.documentType,
          extraction_type: request.extractionType
        });
      }
    }

    return results;
  }

  /**
   * Build extraction prompt based on document type and extraction type
   */
  private buildExtractionPrompt(request: LegalExtractionRequest): string {
    const basePrompts = {
      contract: {
        entities: "Extract all legal entities, parties, dates, monetary amounts, and key terms from this contract.",
        summary: "Provide a comprehensive summary of this contract including parties, key obligations, terms, and conditions.",
        key_terms: "Extract all key terms, definitions, and important clauses from this contract.",
        obligations: "Extract all obligations, duties, and responsibilities for each party in this contract.",
        risks: "Identify potential risks, liabilities, and problematic clauses in this contract.",
        dates: "Extract all dates, deadlines, and time-sensitive provisions from this contract."
      },
      case_law: {
        entities: "Extract case citations, court names, judges, parties, legal issues, and holdings from this case law.",
        summary: "Summarize this case including the facts, legal issues, holding, and reasoning.",
        key_terms: "Extract key legal terms, precedents, and doctrines from this case.",
        obligations: "Extract any legal obligations or duties established by this case.",
        risks: "Identify legal risks and potential precedential impacts of this case.",
        dates: "Extract all relevant dates from this case including filing dates, hearing dates, and decision dates."
      },
      statute: {
        entities: "Extract statutory sections, definitions, penalties, and requirements from this statute.",
        summary: "Provide a summary of this statute including its purpose, scope, and key provisions.",
        key_terms: "Extract definitions, key terms, and important statutory provisions.",
        obligations: "Extract all legal obligations, duties, and compliance requirements from this statute.",
        risks: "Identify potential penalties, violations, and compliance risks under this statute.",
        dates: "Extract effective dates, deadlines, and time-sensitive provisions from this statute."
      },
      evidence: {
        entities: "Extract all relevant facts, names, dates, locations, and evidence markers from this document.",
        summary: "Summarize this evidence including its relevance, credibility, and key facts.",
        key_terms: "Extract key facts, technical terms, and important details from this evidence.",
        obligations: "Extract any procedural requirements or evidentiary standards from this document.",
        risks: "Identify potential admissibility issues or evidentiary problems.",
        dates: "Extract all timestamps, event dates, and chronological information from this evidence."
      },
      motion: {
        entities: "Extract parties, legal standards, arguments, and relief sought from this motion.",
        summary: "Summarize this motion including the relief sought, legal basis, and key arguments.",
        key_terms: "Extract legal standards, procedural requirements, and key arguments.",
        obligations: "Extract any procedural obligations or requirements mentioned in this motion.",
        risks: "Identify potential weaknesses or counterarguments to this motion.",
        dates: "Extract filing deadlines, hearing dates, and time-sensitive requirements."
      },
      brief: {
        entities: "Extract legal arguments, citations, facts, and legal standards from this brief.",
        summary: "Summarize this brief including the main arguments, legal theory, and conclusion.",
        key_terms: "Extract key legal arguments, precedents, and persuasive points.",
        obligations: "Extract any legal obligations or standards discussed in this brief.",
        risks: "Identify potential weaknesses in the legal arguments presented.",
        dates: "Extract all relevant dates and deadlines mentioned in this brief."
      }
    };

    return basePrompts[request.documentType]?.[request.extractionType] ||
           "Extract relevant legal information from this document.";
  }

  /**
   * Get examples for better extraction accuracy
   */
  private getLegalExamples(documentType: string, extractionType: string): Array<{ input: string; output: any }> {
    const examples = {
      contract_entities: [
        {
          input: "This Agreement is entered into on January 15, 2024, between ABC Corp., a Delaware corporation, and XYZ LLC, a California limited liability company.",
          output: {
            parties: ["ABC Corp. (Delaware corporation)", "XYZ LLC (California limited liability company)"],
            dates: ["January 15, 2024"],
            entity_types: ["Delaware corporation", "California limited liability company"]
          }
        }
      ],
      case_law_entities: [
        {
          input: "In Smith v. Jones, 123 F.3d 456 (9th Cir. 2023), the court held that contracts must be interpreted in favor of the non-drafting party.",
          output: {
            case_name: "Smith v. Jones",
            citation: "123 F.3d 456 (9th Cir. 2023)",
            court: "9th Circuit Court of Appeals",
            year: "2023",
            holding: "contracts must be interpreted in favor of the non-drafting party"
          }
        }
      ]
    };

    const key = `${documentType}_${extractionType}`;
    return examples[key as keyof typeof examples] || [];
  }

  /**
   * Calculate confidence score based on extraction quality
   */
  private calculateConfidence(result: any, request: LegalExtractionRequest): number {
    if (!result) return 0;

    let confidence = 0.5; // Base confidence

    // Boost confidence based on extraction type
    if (request.extractionType === 'entities' && Array.isArray(result.entities)) {
      confidence += Math.min(result.entities.length * 0.1, 0.3);
    }

    if (request.extractionType === 'summary' && typeof result === 'string' && result.length > 100) {
      confidence += 0.2;
    }

    if (request.extractionType === 'dates' && Array.isArray(result.dates)) {
      confidence += Math.min(result.dates.length * 0.15, 0.25);
    }

    // Boost confidence for legal-specific models
    if (request.model?.includes('legal')) {
      confidence += 0.1;
    }

    return Math.min(confidence, 1.0);
  }

  /**
   * Check if Ollama service is available
   */
  async isOllamaAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/version`);
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * List available models
   */
  async listAvailableModels(): Promise<string[]> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`);
      if (!response.ok) return [];

      const data = await response.json();
      return data.models?.map((model: any) => model.name) || [];
    } catch {
      return [];
    }
  }

  /**
   * Pull a model if not available
   */
  async ensureModel(modelName: string): Promise<boolean> {
    try {
      const models = await this.listAvailableModels();
      if (models.includes(modelName)) {
        return true;
      }

      // Try to pull the model
      const response = await fetch(`${this.ollamaUrl}/api/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: modelName })
      });

      return response.ok;
    } catch {
      return false;
    }
  }
}

// Export singleton instance
export const langExtractService = new LangExtractOllamaService();