import { langExtractService, type LegalExtractionRequest } from "$lib/services/langextract-ollama-service";
import type { RequestHandler } from './$types';


/**
 * LangExtract + Ollama API Endpoint
 * Provides local LLM processing for legal document extraction
 */

export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json() as LegalExtractionRequest & {
      action?: 'extract' | 'contract_terms' | 'case_citations' | 'dates' | 'summary' | 'risks' | 'batch';
      requests?: LegalExtractionRequest[];
    };

    // Validate required fields
    if (!body.text && !body.requests) {
      return json({
        success: false,
        error: 'Missing required field: text or requests'
      }, { status: 400 });
    }

    // Check if Ollama is available
    const isAvailable = await langExtractService.isOllamaAvailable();
    if (!isAvailable) {
      return json({
        success: false,
        error: 'Ollama service not available. Please ensure Ollama is running on http://localhost:11434'
      }, { status: 503 });
    }

    let result;

    switch (body.action) {
      case 'contract_terms':
        result = await langExtractService.extractContractTerms(body.text, body.model);
        break;

      case 'case_citations':
        result = await langExtractService.extractCaseLawCitations(body.text, body.model);
        break;

      case 'dates':
        result = await langExtractService.extractLegalDates(body.text, body.documentType, body.model);
        break;

      case 'summary':
        result = await langExtractService.generateLegalSummary(body.text, body.documentType || 'contract', body.model);
        break;

      case 'risks':
        result = await langExtractService.extractRiskFactors(body.text, body.documentType, body.model);
        break;

      case 'batch':
        if (!body.requests || !Array.isArray(body.requests)) {
          return json({
            success: false,
            error: 'Batch processing requires requests array'
          }, { status: 400 });
        }
        result = await langExtractService.batchExtract(body.requests);
        break;

      case 'extract':
      default:
        result = await langExtractService.extractLegalEntities(body);
        break;
    }

    return json({
      success: true,
      data: result,
      timestamp: new Date().toISOString(),
      service: 'langextract-ollama'
    });

  } catch (error: any) {
    console.error('LangExtract API error:', error);
    
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

export const GET: RequestHandler = async () => {
  try {
    // Check service status
    const isAvailable = await langExtractService.isOllamaAvailable();
    const models = await langExtractService.listAvailableModels();

    return json({
      success: true,
      status: {
        ollama_available: isAvailable,
        available_models: models,
        service_url: 'http://localhost:11434',
        langextract_version: 'latest'
      },
      capabilities: {
        document_types: ['contract', 'case_law', 'statute', 'evidence', 'motion', 'brief'],
        extraction_types: ['entities', 'summary', 'key_terms', 'obligations', 'risks', 'dates'],
        actions: ['extract', 'contract_terms', 'case_citations', 'dates', 'summary', 'risks', 'batch']
      },
      examples: {
        contract_extraction: {
          method: 'POST',
          url: '/api/legal-ai/langextract',
          body: {
            action: 'contract_terms',
            text: 'This Agreement is entered into on January 15, 2024...',
            documentType: 'contract',
            model: 'gemma2:2b'
          }
        },
        case_citation_extraction: {
          method: 'POST',
          url: '/api/legal-ai/langextract',
          body: {
            action: 'case_citations',
            text: 'In Smith v. Jones, 123 F.3d 456 (9th Cir. 2023)...',
            documentType: 'case_law'
          }
        },
        batch_processing: {
          method: 'POST',
          url: '/api/legal-ai/langextract',
          body: {
            action: 'batch',
            requests: [
              {
                text: 'Contract text...',
                documentType: 'contract',
                extractionType: 'obligations'
              },
              {
                text: 'Case law text...',
                documentType: 'case_law',
                extractionType: 'entities'
              }
            ]
          }
        }
      }
    });
  } catch (error: any) {
    return json({
      success: false,
      error: 'Failed to get service status',
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};