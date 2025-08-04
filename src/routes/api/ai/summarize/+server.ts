import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { langchain } from '$lib/ai/langchain.js';
import { multiLayerCache } from '$lib/cache/multi-layer-cache.js';
import { z } from 'zod';

/**
 * AI Summarization API Endpoint
 * Provides document summarization and analysis using LangChain
 */

const summarizeRequestSchema = z.object({
  content: z.string().min(10).max(100000),
  type: z.enum(['summary', 'analysis', 'extraction', 'comparison']).default('summary'),
  options: z.object({
    maxTokens: z.number().int().min(100).max(4000).optional().default(1000),
    includeEntities: z.boolean().optional().default(true),
    includeKeyTerms: z.boolean().optional().default(true),
    includeRisks: z.boolean().optional().default(true),
    includeCompliance: z.boolean().optional().default(false),
    useCache: z.boolean().optional().default(true),
    cacheTime: z.number().int().optional().default(3600),
    language: z.string().optional().default('en')
  }).optional().default({}),
  customPrompt: z.string().optional(),
  comparisonContent: z.string().optional(),
  extractionTemplate: z.string().optional()
});

export const POST: RequestHandler = async ({ request }) => {
  try {
    const body = await request.json();
    
    // Validate request
    const summaryRequest = summarizeRequestSchema.safeParse(body);
    
    if (!summaryRequest.success) {
      return json({
        success: false,
        error: 'Invalid request parameters',
        details: summaryRequest.error.issues
      }, { status: 400 });
    }

    const { content, type, options, customPrompt, comparisonContent, extractionTemplate } = summaryRequest.data;
    const startTime = Date.now();

    // Check cache if enabled
    let cacheKey = '';
    if (options.useCache) {
      cacheKey = generateCacheKey(content, type, options, customPrompt, comparisonContent, extractionTemplate);
      const cached = await multiLayerCache.get(cacheKey);
      
      if (cached) {
        return json({
          success: true,
          type,
          result: cached,
          metadata: {
            processingTime: 0,
            cached: true,
            contentLength: content.length
          }
        });
      }
    }

    let result: any = {};

    switch (type) {
      case 'summary':
        result = await performSummarization(content, options);
        break;
      
      case 'analysis':
        if (!customPrompt) {
          return json({ 
            success: false, 
            error: 'Custom prompt required for analysis type' 
          }, { status: 400 });
        }
        result = await performCustomAnalysis(content, customPrompt);
        break;
      
      case 'extraction':
        if (!extractionTemplate) {
          return json({ 
            success: false, 
            error: 'Extraction template required for extraction type' 
          }, { status: 400 });
        }
        result = await performInformationExtraction(content, extractionTemplate);
        break;
      
      case 'comparison':
        if (!comparisonContent) {
          return json({ 
            success: false, 
            error: 'Comparison content required for comparison type' 
          }, { status: 400 });
        }
        result = await performDocumentComparison(content, comparisonContent);
        break;
      
      default:
        return json({ 
          success: false, 
          error: 'Unsupported analysis type' 
        }, { status: 400 });
    }

    const processingTime = Date.now() - startTime;

    // Cache result if enabled
    if (options.useCache && cacheKey) {
      await multiLayerCache.set(cacheKey, result, {
        type: 'analysis',
        ttl: options.cacheTime,
        tags: ['summarization', type, 'ai']
      });
    }

    return json({
      success: true,
      type,
      result,
      metadata: {
        processingTime,
        cached: false,
        contentLength: content.length,
        options
      }
    });

  } catch (error) {
    console.error('Summarization error:', error);
    
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Summarization failed',
      details: process.env.NODE_ENV === 'development' ? error : undefined
    }, { status: 500 });
  }
};

async function performSummarization(content: string, options: any) {
  const summary = await langchain.summarizeDocument(content, {
    type: content.length > 4000 ? 'map_reduce' : 'stuff',
    maxTokens: options.maxTokens
  });

  const result: any = {
    summary,
    wordCount: content.split(/\s+/).length,
    characterCount: content.length
  };

  // Extract entities if requested
  if (options.includeEntities) {
    const entitiesTemplate = `
      Extract legal entities from this document:
      - Person names (parties, attorneys, witnesses)
      - Organization names (companies, courts, agencies)
      - Legal terms and concepts
      - Dates and deadlines
      - Monetary amounts
      - Locations and jurisdictions
      
      Return as JSON array: [{"type": "person", "value": "John Doe", "confidence": 0.95}]
    `;
    
    try {
      const entities = await langchain.extractInfo(content, entitiesTemplate);
      result.entities = Array.isArray(entities) ? entities : [];
    } catch {
      result.entities = [];
    }
  }

  // Extract key terms if requested
  if (options.includeKeyTerms) {
    const keyTermsTemplate = `
      Extract the 10-15 most important legal terms and concepts from this document.
      Focus on:
      1. Legal terminology and jargon
      2. Important clauses or sections
      3. Key obligations and rights
      4. Critical conditions or requirements
      
      Return as JSON array of strings.
    `;
    
    try {
      const keyTerms = await langchain.extractInfo(content, keyTermsTemplate);
      result.keyTerms = Array.isArray(keyTerms) ? keyTerms : [];
    } catch {
      result.keyTerms = [];
    }
  }

  // Analyze risks if requested
  if (options.includeRisks) {
    const riskTemplate = `
      Analyze this legal document for potential risks and concerns:
      1. Compliance risks
      2. Financial risks
      3. Legal liability risks
      4. Operational risks
      5. Contractual risks
      
      Return as JSON array: [{"type": "compliance", "severity": "high", "description": "Missing required disclosure"}]
    `;
    
    try {
      const risks = await langchain.extractInfo(content, riskTemplate);
      result.risks = Array.isArray(risks) ? risks : [];
    } catch {
      result.risks = [];
    }
  }

  // Check compliance if requested
  if (options.includeCompliance) {
    const complianceTemplate = `
      Review this document for compliance with common legal and regulatory requirements:
      1. Required disclosures and notices
      2. Regulatory compliance (GDPR, HIPAA, SOX, etc.)
      3. Industry standard adherence
      4. Missing required clauses
      
      Return as JSON object with compliance status and recommendations.
    `;
    
    try {
      const compliance = await langchain.extractInfo(content, complianceTemplate);
      result.compliance = compliance;
    } catch {
      result.compliance = { status: 'unknown', recommendations: [] };
    }
  }

  return result;
}

async function performCustomAnalysis(content: string, prompt: string) {
  const analysis = await langchain.extractInfo(content, prompt);
  
  return {
    prompt,
    analysis,
    contentLength: content.length
  };
}

async function performInformationExtraction(content: string, template: string) {
  const extraction = await langchain.extractInfo(content, template);
  
  return {
    template,
    extraction,
    contentLength: content.length
  };
}

async function performDocumentComparison(content1: string, content2: string) {
  const comparison = await langchain.compareDocuments(content1, content2);
  
  return {
    comparison,
    document1Length: content1.length,
    document2Length: content2.length
  };
}

function generateCacheKey(
  content: string, 
  type: string, 
  options: any, 
  customPrompt?: string,
  comparisonContent?: string,
  extractionTemplate?: string
): string {
  const contentHash = btoa(content.substring(0, 100)).replace(/[/+]/g, '_').substring(0, 20);
  const optionsHash = btoa(JSON.stringify(options)).replace(/[/+]/g, '_').substring(0, 10);
  
  let key = `summary:${type}:${contentHash}:${optionsHash}`;
  
  if (customPrompt) {
    const promptHash = btoa(customPrompt).replace(/[/+]/g, '_').substring(0, 10);
    key += `:${promptHash}`;
  }
  
  if (comparisonContent) {
    const compHash = btoa(comparisonContent.substring(0, 100)).replace(/[/+]/g, '_').substring(0, 10);
    key += `:${compHash}`;
  }
  
  if (extractionTemplate) {
    const templateHash = btoa(extractionTemplate).replace(/[/+]/g, '_').substring(0, 10);
    key += `:${templateHash}`;
  }
  
  return key;
}

export const GET: RequestHandler = async ({ url }) => {
  try {
    // Simple summary endpoint for quick text analysis
    const text = url.searchParams.get('text');
    
    if (!text || text.length < 10) {
      return json({ error: 'Text parameter required (minimum 10 characters)' }, { status: 400 });
    }

    if (text.length > 10000) {
      return json({ error: 'Text too long (maximum 10,000 characters for GET requests)' }, { status: 400 });
    }

    const summary = await langchain.summarizeDocument(text);
    
    return json({
      success: true,
      summary,
      wordCount: text.split(/\s+/).length
    });

  } catch (error) {
    console.error('GET summarization error:', error);
    
    return json({
      success: false,
      error: 'Summarization failed'
    }, { status: 500 });
  }
};