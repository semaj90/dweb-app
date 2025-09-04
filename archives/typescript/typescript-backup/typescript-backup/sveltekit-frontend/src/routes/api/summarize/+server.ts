/**
 * LangChain-Powered Document Summarization API
 * 
 * This endpoint handles long legal documents using LangChain's map-reduce strategy:
 * 1. Split long documents into chunks
 * 2. Summarize each chunk individually (map)
 * 3. Combine chunk summaries into final summary (reduce)
 * 
 * Integrates with your Ollama gemma3-legal model for legal-specific analysis
 */

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// LangChain imports for document processing
import { ChatOpenAI } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { loadSummarizationChain } from 'langchain/chains';
import { PromptTemplate } from '@langchain/core/prompts';

// Environment configuration - connects to your Ollama setup
const OLLAMA_BASE_URL = 'http://localhost:11434/v1';
const LEGAL_MODEL = 'gemma3-legal'; // Your trained legal model

export const POST: RequestHandler = async ({ request }) => {
  try {
    // 1. Parse request body
    const { text, options = {} } = await request.json();

    if (!text || typeof text !== 'string') {
      return json({ 
        error: 'Text content is required.',
        code: 'INVALID_INPUT'
      }, { status: 400 });
    }

    if (text.length < 100) {
      return json({
        error: 'Text must be at least 100 characters for summarization.',
        code: 'TEXT_TOO_SHORT'
      }, { status: 400 });
    }

    const {
      summaryLength = 'medium', // short, medium, long
      includeKeyTerms = true,
      includeLegalAnalysis = true,
      temperature = 0.3,
      chunkSize = 2000,
      chunkOverlap = 200
    } = options;

    console.log(`[Summarize] Processing ${text.length} characters with LangChain map-reduce`);

    // 2. Initialize connection to your Ollama legal model
    const llm = new ChatOpenAI({
      modelName: LEGAL_MODEL,
      temperature,
      openAIApiKey: 'not-needed', // Ollama doesn't need real API key
      configuration: {
        baseURL: OLLAMA_BASE_URL,
      },
      // Legal-specific configuration
      maxTokens: summaryLength === 'short' ? 150 : summaryLength === 'medium' ? 300 : 500,
    });

    // 3. Create legal-optimized text splitter
    // This respects legal document structure (paragraphs, sections, etc.)
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize,
      chunkOverlap,
      separators: [
        '\n\n',      // Paragraph breaks (highest priority)
        '\n',        // Line breaks  
        '. ',        // Sentence endings
        '? ',        // Question endings
        '! ',        // Exclamation endings
        '; ',        // Semicolons (common in legal text)
        ', ',        // Commas
        ' ',         // Spaces (lowest priority)
      ]
    });

    // 4. Split document into manageable chunks
    const docs = await textSplitter.createDocuments([text]);
    console.log(`[Summarize] Split into ${docs.length} chunks for processing`);

    if (docs.length === 0) {
      return json({
        error: 'Failed to split document into processable chunks.',
        code: 'CHUNK_ERROR'
      }, { status: 500 });
    }

    // 5. Create custom prompts for legal document summarization
    const mapPrompt = PromptTemplate.fromTemplate(`
You are an expert legal analyst. Summarize the following legal text chunk concisely, focusing on:
- Key legal concepts and principles
- Important parties, dates, and references
- Critical legal implications or arguments
- Relevant statutes or case law mentioned

Legal text chunk:
{text}

Concise summary:`);

    const combinePrompt = PromptTemplate.fromTemplate(`
You are an expert legal analyst. Combine the following chunk summaries into a comprehensive final summary.

Create a well-structured summary that includes:
1. **Overview**: Main legal issue or document purpose
2. **Key Points**: Most important legal arguments or facts  
3. **Legal Analysis**: Relevant laws, precedents, or implications
4. **Conclusion**: Key outcomes or recommendations

Chunk summaries to combine:
{text}

Final comprehensive summary:`);

    // 6. Load and configure the summarization chain
    // Using map-reduce strategy for handling long documents
    const summarizationChain = loadSummarizationChain(llm, {
      type: 'map_reduce',
      mapPrompt,
      combinePrompt,
      // Combine documents prompt for final step
      combineDocumentsPrimary: combinePrompt,
      returnIntermediateSteps: false, // Set to true for debugging
    });

    console.log('[Summarize] Running LangChain map-reduce summarization...');
    const startTime = Date.now();

    // 7. Execute the summarization chain
    const result = await summarizationChain.invoke({
      input_documents: docs,
    });

    const processingTime = Date.now() - startTime;
    console.log(`[Summarize] Completed in ${processingTime}ms`);

    // 8. Post-process the summary for additional legal insights
    let enhancedSummary = result.text;
    const metadata: any = {
      originalLength: text.length,
      summaryLength: enhancedSummary.length,
      compressionRatio: (text.length / enhancedSummary.length).toFixed(2),
      processingTime,
      chunksProcessed: docs.length,
      model: LEGAL_MODEL,
      timestamp: new Date().toISOString()
    };

    // 9. Extract key legal terms if requested
    if (includeKeyTerms) {
      try {
        const keyTermsPrompt = `Extract the 10 most important legal terms, concepts, or entities from this summary. Return as a JSON array of strings:

${enhancedSummary}

Key terms:`;
        
        const keyTermsResponse = await llm.invoke(keyTermsPrompt);
        
        // Parse legal terms (handle potential JSON parsing issues gracefully)
        try {
          const terms = JSON.parse(keyTermsResponse.content as string);
          if (Array.isArray(terms)) {
            metadata.keyLegalTerms = terms.slice(0, 10); // Limit to 10 terms
          }
        } catch {
          // Fallback: extract terms from text manually
          metadata.keyLegalTerms = extractLegalTermsFromText(enhancedSummary);
        }
      } catch (error: any) {
        console.warn('[Summarize] Failed to extract key terms:', error);
      }
    }

    // 10. Generate legal risk analysis if requested
    if (includeLegalAnalysis) {
      try {
        const riskAnalysisPrompt = `Analyze the legal risk level of this document summary. Rate from 1-5 (1=low risk, 5=high risk) and explain briefly:

${enhancedSummary}

Risk analysis:`;
        
        const riskResponse = await llm.invoke(riskAnalysisPrompt);
        metadata.legalRiskAnalysis = riskResponse.content;
      } catch (error: any) {
        console.warn('[Summarize] Failed to generate legal analysis:', error);
      }
    }

    // 11. Return the complete summarization result
    return json({
      success: true,
      summary: enhancedSummary,
      metadata,
      usage: {
        inputTokens: Math.ceil(text.length / 4), // Rough token estimate
        outputTokens: Math.ceil(enhancedSummary.length / 4),
        model: LEGAL_MODEL,
        processingTime
      }
    });

  } catch (error: any) {
    console.error('[Summarize] LangChain error:', error);
    
    // Handle specific LangChain/Ollama errors gracefully
    let errorMessage = 'Failed to summarize document';
    let errorCode = 'PROCESSING_ERROR';
    
    if (error instanceof Error) {
      if (error.message.includes('ECONNREFUSED')) {
        errorMessage = 'Local LLM service (Ollama) is not available. Please ensure it is running on port 11434.';
        errorCode = 'LLM_UNAVAILABLE';
      } else if (error.message.includes('model')) {
        errorMessage = `Model "${LEGAL_MODEL}" is not available. Please ensure gemma3-legal is installed in Ollama.`;
        errorCode = 'MODEL_UNAVAILABLE';
      } else {
        errorMessage = error.message;
      }
    }

    return json({
      error: errorMessage,
      code: errorCode,
      details: process.env.NODE_ENV === 'development' ? error : undefined
    }, { status: 500 });
  }
};

/**
 * Fallback function to extract legal terms when LLM parsing fails
 */
function extractLegalTermsFromText(text: string): string[] {
  const legalTermRegex = /\b(statute|regulation|ordinance|contract|agreement|liability|damages|negligence|breach|violation|compliance|jurisdiction|precedent|defendant|plaintiff|evidence|testimony|witness|court|judge|jury|appeal|motion|discovery|settlement|verdict|judgment|injunction|subpoena|deposition|arbitration|mediation)\b/gi;
  
  const matches = text.match(legalTermRegex) || [];
  const uniqueTerms = [...new Set(matches.map(term => term.toLowerCase()))];
  
  return uniqueTerms.slice(0, 10);
}

// Optional: Add a GET handler for API documentation
export const GET: RequestHandler = async () => {
  return json({
    service: 'Legal Document Summarization API',
    description: 'LangChain-powered map-reduce summarization for long legal documents',
    model: LEGAL_MODEL,
    endpoint: '/api/summarize',
    methods: ['POST'],
    example: {
      request: {
        text: 'Your long legal document content...',
        options: {
          summaryLength: 'medium',
          includeKeyTerms: true,
          includeLegalAnalysis: true,
          temperature: 0.3
        }
      },
      response: {
        success: true,
        summary: 'Generated summary...',
        metadata: {
          keyLegalTerms: ['contract', 'liability', 'breach'],
          legalRiskAnalysis: 'Risk level: 2/5...'
        }
      }
    }
  });
};