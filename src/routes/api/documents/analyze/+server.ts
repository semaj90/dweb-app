// @ts-nocheck
import { json } from '@sveltejs/kit';
import { db } from '$lib/database/postgres.js';
import { legalDocuments, type NewLegalDocument } from '$lib/database/schema/legal-documents.js';
import { legalOrchestrator } from '$lib/agents/orchestrator.js';
import { qdrantManager } from '$lib/database/qdrant.js';
import type { RequestHandler } from './$types';

/**
 * Document Analysis API with Vector Embeddings
 * Comprehensive legal document processing and analysis
 */

export const POST: RequestHandler = async ({ request }) => {
  try {
    const {
      content,
      title,
      documentType = 'general',
      jurisdiction = 'federal',
      practiceArea,
      fileName,
      fileSize,
      mimeType
    } = await request.json();

    if (!content || !title) {
      return json({ error: 'Content and title are required' }, { status: 400 });
    }

    // Step 1: Generate file hash for deduplication
    const fileHash = await generateFileHash(content);

    // Check if document already exists
    const existingDoc = await db.query.legalDocuments.findFirst({
      where: (documents: any, { eq }: any) => eq(documents.fileHash, fileHash)
    });

    if (existingDoc) {
      return json({
        message: 'Document already exists',
        documentId: existingDoc.id,
        existingAnalysis: existingDoc.analysisResults
      });
    }

    // Step 2: Create initial document record
    const newDocument: NewLegalDocument = {
      title,
      content,
      documentType: documentType as any,
      jurisdiction,
      practiceArea: practiceArea as any,
      fileHash,
      fileName,
      fileSize,
      mimeType,
      processingStatus: 'processing',
      createdAt: new Date(),
      updatedAt: new Date()
    };

    const [insertedDoc] = await db.insert(legalDocuments)
      .values(newDocument)
      .returning();

    // Step 3: Generate embeddings (async)
    const embeddingsPromise = generateDocumentEmbeddings(content, title);

    // Step 4: Perform AI analysis
    const analysisPromise = analyzeDocumentWithAI(content, documentType, jurisdiction, practiceArea);

    // Wait for both operations
    const [embeddings, analysis] = await Promise.all([embeddingsPromise, analysisPromise]);

    // Step 5: Update document with analysis results and embeddings
    const updatedDoc = await db.update(legalDocuments)
      .set({
        analysisResults: analysis,
        contentEmbedding: embeddings.content as any,
        titleEmbedding: embeddings.title as any,
        processingStatus: 'completed',
        updatedAt: new Date()
      })
      .where((doc) => doc.id === insertedDoc.id)
      .returning();

    // Step 6: Store in vector database
    await qdrantManager.upsertDocument({
      id: insertedDoc.id,
      vector: embeddings.content,
      payload: {
        documentId: insertedDoc.id,
        title,
        documentType,
        jurisdiction,
        practiceArea: practiceArea || 'general',
        content: content.substring(0, 1000), // Store first 1000 chars for search
        metadata: {
          fileName,
          fileSize,
          mimeType,
          analysisConfidence: analysis.confidenceLevel
        },
        timestamp: Date.now()
      }
    });

    return json({
      success: true,
      documentId: insertedDoc.id,
      analysis: analysis,
      embeddings: {
        contentDimensions: embeddings.content.length,
        titleDimensions: embeddings.title.length
      },
      processing: {
        status: 'completed',
        processingTime: Date.now() - new Date(insertedDoc.createdAt).getTime()
      }
    });

  } catch (error: any) {
    console.error('Document analysis error:', error);
    return json(
      { error: 'Document analysis failed', details: (error as any)?.message || "Unknown error" },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const documentId = url.searchParams.get('id');
    const limit = parseInt(url.searchParams.get('limit') || '10');
    const status = url.searchParams.get('status');
    const documentType = url.searchParams.get('type');

    if (documentId) {
      // Get specific document analysis
      const document = await db.query.legalDocuments.findFirst({
        where: (docs, { eq }) => eq(docs.id, documentId)
      });

      if (!document) {
        return json({ error: 'Document not found' }, { status: 404 });
      }

      return json({
        document: {
          id: document.id,
          title: document.title,
          documentType: document.documentType,
          jurisdiction: document.jurisdiction,
          practiceArea: document.practiceArea,
          analysisResults: document.analysisResults,
          processingStatus: document.processingStatus,
          createdAt: document.createdAt,
          updatedAt: document.updatedAt
        }
      });
    }

    // Get list of documents with filters
    let query = db.select({
      id: legalDocuments.id,
      title: legalDocuments.title,
      documentType: legalDocuments.documentType,
      jurisdiction: legalDocuments.jurisdiction,
      practiceArea: legalDocuments.practiceArea,
      processingStatus: legalDocuments.processingStatus,
      createdAt: legalDocuments.createdAt,
      analysisResults: legalDocuments.analysisResults
    }).from(legalDocuments);

    // Apply filters
    if (status) {
      query = query.where((docs) => docs.processingStatus === status as any);
    }

    if (documentType) {
      query = query.where((docs) => docs.documentType === documentType as any);
    }

    const documents = await query
      .orderBy((docs) => docs.createdAt)
      .limit(limit);

    return json({ documents });

  } catch (error: any) {
    console.error('Document retrieval error:', error);
    return json(
      { error: 'Failed to retrieve documents', details: (error as any)?.message || "Unknown error" },
      { status: 500 }
    );
  }
};

// Utility functions

async function generateFileHash(content: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(content);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

async function generateDocumentEmbeddings(
  content: string,
  title: string
): Promise<{ content: number[]; title: number[] }> {
  // This would integrate with your embedding service (Ollama, OpenAI, etc.)
  // For now, return mock embeddings
  
  // In production, call your embedding API:
  // const response = await fetch('http://localhost:11434/api/embeddings', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify({
  //     model: 'nomic-embed-text',
  //     prompt: content
  //   })
  // });
  
  return {
    content: Array.from({ length: 384 }, () => Math.random() - 0.5),
    title: Array.from({ length: 384 }, () => Math.random() - 0.5)
  };
}

async function analyzeDocumentWithAI(
  content: string,
  documentType: string,
  jurisdiction: string,
  practiceArea?: string
): Promise<any> {
  try {
    const analysisPrompt = `
      Analyze the following legal document:
      
      Document Type: ${documentType}
      Jurisdiction: ${jurisdiction}
      Practice Area: ${practiceArea || 'General'}
      
      Content: ${content.substring(0, 4000)}...
      
      Please provide:
      1. Key legal entities mentioned
      2. Important terms and obligations
      3. Potential legal risks
      4. Compliance requirements
      5. Sentiment analysis
      6. Document complexity score (1-10)
      7. Confidence level in analysis (1-10)
    `;

    const result = await legalOrchestrator.orchestrate({
      query: analysisPrompt,
      documentType: documentType as any,
      jurisdiction,
      urgency: 'medium',
      requiresMultiAgent: false,
      enableStreaming: false,
      context: { practiceArea }
    });

    // Parse the analysis response into structured data
    const analysisText = result.synthesizedConclusion;
    
    return {
      entities: extractEntities(analysisText),
      keyTerms: extractKeyTerms(analysisText),
      sentimentScore: extractSentimentScore(analysisText),
      complexityScore: extractComplexityScore(analysisText),
      confidenceLevel: result.confidence,
      extractedDates: extractDates(content),
      extractedAmounts: extractAmounts(content),
      parties: extractParties(content),
      obligations: extractObligations(analysisText),
      risks: extractRisks(analysisText),
      rawAnalysis: analysisText,
      processingTime: result.totalProcessingTime,
      agentUsed: result.primaryResponse.agentName
    };
  } catch (error: any) {
    console.error('AI analysis failed:', error);
    return {
      entities: [],
      keyTerms: [],
      sentimentScore: 0,
      complexityScore: 5,
      confidenceLevel: 1,
      extractedDates: [],
      extractedAmounts: [],
      parties: [],
      obligations: [],
      risks: [],
      error: (error as any)?.message || "Unknown error"
    };
  }
}

// Text extraction utilities
function extractEntities(text: string): Array<{ type: string; value: string; confidence: number }> {
  const entities = [];
  const patterns = {
    person: /([A-Z][a-z]+ [A-Z][a-z]+)/g,
    organization: /([A-Z][a-zA-Z\s&.,-]+(?:Inc\.?|Corp\.?|LLC\.?|Ltd\.?))/g,
    location: /([A-Z][a-zA-Z\s]+ (?:City|County|State|Province))/g
  };

  for (const [type, pattern] of Object.entries(patterns)) {
    const matches = text.match(pattern) || [];
    for (const match of matches) {
      entities.push({
        type,
        value: match.trim(),
        confidence: 0.8
      });
    }
  }

  return entities;
}

function extractKeyTerms(text: string): string[] {
  const legalTerms = [
    'contract', 'agreement', 'liability', 'indemnity', 'warranty',
    'breach', 'damages', 'termination', 'jurisdiction', 'arbitration',
    'confidentiality', 'intellectual property', 'force majeure'
  ];

  return legalTerms.filter(term => 
    text.toLowerCase().includes(term.toLowerCase())
  );
}

function extractSentimentScore(text: string): number {
  // Simple sentiment analysis based on keyword presence
  const positiveWords = ['agree', 'benefit', 'advantage', 'profit', 'gain'];
  const negativeWords = ['breach', 'violation', 'penalty', 'damages', 'liability'];
  
  const positive = positiveWords.filter(word => text.toLowerCase().includes(word)).length;
  const negative = negativeWords.filter(word => text.toLowerCase().includes(word)).length;
  
  return (positive - negative) / (positive + negative + 1);
}

function extractComplexityScore(text: string): number {
  // Simple complexity scoring based on text characteristics
  const sentences = text.split(/[.!?]+/).length;
  const avgWordLength = text.split(/\s+/).reduce((sum, word) => sum + word.length, 0) / text.split(/\s+/).length;
  const legalTermDensity = extractKeyTerms(text).length / sentences;
  
  return Math.min(10, Math.max(1, Math.round(avgWordLength * legalTermDensity * sentences / 100)));
}

function extractDates(text: string): string[] {
  const datePattern = /\b(?:\d{1,2}\/\d{1,2}\/\d{4}|\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b/g;
  return text.match(datePattern) || [];
}

function extractAmounts(text: string): string[] {
  const amountPattern = /\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?/g;
  return text.match(amountPattern) || [];
}

function extractParties(text: string): string[] {
  const partyPattern = /\b(?:party|parties|plaintiff|defendant|client|customer|vendor|contractor)\b:?\s*([A-Z][a-zA-Z\s&.,-]+)/gi;
  const matches = text.match(partyPattern) || [];
  return matches.map(match => match.replace(/^(party|parties|plaintiff|defendant|client|customer|vendor|contractor):?\s*/i, '').trim());
}

function extractObligations(text: string): string[] {
  const obligationPattern = /\b(?:shall|must|will|agrees? to|obligated to|required to)\s+([^.!?]+)/gi;
  const matches = text.match(obligationPattern) || [];
  return matches.map(match => match.trim());
}

function extractRisks(text: string): Array<{ type: string; severity: 'low' | 'medium' | 'high'; description: string }> {
  const riskKeywords = {
    high: ['breach', 'violation', 'penalty', 'damages', 'termination'],
    medium: ['liability', 'indemnity', 'warranty', 'compliance'],
    low: ['notice', 'amendment', 'renewal', 'extension']
  };

  const risks = [];
  for (const [severity, keywords] of Object.entries(riskKeywords)) {
    for (const keyword of keywords) {
      if (text.toLowerCase().includes(keyword)) {
        risks.push({
          type: 'contractual',
          severity: severity as 'low' | 'medium' | 'high',
          description: `Document contains ${keyword}-related provisions`
        });
      }
    }
  }

  return risks;
}