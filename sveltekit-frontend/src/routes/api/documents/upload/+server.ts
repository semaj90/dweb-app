import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { parseStringPromise } from "xml2js";
import pdf from "pdf-parse";
import { writeFile, mkdir } from 'fs/promises';
import { randomUUID } from 'crypto';
import path from 'path';
import { db } from '$lib/server/db/index';
import { enhancedEvidence } from '$lib/server/db/enhanced-legal-schema';
import { qdrantService } from '$lib/services/qdrantService';
// @ts-ignore
import xss from "xss"; // XSS protection

// Ensure temp directory exists
const TEMP_DIR = path.join(process.cwd(), 'temp-uploads');

import { ollamaService } from '$lib/services/ollamaService';
import { eq } from 'drizzle-orm';

// XSS-safe text extraction from XML
async function processXml(buffer: Buffer): Promise<string> {
  const xmlString = buffer.toString("utf-8");
  const result = await parseStringPromise(xmlString, {
    explicitArray: false,
    ignoreAttrs: true,
  });
  // TODO: Customize extraction for your XML structure
  const rawText = JSON.stringify(result);
  return xss(rawText); // sanitize output
}

// XSS-safe text extraction from PDF
async function processPdf(buffer: Buffer): Promise<string> {
  const data = await pdf(buffer);
  return xss(data.text); // sanitize output
}

// Enhanced legal analysis using local gemma3-legal model
async function analyzeWithGemma3Legal(
  text: string,
  verbose: boolean,
  thinking: boolean
): Promise<any> {
  try {
    // Use local gemma3-legal model for legal document analysis
    const prompt = `Analyze this legal document and provide:
1. Document type classification
2. Key parties and entities
3. Important dates and monetary amounts
4. Risk assessment (0-100)
5. Legal precedent relevance
6. ${verbose ? 'Detailed analysis' : 'Summary'}

Document text: ${text.substring(0, 4000)}`;

    const analysis = await ollamaService.generateCompletion(prompt);

    // Parse the AI response for structured data
    const entities = extractEntitiesFromAnalysis(analysis);
    const riskScore = extractRiskScore(analysis);
    const caseType = extractCaseType(analysis);

    if (thinking) {
      // Enhanced analysis with deeper reasoning
      const contextPrompt = `Based on your analysis of this document, provide:
1. Similar legal precedents
2. Potential legal issues
3. Recommended actions
4. Confidence assessment

Previous analysis: ${analysis}`;

      const contextAnalysis = await ollamaService.generateCompletion(contextPrompt);

      return {
        summary: analysis,
        synthesizedAnalysis: contextAnalysis,
        entities,
        riskScore,
        caseType,
        confidenceScore: 0.92,
        aiModelVersion: 'gemma3-legal',
        processingStatus: 'completed'
      };
    }

    return {
      summary: analysis,
      entities,
      riskScore,
      caseType,
      confidenceScore: 0.88,
      aiModelVersion: 'gemma3-legal',
      processingStatus: 'completed'
    };
  } catch (error) {
    console.error('Gemma3 Legal analysis failed:', error);
    return {
      summary: 'AI analysis temporarily unavailable',
      entities: { parties: [], dates: [], monetary: [], clauses: [], jurisdictions: [], caseTypes: [] },
      riskScore: 0,
      caseType: 'unknown',
      confidenceScore: 0.0,
      aiModelVersion: 'gemma3-legal',
      processingStatus: 'error'
    };
  }
}

// Helper functions for parsing AI analysis
function extractEntitiesFromAnalysis(analysisText: string) {
  // Simple regex-based entity extraction - can be enhanced with NER
  const parties = (analysisText.match(/parties?[:\s]+([^\n.]+)/gi) || []).map(m => m.split(':')[1]?.trim()).filter(Boolean);
  const dates = (analysisText.match(/\b\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}\b/g) || []);
  const monetary = (analysisText.match(/\$[\d,]+(?:\.\d{2})?/g) || []);
  
  return {
    parties: parties.slice(0, 10), // Limit results
    dates: dates.slice(0, 10),
    monetary: monetary.slice(0, 10),
    clauses: [],
    jurisdictions: [],
    caseTypes: []
  };
}

function extractRiskScore(analysisText: string): number {
  const riskMatch = analysisText.match(/risk[:\s]+(\d+)/i);
  return riskMatch ? parseInt(riskMatch[1]) : 25; // Default medium-low risk
}

function extractCaseType(analysisText: string): string {
  const lowerText = analysisText.toLowerCase();
  if (lowerText.includes('contract')) return 'contract';
  if (lowerText.includes('litigation')) return 'litigation';
  if (lowerText.includes('regulatory')) return 'regulatory';
  if (lowerText.includes('compliance')) return 'compliance';
  return 'document';
}

export const POST: RequestHandler = async ({ request }) => {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const verbose = formData.get("verbose") === "true";
    const thinking = formData.get("thinking") === "true";

    if (!file) {
      return json({ error: "No file uploaded" }, { status: 400 });
    }

    // File validation: limit size, check type, sanitize name
    if (file.size > 10 * 1024 * 1024) {
      return json({ error: "File too large (max 10MB)" }, { status: 400 });
    }
    const allowedTypes = ["application/pdf", "text/xml", "application/xml"];
    if (!allowedTypes.includes(file.type) && !file.name.endsWith(".xml")) {
      return json({ error: "Unsupported file type" }, { status: 400 });
    }
    const safeFileName = xss(file.name);

    const buffer = Buffer.from(await file.arrayBuffer());
    let textContent = "";

    if (file.type === "application/pdf") {
      textContent = await processPdf(buffer);
    } else if (
      file.type === "text/xml" ||
      file.type === "application/xml" ||
      file.name.endsWith(".xml")
    ) {
      textContent = await processXml(buffer);
    } else {
      return json({ error: "Unsupported file type" }, { status: 400 });
    }

    // XSS sanitize extracted text
    textContent = xss(textContent);

    // Generate embedding using local nomic-embed-text via Ollama
    let embedding;
    try {
      const embeddingResponse = await ollamaService.generateEmbedding(
        textContent.substring(0, 2000) // Limit for embedding
      );
      embedding = embeddingResponse;
    } catch (error) {
      console.error('Embedding generation failed:', error);
      embedding = new Array(384).fill(0); // Fallback zero vector
    }

    // Perform AI analysis with gemma3-legal
    const analysis = await analyzeWithGemma3Legal(textContent, verbose, thinking);
    
    const docId = randomUUID();

    // Store in enhanced evidence table with proper schema
    const evidenceRecord = await db.insert(enhancedEvidence).values({
      id: docId,
      caseId: randomUUID(), // Generate case ID or get from form data
      title: safeFileName || "Uploaded Document",
      content: textContent,
      summary: analysis.summary,
      caseType: analysis.caseType,
      jurisdiction: 'federal', // Default, could be extracted from analysis
      entities: analysis.entities,
      tags: [],
      riskScore: analysis.riskScore,
      confidenceScore: analysis.confidenceScore.toString(),
      embedding: embedding,
      processingStatus: analysis.processingStatus,
      aiModelVersion: analysis.aiModelVersion,
      createdBy: randomUUID(), // Get from session or form
    }).returning();

    // Store in Qdrant for hybrid vector search
    try {
      await qdrantService.client.upsert('legal_documents', {
        wait: true,
        points: [{
          id: docId,
          vector: embedding,
          payload: {
            title: safeFileName || "Document",
            content: textContent.substring(0, 1000), // Store excerpt
            type: "document",
            case_id: randomUUID(),
            metadata: {
              caseType: analysis.caseType,
              jurisdiction: 'federal',
              riskScore: analysis.riskScore,
              entities: analysis.entities
            }
          }
        }]
      });
      
      console.log('✅ Document stored in Qdrant successfully');
        
    } catch (qdrantError) {
      console.error('Qdrant storage failed:', qdrantError);
      // Continue without Qdrant - PostgreSQL vector storage is still available
    }

    // TODO: Log analytics event for document upload and analysis
    
    // Return successful response with analysis
    const response = {
      id: docId,
      fileName: safeFileName,
      title: safeFileName || "Uploaded Document",
      summary: analysis.summary,
      entities: analysis.entities,
      riskScore: analysis.riskScore,
      caseType: analysis.caseType,
      confidenceScore: analysis.confidenceScore,
      aiModelVersion: analysis.aiModelVersion,
      processingStatus: analysis.processingStatus,
      synthesizedAnalysis: analysis.synthesizedAnalysis || null
    };

    return json(response);
  } catch (error: any) {
    // TODO: Log error to error tracking system/markdown
    console.error("Upload error:", error);
    return json(
      { error: error.message || "Failed to process file" },
      { status: 500 }
    );
  }
};

// TODO: Add POST endpoints for voice-to-text and TTS (voice output)
// TODO: Add endpoints for user analytics, recommendations, and enhanced RAG
// TODO: Add SSR/context hydration and golden ratio layout support
// TODO: Add backup/infra checks for Qdrant, Neo4j, Nomic, Ollama, pgvector, service workers
