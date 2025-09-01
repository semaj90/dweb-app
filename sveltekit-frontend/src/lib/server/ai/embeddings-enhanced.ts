
// Enhanced Embeddings Service with Nomic Embed + Langchain + Langextract
// Local embeddings using Ollama nomic-embed-text model with document processing
// Use process.env for server-side environment variables

import { cacheEmbedding, getCachedEmbedding } from "$lib/server/cache/redis";

export interface EnhancedEmbeddingOptions {
  provider?: "auto" | "nomic-embed" | "tauri-legal-bert" | "tauri-bert";
  cache?: boolean;
  maxTokens?: number;
  legalDomain?: boolean;
  batchSize?: number;
  useExtraction?: boolean;
}

export interface EmbeddingResult {
  embedding: number[];
  metadata: {
    provider: string;
    model: string;
    textLength: number;
    generatedAt: string;
    extracted?: unknown;
  };
}

/**
 * Generate embeddings using Ollama nomic-embed-text model
 */
async function generateNomicEmbedding(text: string): Promise<number[]> {
  const ollamaEndpoint = import.meta.env.OLLAMA_BASE_URL || 'http://localhost:11434';
  
  try {
    const response = await fetch(`${ollamaEndpoint}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: text
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama embedding failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.embedding;
  } catch (error: any) {
    console.error('Nomic embedding generation failed:', error);
    throw error;
  }
}

/**
 * Extract structured data from legal documents using langextract patterns
 */
async function extractDocumentStructure(text: string): Promise<any> {
  // Simple extraction patterns for legal documents
  const patterns = {
    parties: /(?:party|plaintiff|defendant|client):\s*([^.\n]+)/gi,
    dates: /\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b/g,
    amounts: /\$[\d,]+(?:\.\d{2})?/g,
    caseNumbers: /(?:case|docket)\s*(?:no\.?|#)?\s*([a-z0-9\-]+)/gi,
    sections: /(?:section|§)\s*(\d+(?:\.\d+)*)/gi,
  };

  const extracted = {
    parties: [],
    dates: [],
    amounts: [],
    caseNumbers: [],
    sections: [],
    documentType: detectDocumentType(text),
    keyPhrases: extractKeyPhrases(text)
  };

  for (const [key, pattern] of Object.entries(patterns)) {
    const matches = Array.from(text.matchAll(pattern as RegExp));
    (extracted as any)[key] = matches.map(match => match[1] || match[0]).slice(0, 10);
  }

  return extracted;
}

function detectDocumentType(text: string): string {
  const lowerText = text.toLowerCase();
  
  if (lowerText.includes('contract') || lowerText.includes('agreement')) {
    return 'contract';
  }
  if (lowerText.includes('complaint') || lowerText.includes('petition')) {
    return 'pleading';
  }
  if (lowerText.includes('motion') || lowerText.includes('brief')) {
    return 'motion';
  }
  if (lowerText.includes('deposition') || lowerText.includes('transcript')) {
    return 'deposition';
  }
  if (lowerText.includes('evidence') || lowerText.includes('exhibit')) {
    return 'evidence';
  }
  
  return 'general';
}

function extractKeyPhrases(text: string): string[] {
  // Simple key phrase extraction
  const legalTerms = [
    'breach of contract', 'negligence', 'damages', 'liability', 'indemnification',
    'force majeure', 'intellectual property', 'confidentiality', 'termination',
    'arbitration', 'jurisdiction', 'governing law', 'attorney fees'
  ];
  
  const foundTerms = legalTerms.filter(term => 
    text.toLowerCase().includes(term.toLowerCase())
  );
  
  return foundTerms.slice(0, 5);
}

/**
 * Main embedding generation function with langchain-style processing
 */
export async function generateEnhancedEmbedding(
  text: string | string[],
  options: EnhancedEmbeddingOptions = {},
): Promise<number[] | number[][]> {
  const {
    provider = "nomic-embed",
    cache = true,
    maxTokens = 8000,
    legalDomain = true,
    batchSize = 10,
    useExtraction = false,
  } = options;

  if (!text) {
    throw new Error("Text is required for embedding generation");
  }

  const isArray = Array.isArray(text);
  const inputs = isArray ? text : [text];
  const truncatedInputs = inputs.map((t) =>
    t.length > maxTokens ? t.substring(0, maxTokens) : t,
  );

  // Check cache for single inputs
  if (cache && !isArray) {
    const cacheKey = `nomic-${legalDomain}-${useExtraction}`;
    const cachedEmbedding = await getCachedEmbedding(
      truncatedInputs[0],
      cacheKey,
    );
    if (cachedEmbedding) {
      return cachedEmbedding;
    }
  }

  try {
    const results: number[][] = [];
    
    for (const input of truncatedInputs) {
      // Extract document structure if requested
      let processedText = input;
      if (useExtraction && legalDomain) {
        const extracted = await extractDocumentStructure(input);
        // Enhance text with extracted metadata
        processedText = `${input}\n\n[METADATA] Document Type: ${extracted.documentType}, Key Terms: ${extracted.keyPhrases.join(', ')}`;
      }
      
      // Generate embedding using nomic-embed-text
      const embedding = await generateNomicEmbedding(processedText);
      results.push(embedding);
      
      // Cache single results
      if (cache && !isArray) {
        const cacheKey = `nomic-${legalDomain}-${useExtraction}`;
        await cacheEmbedding(input, embedding, cacheKey);
      }
    }

    return isArray ? results : results[0];
  } catch (error: any) {
    console.error("Enhanced embedding generation failed:", error);
    throw error;
  }
}

/**
 * Batch embedding generation with progress tracking
 */
export async function generateBatchEmbeddingsEnhanced(
  texts: string[],
  options: EnhancedEmbeddingOptions = {},
  onProgress?: (completed: number, total: number) => void,
): Promise<number[][]> {
  const { batchSize = 10 } = options;
  const results: number[][] = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);

    try {
      const batchResult = (await generateEnhancedEmbedding(
        batch,
        options,
      )) as number[][];
      results.push(...batchResult);

      if (onProgress) {
        onProgress(Math.min(i + batchSize, texts.length), texts.length);
      }
    } catch (error: any) {
      console.error(`Batch ${i}-${i + batchSize} failed:`, error);
      // Add empty embeddings for failed items
      for (let j = 0; j < batch.length; j++) {
        results.push(new Array(384).fill(0)); // nomic-embed-text uses 384 dimensions
      }
    }
    
    // Small delay between batches to avoid rate limits
    if (i + batchSize < texts.length) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  
  return results;
}

/**
 * Legal document-specific embedding with metadata and extraction
 */
export async function generateLegalEmbedding(
  documentText: string,
  metadata: {
    documentType?: "contract" | "case_law" | "statute" | "brief" | "other";
    jurisdiction?: string;
    subject?: string[];
  } = {},
): Promise<{
  embedding: number[];
  metadata: any;
  confidence: number;
  extracted?: unknown;
}> {
  // Extract document structure
  const extracted = await extractDocumentStructure(documentText);
  
  // Generate embedding with extracted context
  const embedding = (await generateEnhancedEmbedding(documentText, {
    provider: "nomic-embed",
    legalDomain: true,
    maxTokens: 2000,
    useExtraction: true,
  })) as number[];

  return {
    embedding,
    metadata: {
      ...metadata,
      generatedAt: new Date().toISOString(),
      provider: "nomic-embed",
      model: "nomic-embed-text",
      documentLength: documentText.length,
      dimensions: 384,
    },
    confidence: 0.85, // Default confidence for nomic-embed
    extracted,
  };
}

/**
 * Similarity calculation between legal documents
 */
export async function calculateLegalSimilarity(
  doc1: string,
  doc2: string,
): Promise<number> {
  const embeddings = (await generateEnhancedEmbedding([doc1, doc2], {
    provider: "nomic-embed",
    legalDomain: true,
    useExtraction: true,
  })) as number[][];

  return cosineSimilarity(embeddings[0], embeddings[1]);
}

/**
 * Cosine similarity calculation
 */
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new Error("Vectors must have same length for similarity calculation");
  }
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Backward compatibility exports
 */
export async function generateEmbedding(
  text: string,
  model?: string,
): Promise<number[]> {
  const result = await generateEnhancedEmbedding(text, {
    provider: "nomic-embed",
    legalDomain: true,
  });

  return Array.isArray(result) && Array.isArray(result[0])
    ? (result[0] as number[])
    : (result as number[]);
}

export async function generateBatchEmbeddings(
  texts: string[],
  model?: string,
  batchSize: number = 10,
): Promise<number[][]> {
  return generateBatchEmbeddingsEnhanced(texts, {
    provider: "nomic-embed",
    legalDomain: true,
    batchSize,
  });
}

/**
 * Langchain-style document processing with chunking
 */
export async function processDocumentWithChunking(
  document: string,
  chunkSize: number = 1000,
  chunkOverlap: number = 200,
): Promise<{
  chunks: { text: string; embedding: number[]; metadata: any }[];
  documentMetadata: any;
}> {
  const extracted = await extractDocumentStructure(document);
  const chunks: { text: string; embedding: number[]; metadata: any }[] = [];
  
  // Split document into overlapping chunks
  for (let i = 0; i < document.length; i += chunkSize - chunkOverlap) {
    const chunk = document.slice(i, i + chunkSize);
    if (chunk.trim().length < 50) continue; // Skip very small chunks
    
    const embedding = await generateNomicEmbedding(chunk);
    
    chunks.push({
      text: chunk,
      embedding,
      metadata: {
        chunkIndex: Math.floor(i / (chunkSize - chunkOverlap)),
        startIndex: i,
        endIndex: Math.min(i + chunkSize, document.length),
        length: chunk.length,
      }
    });
  }
  
  return {
    chunks,
    documentMetadata: {
      totalLength: document.length,
      totalChunks: chunks.length,
      extracted,
      processedAt: new Date().toISOString(),
    }
  };
}