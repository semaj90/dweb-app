import { eq, sql as drizzleSql, and, gte } from "drizzle-orm";
// Fallback schema import - will gracefully degrade if schema not available
let schema: any;
try {
  schema = require("$lib/server/db/unified-schema");
} catch {
  // Provide minimal fallback schema structure
  schema = {
    documents: { title: "", content: "", titleEmbedding: [], contentEmbedding: [], metadata: {} },
    documentVectors: {},
    evidence: {},
    cases: {}
  };
}
import Redis from "ioredis";
import { createHash } from "crypto";
// RAG Pipeline with PostgreSQL + pgvector + LangChain + Ollama
// (Header line previously corrupted; cleaned.)

import { Ollama } from "@langchain/community/llms/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { StringOutputParser } from '@langchain/core/output_parsers';
import type { Document as LangChainDocument } from "@langchain/core/documents";
const postgres = require("postgres");
import { drizzle } from "drizzle-orm/postgres-js";

// Import schema directly (same path used across project). If it fails at runtime we degrade gracefully.

// Configuration
const EMBEDDING_MODEL = 'nomic-embed-text:latest';
const EMBEDDING_DIMENSIONS = 768;
const LLM_MODEL = 'gemma3-legal:latest';
const OLLAMA_BASE_URL = process.env.OLLAMA_URL || 'http://localhost:11434';

// Initialize PostgreSQL connection
const sql = postgres({
  host: process.env.DATABASE_HOST || 'localhost',
  port: parseInt(process.env.DATABASE_PORT || '5432'),
  database: process.env.DATABASE_NAME || 'legal_ai_db',
  username: process.env.DATABASE_USER || 'legal_admin',
  password: process.env.DATABASE_PASSWORD || '123456',
  max: 20,
  idle_timeout: 20,
  prepare: true,
  ssl: process.env.NODE_ENV === 'production' ? 'require' : false,
});

const db = drizzle(sql, { schema });

// Initialize Redis for caching
const redis = new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: parseInt(process.env.REDIS_PORT || '6379'),
  db: parseInt(process.env.REDIS_DB || '0'),
  maxRetriesPerRequest: 3,
  enableReadyCheck: true,
  lazyConnect: false,
  retryStrategy: (times) => Math.min(times * 50, 2000),
});

// Initialize LangChain components
const embeddings = new OllamaEmbeddings({
  baseUrl: OLLAMA_BASE_URL,
  model: EMBEDDING_MODEL,
  requestOptions: {
    useMMap: true,
    numThread: 8,
  },
});

const llm = new Ollama({
  baseUrl: OLLAMA_BASE_URL,
  model: LLM_MODEL,
  temperature: 0.3, // Lower for legal accuracy
  numCtx: 8192,
  numPredict: 2048,
  topK: 40,
  topP: 0.9,
  repeatPenalty: 1.1,
  callbacks: [
    {
      handleLLMStart: async (llm, prompts) => {
        console.log(`[RAG] LLM Started: ${LLM_MODEL}`);
      },
      handleLLMEnd: async (output) => {
        console.log(`[RAG] LLM Completed`);
      },
      handleLLMError: async (err) => {
        console.error('[RAG] LLM Error:', err);
      },
    },
  ],
});

// Text splitter for legal documents
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1500, // Larger for legal context
  chunkOverlap: 300,
  separators: [
    '\n\nSECTION',
    '\n\nARTICLE',
    '\n\nCLAUSE', // Legal sections
    '\n\n§',
    '\n\n¶', // Legal symbols
    '\n\n',
    '\n',
    '.',
    '!',
    '?',
    ';',
    ':',
    ' ',
    '',
  ],
  keepSeparator: true,
});

export class LegalRAGPipeline {
  private initialized = false;

  async initialize() {
    if (this.initialized) return;

    // Test database connection
    const testResult = await sql`SELECT 1 as test`;
    console.log('[RAG] Database connected:', testResult[0].test === 1);

    // Test Redis connection
    await redis.set('health-check', 'ok');
    console.log('[RAG] Redis connected');

    // Test Ollama connection
    const testEmbedding = await embeddings.embedQuery('test');
    console.log('[RAG] Embeddings working:', testEmbedding.length === EMBEDDING_DIMENSIONS);

    this.initialized = true;
  }

  // === DOCUMENT INGESTION ===

  async ingestLegalDocument(params: {
    title: string;
    content: string;
    documentType: string;
    metadata?: Record<string, any>;
    caseId?: string;
    userId: string;
  }) {
    const startTime = Date.now();
    const { title, content, documentType, metadata = {}, caseId, userId } = params;

    try {
      // 1. Create main document record
      const [document] = await db
        .insert(schema.legalDocuments)
        .values({
          title,
          content: content.substring(0, 10000), // Store first 10k chars for preview
          fullText: content,
          documentType,
          keywords: metadata.keywords || [],
          topics: metadata.topics || [],
          jurisdiction: metadata.jurisdiction,
          caseId,
          createdBy: userId,
        })
        .returning();

      console.log(`[RAG] Created document: ${document.id}`);

      // 2. Generate document-level embedding
      const docEmbedding = await this.generateEmbedding(`${title}\n${content.substring(0, 2000)}`);

      // Store as JSON string for now (pgvector requires array format)
      await db
        .update(schema.legalDocuments)
        .set({ embedding: JSON.stringify(docEmbedding) })
        .where(eq(schema.legalDocuments.id, document.id));

      // 3. Split into chunks for detailed retrieval
      const chunks = await this.smartLegalChunking(content);
      console.log(`[RAG] Split into ${chunks.length} chunks`);

      // 4. Process chunks in batches
      const BATCH_SIZE = 10;
      for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
        const batch = chunks.slice(i, i + BATCH_SIZE);

        const chunkRecords = await Promise.all(
          batch.map(async (chunk, idx) => {
            const embedding = await this.generateEmbedding(chunk);

            return {
              documentId: document.id,
              documentType,
              chunkIndex: i + idx,
              content: chunk,
              embedding: JSON.stringify(embedding),
              metadata: {
                title,
                position: i + idx,
                totalChunks: chunks.length,
                ...metadata,
              },
            };
          })
        );

        await db.insert(schema.documentChunks).values(chunkRecords);
        console.log(
          `[RAG] Processed batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(chunks.length / BATCH_SIZE)}`
        );
      }

      // 5. Auto-generate tags using AI
      const tags = await this.generateAutoTags(content, documentType);

      for (const tag of tags) {
        await db.insert(schema.autoTags).values({
          entityId: document.id,
          entityType: 'document',
          tag: tag.tag,
          confidence: tag.confidence.toString(),
          source: 'ai_analysis',
          model: LLM_MODEL,
        });
      }

      const processingTime = Date.now() - startTime;
      console.log(`[RAG] Document ingestion completed in ${processingTime}ms`);

      return {
        documentId: document.id,
        chunksCreated: chunks.length,
        tags: tags.map((t) => t.tag),
        processingTime,
      };
    } catch (error: any) {
      console.error('[RAG] Ingestion error:', error);
      throw error;
    }
  }

  // === RETRIEVAL & SEARCH ===

  async hybridSearch(params: {
    query: string;
    caseId?: string;
    documentType?: string;
    limit?: number;
    threshold?: number;
  }): Promise<LangChainDocument[]> {
    const { query, caseId, documentType, limit = 10, threshold = 0.5 } = params;

    try {
      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(query);

      // Perform vector similarity search
      const vectorResults = await sql`
        SELECT
          dc.id,
          dc.content,
          dc.metadata,
          dc.document_id,
          1 - (dc.embedding::vector <=> ${JSON.stringify(queryEmbedding)}::vector) as similarity
        FROM document_chunks dc
        WHERE
          1 - (dc.embedding::vector <=> ${JSON.stringify(queryEmbedding)}::vector) > ${threshold}
          ${caseId ? sql`AND dc.metadata->>'caseId' = ${caseId}` : sql``}
          ${documentType ? sql`AND dc.document_type = ${documentType}` : sql``}
        ORDER BY dc.embedding::vector <=> ${JSON.stringify(queryEmbedding)}::vector
        LIMIT ${limit * 2}
      `;

      // Perform keyword search
      const keywordResults = await sql`
        SELECT
          dc.id,
          dc.content,
          dc.metadata,
          ts_rank(to_tsvector('english', dc.content),
                  plainto_tsquery('english', ${query})) as text_rank
        FROM document_chunks dc
        WHERE
          to_tsvector('english', dc.content) @@ plainto_tsquery('english', ${query})
          ${caseId ? sql`AND dc.metadata->>'caseId' = ${caseId}` : sql``}
          ${documentType ? sql`AND dc.document_type = ${documentType}` : sql``}
        ORDER BY text_rank DESC
        LIMIT ${limit}
      `;

      // Combine and deduplicate results
      const combinedResults = new Map<string, any>();

      // Add vector results with higher weight
      vectorResults.forEach((r) => {
        combinedResults.set(r.id, {
          ...r,
          score: r.similarity * 0.7,
        });
      });

      // Add or update with keyword results
      keywordResults.forEach((r) => {
        const existing = combinedResults.get(r.id);
        if (existing) {
          existing.score += r.text_rank * 0.3;
        } else {
          combinedResults.set(r.id, {
            ...r,
            score: r.text_rank * 0.3,
          });
        }
      });

      // Sort by combined score and convert to Documents
      const sortedResults = Array.from(combinedResults.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);

      return sortedResults.map(
        (r): LangChainDocument => ({
            pageContent: r.content,
            metadata: {
              ...r.metadata,
              documentId: r.document_id,
              score: r.score,
              similarity: r.similarity || 0,
              textRank: r.text_rank || 0,
            },
          })
      );
    } catch (error: any) {
      console.error('[RAG] Search error:', error);
      throw error;
    }
  }

  // === QUESTION ANSWERING ===

  async answerLegalQuestion(params: {
    question: string;
    caseId?: string;
    userId: string;
    conversationContext?: string;
  }) {
    const startTime = Date.now();
    const { question, caseId, userId, conversationContext } = params;

    try {
      // 1. Retrieve relevant context
      const relevantDocs = await this.hybridSearch({
        query: question,
        caseId,
        limit: 5,
        threshold: 0.6,
      });

      if (relevantDocs.length === 0) {
        return {
          answer:
            "I couldn't find relevant information in the knowledge base to answer your question. Please provide more context or try rephrasing your question.",
          sources: [],
          confidence: 0,
        };
      }

      // 2. Build context from retrieved documents
      const context = relevantDocs
        .map((doc, idx) => `[Source ${idx + 1}]:\n${doc.pageContent}`)
        .join('\n\n---\n\n');

      // 3. Create prompt with legal context
      const promptTemplate = PromptTemplate.fromTemplate(`
You are a legal AI assistant with expertise in legal analysis. Answer the question based ONLY on the provided context.

${conversationContext ? `Previous Conversation Context:\n${conversationContext}\n\n` : ''}

Legal Context:
{context}

Question: {question}

Instructions:
1. Provide a clear, accurate answer based on the context
2. Cite specific sources using [Source N] notation
3. Identify any legal principles or precedents mentioned
4. Note any important caveats or limitations
5. If the context doesn't fully answer the question, clearly state what information is missing

Answer:
      `);

      // 4. Create chain and generate answer
      const chain = RunnableSequence.from([
        {
          context: () => context,
          question: new RunnablePassthrough(),
        },
        promptTemplate,
        llm,
        new StringOutputParser(),
      ]);

      const chainResult = await chain.invoke(question);
      const answer = typeof chainResult === 'string' ? chainResult : chainResult.parse || '';

      // 5. Extract confidence and key points
      const analysis = await this.analyzeAnswer(answer, relevantDocs);

      // 6. Log the query for future improvement
      const queryEmbedding = await this.generateEmbedding(question);

      await db.insert(schema.userAiQueries).values({
        userId,
        caseId,
        query: question,
        response: answer,
        model: LLM_MODEL,
        queryType: 'legal_research',
        confidence: analysis.confidence.toString(),
        processingTime: Date.now() - startTime,
        contextUsed: relevantDocs.map((d) => d.metadata.documentId),
        embedding: JSON.stringify(queryEmbedding),
        metadata: {
          sourcesCount: relevantDocs.length,
          keyPoints: analysis.keyPoints,
        },
      });

      return {
        answer,
        sources: relevantDocs.map((d) => ({
          id: d.metadata.documentId,
          title: d.metadata.title,
          score: d.metadata.score,
        })),
        confidence: analysis.confidence,
        keyPoints: analysis.keyPoints,
        processingTime: Date.now() - startTime,
      };
    } catch (error: any) {
      console.error('[RAG] QA error:', error);

      // Log failed query
      await db.insert(schema.userAiQueries).values({
        userId,
        caseId,
        query: question,
        response: '',
        model: LLM_MODEL,
        isSuccessful: false,
        errorMessage: error.message,
        processingTime: Date.now() - startTime,
      });

      throw error;
    }
  }

  // === LEGAL ANALYSIS CHAINS ===

  async analyzeContract(contractText: string) {
    const contractPrompt = PromptTemplate.fromTemplate(`
You are a legal expert specializing in contract analysis. Analyze the following contract and provide a structured assessment.

Contract:
{contract}

Provide your analysis in the following format:

1. CONTRACT TYPE & PARTIES
- Type of contract
- Parties involved
- Governing law/jurisdiction

2. KEY TERMS & OBLIGATIONS
- Primary obligations of each party
- Payment terms
- Duration and termination clauses
- Deliverables/milestones

3. RISK ASSESSMENT
- Potential risks for each party
- Liability limitations
- Indemnification clauses
- Force majeure provisions

4. LEGAL ISSUES
- Ambiguous terms requiring clarification
- Potential enforceability issues
- Missing standard clauses
- Compliance considerations

5. RECOMMENDATIONS
- Suggested modifications
- Points for negotiation
- Additional clauses to consider

Provide specific clause references where applicable.
    `);

    const chain = RunnableSequence.from([contractPrompt, llm, new StringOutputParser()]);

    const chainResult = await chain.invoke({ contract: contractText });
    const analysis = typeof chainResult === 'string' ? chainResult : chainResult.parse || '';
    return this.parseContractAnalysis(analysis);
  }

  async correlateEvidence(evidenceIds: string[]) {
    // Fetch evidence content
    const evidenceRecords = await db
      .select()
      .from(schema.evidence)
      .where(drizzleSql`${schema.evidence.id} = ANY(${evidenceIds})`);

    const correlationPrompt = PromptTemplate.fromTemplate(`
As a legal analyst, examine the relationships between these pieces of evidence:

${evidenceRecords
  .map(
    (e, i) => `Evidence ${i + 1} (${e.title}):
${e.description}
${e.summary || ''}`
  )
  .join('\n\n')}

Provide a comprehensive analysis covering:

1. DIRECT CONNECTIONS
- Explicit relationships between evidence items
- Common entities, dates, or locations
- Causal relationships

2. TIMELINE ANALYSIS
- Chronological sequence of events
- Temporal gaps or overlaps
- Critical time periods

3. CONTRADICTIONS & INCONSISTENCIES
- Conflicting information
- Gaps in the narrative
- Unexplained discrepancies

4. CORROBORATION
- Mutually supporting evidence
- Independent verification points
- Strength of combined evidence

5. LEGAL SIGNIFICANCE
- Combined probative value
- Impact on case theory
- Potential weaknesses

6. RECOMMENDED ACTIONS
- Additional evidence needed
- Further investigation required
- Strategic considerations

Analysis:
    `);

    const chain = RunnableSequence.from([correlationPrompt, llm, new StringOutputParser()]);

    return await chain.invoke({});
  }

  // === HELPER METHODS ===

  private async generateEmbedding(text: string): Promise<number[]> {
    // Check cache first
    const cacheKey = `embedding:${this.hashText(text)}`;
    const cached = await redis.get(cacheKey);

    if (cached) {
      return JSON.parse(cached);
    }

    // Generate new embedding
    const embedding = await embeddings.embedQuery(text);

    // Cache for 24 hours
    await redis.set(cacheKey, JSON.stringify(embedding));

    return embedding;
  }

  private async smartLegalChunking(content: string): Promise<string[]> {
    const chunks: string[] = [];

    // Legal document patterns
    const sectionPatterns = [
      /(?:^|\n)(?:SECTION|ARTICLE|CLAUSE|PARAGRAPH)\s+[\d.]+[^\n]*/gi,
      /(?:^|\n)§\s*[\d.]+[^\n]*/g,
      /(?:^|\n)\d+\.\s+[A-Z][^\n]+/g, // Numbered sections with capital start
      /(?:^|\n)\([a-z]\)\s+[^\n]+/g, // Lettered subsections
    ];

    // Try to split by legal structure
    let structuredChunks: string[] = [];

    for (const pattern of sectionPatterns) {
      const matches = content.match(pattern);
      if (matches && matches.length > 0) {
        const sections = content.split(pattern);

        // Recombine pattern matches with their content
        for (let i = 0; i < sections.length; i++) {
          if (sections[i].trim().length > 50) {
            structuredChunks.push(sections[i].trim());
          }
        }

        if (structuredChunks.length > 0) break;
      }
    }

    // If no legal structure found, use standard chunking
    if (structuredChunks.length === 0) {
      const docs = await textSplitter.createDocuments([content]);
      structuredChunks = docs.map((d) => d.pageContent);
    }

    // Further split large chunks if needed
    for (const chunk of structuredChunks) {
      if (chunk.length > 2000) {
        const subDocs = await textSplitter.createDocuments([chunk]);
        chunks.push(...subDocs.map((d) => d.pageContent));
      } else {
        chunks.push(chunk);
      }
    }

    return chunks;
  }

  private async generateAutoTags(
    content: string,
    documentType: string
  ): Promise<Array<{ tag: string; confidence: number }>> {
    const tagPrompt = PromptTemplate.fromTemplate(`
Extract relevant legal tags from this {documentType} document.
Focus on: legal concepts, parties, jurisdictions, case types, and key topics.

Document excerpt:
{content}

Return ONLY a JSON array of tags with confidence scores (0-1):
[{"tag": "contract law", "confidence": 0.95}, ...]
    `);

    const chain = RunnableSequence.from([tagPrompt, llm, new StringOutputParser()]);

    try {
      const chainResult = await chain.invoke({
        documentType,
        content: content.substring(0, 3000), // Use first 3000 chars for tagging
      });

      const response = typeof chainResult === 'string' ? chainResult : chainResult.parse || '';

      // Extract JSON from response
      const jsonMatch = response.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }

      return [];
    } catch (error: any) {
      console.error('[RAG] Auto-tagging error:', error);
      return [];
    }
  }

  private async analyzeAnswer(answer: string, sources: LangChainDocument[]) {
    // Simple confidence calculation based on source relevance
    const avgScore = sources.reduce((sum, doc) => sum + (doc.metadata?.score || 0), 0) / sources.length;
    const confidence = Math.min(0.95, avgScore);

    // Extract key points (simplified - could use LLM for better extraction)
    const keyPoints = answer
      .split('\n')
      .filter((line) => line.match(/^\d+\.|^-|^•/))
      .slice(0, 5)
      .map((line) => line.replace(/^[\d.•-]\s*/, '').trim());

    return {
      confidence,
      keyPoints,
    };
  }

  private parseContractAnalysis(analysis: string) {
    const sections = {
      contractType: '',
      parties: [],
      keyTerms: [],
      risks: [],
      legalIssues: [],
      recommendations: [],
    };

    // Parse the structured response
    const lines = analysis.split('\n');
    let currentSection = '';

    for (const line of lines) {
      if (line.includes('CONTRACT TYPE')) currentSection = 'type';
      else if (line.includes('KEY TERMS')) currentSection = 'terms';
      else if (line.includes('RISK')) currentSection = 'risks';
      else if (line.includes('LEGAL ISSUES')) currentSection = 'issues';
      else if (line.includes('RECOMMENDATIONS')) currentSection = 'recommendations';
      else if (line.trim() && currentSection) {
        const trimmed = line.replace(/^[-•*]\s*/, '').trim();

        switch (currentSection) {
          case 'type':
            if (!sections.contractType) sections.contractType = trimmed;
            break;
          case 'terms':
            sections.keyTerms.push(trimmed);
            break;
          case 'risks':
            sections.risks.push(trimmed);
            break;
          case 'issues':
            sections.legalIssues.push(trimmed);
            break;
          case 'recommendations':
            sections.recommendations.push(trimmed);
            break;
        }
      }
    }

    return sections;
  }

  private hashText(text: string): string {
    return createHash('sha256').update(text).digest('hex');
  }

  // === CLEANUP ===

  async close() {
    // Cleanup handled by connection pool
    await sql.end();
  }
}

// Export singleton instance
export const ragPipeline = new LegalRAGPipeline();
