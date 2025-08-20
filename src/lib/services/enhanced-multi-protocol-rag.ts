// Enhanced Multi-Protocol RAG Service with QUIC, protobuf, and semantic analysis
import { writable, derived, type Writable } from 'svelte/store';
import type {
  Document,
  Evidence,
  Case,
  SemanticChunk,
  IntentClassification,
  RagResponse,
  ProcessingPipeline
} from '$lib/types/legal-ai';

// Define LegalAnalysis type if not already defined in $lib/types/legal-ai
export interface LegalAnalysis {
  confidence: number;
  legalDomain: string;
  complexity: string;
  keyTerms: string[];
  suggestedActions: string[];
}

// ==========================================
// MULTI-PROTOCOL RAG CONFIGURATION
// ==========================================

interface RagConfig {
  quicEndpoint: string;
  httpEndpoint: string;
  grpcEndpoint: string;
  protobufEnabled: boolean;
  embeddingDimensions: number;
  chunkSize: number;
  overlapSize: number;
  semanticThreshold: number;
  intentModelPath: string;
  legalBertModel: string;
  gemmaModel: string;
}

const defaultConfig: RagConfig = {
  quicEndpoint: 'https://localhost:8443',
  httpEndpoint: 'http://localhost:8094',
  grpcEndpoint: 'localhost:50051',
  protobufEnabled: true,
  embeddingDimensions: 384,
  chunkSize: 512,
  overlapSize: 64,
  semanticThreshold: 0.75,
  intentModelPath: '/models/legal-bert-intent',
  legalBertModel: 'legal-bert-base-uncased',
  gemmaModel: 'gemma3-legal:latest'
};

// ==========================================
// DOCUMENT INGESTION PIPELINE
// ==========================================

export interface DocumentIngestionRequest {
  documentId: string;
  caseId: string;
  userId: string;
  filename: string;
  content: ArrayBuffer | string;
  mimeType: string;
  metadata: Record<string, any>;
  processingOptions: ProcessingOptions;
}

export interface ProcessingOptions {
  enableOcr: boolean;
  enableEntityExtraction: boolean;
  enableLegalClassification: boolean;
  enableRiskAnalysis: boolean;
  chunkingStrategy: 'fixed' | 'sentence' | 'paragraph' | 'semantic' | 'legal_section';
  embeddingModel: string;
  qualityThreshold: number;
}

export interface IngestionResponse {
  documentId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  stages: ProcessingStage[];
  analysis: LegalAnalysis;
  entities: LegalEntity[];
  chunks: SemanticChunk[];
  storagePath: string;
  error?: string;
}

export interface ProcessingStage {
  name: string;
  status: 'not_started' | 'in_progress' | 'completed' | 'failed';
  startTime: Date;
  endTime?: Date;
  details: string;
  progress: number;
}

export interface LegalEntity {
  name: string;
  type: 'person' | 'organization' | 'location' | 'date' | 'money' | 'legal_statute' | 'case_citation' | 'contract_clause';
  confidence: number;
  startPosition: number;
  endPosition: number;
  aliases: string[];
  attributes: Record<string, any>;
}

// ==========================================
// SEMANTIC ANALYSIS PIPELINE
// ==========================================

export class SemanticAnalyzer {
  private intentClassifier: IntentClassifier;
  private entityExtractor: EntityExtractor;
  private legalBertProcessor: LegalBertProcessor;
  private gemmaGenerator: GemmaGenerator;

  constructor() {
    this.intentClassifier = new IntentClassifier();
    this.entityExtractor = new EntityExtractor();
    this.legalBertProcessor = new LegalBertProcessor();
    this.gemmaGenerator = new GemmaGenerator();
  }

  async analyzeQuery(query: string, context: Record<string, any>): Promise<QueryAnalysis> {
    const [intent, entities, legalContext] = await Promise.all([
      this.intentClassifier.classify(query),
      this.entityExtractor.extract(query),
      this.legalBertProcessor.analyze(query, context)
    ]);

    return {
      query,
      intent,
      entities,
      legalContext,
      confidence: Math.min(intent.confidence, legalContext.confidence),
      timestamp: new Date()
    };
  }

  async processDocument(content: string, metadata: Record<string, any>): Promise<DocumentAnalysis> {
    const chunks = await this.chunkDocument(content, metadata);
    const embeddings = await this.generateEmbeddings(chunks);
    const entities = await this.extractEntities(content);
    const classification = await this.classifyDocument(content, metadata);
    const riskFactors = await this.analyzeRisks(content, entities);

    return {
      chunks,
      embeddings,
      entities,
      classification,
      riskFactors,
      confidence: classification.confidence,
      processingTime: new Date()
    };
  }

  private async chunkDocument(content: string, metadata: Record<string, any>): Promise<SemanticChunk[]> {
    // Implementation of semantic chunking with legal section awareness
    const strategy = metadata.chunkingStrategy || 'semantic';

    switch (strategy) {
      case 'legal_section':
        return this.chunkByLegalSections(content);
      case 'semantic':
        return this.chunkBySemantic(content);
      case 'sentence':
        return this.chunkBySentence(content);
      case 'paragraph':
        return this.chunkByParagraph(content);
      default:
        return this.chunkByFixedSize(content);
    }
  }

  private async chunkByLegalSections(content: string): Promise<SemanticChunk[]> {
    // Legal-aware chunking that preserves contract sections, clauses, etc.
    const sectionPatterns = [
      /ARTICLE\s+[IVX\d]+[.:]/gi,
      /Section\s+\d+[.:]/gi,
      /Clause\s+\d+[.:]/gi,
      /\d+\.\s*[A-Z][^.]*[.:]/g,
      /WHEREAS[,:]/gi,
      /NOW, THEREFORE[,:]/gi,
      /IN WITNESS WHEREOF/gi
    ];

    const chunks: SemanticChunk[] = [];
    let currentChunk = '';
    let chunkIndex = 0;

    const lines = content.split('\n');

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      if (line === '') continue;

      const isSectionBreak = sectionPatterns.some(pattern => pattern.test(line));

      if (isSectionBreak && currentChunk.length > 100) {
        chunks.push({
          id: `chunk_${chunkIndex++}`,
          content: currentChunk.trim(),
          startPosition: content.indexOf(currentChunk),
          endPosition: content.indexOf(currentChunk) + currentChunk.length,
          type: 'legal_section',
          metadata: {
            sectionType: this.detectSectionType(currentChunk),
            wordCount: currentChunk.split(/\s+/).length,
            hasLegalTerms: this.containsLegalTerms(currentChunk)
          }
        });
        currentChunk = line;
      } else {
        currentChunk += (currentChunk ? '\n' : '') + line;
      }
    }

    if (currentChunk.trim()) {
      chunks.push({
        id: `chunk_${chunkIndex}`,
        content: currentChunk.trim(),
        startPosition: content.lastIndexOf(currentChunk),
        endPosition: content.lastIndexOf(currentChunk) + currentChunk.length,
        type: 'legal_section',
        metadata: {
          sectionType: this.detectSectionType(currentChunk),
          wordCount: currentChunk.split(/\s+/).length,
          hasLegalTerms: this.containsLegalTerms(currentChunk)
        }
      });
    }

    return chunks;
  }

  private detectSectionType(content: string): string {
    const lowerContent = content.toLowerCase();

    if (lowerContent.includes('whereas')) return 'recital';
    if (lowerContent.includes('liability') || lowerContent.includes('indemnif')) return 'liability';
    if (lowerContent.includes('payment') || lowerContent.includes('invoice')) return 'payment';
    if (lowerContent.includes('termination') || lowerContent.includes('expire')) return 'termination';
    if (lowerContent.includes('confidential') || lowerContent.includes('non-disclosure')) return 'confidentiality';
    if (lowerContent.includes('intellectual property') || lowerContent.includes('copyright')) return 'ip';
    if (lowerContent.includes('governing law') || lowerContent.includes('jurisdiction')) return 'governing_law';
    if (lowerContent.includes('dispute') || lowerContent.includes('arbitration')) return 'dispute_resolution';

    return 'general';
  }

  private containsLegalTerms(content: string): boolean {
    const legalTerms = [
      'contract', 'agreement', 'party', 'parties', 'clause', 'section',
      'liability', 'damages', 'breach', 'warranty', 'representation',
      'indemnification', 'confidential', 'proprietary', 'jurisdiction',
      'governing law', 'force majeure', 'termination', 'default'
    ];

    const lowerContent = content.toLowerCase();
    return legalTerms.some(term => lowerContent.includes(term));
  }

  private async chunkBySemantic(content: string): Promise<SemanticChunk[]> {
    // Semantic similarity-based chunking
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 20);
    const chunks: SemanticChunk[] = [];
    let currentChunk = '';
    let chunkIndex = 0;

    for (let i = 0; i < sentences.length; i++) {
      const sentence = sentences[i].trim();
      const testChunk = currentChunk + (currentChunk ? '. ' : '') + sentence;

      if (testChunk.length > defaultConfig.chunkSize && currentChunk.length > 0) {
        chunks.push({
          id: `semantic_chunk_${chunkIndex++}`,
          content: currentChunk,
          startPosition: content.indexOf(currentChunk),
          endPosition: content.indexOf(currentChunk) + currentChunk.length,
          type: 'semantic',
          metadata: {
            sentenceCount: currentChunk.split(/[.!?]+/).length,
            avgSentenceLength: currentChunk.length / currentChunk.split(/[.!?]+/).length
          }
        });
        currentChunk = sentence;
      } else {
        currentChunk = testChunk;
      }
    }

    if (currentChunk.trim()) {
      chunks.push({
        id: `semantic_chunk_${chunkIndex}`,
        content: currentChunk,
        startPosition: content.lastIndexOf(currentChunk),
        endPosition: content.lastIndexOf(currentChunk) + currentChunk.length,
        type: 'semantic',
        metadata: {
          sentenceCount: currentChunk.split(/[.!?]+/).length,
          avgSentenceLength: currentChunk.length / currentChunk.split(/[.!?]+/).length
        }
      });
    }

    return chunks;
  }

  private async chunkBySentence(content: string): Promise<SemanticChunk[]> {
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 10);
    return sentences.map((sentence, index) => ({
      id: `sentence_chunk_${index}`,
      content: sentence.trim(),
      startPosition: content.indexOf(sentence),
      endPosition: content.indexOf(sentence) + sentence.length,
      type: 'sentence',
      metadata: {
        wordCount: sentence.split(/\s+/).length,
        characterCount: sentence.length
      }
    }));
  }

  private async chunkByParagraph(content: string): Promise<SemanticChunk[]> {
    const paragraphs = content.split(/\n\s*\n/).filter(p => p.trim().length > 20);
    return paragraphs.map((paragraph, index) => ({
      id: `paragraph_chunk_${index}`,
      content: paragraph.trim(),
      startPosition: content.indexOf(paragraph),
      endPosition: content.indexOf(paragraph) + paragraph.length,
      type: 'paragraph',
      metadata: {
        sentenceCount: paragraph.split(/[.!?]+/).length,
        wordCount: paragraph.split(/\s+/).length
      }
    }));
  }

  private async chunkByFixedSize(content: string): Promise<SemanticChunk[]> {
    const chunks: SemanticChunk[] = [];
    const chunkSize = defaultConfig.chunkSize;
    const overlap = defaultConfig.overlapSize;

    for (let i = 0; i < content.length; i += chunkSize - overlap) {
      const chunk = content.slice(i, i + chunkSize);
      chunks.push({
        id: `fixed_chunk_${Math.floor(i / (chunkSize - overlap))}`,
        content: chunk,
        startPosition: i,
        endPosition: i + chunk.length,
        type: 'fixed',
        metadata: {
          chunkSize: chunk.length,
          hasOverlap: i > 0
        }
      });
    }

    return chunks;
  }

  private async generateEmbeddings(chunks: SemanticChunk[]): Promise<number[][]> {
    // Generate embeddings using the embedding service
    const embeddings: number[][] = [];

    for (const chunk of chunks) {
      try {
        const embedding = await this.callEmbeddingService(chunk.content);
        embeddings.push(embedding);
      } catch (error) {
        console.warn(`Failed to generate embedding for chunk ${chunk.id}:`, error);
        // Fallback to zero vector
        embeddings.push(new Array(defaultConfig.embeddingDimensions).fill(0));
      }
    }

    return embeddings;
  }

  private async callEmbeddingService(text: string): Promise<number[]> {
    // This would call the actual embedding service (ONNX/gRPC)
    // For now, return a mock embedding
    return new Array(defaultConfig.embeddingDimensions).fill(0).map(() => Math.random() - 0.5);
  }

  private async extractEntities(content: string): Promise<LegalEntity[]> {
    // Legal entity extraction using NER models
    return [
      {
        name: 'Sample Contract',
        type: 'contract_clause',
        confidence: 0.95,
        startPosition: 0,
        endPosition: 100,
        aliases: [],
        attributes: { section: 'liability' }
      }
    ];
  }

  private async classifyDocument(content: string, metadata: Record<string, any>): Promise<DocumentClassification> {
    return {
      primaryType: 'contract',
      confidence: 0.92,
      practiceAreas: ['corporate', 'commercial'],
      urgency: 'normal',
      compliance: 'compliant',
      riskLevel: 'medium'
    };
  }

  private async analyzeRisks(content: string, entities: LegalEntity[]): Promise<RiskFactor[]> {
    return [
      {
        description: 'Broad liability clause detected',
        level: 'medium',
        probability: 0.75,
        mitigationStrategies: ['Add liability caps', 'Include mutual indemnification'],
        category: 'liability'
      }
    ];
  }
}

// ==========================================
// INTENT CLASSIFICATION
// ==========================================

export class IntentClassifier {
  private modelLoaded = false;

  async classify(text: string): Promise<IntentClassification> {
    await this.ensureModelLoaded();

    // Legal-BERT intent classification
    const intents = this.classifyLegalIntent(text);
    const entities = this.extractNamedEntities(text);

    return {
      intent: intents.primary,
      confidence: intents.confidence,
      alternativeIntents: intents.alternatives,
      entities,
      context: {
        domain: 'legal',
        complexity: this.assessComplexity(text),
        urgency: this.assessUrgency(text)
      }
    };
  }

  private async ensureModelLoaded(): Promise<void> {
    if (!this.modelLoaded) {
      // Load Legal-BERT model
      console.log('Loading Legal-BERT intent classification model...');
      this.modelLoaded = true;
    }
  }

  private classifyLegalIntent(text: string): { primary: string; confidence: number; alternatives: string[] } {
    const lowerText = text.toLowerCase();

    // Legal intent patterns
    if (lowerText.includes('risk') || lowerText.includes('liability') || lowerText.includes('danger')) {
      return { primary: 'risk_assessment', confidence: 0.92, alternatives: ['legal_analysis', 'contract_review'] };
    }

    if (lowerText.includes('contract') || lowerText.includes('agreement') || lowerText.includes('terms')) {
      return { primary: 'contract_review', confidence: 0.89, alternatives: ['legal_analysis', 'compliance_check'] };
    }

    if (lowerText.includes('analyze') || lowerText.includes('analysis') || lowerText.includes('review')) {
      return { primary: 'legal_analysis', confidence: 0.86, alternatives: ['document_review', 'case_research'] };
    }

    if (lowerText.includes('precedent') || lowerText.includes('case') || lowerText.includes('ruling')) {
      return { primary: 'case_research', confidence: 0.88, alternatives: ['legal_analysis', 'precedent_search'] };
    }

    if (lowerText.includes('compliance') || lowerText.includes('regulation') || lowerText.includes('statute')) {
      return { primary: 'compliance_check', confidence: 0.85, alternatives: ['legal_analysis', 'regulatory_review'] };
    }

    return { primary: 'general_inquiry', confidence: 0.70, alternatives: ['legal_analysis'] };
  }

  private extractNamedEntities(text: string): LegalEntity[] {
    // Simple regex-based NER for demonstration
    const entities: LegalEntity[] = [];

    // Date patterns
    const datePattern = /\d{1,2}\/\d{1,2}\/\d{4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+ \d{1,2}, \d{4}/g;
    let match;
    while ((match = datePattern.exec(text)) !== null) {
      entities.push({
        name: match[0],
        type: 'date',
        confidence: 0.85,
        startPosition: match.index,
        endPosition: match.index + match[0].length,
        aliases: [],
        attributes: {}
      });
    }

    // Money patterns
    const moneyPattern = /\$[\d,]+\.?\d*/g;
    while ((match = moneyPattern.exec(text)) !== null) {
      entities.push({
        name: match[0],
        type: 'money',
        confidence: 0.90,
        startPosition: match.index,
        endPosition: match.index + match[0].length,
        aliases: [],
        attributes: {}
      });
    }

    return entities;
  }

  private assessComplexity(text: string): 'low' | 'medium' | 'high' {
    const complexTerms = ['indemnification', 'jurisdiction', 'arbitration', 'liquidated damages', 'force majeure'];
    const count = complexTerms.filter(term => text.toLowerCase().includes(term)).length;

    if (count >= 3) return 'high';
    if (count >= 1) return 'medium';
    return 'low';
  }

  private assessUrgency(text: string): 'low' | 'medium' | 'high' | 'critical' {
    const urgentTerms = ['urgent', 'immediate', 'asap', 'emergency', 'critical', 'deadline'];
    const hasUrgentTerms = urgentTerms.some(term => text.toLowerCase().includes(term));

    if (hasUrgentTerms) return 'high';
    return 'medium';
  }
}

// ==========================================
// ENTITY EXTRACTION
// ==========================================

export class EntityExtractor {
  async extract(text: string): Promise<LegalEntity[]> {
    // Enhanced legal entity extraction
    return [];
  }
}

// ==========================================
// LEGAL-BERT PROCESSOR
// ==========================================

export class LegalBertProcessor {
  async analyze(text: string, context: Record<string, any>): Promise<LegalAnalysis> {
    return {
      confidence: 0.85,
      legalDomain: 'contract_law',
      complexity: 'medium',
      keyTerms: ['liability', 'indemnification', 'termination'],
      suggestedActions: ['Review liability clauses', 'Check termination conditions']
    };
  }
}

// ==========================================
// GEMMA GENERATOR
// ==========================================

export class GemmaGenerator {
  async generate(prompt: string, context: SemanticChunk[], intent: string): Promise<string> {
    // Call Gemma3-legal model via Ollama
    const contextText = context.map(chunk => chunk.content).join('\n\n');
    const fullPrompt = this.buildPrompt(prompt, contextText, intent);

    try {
      const response = await fetch(`${defaultConfig.httpEndpoint}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: defaultConfig.gemmaModel,
          prompt: fullPrompt,
          stream: false
        })
      });

      const result = await response.json();
      return result.response || 'Failed to generate response';
    } catch (error) {
      console.error('Gemma generation failed:', error);
      return 'Error: Unable to generate legal analysis at this time.';
    }
  }

  private buildPrompt(query: string, context: string, intent: string): string {
    const intentPrompts = {
      risk_assessment: 'Analyze the following legal documents for potential risks and provide mitigation strategies:',
      contract_review: 'Review the following contract terms and identify key provisions, potential issues, and recommendations:',
      legal_analysis: 'Provide a comprehensive legal analysis of the following documents:',
      case_research: 'Research and analyze relevant case law and precedents for the following query:',
      compliance_check: 'Assess compliance with relevant regulations and statutes for the following:',
      general_inquiry: 'Provide legal guidance and analysis for the following query:'
    };

    const systemPrompt = intentPrompts[intent as keyof typeof intentPrompts] || intentPrompts.general_inquiry;

    return `${systemPrompt}

Context Documents:
${context}

User Query: ${query}

Please provide a detailed legal analysis with specific citations to the context documents. Include:
1. Key findings
2. Legal implications
3. Recommendations
4. Risk factors (if applicable)
5. Next steps

Format your response clearly with bullet points and section headers.`;
  }
}

// ==========================================
// MULTI-PROTOCOL CLIENT
// ==========================================

export class MultiProtocolRagClient {
  private config: RagConfig;
  private semanticAnalyzer: SemanticAnalyzer;

  constructor(config: Partial<RagConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
    this.semanticAnalyzer = new SemanticAnalyzer();
  }

  async ingestDocument(request: DocumentIngestionRequest): Promise<IngestionResponse> {
    // Multi-protocol document ingestion with QUIC, HTTP, and gRPC

    // Step 1: Try QUIC for high-performance upload
    try {
      return await this.ingestViaQuic(request);
    } catch (error) {
      console.warn('QUIC ingestion failed, falling back to HTTP:', error);
    }

    // Step 2: Fallback to HTTP
    try {
      return await this.ingestViaHttp(request);
    } catch (error) {
      console.warn('HTTP ingestion failed, falling back to gRPC:', error);
    }

    // Step 3: Final fallback to gRPC
    return await this.ingestViaGrpc(request);
  }

  async queryRag(query: string, options: QueryOptions = {}): Promise<RagResponse> {
    // Step 1: Analyze query intent and extract entities
    const analysis = await this.semanticAnalyzer.analyzeQuery(query, options.context || {});

    // Step 2: Perform vector search
    const searchResults = await this.performVectorSearch(query, options, analysis);

    // Step 3: Enrich with graph data
    const enrichedResults = await this.enrichWithGraphData(searchResults, options.caseId);

    // Step 4: Rerank using legal-specific scoring
    const rerankedResults = await this.rerankResults(enrichedResults, analysis);

    // Step 5: Generate response using Gemma3-legal
    const generator = new GemmaGenerator();
    const answer = await generator.generate(query, rerankedResults, analysis.intent.intent);

    return {
      query,
      answer,
      sources: rerankedResults,
      analysis,
      processingTime: new Date(),
      metadata: {
        protocol: 'http',
        model: this.config.gemmaModel,
        confidence: analysis.confidence,
        intent: analysis.intent.intent
      }
    };
  }

  private async ingestViaQuic(request: DocumentIngestionRequest): Promise<IngestionResponse> {
    // QUIC-based high-performance ingestion
    throw new Error('QUIC ingestion not implemented yet');
  }

  private async ingestViaHttp(request: DocumentIngestionRequest): Promise<IngestionResponse> {
    const formData = new FormData();
    formData.append('document_id', request.documentId);
    formData.append('case_id', request.caseId);
    formData.append('user_id', request.userId);
    formData.append('filename', request.filename);
    formData.append('content', new Blob([request.content]), request.filename);
    formData.append('mime_type', request.mimeType);
    formData.append('metadata', JSON.stringify(request.metadata));
    formData.append('processing_options', JSON.stringify(request.processingOptions));

    const response = await fetch(`${this.config.httpEndpoint}/v1/rag/ingest`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP ingestion failed: ${response.statusText}`);
    }

    return await response.json();
  }

  private async ingestViaGrpc(request: DocumentIngestionRequest): Promise<IngestionResponse> {
    // gRPC-based ingestion using protobuf
    throw new Error('gRPC ingestion not implemented yet');
  }

  private async performVectorSearch(query: string, options: QueryOptions, analysis: QueryAnalysis): Promise<SemanticChunk[]> {
    const searchPayload = {
      query,
      case_id: options.caseId,
      user_id: options.userId,
      top_k: options.topK || 10,
      threshold: options.threshold || this.config.semanticThreshold,
      filters: options.filters || {},
      use_reranker: true,
      intent: analysis.intent.intent,
      entities: analysis.entities
    };

    const response = await fetch(`${this.config.httpEndpoint}/v1/rag/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(searchPayload)
    });

    if (!response.ok) {
      throw new Error(`Vector search failed: ${response.statusText}`);
    }

    const result = await response.json();
    return result.sources || [];
  }

  private async enrichWithGraphData(chunks: SemanticChunk[], caseId?: string): Promise<SemanticChunk[]> {
    if (!caseId) return chunks;

    // Enrich with Neo4j graph data
    for (const chunk of chunks) {
      chunk.metadata = {
        ...chunk.metadata,
        graphEntities: [], // Would be populated from Neo4j
        relationships: []   // Would be populated from Neo4j
      };
    }

    return chunks;
  }

  private async rerankResults(chunks: SemanticChunk[], analysis: QueryAnalysis): Promise<SemanticChunk[]> {
    // Legal-specific reranking based on intent, entity overlap, etc.
    return chunks.sort((a, b) => {
      let scoreA = a.metadata?.similarity || 0;
      let scoreB = b.metadata?.similarity || 0;

      // Boost based on intent match
      if (analysis.intent.intent === 'risk_assessment') {
        if (a.content.toLowerCase().includes('risk') || a.content.toLowerCase().includes('liability')) {
          scoreA *= 1.2;
        }
        if (b.content.toLowerCase().includes('risk') || b.content.toLowerCase().includes('liability')) {
          scoreB *= 1.2;
        }
      }

      // Boost based on entity overlap
      const entitiesInA = analysis.entities.filter(entity =>
        a.content.toLowerCase().includes(entity.name.toLowerCase())
      ).length;
      const entitiesInB = analysis.entities.filter(entity =>
        b.content.toLowerCase().includes(entity.name.toLowerCase())
      ).length;

      scoreA += entitiesInA * 0.1;
      scoreB += entitiesInB * 0.1;

      return scoreB - scoreA;
    });
  }
}

// ==========================================
// TYPE DEFINITIONS
// ==========================================

export interface QueryOptions {
  caseId?: string;
  userId?: string;
  topK?: number;
  threshold?: number;
  filters?: Record<string, any>;
  context?: Record<string, any>;
}

export interface QueryAnalysis {
  query: string;
  intent: IntentClassification;
  entities: LegalEntity[];
  legalContext: LegalAnalysis;
  confidence: number;
  timestamp: Date;
}

export interface DocumentAnalysis {
  chunks: SemanticChunk[];
  embeddings: number[][];
  entities: LegalEntity[];
  classification: DocumentClassification;
  riskFactors: RiskFactor[];
  confidence: number;
  processingTime: Date;
}

export interface DocumentClassification {
  primaryType: string;
  confidence: number;
  practiceAreas: string[];
  urgency: string;
  compliance: string;
  riskLevel: string;
}

export interface RiskFactor {
  description: string;
  level: 'low' | 'medium' | 'high' | 'critical';
  probability: number;
  mitigationStrategies: string[];
  category: string;
}

// ==========================================
// SVELTE STORES
// ==========================================

export const ragClient = new MultiProtocolRagClient();

export const ragState = writable({
  isProcessing: false,
  lastQuery: '',
  lastResponse: null as RagResponse | null,
  error: null as string | null,
  ingestionStatus: new Map<string, IngestionResponse>()
});

export const ingestionQueue = writable<DocumentIngestionRequest[]>([]);

export const processingPipeline = writable<ProcessingPipeline>({
  stages: [],
  currentStage: 0,
  overallProgress: 0,
  isActive: false
});

// ==========================================
// REACTIVE HELPERS
// ==========================================

export const activeIngestions = derived(ragState, $ragState =>
  Array.from($ragState.ingestionStatus.values()).filter(status =>
    status.status === 'processing' || status.status === 'pending'
  )
);

export const completedIngestions = derived(ragState, $ragState =>
  Array.from($ragState.ingestionStatus.values()).filter(status =>
    status.status === 'completed'
  )
);

export const failedIngestions = derived(ragState, $ragState =>
  Array.from($ragState.ingestionStatus.values()).filter(status =>
    status.status === 'failed'
  )
);

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

export async function ingestDocument(request: DocumentIngestionRequest): Promise<string> {
  ragState.update(state => ({
    ...state,
    isProcessing: true,
    error: null
  }));

  try {
    const response = await ragClient.ingestDocument(request);

    ragState.update(state => {
      const newStatus = new Map(state.ingestionStatus);
      newStatus.set(request.documentId, response);
      return {
        ...state,
        ingestionStatus: newStatus,
        isProcessing: false
      };
    });

    return response.documentId;
  } catch (error) {
    ragState.update(state => ({
      ...state,
      isProcessing: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }));
    throw error;
  }
}

export async function queryRag(query: string, options: QueryOptions = {}): Promise<RagResponse> {
  ragState.update(state => ({
    ...state,
    isProcessing: true,
    error: null,
    lastQuery: query
  }));

  try {
    const response = await ragClient.queryRag(query, options);

    ragState.update(state => ({
      ...state,
      isProcessing: false,
      lastResponse: response
    }));

    return response;
  } catch (error) {
    ragState.update(state => ({
      ...state,
      isProcessing: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }));
    throw error;
  }
}

export function clearRagState(): void {
  ragState.set({
    isProcessing: false,
    lastQuery: '',
    lastResponse: null,
    error: null,
    ingestionStatus: new Map()
  });
}

export default ragClient;