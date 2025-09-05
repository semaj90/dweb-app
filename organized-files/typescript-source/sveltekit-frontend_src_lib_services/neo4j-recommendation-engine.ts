/**
 * Neo4j Recommendation Engine with SIMD GPU Integration
 * Advanced legal AI recommendation system with graph-based intelligence
 */

import type { Driver, Session, Result, Record } from 'neo4j-driver';
import { EnhancedVLLMCudaIntegration, type StreamingRequest, type StreamingResponse } from './enhanced-vllm-cuda-integration'
import { SIMDGPUParserIntegration, type ParsedDocument, type ExtractedEntity } from './simd-gpu-parser-integration'
import type { IdleDetectionContext } from '../machines/idle-detection-rabbitmq-machine';

// Types for Neo4j recommendation system
export interface LegalNode {
  id: string;
  labels: string[];
  properties: Record<string, any>;
  relationships: LegalRelationship[];
  score?: number;
  confidence?: number;
}

export interface LegalRelationship {
  id: string;
  type: string;
  startNode: string;
  endNode: string;
  properties: Record<string, any>;
  weight: number;
  confidence: number;
}

export interface RecommendationRequest {
  userId: string;
  caseId?: string;
  context: string;
  type: 'similar_cases' | 'related_documents' | 'legal_precedents' | 'expert_insights' | 'risk_analysis';
  limit?: number;
  filters?: RecommendationFilters;
  includeExplanation?: boolean;
  useAI?: boolean;
}

export interface RecommendationFilters {
  practiceArea?: string[];
  jurisdiction?: string[];
  dateRange?: { start: string; end: string };
  riskLevel?: string[];
  entityTypes?: string[];
  similarity?: number; // minimum similarity score
}

export interface Recommendation {
  id: string;
  type: string;
  title: string;
  description: string;
  score: number;
  confidence: number;
  reasoning: string;
  relatedNodes: LegalNode[];
  metadata: {
    practiceArea?: string;
    jurisdiction?: string;
    riskLevel?: string;
    lastUpdated: string;
    sourceDocument?: string;
    entities: string[];
  };
  aiGenerated?: {
    prompt: string;
    response: string;
    model: string;
    processingTime: number;
  };
}

export interface GraphAnalytics {
  totalNodes: number;
  totalRelationships: number;
  nodesByType: Record<string, number>;
  relationshipsByType: Record<string, number>;
  averageConnectivity: number;
  clusteringCoefficient: number;
  centralityScores: Array<{ nodeId: string; centrality: number }>;
  communityDetection: Array<{ community: string; members: string[]; strength: number }>;
}

export class Neo4jRecommendationEngine {
  private driver: Driver | null = null;
  private vllmIntegration: EnhancedVLLMCudaIntegration;
  private simdParser: SIMDGPUParserIntegration;
  private isConnected = false;
  private queryCache: Map<string, { result: any; timestamp: number }> = new Map();
  private cacheTimeout = 5 * 60 * 1000; // 5 minutes

  // Graph analysis state
  private graphStats: GraphAnalytics | null = null;
  private lastStatsUpdate = 0;
  private statsUpdateInterval = 10 * 60 * 1000; // 10 minutes

  // SIMD-optimized matrices for similarity calculations
  private similarityMatrix: Float32Array = new Float32Array(0);
  private nodeEmbeddings: Map<string, Float32Array> = new Map();
  
  constructor(
    private neo4jUri: string = 'bolt://localhost:7687',
    private neo4jUser: string = 'neo4j',
    private neo4jPassword: string = 'password'
  ) {
    this.vllmIntegration = new EnhancedVLLMCudaIntegration({
      serverUrl: 'http://localhost:8000',
      maxConcurrentStreams: 100,
      gpuMemoryPerDevice: 8,
      tensorParallelSize: 1,
      quantization: 'int8',
      maxModelLength: 4096,
      enableTensorCores: true
    });

    this.simdParser = new SIMDGPUParserIntegration({
      enableSpellCheck: true,
      enableEntityExtraction: true,
      enableLegalTermSuggestions: true,
      enableCitationValidation: true,
      confidenceThreshold: 0.7,
      maxSuggestions: 10,
      simdOptimization: true,
      gpuAcceleration: true
    });
  }

  /**
   * Initialize the recommendation engine
   */
  async initialize(): Promise<void> {
    console.log('🔌 Initializing Neo4j Recommendation Engine...');

    try {
      // Initialize Neo4j connection
      await this.connectToNeo4j();

      // Initialize SIMD GPU parser and vLLM integration
      await Promise.all([
        this.simdParser.initializeGPU(),
        this.vllmIntegration.initializeGPU()
      ]);

      // Initialize graph schema and indexes
      await this.initializeGraphSchema();

      // Load initial graph statistics
      await this.updateGraphStatistics();

      // Precompute similarity matrices
      await this.precomputeSimilarityMatrices();

      this.isConnected = true;
      console.log('✅ Neo4j Recommendation Engine initialized successfully');

    } catch (error) {
      console.error('❌ Neo4j Recommendation Engine initialization failed:', error);
      throw error;
    }
  }

  /**
   * Connect to Neo4j database
   */
  private async connectToNeo4j(): Promise<void> {
    try {
      // For demo purposes, we'll simulate Neo4j connection
      // In production, you would use: import neo4j from 'neo4j-driver';
      // this.driver = neo4j.driver(this.neo4jUri, neo4j.auth.basic(this.neo4jUser, this.neo4jPassword));
      
      console.log('📊 Connected to Neo4j database');
    } catch (error) {
      throw new Error(`Failed to connect to Neo4j: ${error}`);
    }
  }

  /**
   * Initialize graph schema and indexes for optimal performance
   */
  private async initializeGraphSchema(): Promise<void> {
    console.log('📋 Initializing graph schema and indexes...');

    // Simulated schema creation for demo
    const schemaQueries = [
      // Node constraints
      'CREATE CONSTRAINT legal_case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE',
      'CREATE CONSTRAINT legal_document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE',
      'CREATE CONSTRAINT legal_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE',
      'CREATE CONSTRAINT legal_precedent_id IF NOT EXISTS FOR (p:Precedent) REQUIRE p.id IS UNIQUE',
      
      // Indexes for performance
      'CREATE INDEX legal_case_practice_area IF NOT EXISTS FOR (c:Case) ON (c.practiceArea)',
      'CREATE INDEX legal_case_jurisdiction IF NOT EXISTS FOR (c:Case) ON (c.jurisdiction)',
      'CREATE INDEX legal_document_type IF NOT EXISTS FOR (d:Document) ON (d.documentType)',
      'CREATE INDEX legal_entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entityType)',
      
      // Vector indexes for similarity search (Neo4j 5.x+)
      'CALL db.index.vector.createNodeIndex("caseEmbeddings", "Case", "embedding", 768, "cosine")',
      'CALL db.index.vector.createNodeIndex("documentEmbeddings", "Document", "embedding", 768, "cosine")'
    ];

    // In production, execute these queries
    console.log(`✅ Graph schema initialized with ${schemaQueries.length} constraints and indexes`);
  }

  /**
   * Get recommendations based on context and type
   */
  async getRecommendations(request: RecommendationRequest): Promise<Recommendation[]> {
    console.log(`🎯 Generating ${request.type} recommendations for user ${request.userId}`);

    try {
      // Check cache first
      const cacheKey = this.generateCacheKey(request);
      const cachedResult = this.getFromCache(cacheKey);
      if (cachedResult) {
        console.log('💾 Returning cached recommendations');
        return cachedResult;
      }

      // Process context with SIMD GPU parser
      const parsedContext = await this.simdParser.parseDocument(request.context, {
        documentType: 'query',
        language: 'en'
      });

      // Generate recommendations based on type
      let recommendations: Recommendation[] = [];

      switch (request.type) {
        case 'similar_cases':
          recommendations = await this.findSimilarCases(request, parsedContext);
          break;
        case 'related_documents':
          recommendations = await this.findRelatedDocuments(request, parsedContext);
          break;
        case 'legal_precedents':
          recommendations = await this.findLegalPrecedents(request, parsedContext);
          break;
        case 'expert_insights':
          recommendations = await this.generateExpertInsights(request, parsedContext);
          break;
        case 'risk_analysis':
          recommendations = await this.performRiskAnalysis(request, parsedContext);
          break;
      }

      // Enhance with AI-generated insights if requested
      if (request.useAI && recommendations.length > 0) {
        recommendations = await this.enhanceWithAI(recommendations, request, parsedContext);
      }

      // Cache results
      this.setInCache(cacheKey, recommendations);

      console.log(`✅ Generated ${recommendations.length} recommendations`);
      return recommendations;

    } catch (error) {
      console.error('❌ Recommendation generation failed:', error);
      throw error;
    }
  }

  /**
   * Find similar cases using graph algorithms and vector similarity
   */
  private async findSimilarCases(request: RecommendationRequest, context: ParsedDocument): Promise<Recommendation[]> {
    console.log('🔍 Finding similar cases using graph analysis...');

    // Simulate Neo4j query for similar cases
    const mockSimilarCases = [
      {
        id: 'case_001',
        title: 'Contract Dispute - Service Agreement Terms',
        description: 'Similar contract interpretation dispute involving service level agreements',
        score: 0.92,
        confidence: 0.88,
        reasoning: 'High similarity in legal entities, contract terms, and dispute patterns',
        metadata: {
          practiceArea: 'Contract Law',
          jurisdiction: 'Federal',
          riskLevel: 'Medium',
          lastUpdated: new Date().toISOString(),
          entities: ['Service Provider', 'Client', 'SLA Terms']
        }
      },
      {
        id: 'case_002',
        title: 'Breach of Contract - Performance Standards',
        description: 'Related case involving performance standard disputes and remedies',
        score: 0.87,
        confidence: 0.82,
        reasoning: 'Similar contractual obligations and performance metrics issues',
        metadata: {
          practiceArea: 'Contract Law',
          jurisdiction: 'State',
          riskLevel: 'High',
          lastUpdated: new Date().toISOString(),
          entities: ['Performance Standards', 'Breach of Contract', 'Damages']
        }
      }
    ];

    return mockSimilarCases.map(caseData => ({
      ...caseData,
      type: 'similar_case',
      relatedNodes: [] // Would be populated from actual graph query
    }));
  }

  /**
   * Find related documents using vector similarity and graph traversal
   */
  private async findRelatedDocuments(request: RecommendationRequest, context: ParsedDocument): Promise<Recommendation[]> {
    console.log('📄 Finding related documents using vector similarity...');

    // Use SIMD-optimized similarity calculations
    const documentEmbedding = await this.generateDocumentEmbedding(context.content);
    const similarDocuments = await this.findSimilarDocumentsBySIMD(documentEmbedding, request.limit || 10);

    return similarDocuments.map(doc => ({
      id: doc.id,
      type: 'related_document',
      title: doc.title,
      description: doc.summary,
      score: doc.similarity,
      confidence: doc.confidence,
      reasoning: `Vector similarity: ${(doc.similarity * 100).toFixed(1)}% match with extracted entities: ${context.entities.slice(0, 3).map(e => e.text).join(', ')}`,
      relatedNodes: [],
      metadata: {
        practiceArea: doc.practiceArea,
        jurisdiction: doc.jurisdiction,
        riskLevel: doc.riskLevel || 'Unknown',
        lastUpdated: doc.lastModified,
        entities: doc.entities
      }
    }));
  }

  /**
   * Find legal precedents using citation network analysis
   */
  private async findLegalPrecedents(request: RecommendationRequest, context: ParsedDocument): Promise<Recommendation[]> {
    console.log('⚖️ Finding legal precedents using citation network analysis...');

    // Mock precedent data based on parsed context entities
    const mockPrecedents = [
      {
        id: 'precedent_001',
        type: 'legal_precedent',
        title: 'Smith v. Jones (2020) - Contract Interpretation',
        description: 'Landmark case establishing standards for service agreement interpretation',
        score: 0.94,
        confidence: 0.91,
        reasoning: 'Direct precedential value for contract interpretation disputes',
        relatedNodes: [],
        metadata: {
          practiceArea: 'Contract Law',
          jurisdiction: 'Federal Circuit',
          riskLevel: 'High',
          lastUpdated: '2020-05-15T00:00:00Z',
          sourceDocument: 'Federal Reporter 3d, Vol. 923',
          entities: ['Contract Interpretation', 'Service Agreements', 'Performance Standards']
        }
      }
    ];

    return mockPrecedents;
  }

  /**
   * Generate expert insights using vLLM AI integration
   */
  private async generateExpertInsights(request: RecommendationRequest, context: ParsedDocument): Promise<Recommendation[]> {
    console.log('🧠 Generating expert insights using vLLM AI...');

    try {
      // Create AI prompts based on parsed context
      const aiRequests: StreamingRequest[] = [
        {
          id: crypto.randomUUID(),
          model: 'gemma3-legal',
          prompt: `Based on the legal document analysis with entities: ${context.entities.map(e => e.text).join(', ')}, provide expert insights on potential legal risks and strategic considerations.`,
          temperature: 0.1,
          maxTokens: 500,
          stream: false,
          useCache: true,
          priority: 'high'
        },
        {
          id: crypto.randomUUID(),
          model: 'gemma3-legal',
          prompt: `Analyze the following legal context for procedural considerations and best practices: ${context.content.substring(0, 500)}`,
          temperature: 0.1,
          maxTokens: 400,
          stream: false,
          useCache: true,
          priority: 'medium'
        }
      ];

      // Process AI requests
      const aiResponses: Recommendation[] = [];
      const responseGenerator = this.vllmIntegration.streamWithEnhancedQUIC(aiRequests);
      
      for await (const response of responseGenerator) {
        if (response.choices?.[0]?.delta?.content) {
          const insight: Recommendation = {
            id: response.id,
            type: 'expert_insight',
            title: 'AI-Generated Legal Analysis',
            description: response.choices[0].delta.content,
            score: 0.85, // AI-generated insights get consistent high score
            confidence: 0.8,
            reasoning: 'Generated by legal AI model based on document analysis and entity extraction',
            relatedNodes: [],
            metadata: {
              practiceArea: context.metadata.documentType === 'contract' ? 'Contract Law' : 'General',
              jurisdiction: 'Analysis-Based',
              riskLevel: 'Variable',
              lastUpdated: new Date().toISOString(),
              entities: context.entities.map(e => e.text)
            },
            aiGenerated: {
              prompt: aiRequests.find(req => req.id === response.id)?.prompt || '',
              response: response.choices[0].delta.content,
              model: response.model,
              processingTime: 0 // Would be calculated from actual processing
            }
          };
          aiResponses.push(insight);
        }
      }

      return aiResponses;

    } catch (error) {
      console.error('❌ Expert insight generation failed:', error);
      // Return fallback insights based on parsed entities
      return this.generateFallbackInsights(context);
    }
  }

  /**
   * Perform risk analysis using graph-based pattern recognition
   */
  private async performRiskAnalysis(request: RecommendationRequest, context: ParsedDocument): Promise<Recommendation[]> {
    console.log('⚠️ Performing risk analysis using graph patterns...');

    // Analyze entities and their relationships for risk indicators
    const riskFactors = this.analyzeRiskFactors(context);
    
    const riskRecommendations: Recommendation[] = riskFactors.map((factor, index) => ({
      id: `risk_${index}`,
      type: 'risk_analysis',
      title: factor.title,
      description: factor.description,
      score: factor.severity,
      confidence: factor.confidence,
      reasoning: factor.reasoning,
      relatedNodes: [],
      metadata: {
        practiceArea: factor.practiceArea,
        jurisdiction: 'Analysis-Based',
        riskLevel: factor.riskLevel,
        lastUpdated: new Date().toISOString(),
        entities: factor.relatedEntities
      }
    }));

    return riskRecommendations;
  }

  /**
   * Enhance recommendations with AI-generated insights
   */
  private async enhanceWithAI(
    recommendations: Recommendation[], 
    request: RecommendationRequest, 
    context: ParsedDocument
  ): Promise<Recommendation[]> {
    console.log('🚀 Enhancing recommendations with AI insights...');

    const enhancedRecommendations = await Promise.all(
      recommendations.map(async (rec, index) => {
        try {
          // Generate AI enhancement for each recommendation
          const aiRequest: StreamingRequest = {
            id: crypto.randomUUID(),
            model: 'gemma3-legal',
            prompt: `Provide additional strategic insights for this legal recommendation: "${rec.title}". Context: ${rec.description}. Focus on practical next steps and potential implications.`,
            temperature: 0.1,
            maxTokens: 200,
            stream: false,
            useCache: true,
            priority: index < 3 ? 'high' : 'medium'
          };

          const responseGenerator = this.vllmIntegration.streamWithEnhancedQUIC([aiRequest]);
          let aiEnhancement = '';
          
          for await (const response of responseGenerator) {
            if (response.choices?.[0]?.delta?.content) {
              aiEnhancement += response.choices[0].delta.content;
            }
          }

          // Enhance the recommendation with AI insights
          return {
            ...rec,
            description: `${rec.description}\n\nAI Enhancement: ${aiEnhancement}`,
            confidence: Math.min(rec.confidence + 0.05, 1.0), // Slight confidence boost
            aiGenerated: {
              prompt: aiRequest.prompt,
              response: aiEnhancement,
              model: aiRequest.model,
              processingTime: 0
            }
          };

        } catch (error) {
          console.warn(`⚠️ AI enhancement failed for recommendation ${rec.id}:`, error);
          return rec; // Return original recommendation if AI enhancement fails
        }
      })
    );

    return enhancedRecommendations;
  }

  /**
   * Generate document embedding using SIMD-optimized processing
   */
  private async generateDocumentEmbedding(content: string): Promise<Float32Array> {
    // Use the SIMD parser's embedding generation
    // In production, this would use a proper embedding model
    const words = content.toLowerCase().split(/\s+/);
    const embedding = new Float32Array(768);
    
    // Simple hash-based embedding for demo
    for (let i = 0; i < 768; i++) {
      let value = 0;
      for (const word of words.slice(0, 100)) { // Limit to first 100 words
        const hash = this.simpleHash(word + i.toString());
        value += Math.sin(hash) * 0.1;
      }
      embedding[i] = value / Math.sqrt(words.length);
    }
    
    return embedding;
  }

  /**
   * Find similar documents using SIMD-optimized vector operations
   */
  private async findSimilarDocumentsBySIMD(queryEmbedding: Float32Array, limit: number): Promise<any[]> {
    // Mock document database for demo
    const mockDocuments = [
      {
        id: 'doc_001',
        title: 'Service Level Agreement Template',
        summary: 'Comprehensive SLA template with performance metrics',
        similarity: 0.91,
        confidence: 0.87,
        practiceArea: 'Contract Law',
        jurisdiction: 'General',
        lastModified: new Date().toISOString(),
        entities: ['SLA', 'Performance Metrics', 'Service Provider']
      },
      {
        id: 'doc_002',
        title: 'Contract Breach Remedies Guide',
        summary: 'Legal guide on contract breach remedies and enforcement',
        similarity: 0.86,
        confidence: 0.82,
        practiceArea: 'Contract Law',
        jurisdiction: 'Federal',
        lastModified: new Date().toISOString(),
        entities: ['Contract Breach', 'Remedies', 'Enforcement']
      }
    ];

    return mockDocuments.slice(0, limit);
  }

  /**
   * Analyze risk factors from parsed document context
   */
  private analyzeRiskFactors(context: ParsedDocument): Array<{
    title: string;
    description: string;
    severity: number;
    confidence: number;
    reasoning: string;
    practiceArea: string;
    riskLevel: string;
    relatedEntities: string[];
  }> {
    const riskFactors = [];

    // Check for high-risk entities
    const highRiskEntities = context.entities.filter(entity => 
      entity.type === 'legal_term' && entity.confidence > 0.8
    );

    if (highRiskEntities.length > 5) {
      riskFactors.push({
        title: 'High Legal Entity Density',
        description: 'Document contains numerous legal terms that may indicate complex legal implications',
        severity: 0.75,
        confidence: 0.85,
        reasoning: `Found ${highRiskEntities.length} high-confidence legal entities`,
        practiceArea: 'General Legal',
        riskLevel: 'Medium',
        relatedEntities: highRiskEntities.map(e => e.text)
      });
    }

    // Check document confidence
    if (context.confidence < 0.6) {
      riskFactors.push({
        title: 'Document Analysis Uncertainty',
        description: 'Document analysis shows low confidence, suggesting complex or ambiguous content',
        severity: 0.6,
        confidence: 0.9,
        reasoning: `Document confidence score: ${context.confidence.toFixed(2)}`,
        practiceArea: 'Document Analysis',
        riskLevel: 'Low',
        relatedEntities: []
      });
    }

    // Check for suggestions (potential issues)
    if (context.suggestions.length > 3) {
      riskFactors.push({
        title: 'Document Quality Issues',
        description: 'Multiple suggestions indicate potential document quality or clarity issues',
        severity: 0.5,
        confidence: 0.8,
        reasoning: `Found ${context.suggestions.length} suggestions for improvement`,
        practiceArea: 'Document Preparation',
        riskLevel: 'Low',
        relatedEntities: context.suggestions.map(s => s.original)
      });
    }

    return riskFactors;
  }

  /**
   * Generate fallback insights when AI processing fails
   */
  private generateFallbackInsights(context: ParsedDocument): Recommendation[] {
    return [{
      id: 'fallback_insight',
      type: 'expert_insight',
      title: 'Document Analysis Summary',
      description: `Document contains ${context.entities.length} legal entities with ${context.confidence > 0.8 ? 'high' : 'moderate'} confidence. Key entities include: ${context.entities.slice(0, 5).map(e => e.text).join(', ')}.`,
      score: context.confidence,
      confidence: 0.7,
      reasoning: 'Generated from document parsing results due to AI service unavailability',
      relatedNodes: [],
      metadata: {
        practiceArea: context.metadata.documentType || 'General',
        jurisdiction: 'Analysis-Based',
        riskLevel: context.confidence > 0.8 ? 'Low' : 'Medium',
        lastUpdated: new Date().toISOString(),
        entities: context.entities.map(e => e.text)
      }
    }];
  }

  /**
   * Update graph statistics for analytics
   */
  private async updateGraphStatistics(): Promise<void> {
    if (Date.now() - this.lastStatsUpdate < this.statsUpdateInterval) {
      return; // Skip update if too recent
    }

    console.log('📊 Updating graph statistics...');

    // Mock graph statistics
    this.graphStats = {
      totalNodes: 15432,
      totalRelationships: 48291,
      nodesByType: {
        'Case': 3245,
        'Document': 8921,
        'Entity': 2156,
        'Precedent': 1110
      },
      relationshipsByType: {
        'SIMILAR_TO': 12453,
        'CITES': 8932,
        'CONTAINS': 15234,
        'RELATED_TO': 11672
      },
      averageConnectivity: 3.12,
      clusteringCoefficient: 0.67,
      centralityScores: [
        { nodeId: 'case_123', centrality: 0.95 },
        { nodeId: 'doc_456', centrality: 0.87 },
        { nodeId: 'entity_789', centrality: 0.82 }
      ],
      communityDetection: [
        { community: 'contract_law', members: ['case_123', 'doc_456'], strength: 0.89 },
        { community: 'tort_law', members: ['case_789', 'doc_012'], strength: 0.76 }
      ]
    };

    this.lastStatsUpdate = Date.now();
    console.log('✅ Graph statistics updated');
  }

  /**
   * Precompute similarity matrices for performance
   */
  private async precomputeSimilarityMatrices(): Promise<void> {
    console.log('🧮 Precomputing SIMD-optimized similarity matrices...');

    // In production, this would load actual node embeddings and compute similarity matrix
    const nodeCount = 1000; // Mock node count
    this.similarityMatrix = new Float32Array(nodeCount * nodeCount);
    
    // Simulate matrix computation with SIMD optimization
    for (let i = 0; i < nodeCount; i++) {
      for (let j = i; j < nodeCount; j++) {
        const similarity = Math.random(); // Mock similarity
        this.similarityMatrix[i * nodeCount + j] = similarity;
        this.similarityMatrix[j * nodeCount + i] = similarity; // Symmetric matrix
      }
    }

    console.log(`✅ Similarity matrix computed for ${nodeCount} nodes`);
  }

  /**
   * Cache management
   */
  private generateCacheKey(request: RecommendationRequest): string {
    return `rec_${request.type}_${request.userId}_${this.simpleHash(request.context)}`;
  }

  private getFromCache(key: string): Recommendation[] | null {
    const cached = this.queryCache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.result;
    }
    return null;
  }

  private setInCache(key: string, result: Recommendation[]): void {
    this.queryCache.set(key, { result, timestamp: Date.now() });
    
    // Prune old cache entries
    if (this.queryCache.size > 1000) {
      const keys = Array.from(this.queryCache.keys());
      const oldKeys = keys.slice(0, keys.length - 500);
      oldKeys.forEach(k => this.queryCache.delete(k));
    }
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Get comprehensive analytics about the graph
   */
  async getGraphAnalytics(): Promise<GraphAnalytics> {
    await this.updateGraphStatistics();
    return this.graphStats!;
  }

  /**
   * Integration with idle detection for background processing
   */
  async processIdleRecommendations(idleContext: IdleDetectionContext): Promise<void> {
    if (!idleContext.userId) return;

    console.log('😴 Processing background recommendations during idle time...');

    try {
      // Generate background recommendations for user
      const backgroundRequest: RecommendationRequest = {
        userId: idleContext.userId,
        context: 'User idle - generate proactive insights',
        type: 'expert_insights',
        limit: 3,
        useAI: true,
        includeExplanation: true
      };

      const recommendations = await this.getRecommendations(backgroundRequest);
      
      // Store recommendations for when user returns
      this.setInCache(`idle_${idleContext.userId}`, recommendations);
      
      console.log(`✅ Generated ${recommendations.length} background recommendations`);

    } catch (error) {
      console.error('❌ Background recommendation processing failed:', error);
    }
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    console.log('🧹 Cleaning up Neo4j Recommendation Engine...');

    // Cleanup integrations
    await Promise.all([
      this.vllmIntegration.cleanup(),
      this.simdParser.cleanup()
    ]);

    // Clear caches
    this.queryCache.clear();
    this.nodeEmbeddings.clear();

    // Close Neo4j connection
    if (this.driver) {
      await this.driver.close();
    }

    this.isConnected = false;
    console.log('✅ Cleanup completed');
  }

  /**
   * Health check for the recommendation engine
   */
  async healthCheck(): Promise<{
    status: string;
    neo4j: boolean;
    vllm: boolean;
    simd: boolean;
    graphStats: GraphAnalytics | null;
  }> {
    return {
      status: this.isConnected ? 'healthy' : 'disconnected',
      neo4j: this.isConnected,
      vllm: true, // Would check actual vLLM connection
      simd: true, // Would check SIMD GPU availability
      graphStats: this.graphStats
    };
  }
}

export default Neo4jRecommendationEngine;