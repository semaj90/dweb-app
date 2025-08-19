/**
 * Transformers for Summarization Pipeline to Neo4j
 * Complete AI pipeline with graph knowledge integration
 */

import { Driver, Session, Record, auth, driver } from 'neo4j-driver';
import { langChainOllamaService } from './langchain-ollama-llama-integration';
import { vectorProxy } from './grpc-quic-vector-proxy';

export interface DocumentSummary {
  id: string;
  title: string;
  summary: string;
  keyPoints: string[];
  entities: LegalEntity[];
  relationships: Relationship[];
  confidence: number;
  processingTime: number;
  graphNodes: GraphNode[];
}

export interface LegalEntity {
  id: string;
  type: 'person' | 'organization' | 'case' | 'statute' | 'precedent' | 'contract';
  name: string;
  attributes: Record<string, any>;
  confidence: number;
}

export interface Relationship {
  from: string;
  to: string;
  type: string;
  strength: number;
  metadata: Record<string, any>;
}

export interface GraphNode {
  id: string;
  labels: string[];
  properties: Record<string, any>;
  embedding?: number[];
}

export interface SummarizationConfig {
  neo4j: {
    uri: string;
    username: string;
    password: string;
    database?: string;
  };
  transformers: {
    model: 'gemma3-legal';
    maxTokens: number;
    temperature: number;
    chunkSize: number;
    overlapSize: number;
  };
  graph: {
    enableRelationshipExtraction: boolean;
    enableEntityLinking: boolean;
    confidenceThreshold: number;
    maxDepth: number;
  };
}

/**
 * Neo4j Transformers Summarization Pipeline
 * Integrates with LangChain, Vector Proxy, and Graph Database
 */
export class Neo4jTransformersSummarization {
  private driver: Driver | null = null;
  private session: Session | null = null;
  private config: SummarizationConfig;
  private isInitialized = false;

  constructor(config: Partial<SummarizationConfig> = {}) {
    this.config = {
      neo4j: {
        uri: 'bolt://localhost:7687',
        username: 'neo4j',
        password: 'legal-ai-2024',
        database: 'legal-graph',
        ...config.neo4j
      },
      transformers: {
        model: 'gemma3-legal',
        maxTokens: 2048,
        temperature: 0.1,
        chunkSize: 1000,
        overlapSize: 200,
        ...config.transformers
      },
      graph: {
        enableRelationshipExtraction: true,
        enableEntityLinking: true,
        confidenceThreshold: 0.7,
        maxDepth: 3,
        ...config.graph
      }
    };
  }

  /**
   * Initialize Neo4j connection and graph schema
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('🔗 Initializing Neo4j transformers summarization pipeline...');

    try {
      // Initialize Neo4j driver
      this.driver = driver(
        this.config.neo4j.uri,
        auth.basic(this.config.neo4j.username, this.config.neo4j.password)
      );

      // Test connection
      await this.driver.verifyConnectivity();
      this.session = this.driver.session({ database: this.config.neo4j.database });

      // Initialize graph schema
      await this.initializeGraphSchema();

      // Test integrations
      await this.testServiceIntegrations();

      this.isInitialized = true;
      console.log('✅ Neo4j transformers pipeline initialized successfully');

    } catch (error) {
      console.error('❌ Neo4j initialization failed:', error);
      throw error;
    }
  }

  /**
   * Initialize Neo4j graph schema for legal documents
   */
  private async initializeGraphSchema(): Promise<void> {
    const schemaQueries = [
      // Create constraints and indexes
      `CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE`,
      `CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE`,
      `CREATE CONSTRAINT IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE`,
      `CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE`,
      
      // Create vector indexes for embeddings (if Neo4j supports them)
      `CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.embedding)`,
      `CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.embedding)`,
      
      // Create text indexes for full-text search
      `CREATE FULLTEXT INDEX IF NOT EXISTS documentFullText FOR (d:Document) ON EACH [d.title, d.content, d.summary]`,
      `CREATE FULLTEXT INDEX IF NOT EXISTS entityFullText FOR (e:Entity) ON EACH [e.name, e.description]`
    ];

    for (const query of schemaQueries) {
      try {
        await this.session!.run(query);
        console.log(`✅ Schema query executed: ${query.slice(0, 50)}...`);
      } catch (error) {
        console.warn(`⚠️ Schema query failed (may already exist): ${query.slice(0, 50)}...`);
      }
    }
  }

  /**
   * Test integration with other services
   */
  private async testServiceIntegrations(): Promise<void> {
    try {
      // Test LangChain Ollama service
      const status = langChainOllamaService.getStatus();
      console.log(`  ✅ LangChain Ollama: ${status.initialized ? 'Connected' : 'Not initialized'}`);

      // Test Vector Proxy
      const vectorHealth = await vectorProxy.healthCheck();
      const healthyProtocols = Object.entries(vectorHealth).filter(([, v]) => v.status === 'healthy').length;
      console.log(`  ✅ Vector Proxy: ${healthyProtocols}/3 protocols healthy`);

    } catch (error) {
      console.warn('⚠️ Service integration test failed:', error);
    }
  }

  /**
   * Process legal document through complete pipeline
   */
  async processDocument(
    documentId: string,
    title: string,
    content: string,
    metadata: Record<string, any> = {}
  ): Promise<DocumentSummary> {
    await this.initialize();

    const startTime = performance.now();
    console.log(`📄 Processing document: ${title} (${content.length} chars)`);

    try {
      // Step 1: Generate summary using LangChain + gemma3-legal
      const summary = await this.generateDocumentSummary(content);
      console.log('✅ Summary generated');

      // Step 2: Extract legal entities using transformers
      const entities = await this.extractLegalEntities(content);
      console.log(`✅ Extracted ${entities.length} entities`);

      // Step 3: Generate embeddings for semantic search
      const embedding = await langChainOllamaService.generateEmbedding(content);
      const summaryEmbedding = await langChainOllamaService.generateEmbedding(summary);
      console.log('✅ Embeddings generated');

      // Step 4: Extract relationships between entities
      const relationships = await this.extractRelationships(entities, content);
      console.log(`✅ Extracted ${relationships.length} relationships`);

      // Step 5: Store in Neo4j graph database
      const graphNodes = await this.storeInGraph(documentId, title, content, summary, entities, relationships, embedding);
      console.log(`✅ Stored ${graphNodes.length} graph nodes`);

      // Step 6: Store embeddings in vector proxy for search
      await this.storeVectorEmbeddings(documentId, embedding, summaryEmbedding, metadata);
      console.log('✅ Vector embeddings stored');

      const processingTime = performance.now() - startTime;

      const result: DocumentSummary = {
        id: documentId,
        title,
        summary,
        keyPoints: this.extractKeyPoints(summary),
        entities,
        relationships,
        confidence: 0.85,
        processingTime,
        graphNodes
      };

      console.log(`🎯 Document processing complete (${processingTime.toFixed(2)}ms)`);
      return result;

    } catch (error) {
      console.error('❌ Document processing failed:', error);
      throw error;
    }
  }

  /**
   * Generate comprehensive summary using gemma3-legal model
   */
  private async generateDocumentSummary(content: string): Promise<string> {
    try {
      const summaryResult = await langChainOllamaService.ragQuery(
        `Provide a comprehensive legal summary of this document. Focus on:
        - Key legal concepts and principles
        - Parties involved and their roles
        - Important dates and deadlines
        - Legal obligations and rights
        - Potential issues or concerns
        
        Document content: ${content.slice(0, 4000)}`,
        [],
        true // Use GPU acceleration
      );

      return summaryResult.answer || 'Summary could not be generated';

    } catch (error) {
      console.error('❌ Summary generation failed:', error);
      return 'Summary generation failed due to processing error';
    }
  }

  /**
   * Extract legal entities using advanced NLP and transformers
   */
  private async extractLegalEntities(content: string): Promise<LegalEntity[]> {
    const entities: LegalEntity[] = [];

    try {
      // Use LangChain for entity extraction
      const entityExtractionResult = await langChainOllamaService.ragQuery(
        `Extract legal entities from this document. Return a JSON array with the following format:
        [{"type": "person|organization|case|statute|precedent|contract", "name": "entity name", "context": "surrounding context"}]
        
        Document: ${content.slice(0, 3000)}`,
        [],
        true
      );

      // Parse entity extraction results
      try {
        const extractedEntities = JSON.parse(entityExtractionResult.answer || '[]');
        
        for (let i = 0; i < extractedEntities.length; i++) {
          const entity = extractedEntities[i];
          entities.push({
            id: `entity-${crypto.randomUUID()}`,
            type: entity.type || 'person',
            name: entity.name || `Unknown Entity ${i + 1}`,
            attributes: {
              context: entity.context || '',
              source: 'transformers-extraction',
              confidence: 0.8
            },
            confidence: 0.8
          });
        }
      } catch (parseError) {
        console.warn('⚠️ Entity extraction JSON parsing failed, using regex fallback');
        
        // Fallback: regex-based entity extraction
        entities.push(...this.extractEntitiesWithRegex(content));
      }

      // Additional pattern-based extraction
      entities.push(...this.extractEntitiesWithRegex(content));

      // Deduplicate entities by name
      const uniqueEntities = entities.filter((entity, index, self) =>
        index === self.findIndex(e => e.name.toLowerCase() === entity.name.toLowerCase())
      );

      console.log(`🎯 Extracted ${uniqueEntities.length} unique entities`);
      return uniqueEntities;

    } catch (error) {
      console.error('❌ Entity extraction failed:', error);
      return this.extractEntitiesWithRegex(content);
    }
  }

  /**
   * Regex-based entity extraction fallback
   */
  private extractEntitiesWithRegex(content: string): LegalEntity[] {
    const entities: LegalEntity[] = [];

    // Extract case names (Pattern: Party v. Party)
    const caseNames = content.match(/[A-Z][a-zA-Z\s]+ v\. [A-Z][a-zA-Z\s]+/g) || [];
    caseNames.forEach(name => {
      entities.push({
        id: `case-${crypto.randomUUID()}`,
        type: 'case',
        name: name.trim(),
        attributes: { extractionMethod: 'regex', pattern: 'case_name' },
        confidence: 0.7
      });
    });

    // Extract people with titles
    const people = content.match(/(?:Mr\.|Ms\.|Dr\.|Judge|Justice|Attorney)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g) || [];
    people.forEach(person => {
      entities.push({
        id: `person-${crypto.randomUUID()}`,
        type: 'person',
        name: person.trim(),
        attributes: { extractionMethod: 'regex', pattern: 'titled_person' },
        confidence: 0.6
      });
    });

    // Extract organizations
    const orgs = content.match(/\b[A-Z][a-zA-Z\s]*(?:Inc\.|Corp\.|LLC|Ltd\.|Company|Corporation|Association)\b/g) || [];
    orgs.forEach(org => {
      entities.push({
        id: `org-${crypto.randomUUID()}`,
        type: 'organization',
        name: org.trim(),
        attributes: { extractionMethod: 'regex', pattern: 'organization' },
        confidence: 0.6
      });
    });

    return entities.slice(0, 30); // Limit to 30 entities
  }

  /**
   * Extract relationships between entities
   */
  private async extractRelationships(entities: LegalEntity[], content: string): Promise<Relationship[]> {
    const relationships: Relationship[] = [];

    try {
      // Use AI to identify relationships
      const relationshipPrompt = `
        Analyze the relationships between these entities in the legal document:
        ${entities.map(e => `${e.type}: ${e.name}`).join('\n')}
        
        Return JSON array of relationships in format:
        [{"from": "entity1", "to": "entity2", "type": "relationship_type", "strength": 0.8}]
        
        Common legal relationships: represents, employed_by, sued_by, cited_in, related_to, owns, contracts_with
        
        Document context: ${content.slice(0, 2000)}
      `;

      const relationshipResult = await langChainOllamaService.ragQuery(relationshipPrompt, [], true);

      try {
        const extractedRelationships = JSON.parse(relationshipResult.answer || '[]');
        
        extractedRelationships.forEach((rel: any) => {
          const fromEntity = entities.find(e => e.name === rel.from);
          const toEntity = entities.find(e => e.name === rel.to);
          
          if (fromEntity && toEntity) {
            relationships.push({
              from: fromEntity.id,
              to: toEntity.id,
              type: rel.type || 'related_to',
              strength: rel.strength || 0.5,
              metadata: {
                extractionMethod: 'ai_transformers',
                source: 'gemma3-legal'
              }
            });
          }
        });

      } catch (parseError) {
        console.warn('⚠️ Relationship JSON parsing failed, using heuristic approach');
        relationships.push(...this.extractHeuristicRelationships(entities, content));
      }

    } catch (error) {
      console.warn('⚠️ AI relationship extraction failed, using heuristic fallback:', error);
      relationships.push(...this.extractHeuristicRelationships(entities, content));
    }

    console.log(`🔗 Extracted ${relationships.length} relationships`);
    return relationships;
  }

  /**
   * Heuristic relationship extraction based on proximity and patterns
   */
  private extractHeuristicRelationships(entities: LegalEntity[], content: string): Relationship[] {
    const relationships: Relationship[] = [];
    const contentLower = content.toLowerCase();

    for (let i = 0; i < entities.length; i++) {
      for (let j = i + 1; j < entities.length; j++) {
        const entity1 = entities[i];
        const entity2 = entities[j];

        // Check if entities appear close together in text
        const entity1Pos = contentLower.indexOf(entity1.name.toLowerCase());
        const entity2Pos = contentLower.indexOf(entity2.name.toLowerCase());

        if (entity1Pos !== -1 && entity2Pos !== -1) {
          const distance = Math.abs(entity1Pos - entity2Pos);
          
          // If entities appear within 200 characters, they might be related
          if (distance < 200) {
            let relationshipType = 'mentioned_together';
            let strength = Math.max(0.3, 1 - (distance / 200));

            // Infer relationship type based on patterns
            const contextBetween = content.slice(
              Math.min(entity1Pos, entity2Pos),
              Math.max(entity1Pos, entity2Pos) + Math.max(entity1.name.length, entity2.name.length)
            );

            if (contextBetween.includes(' v. ') || contextBetween.includes(' vs ')) {
              relationshipType = 'legal_opposition';
              strength = 0.9;
            } else if (contextBetween.includes('represent') || contextBetween.includes('attorney')) {
              relationshipType = 'legal_representation';
              strength = 0.8;
            } else if (contextBetween.includes('employ') || contextBetween.includes('work')) {
              relationshipType = 'employment';
              strength = 0.7;
            }

            relationships.push({
              from: entity1.id,
              to: entity2.id,
              type: relationshipType,
              strength,
              metadata: {
                extractionMethod: 'heuristic',
                distance,
                contextSnippet: contextBetween.slice(0, 100)
              }
            });
          }
        }
      }
    }

    return relationships;
  }

  /**
   * Store document, entities, and relationships in Neo4j graph
   */
  private async storeInGraph(
    documentId: string,
    title: string,
    content: string,
    summary: string,
    entities: LegalEntity[],
    relationships: Relationship[],
    embedding: number[]
  ): Promise<GraphNode[]> {
    if (!this.session) throw new Error('Neo4j session not initialized');

    const graphNodes: GraphNode[] = [];

    try {
      // Store document node
      const documentQuery = `
        MERGE (d:Document {id: $documentId})
        SET d.title = $title,
            d.content = $content,
            d.summary = $summary,
            d.embedding = $embedding,
            d.wordCount = $wordCount,
            d.updatedAt = datetime(),
            d.source = 'transformers_pipeline'
        RETURN d
      `;

      const documentResult = await this.session.run(documentQuery, {
        documentId,
        title,
        content: content.slice(0, 10000), // Limit content size
        summary,
        embedding,
        wordCount: content.split(/\s+/).length
      });

      if (documentResult.records.length > 0) {
        graphNodes.push({
          id: documentId,
          labels: ['Document'],
          properties: { title, summary, wordCount: content.split(/\s+/).length },
          embedding
        });
      }

      // Store entities
      for (const entity of entities) {
        const entityQuery = `
          MERGE (e:Entity:${entity.type.charAt(0).toUpperCase() + entity.type.slice(1)} {id: $entityId})
          SET e.name = $name,
              e.type = $type,
              e.confidence = $confidence,
              e.attributes = $attributes,
              e.updatedAt = datetime()
          RETURN e
        `;

        const entityResult = await this.session.run(entityQuery, {
          entityId: entity.id,
          name: entity.name,
          type: entity.type,
          confidence: entity.confidence,
          attributes: entity.attributes
        });

        if (entityResult.records.length > 0) {
          graphNodes.push({
            id: entity.id,
            labels: ['Entity', entity.type.charAt(0).toUpperCase() + entity.type.slice(1)],
            properties: { name: entity.name, type: entity.type, confidence: entity.confidence }
          });
        }

        // Connect entity to document
        await this.session.run(`
          MATCH (d:Document {id: $documentId})
          MATCH (e:Entity {id: $entityId})
          MERGE (d)-[:MENTIONS {confidence: $confidence}]->(e)
        `, {
          documentId,
          entityId: entity.id,
          confidence: entity.confidence
        });
      }

      // Store relationships between entities
      for (const relationship of relationships) {
        const relationshipQuery = `
          MATCH (from:Entity {id: $fromId})
          MATCH (to:Entity {id: $toId})
          MERGE (from)-[r:${relationship.type.toUpperCase().replace(/[^A-Z_]/g, '_')} {
            strength: $strength,
            metadata: $metadata,
            createdAt: datetime()
          }]->(to)
          RETURN r
        `;

        await this.session.run(relationshipQuery, {
          fromId: relationship.from,
          toId: relationship.to,
          strength: relationship.strength,
          metadata: relationship.metadata
        });
      }

      console.log(`🌐 Graph storage complete: ${graphNodes.length} nodes, ${relationships.length} relationships`);
      return graphNodes;

    } catch (error) {
      console.error('❌ Graph storage failed:', error);
      throw error;
    }
  }

  /**
   * Store vector embeddings via proxy for search functionality
   */
  private async storeVectorEmbeddings(
    documentId: string,
    documentEmbedding: number[],
    summaryEmbedding: number[],
    metadata: Record<string, any>
  ): Promise<void> {
    try {
      // Store document embedding
      await vectorProxy.store(
        `doc-${documentId}`,
        documentEmbedding,
        {
          ...metadata,
          type: 'document',
          model: 'nomic-embed-text',
          neo4j_node_id: documentId
        }
      );

      // Store summary embedding
      await vectorProxy.store(
        `summary-${documentId}`,
        summaryEmbedding,
        {
          ...metadata,
          type: 'summary',
          model: 'nomic-embed-text',
          neo4j_node_id: documentId,
          parent_document: documentId
        }
      );

      console.log('📊 Vector embeddings stored successfully');

    } catch (error) {
      console.error('❌ Vector embedding storage failed:', error);
      // Don't throw - this is not critical for main processing
    }
  }

  /**
   * Extract key points from summary
   */
  private extractKeyPoints(summary: string): string[] {
    // Split summary into sentences and extract key points
    const sentences = summary.split(/[.!?]+/).filter(s => s.trim().length > 10);
    
    const keyPoints = sentences
      .filter(sentence => {
        const lower = sentence.toLowerCase();
        return lower.includes('key') || 
               lower.includes('important') || 
               lower.includes('significant') ||
               lower.includes('must') ||
               lower.includes('shall') ||
               lower.includes('required');
      })
      .map(sentence => sentence.trim())
      .slice(0, 5); // Top 5 key points

    return keyPoints.length > 0 ? keyPoints : sentences.slice(0, 3);
  }

  /**
   * Search documents using graph traversal and vector similarity
   */
  async searchDocuments(
    query: string,
    options: {
      includeEntities?: boolean;
      includeRelationships?: boolean;
      maxDepth?: number;
      vectorThreshold?: number;
      limit?: number;
    } = {}
  ): Promise<DocumentSummary[]> {
    await this.initialize();

    const searchOptions = {
      includeEntities: true,
      includeRelationships: true,
      maxDepth: 2,
      vectorThreshold: 0.7,
      limit: 10,
      ...options
    };

    console.log(`🔍 Searching documents: "${query}"`);

    try {
      // Step 1: Generate query embedding for vector search
      const queryEmbedding = await langChainOllamaService.generateEmbedding(query);

      // Step 2: Vector similarity search via proxy
      const vectorResults = await vectorProxy.search(queryEmbedding, {
        query,
        threshold: searchOptions.vectorThreshold,
        limit: searchOptions.limit * 2,
        useGPU: true
      });

      // Step 3: Get document IDs from vector results
      const documentIds = vectorResults.success ? 
        (vectorResults.data?.map((result: any) => result.metadata?.neo4j_node_id).filter(Boolean) || []) : 
        [];

      // Step 4: Graph traversal to get full document context
      const searchResults: DocumentSummary[] = [];

      for (const docId of documentIds.slice(0, searchOptions.limit)) {
        try {
          const graphQuery = `
            MATCH (d:Document {id: $documentId})
            OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
            OPTIONAL MATCH (e)-[r]-(related:Entity)
            RETURN d, collect(DISTINCT e) as entities, collect(DISTINCT {rel: r, entity: related}) as relationships
          `;

          const result = await this.session!.run(graphQuery, { documentId: docId });

          if (result.records.length > 0) {
            const record = result.records[0];
            const doc = record.get('d').properties;
            const entities = record.get('entities').map((e: any) => ({
              id: e.properties.id,
              type: e.properties.type,
              name: e.properties.name,
              confidence: e.properties.confidence,
              attributes: e.properties.attributes || {}
            }));

            const relationships = record.get('relationships')
              .filter((r: any) => r.rel !== null)
              .map((r: any) => ({
                from: r.rel.start,
                to: r.rel.end,
                type: r.rel.type,
                strength: r.rel.properties.strength || 0.5,
                metadata: r.rel.properties.metadata || {}
              }));

            searchResults.push({
              id: doc.id,
              title: doc.title,
              summary: doc.summary,
              keyPoints: this.extractKeyPoints(doc.summary || ''),
              entities: searchOptions.includeEntities ? entities : [],
              relationships: searchOptions.includeRelationships ? relationships : [],
              confidence: 0.8,
              processingTime: 0,
              graphNodes: []
            });
          }

        } catch (error) {
          console.warn(`⚠️ Failed to fetch document ${docId}:`, error);
        }
      }

      console.log(`📊 Document search complete: ${searchResults.length} results`);
      return searchResults;

    } catch (error) {
      console.error('❌ Document search failed:', error);
      throw error;
    }
  }

  /**
   * Get document relationships and entity connections
   */
  async getDocumentConnections(documentId: string, maxDepth = 2): Promise<{
    connectedDocuments: DocumentSummary[];
    entityNetwork: LegalEntity[];
    relationshipPaths: Relationship[][];
  }> {
    await this.initialize();

    try {
      const connectionQuery = `
        MATCH (d:Document {id: $documentId})
        MATCH path = (d)-[:MENTIONS*1..${maxDepth}]-(connected:Document)
        WHERE connected.id <> $documentId
        RETURN DISTINCT connected, length(path) as distance
        ORDER BY distance ASC
        LIMIT 20
      `;

      const result = await this.session!.run(connectionQuery, { documentId });
      
      const connectedDocuments: DocumentSummary[] = [];
      
      for (const record of result.records) {
        const connectedDoc = record.get('connected').properties;
        
        // Get entities for connected document
        const entitiesResult = await this.session!.run(`
          MATCH (d:Document {id: $docId})-[:MENTIONS]->(e:Entity)
          RETURN e
        `, { docId: connectedDoc.id });

        const entities = entitiesResult.records.map(r => {
          const e = r.get('e').properties;
          return {
            id: e.id,
            type: e.type,
            name: e.name,
            confidence: e.confidence,
            attributes: e.attributes || {}
          };
        });

        connectedDocuments.push({
          id: connectedDoc.id,
          title: connectedDoc.title,
          summary: connectedDoc.summary,
          keyPoints: this.extractKeyPoints(connectedDoc.summary || ''),
          entities,
          relationships: [],
          confidence: 0.8,
          processingTime: 0,
          graphNodes: []
        });
      }

      console.log(`🌐 Found ${connectedDocuments.length} connected documents`);
      
      return {
        connectedDocuments,
        entityNetwork: [],
        relationshipPaths: []
      };

    } catch (error) {
      console.error('❌ Failed to get document connections:', error);
      throw error;
    }
  }

  /**
   * Generate comprehensive legal analysis using graph context
   */
  async generateGraphEnhancedAnalysis(
    query: string,
    documentIds: string[] = []
  ): Promise<{
    analysis: string;
    relevantDocuments: DocumentSummary[];
    entityInsights: LegalEntity[];
    confidence: number;
  }> {
    await this.initialize();

    console.log(`⚖️ Generating graph-enhanced legal analysis for: "${query}"`);

    try {
      // Step 1: Get relevant documents via search if none provided
      let relevantDocuments: DocumentSummary[] = [];
      if (documentIds.length === 0) {
        relevantDocuments = await this.searchDocuments(query, { limit: 5 });
      } else {
        // Get specific documents
        for (const docId of documentIds) {
          const searchResult = await this.searchDocuments(docId, { limit: 1 });
          relevantDocuments.push(...searchResult);
        }
      }

      // Step 2: Extract all entities from relevant documents
      const allEntities = relevantDocuments.flatMap(doc => doc.entities);
      const uniqueEntities = allEntities.filter((entity, index, self) =>
        index === self.findIndex(e => e.id === entity.id)
      );

      // Step 3: Get graph context
      const graphContext = relevantDocuments.map(doc => 
        `Document: ${doc.title}\nSummary: ${doc.summary}\nEntities: ${doc.entities.map(e => e.name).join(', ')}`
      ).join('\n\n');

      // Step 4: Generate enhanced analysis using all context
      const analysisPrompt = `
        As a legal AI assistant, provide a comprehensive analysis of the following query using the provided graph context.
        
        Query: ${query}
        
        Graph Context:
        ${graphContext}
        
        Available Entities: ${uniqueEntities.map(e => `${e.type}: ${e.name}`).join(', ')}
        
        Please provide:
        1. Direct analysis of the query
        2. Relevant legal precedents and connections
        3. Entity relationships that matter
        4. Potential legal implications
        5. Recommendations for further investigation
        
        Analysis:
      `;

      const analysisResult = await langChainOllamaService.ragQuery(
        analysisPrompt,
        relevantDocuments.map(doc => doc.summary),
        true // Use GPU acceleration
      );

      const result = {
        analysis: analysisResult.answer || 'Analysis could not be generated',
        relevantDocuments,
        entityInsights: uniqueEntities.slice(0, 10),
        confidence: analysisResult.confidence || 0.75
      };

      console.log(`✅ Graph-enhanced analysis complete (confidence: ${result.confidence})`);
      return result;

    } catch (error) {
      console.error('❌ Graph-enhanced analysis failed:', error);
      throw error;
    }
  }

  /**
   * Batch process multiple documents
   */
  async batchProcessDocuments(
    documents: Array<{
      id: string;
      title: string;
      content: string;
      metadata?: Record<string, any>;
    }>,
    batchSize = 3
  ): Promise<DocumentSummary[]> {
    await this.initialize();

    console.log(`📚 Batch processing ${documents.length} documents (batch size: ${batchSize})`);

    const results: DocumentSummary[] = [];

    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);
      console.log(`⚡ Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(documents.length / batchSize)}`);

      const batchPromises = batch.map(doc =>
        this.processDocument(doc.id, doc.title, doc.content, doc.metadata)
      );

      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);

      // Small delay between batches to prevent overwhelming services
      if (i + batchSize < documents.length) {
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }

    console.log(`✅ Batch processing complete: ${results.length} documents processed`);
    return results;
  }

  /**
   * Get service status and health
   */
  async getStatus(): Promise<{
    neo4j: { connected: boolean; database: string };
    integrations: { langchain: boolean; vectorProxy: boolean };
    performance: { documentsProcessed: number; avgProcessingTime: number };
  }> {
    const status = {
      neo4j: { connected: false, database: this.config.neo4j.database || 'default' },
      integrations: { langchain: false, vectorProxy: false },
      performance: { documentsProcessed: 0, avgProcessingTime: 0 }
    };

    try {
      // Test Neo4j connection
      if (this.session) {
        const result = await this.session.run('RETURN 1 as test');
        status.neo4j.connected = result.records.length > 0;
      }

      // Test LangChain integration
      const langchainStatus = langChainOllamaService.getStatus();
      status.integrations.langchain = langchainStatus.initialized;

      // Test vector proxy
      const vectorHealth = await vectorProxy.healthCheck();
      status.integrations.vectorProxy = Object.values(vectorHealth).some((v: any) => v.status === 'healthy');

      // Get document count from Neo4j
      if (status.neo4j.connected) {
        const docCountResult = await this.session!.run('MATCH (d:Document) RETURN count(d) as count');
        status.performance.documentsProcessed = docCountResult.records[0]?.get('count')?.toNumber() || 0;
      }

    } catch (error) {
      console.warn('⚠️ Status check failed:', error);
    }

    return status;
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    try {
      if (this.session) {
        await this.session.close();
        this.session = null;
      }
      
      if (this.driver) {
        await this.driver.close();
        this.driver = null;
      }
      
      this.isInitialized = false;
      console.log('🧹 Neo4j transformers pipeline cleaned up');
    } catch (error) {
      console.error('❌ Cleanup failed:', error);
    }
  }
}

// Global service instance
export const neo4jSummarization = new Neo4jTransformersSummarization({
  neo4j: {
    uri: 'bolt://localhost:7687',
    username: 'neo4j',
    password: 'legal-ai-2024',
    database: 'legal-graph'
  },
  transformers: {
    model: 'gemma3-legal',
    maxTokens: 2048,
    temperature: 0.1,
    chunkSize: 1000,
    overlapSize: 200
  },
  graph: {
    enableRelationshipExtraction: true,
    enableEntityLinking: true,
    confidenceThreshold: 0.7,
    maxDepth: 3
  }
});

// Auto-initialize on import (server-side only)
if (typeof window === 'undefined') {
  neo4jSummarization.initialize().catch(console.warn);
}

// Legal AI specific operations
export class LegalGraphOperations {
  /**
   * Process legal case with full graph analysis
   */
  static async processLegalCase(
    caseId: string,
    caseTitle: string,
    documents: Array<{ title: string; content: string }>
  ): Promise<{
    caseAnalysis: string;
    documentSummaries: DocumentSummary[];
    entityNetwork: LegalEntity[];
    precedentConnections: any[];
  }> {
    console.log(`⚖️ Processing legal case: ${caseTitle}`);

    // Process all case documents
    const documentSummaries = await neo4jSummarization.batchProcessDocuments(
      documents.map((doc, index) => ({
        id: `${caseId}-doc-${index}`,
        title: doc.title,
        content: doc.content,
        metadata: { caseId, caseTitle }
      }))
    );

    // Generate case-wide analysis
    const caseAnalysis = await neo4jSummarization.generateGraphEnhancedAnalysis(
      `Provide a comprehensive legal analysis for case: ${caseTitle}`,
      documentSummaries.map(doc => doc.id)
    );

    // Extract entity network
    const entityNetwork = documentSummaries.flatMap(doc => doc.entities);
    const uniqueEntityNetwork = entityNetwork.filter((entity, index, self) =>
      index === self.findIndex(e => e.name === entity.name)
    );

    return {
      caseAnalysis: caseAnalysis.analysis,
      documentSummaries,
      entityNetwork: uniqueEntityNetwork,
      precedentConnections: [] // Future: implement precedent analysis
    };
  }

  /**
   * Search for similar cases using graph connections
   */
  static async findSimilarCases(
    query: string,
    currentCaseId?: string
  ): Promise<DocumentSummary[]> {
    const searchResults = await neo4jSummarization.searchDocuments(query, {
      includeEntities: true,
      includeRelationships: true,
      limit: 15
    });

    // Filter out current case if provided
    return currentCaseId ? 
      searchResults.filter(doc => !doc.id.includes(currentCaseId)) : 
      searchResults;
  }

  /**
   * Analyze legal precedents and citations
   */
  static async analyzePrecedents(
    documentContent: string
  ): Promise<{
    citations: string[];
    precedentAnalysis: string;
    relevanceScores: Record<string, number>;
  }> {
    // Extract citations using regex
    const citations = documentContent.match(/\d+\s+[A-Z][a-z]+\.?\s*\d+/g) || [];
    
    // Generate precedent analysis
    const precedentAnalysis = await neo4jSummarization.generateGraphEnhancedAnalysis(
      `Analyze the legal precedents and citations in this document: ${documentContent.slice(0, 2000)}`
    );

    // Mock relevance scores (future: implement citation analysis)
    const relevanceScores: Record<string, number> = {};
    citations.forEach(citation => {
      relevanceScores[citation] = Math.random() * 0.5 + 0.5; // 0.5-1.0 range
    });

    return {
      citations: [...new Set(citations)],
      precedentAnalysis: precedentAnalysis.analysis,
      relevanceScores
    };
  }
}

export { LegalGraphOperations };