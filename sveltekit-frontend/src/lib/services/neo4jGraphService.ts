/**
 * Neo4j Graph Service - Enhanced RAG with Knowledge Graph
 * Integrates graph relationships with vector search for contextual AI
 */

import { writable, derived } from "svelte/store";
import { productionServiceClient } from "$lib/api/production-client";

export interface GraphNode {
  id: string;
  labels: string[];
  properties: Record<string, any>;
  embedding?: number[];
}

export interface GraphRelationship {
  id: string;
  type: string;
  startNode: string;
  endNode: string;
  properties: Record<string, any>;
}

export interface GraphQuery {
  cypher: string;
  parameters?: Record<string, any>;
  includeEmbeddings?: boolean;
  maxResults?: number;
}

export interface GraphSearchResult {
  nodes: GraphNode[];
  relationships: GraphRelationship[];
  paths: Array<{
    nodes: GraphNode[];
    relationships: GraphRelationship[];
    similarity?: number;
  }>;
  metadata: {
    queryTime: number;
    resultCount: number;
    graphTraversalDepth: number;
  };
}

export interface LegalGraphSchema {
  // Legal Entity Types
  case: {
    properties: ['caseNumber', 'title', 'status', 'jurisdiction', 'category'];
    relationships: ['HAS_EVIDENCE', 'INVOLVES_PERSON', 'CITES_PRECEDENT', 'RELATED_TO'];
  };
  evidence: {
    properties: ['type', 'description', 'collectedDate', 'location', 'chainOfCustody'];
    relationships: ['BELONGS_TO_CASE', 'ANALYZED_BY', 'REFERENCES', 'CONTRADICTS'];
  };
  person: {
    properties: ['name', 'role', 'dateOfBirth', 'address', 'occupation'];
    relationships: ['INVOLVED_IN', 'WITNESS_OF', 'VICTIM_OF', 'SUSPECT_IN', 'KNOWS'];
  };
  precedent: {
    properties: ['citation', 'court', 'year', 'category', 'outcome', 'summary'];
    relationships: ['CITED_BY', 'OVERRULES', 'DISTINGUISHES', 'FOLLOWS'];
  };
  document: {
    properties: ['filename', 'type', 'content', 'dateCreated', 'author'];
    relationships: ['ATTACHED_TO', 'REFERENCES', 'AMENDED_BY', 'SUPERSEDES'];
  };
}

export interface Neo4jStats {
  // Database Stats
  nodeCount: number;
  relationshipCount: number;
  labelCount: number;
  propertyKeyCount: number;
  
  // Performance Stats
  avgQueryTime: number;
  queryCount: number;
  cacheHitRatio: number;
  
  // Graph Stats
  maxDepth: number;
  avgNodeDegree: number;
  connectedComponents: number;
  
  // Vector Integration
  nodesWithEmbeddings: number;
  embeddingDimensions: number;
  vectorSimilarityQueries: number;
}

const initialStats: Neo4jStats = {
  nodeCount: 0,
  relationshipCount: 0,
  labelCount: 0,
  propertyKeyCount: 0,
  avgQueryTime: 0,
  queryCount: 0,
  cacheHitRatio: 0,
  maxDepth: 0,
  avgNodeDegree: 0,
  connectedComponents: 0,
  nodesWithEmbeddings: 0,
  embeddingDimensions: 768,
  vectorSimilarityQueries: 0
};

// Core stores
export const neo4jStatsStore = writable<Neo4jStats>(initialStats);
export const graphSchemaStore = writable<LegalGraphSchema>({
  case: {
    properties: ['caseNumber', 'title', 'status', 'jurisdiction', 'category'],
    relationships: ['HAS_EVIDENCE', 'INVOLVES_PERSON', 'CITES_PRECEDENT', 'RELATED_TO']
  },
  evidence: {
    properties: ['type', 'description', 'collectedDate', 'location', 'chainOfCustody'],
    relationships: ['BELONGS_TO_CASE', 'ANALYZED_BY', 'REFERENCES', 'CONTRADICTS']
  },
  person: {
    properties: ['name', 'role', 'dateOfBirth', 'address', 'occupation'],
    relationships: ['INVOLVED_IN', 'WITNESS_OF', 'VICTIM_OF', 'SUSPECT_IN', 'KNOWS']
  },
  precedent: {
    properties: ['citation', 'court', 'year', 'category', 'outcome', 'summary'],
    relationships: ['CITED_BY', 'OVERRULES', 'DISTINGUISHES', 'FOLLOWS']
  },
  document: {
    properties: ['filename', 'type', 'content', 'dateCreated', 'author'],
    relationships: ['ATTACHED_TO', 'REFERENCES', 'AMENDED_BY', 'SUPERSEDES']
  }
});

// Derived stores
export const isGraphHealthy = derived(
  neo4jStatsStore,
  $stats => $stats.nodeCount > 0 && $stats.avgQueryTime < 1000
);

export const graphComplexity = derived(
  neo4jStatsStore,
  $stats => {
    if ($stats.nodeCount === 0) return 'empty';
    if ($stats.nodeCount < 1000) return 'simple';
    if ($stats.nodeCount < 10000) return 'moderate';
    return 'complex';
  }
);

export class Neo4jGraphService {
  private statsUpdateInterval: NodeJS.Timeout | null = null;

  constructor() {
    this.startStatsCollection();
  }

  /**
   * Enhanced RAG search combining vector similarity with graph traversal
   */
  async enhancedRAGSearch(query: string, options: {
    userId: string;
    caseId?: string;
    maxDepth?: number;
    includeRelatedCases?: boolean;
    includePrecedents?: boolean;
    semanticThreshold?: number;
  }): Promise<{
    vectorResults: Array<{ content: string; similarity: number; nodeId: string }>;
    graphContext: GraphSearchResult;
    enhancedResponse: string;
    reasoning: string[];
  }> {
    const startTime = Date.now();

    try {
      // Step 1: Vector similarity search
      const vectorQuery = `
        CALL db.index.vector.queryNodes('case_embeddings', $queryEmbedding, $topK)
        YIELD node, score
        WHERE score > $threshold
        RETURN node.caseNumber as caseNumber, node.title as title, 
               node.description as content, score as similarity, 
               elementId(node) as nodeId
        ORDER BY score DESC
      `;

      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(query);

      const vectorResults = await this.executeQuery({
        cypher: vectorQuery,
        parameters: {
          queryEmbedding,
          topK: 10,
          threshold: options.semanticThreshold || 0.7
        }
      });

      // Step 2: Graph context expansion
      const contextQueries = await this.buildContextQueries(vectorResults.nodes, options);
      const graphContext = await this.executeMultipleQueries(contextQueries);

      // Step 3: Enhanced response generation
      const ragContext = this.buildRAGContext(vectorResults, graphContext);
      const enhancedResponse = await this.generateEnhancedResponse(query, ragContext);

      return {
        vectorResults: vectorResults.nodes.map((node: any) => ({
          content: node.properties.content || node.properties.title,
          similarity: node.properties.similarity,
          nodeId: node.id
        })),
        graphContext,
        enhancedResponse: enhancedResponse.response,
        reasoning: enhancedResponse.reasoning
      };

    } catch (error) {
      console.error('Enhanced RAG search failed:', error);
      throw error;
    }
  }

  /**
   * Find legal precedents using graph traversal
   */
  async findLegalPrecedents(caseId: string, options: {
    maxDepth?: number;
    jurisdiction?: string;
    category?: string;
    yearRange?: [number, number];
  }): Promise<GraphSearchResult> {
    const query = `
      MATCH (case:Case {id: $caseId})
      MATCH path = (case)-[:CITES_PRECEDENT|RELATED_TO*1..${options.maxDepth || 3}]->(precedent:Precedent)
      WHERE ($jurisdiction IS NULL OR precedent.jurisdiction = $jurisdiction)
        AND ($category IS NULL OR precedent.category = $category)
        AND ($minYear IS NULL OR precedent.year >= $minYear)
        AND ($maxYear IS NULL OR precedent.year <= $maxYear)
      WITH path, precedent,
           gds.similarity.cosine(case.embedding, precedent.embedding) as similarity
      WHERE similarity > 0.6
      RETURN path, precedent, similarity
      ORDER BY similarity DESC, precedent.year DESC
      LIMIT 20
    `;

    return this.executeQuery({
      cypher: query,
      parameters: {
        caseId,
        jurisdiction: options.jurisdiction,
        category: options.category,
        minYear: options.yearRange?.[0],
        maxYear: options.yearRange?.[1]
      }
    });
  }

  /**
   * Analyze case relationships and patterns
   */
  async analyzeCaseRelationships(caseId: string): Promise<{
    similarCases: GraphNode[];
    keyPeople: GraphNode[];
    criticalEvidence: GraphNode[];
    citedPrecedents: GraphNode[];
    relationshipStrength: Record<string, number>;
  }> {
    const analysisQueries = [
      // Similar cases
      `
        MATCH (case:Case {id: $caseId})
        MATCH (similarCase:Case)
        WHERE similarCase <> case
        WITH case, similarCase,
             gds.similarity.cosine(case.embedding, similarCase.embedding) as similarity
        WHERE similarity > 0.8
        RETURN similarCase
        ORDER BY similarity DESC
        LIMIT 10
      `,
      
      // Key people
      `
        MATCH (case:Case {id: $caseId})-[r:INVOLVES_PERSON]->(person:Person)
        WITH person, count(r) as involvement_count,
             avg(r.importance) as avg_importance
        WHERE involvement_count > 1 OR avg_importance > 0.7
        RETURN person
        ORDER BY involvement_count DESC, avg_importance DESC
        LIMIT 10
      `,
      
      // Critical evidence
      `
        MATCH (case:Case {id: $caseId})-[:HAS_EVIDENCE]->(evidence:Evidence)
        WHERE evidence.importance > 0.8 OR evidence.type IN ['DNA', 'Video', 'Document']
        RETURN evidence
        ORDER BY evidence.importance DESC, evidence.collectedDate DESC
        LIMIT 15
      `,
      
      // Cited precedents
      `
        MATCH (case:Case {id: $caseId})-[:CITES_PRECEDENT]->(precedent:Precedent)
        RETURN precedent
        ORDER BY precedent.year DESC, precedent.importance DESC
        LIMIT 10
      `
    ];

    const results = await Promise.all(
      analysisQueries.map(cypher => this.executeQuery({ cypher, parameters: { caseId } }))
    );

    return {
      similarCases: results[0].nodes,
      keyPeople: results[1].nodes,
      criticalEvidence: results[2].nodes,
      citedPrecedents: results[3].nodes,
      relationshipStrength: this.calculateRelationshipStrengths(results)
    };
  }

  /**
   * Create or update graph nodes with embeddings
   */
  async createNodeWithEmbedding(
    label: string,
    properties: Record<string, any>,
    textContent: string
  ): Promise<string> {
    try {
      // Generate embedding for the text content
      const embedding = await this.generateEmbedding(textContent);

      const query = `
        CREATE (n:${label} $properties)
        SET n.embedding = $embedding,
            n.textContent = $textContent,
            n.createdAt = datetime(),
            n.updatedAt = datetime()
        RETURN elementId(n) as nodeId
      `;

      const result = await this.executeQuery({
        cypher: query,
        parameters: {
          properties,
          embedding,
          textContent
        }
      });

      return result.nodes[0]?.id || '';

    } catch (error) {
      console.error('Failed to create node with embedding:', error);
      throw error;
    }
  }

  /**
   * Update embeddings for existing nodes
   */
  async updateNodeEmbeddings(nodeIds: string[]): Promise<void> {
    const batchSize = 50;
    
    for (let i = 0; i < nodeIds.length; i += batchSize) {
      const batch = nodeIds.slice(i, i + batchSize);
      
      const query = `
        UNWIND $nodeIds as nodeId
        MATCH (n) WHERE elementId(n) = nodeId AND n.textContent IS NOT NULL
        WITH n, n.textContent as text
        CALL {
          WITH text
          // Call embedding service
          RETURN $embedding as embedding
        }
        SET n.embedding = embedding,
            n.updatedAt = datetime()
        RETURN count(n) as updatedCount
      `;

      // This would require a custom procedure to call the embedding service
      // For now, we'll use the productionServiceClient
      await this.executeQuery({
        cypher: query,
        parameters: {
          nodeIds: batch,
          embedding: [] // Placeholder - would be generated per node
        }
      });
    }
  }

  /**
   * Execute graph query
   */
  async executeQuery(query: GraphQuery): Promise<GraphSearchResult> {
    const startTime = Date.now();

    try {
      const response = await productionServiceClient.execute('neo4j.query', {
        cypher: query.cypher,
        parameters: query.parameters || {},
        includeEmbeddings: query.includeEmbeddings || false,
        maxResults: query.maxResults || 1000
      });

      const queryTime = Date.now() - startTime;
      
      // Update stats
      neo4jStatsStore.update(stats => ({
        ...stats,
        queryCount: stats.queryCount + 1,
        avgQueryTime: (stats.avgQueryTime + queryTime) / 2
      }));

      return {
        nodes: response.nodes || [],
        relationships: response.relationships || [],
        paths: response.paths || [],
        metadata: {
          queryTime,
          resultCount: (response.nodes?.length || 0) + (response.relationships?.length || 0),
          graphTraversalDepth: response.maxDepth || 0
        }
      };

    } catch (error) {
      console.error('Neo4j query failed:', error);
      throw error;
    }
  }

  /**
   * Execute multiple queries in parallel
   */
  async executeMultipleQueries(queries: GraphQuery[]): Promise<GraphSearchResult> {
    const results = await Promise.all(
      queries.map(query => this.executeQuery(query))
    );

    // Merge results
    const mergedResult: GraphSearchResult = {
      nodes: results.flatMap(r => r.nodes),
      relationships: results.flatMap(r => r.relationships),
      paths: results.flatMap(r => r.paths),
      metadata: {
        queryTime: results.reduce((sum, r) => sum + r.metadata.queryTime, 0),
        resultCount: results.reduce((sum, r) => sum + r.metadata.resultCount, 0),
        graphTraversalDepth: Math.max(...results.map(r => r.metadata.graphTraversalDepth))
      }
    };

    return mergedResult;
  }

  /**
   * Generate embedding for text
   */
  private async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await productionServiceClient.execute('embedding.generate', {
        text,
        model: 'nomic-embed-text',
        dimensions: 768
      });

      return response.embedding || [];
    } catch (error) {
      console.error('Embedding generation failed:', error);
      return new Array(768).fill(0); // Fallback zero vector
    }
  }

  /**
   * Build context queries for graph expansion
   */
  private buildContextQueries(nodes: GraphNode[], options: any): GraphQuery[] {
    return [
      // Related cases query
      {
        cypher: `
          UNWIND $nodeIds as nodeId
          MATCH (n) WHERE elementId(n) = nodeId
          MATCH (n)-[:RELATED_TO|SIMILAR_TO]-(related:Case)
          RETURN related
          LIMIT 20
        `,
        parameters: { nodeIds: nodes.map(n => n.id) }
      },
      
      // Evidence chain query
      {
        cypher: `
          UNWIND $nodeIds as nodeId
          MATCH (n) WHERE elementId(n) = nodeId
          MATCH (n)-[:HAS_EVIDENCE]->(evidence:Evidence)
          MATCH (evidence)-[:REFERENCES|CONTRADICTS]-(relatedEvidence:Evidence)
          RETURN evidence, relatedEvidence
          LIMIT 50
        `,
        parameters: { nodeIds: nodes.map(n => n.id) }
      }
    ];
  }

  /**
   * Build RAG context from graph results
   */
  private buildRAGContext(vectorResults: any, graphContext: GraphSearchResult): string {
    const context = [];
    
    // Add vector search results
    context.push('SIMILAR CASES:');
    vectorResults.nodes.forEach((node: any, index: number) => {
      context.push(`${index + 1}. ${node.properties.title} (similarity: ${(node.properties.similarity * 100).toFixed(1)}%)`);
      if (node.properties.content) {
        context.push(`   ${node.properties.content.substring(0, 200)}...`);
      }
    });

    // Add graph relationships
    if (graphContext.relationships.length > 0) {
      context.push('\nRELATED INFORMATION:');
      graphContext.relationships.forEach((rel, index) => {
        context.push(`${index + 1}. ${rel.type} relationship found`);
      });
    }

    return context.join('\n');
  }

  /**
   * Generate enhanced response using RAG context
   */
  private async generateEnhancedResponse(query: string, context: string): Promise<{
    response: string;
    reasoning: string[];
  }> {
    try {
      const response = await productionServiceClient.execute('rag.enhanced_generate', {
        query,
        context,
        model: 'gemma3-legal',
        includeReasoning: true,
        useGraphContext: true
      });

      return {
        response: response.response || 'No response generated',
        reasoning: response.reasoning || []
      };
    } catch (error) {
      console.error('Enhanced response generation failed:', error);
      return {
        response: 'Sorry, I encountered an error generating the response.',
        reasoning: ['Error in response generation']
      };
    }
  }

  /**
   * Calculate relationship strengths
   */
  private calculateRelationshipStrengths(results: GraphSearchResult[]): Record<string, number> {
    const strengths: Record<string, number> = {};
    
    results.forEach(result => {
      result.relationships.forEach(rel => {
        const key = `${rel.startNode}-${rel.type}-${rel.endNode}`;
        strengths[key] = (strengths[key] || 0) + 1;
      });
    });

    return strengths;
  }

  /**
   * Start collecting Neo4j statistics
   */
  private startStatsCollection(): void {
    if (this.statsUpdateInterval) {
      clearInterval(this.statsUpdateInterval);
    }

    this.statsUpdateInterval = setInterval(async () => {
      await this.updateStats();
    }, 30000); // Update every 30 seconds

    // Initial stats update
    this.updateStats();
  }

  /**
   * Update Neo4j statistics
   */
  private async updateStats(): Promise<void> {
    try {
      const statsQuery = `
        MATCH (n) 
        OPTIONAL MATCH ()-[r]->()
        RETURN count(DISTINCT n) as nodeCount,
               count(DISTINCT r) as relationshipCount,
               count(DISTINCT labels(n)) as labelCount
      `;

      const result = await this.executeQuery({ cypher: statsQuery });
      
      if (result.nodes.length > 0) {
        const stats = result.nodes[0].properties;
        
        neo4jStatsStore.update(current => ({
          ...current,
          nodeCount: stats.nodeCount || 0,
          relationshipCount: stats.relationshipCount || 0,
          labelCount: stats.labelCount || 0
        }));
      }

    } catch (error) {
      console.error('Failed to update Neo4j stats:', error);
    }
  }

  /**
   * Cleanup resources
   */
  destroy(): void {
    if (this.statsUpdateInterval) {
      clearInterval(this.statsUpdateInterval);
      this.statsUpdateInterval = null;
    }
  }
}

// Singleton instance
export const neo4jGraphService = new Neo4jGraphService();