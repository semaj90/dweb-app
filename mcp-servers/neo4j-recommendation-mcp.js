#!/usr/bin/env node
/**
 * Neo4j Recommendation Engine MCP Server
 * Provides intelligent legal AI recommendations through Model Context Protocol
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import neo4j from 'neo4j-driver';
import crypto from 'crypto';

class Neo4jRecommendationMCP {
  constructor() {
    this.server = new Server(
      {
        name: 'neo4j-recommendation-engine',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
          resources: {},
        },
      }
    );

    // Neo4j connection
    this.driver = null;
    this.initializeNeo4j();
    
    // Setup MCP handlers
    this.setupHandlers();
  }

  async initializeNeo4j() {
    const uri = process.env.NEO4J_URI || 'bolt://localhost:7687';
    const username = process.env.NEO4J_USERNAME || 'neo4j';
    const password = process.env.NEO4J_PASSWORD || 'password';

    try {
      this.driver = neo4j.driver(uri, neo4j.auth.basic(username, password));
      await this.driver.verifyConnectivity();
      console.error('✅ Neo4j MCP Server connected');
    } catch (error) {
      console.error('❌ Neo4j connection failed:', error.message);
    }
  }

  setupHandlers() {
    // List available tools
    this.server.setRequestHandler('tools/list', async () => ({
      tools: [
        {
          name: 'get_legal_recommendations',
          description: 'Get intelligent recommendations for legal cases and evidence analysis',
          inputSchema: {
            type: 'object',
            properties: {
              userId: { type: 'string', description: 'User ID for personalized recommendations' },
              currentCaseId: { type: 'string', description: 'Current case ID for contextual suggestions' },
              sessionContext: {
                type: 'object',
                properties: {
                  currentPage: { type: 'string' },
                  searchQuery: { type: 'string' },
                  activeFilters: { type: 'object' }
                }
              },
              userActivity: {
                type: 'object',
                properties: {
                  recentSearches: { type: 'array', items: { type: 'string' } },
                  viewedCases: { type: 'array', items: { type: 'string' } },
                  analyzedEvidence: { type: 'array', items: { type: 'string' } },
                  sessionDuration: { type: 'number' }
                }
              }
            },
            required: ['userId']
          }
        },
        {
          name: 'search_legal_concepts',
          description: 'Search legal concepts and case types using semantic analysis',
          inputSchema: {
            type: 'object',
            properties: {
              query: { type: 'string', description: 'Search query for legal concepts' },
              userId: { type: 'string', description: 'User ID for personalized results' },
              limit: { type: 'number', default: 10, description: 'Maximum results to return' }
            },
            required: ['query', 'userId']
          }
        },
        {
          name: 'get_evidence_analysis_suggestions',
          description: 'Get analysis suggestions for uploaded evidence',
          inputSchema: {
            type: 'object',
            properties: {
              evidenceIds: { type: 'array', items: { type: 'string' } },
              caseId: { type: 'string' },
              evidenceTypes: { type: 'array', items: { type: 'string' } }
            },
            required: ['evidenceIds', 'caseId']
          }
        },
        {
          name: 'track_user_behavior',
          description: 'Track user behavior for improving recommendations',
          inputSchema: {
            type: 'object',
            properties: {
              userId: { type: 'string' },
              action: { type: 'string' },
              context: { type: 'object' }
            },
            required: ['userId', 'action']
          }
        },
        {
          name: 'get_case_similarity',
          description: 'Find similar cases based on case characteristics',
          inputSchema: {
            type: 'object',
            properties: {
              caseId: { type: 'string' },
              similarityThreshold: { type: 'number', default: 0.7 },
              limit: { type: 'number', default: 5 }
            },
            required: ['caseId']
          }
        }
      ]
    }));

    // Handle tool calls
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'get_legal_recommendations':
            return await this.getLegalRecommendations(args);
          
          case 'search_legal_concepts':
            return await this.searchLegalConcepts(args);
          
          case 'get_evidence_analysis_suggestions':
            return await this.getEvidenceAnalysisSuggestions(args);
          
          case 'track_user_behavior':
            return await this.trackUserBehavior(args);
          
          case 'get_case_similarity':
            return await this.getCaseSimilarity(args);
          
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error: ${error.message}`
            }
          ],
          isError: true
        };
      }
    });

    // List available resources
    this.server.setRequestHandler('resources/list', async () => ({
      resources: [
        {
          uri: 'neo4j://legal-concepts',
          name: 'Legal Concepts Database',
          description: 'Knowledge graph of legal concepts and relationships',
          mimeType: 'application/json'
        },
        {
          uri: 'neo4j://user-patterns',
          name: 'User Behavior Patterns',
          description: 'Analyzed user behavior patterns for recommendations',
          mimeType: 'application/json'
        },
        {
          uri: 'neo4j://case-types',
          name: 'Legal Case Types',
          description: 'Categorized legal case types with analysis patterns',
          mimeType: 'application/json'
        }
      ]
    }));

    // Handle resource reads
    this.server.setRequestHandler('resources/read', async (request) => {
      const { uri } = request.params;
      
      switch (uri) {
        case 'neo4j://legal-concepts':
          return await this.readLegalConcepts();
        case 'neo4j://user-patterns':
          return await this.readUserPatterns();
        case 'neo4j://case-types':
          return await this.readCaseTypes();
        default:
          throw new Error(`Unknown resource: ${uri}`);
      }
    });
  }

  // Tool implementation methods
  async getLegalRecommendations(args) {
    const session = this.driver.session();
    
    try {
      // Get behavior-based recommendations
      const behaviorQuery = `
        // Find user's behavior pattern or create default
        MERGE (u:User {id: $userId})
        OPTIONAL MATCH (u)-[:HAS_PATTERN]->(ubp:UserBehaviorPattern)
        WITH u, COALESCE(ubp, {name: 'general-user', recommendations: ['semantic-search', 'document-analysis']}) as pattern
        
        // Get AI capabilities for this pattern
        MATCH (ai:AICapability)
        WHERE ai.name IN ['semantic-search', 'legal-qa', 'document-classification']
        
        RETURN {
          userId: u.id,
          pattern: pattern.name,
          recommendations: collect({
            id: 'ai-' + ai.id,
            type: 'feature',
            title: 'Try ' + ai.name,
            description: ai.description,
            relevanceScore: 0.8,
            reasoning: 'AI capability recommendation',
            metadata: {
              accuracy: ai.accuracy,
              applications: ai.applications
            }
          })
        } as result
      `;

      const result = await session.run(behaviorQuery, { userId: args.userId });
      
      const recommendations = result.records.length > 0 
        ? result.records[0].get('result').recommendations 
        : [];

      // Add contextual recommendations if case ID provided
      if (args.currentCaseId) {
        const contextualQuery = `
          MATCH (c:Case {id: $caseId})-[:OF_TYPE]->(ct:CaseType)
          MATCH (ct)-[:USES_PATTERN]->(ip:InvestigationPattern)
          RETURN collect({
            id: 'pattern-' + ip.id,
            type: 'analysis',
            title: 'Apply ' + ip.name,
            description: ip.description,
            relevanceScore: 0.9,
            reasoning: 'Recommended for this case type',
            metadata: {
              steps: ip.steps,
              complexity: ip.complexity
            }
          }) as contextual
        `;

        const contextualResult = await session.run(contextualQuery, { 
          caseId: args.currentCaseId 
        });
        
        const contextual = contextualResult.records.length > 0 
          ? contextualResult.records[0].get('contextual') 
          : [];
        
        recommendations.push(...contextual);
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              recommendations: recommendations.slice(0, 10), // Limit results
              total: recommendations.length,
              userId: args.userId,
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    } finally {
      await session.close();
    }
  }

  async searchLegalConcepts(args) {
    const session = this.driver.session();
    
    try {
      const query = `
        // Full-text search across legal concepts
        CALL db.index.fulltext.queryNodes('legal_entities_fulltext', $searchQuery)
        YIELD node, score
        WHERE labels(node)[0] IN ['LegalConcept', 'CaseType', 'EvidenceType']
        
        // Get related information
        OPTIONAL MATCH (node)-[r]-(related)
        WHERE labels(related)[0] IN ['LegalConcept', 'CaseType']
        
        WITH node, score, collect(DISTINCT {
          name: related.name,
          relationship: type(r)
        })[0..3] as related
        
        RETURN {
          id: node.id,
          name: node.name,
          type: labels(node)[0],
          category: node.category,
          description: node.description,
          complexity: node.complexity,
          searchScore: score,
          relatedItems: related
        } as result
        ORDER BY score DESC
        LIMIT $limit
      `;

      const result = await session.run(query, {
        searchQuery: args.query,
        limit: args.limit || 10
      });

      const concepts = result.records.map(record => record.get('result'));

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              query: args.query,
              results: concepts,
              total: concepts.length,
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    } finally {
      await session.close();
    }
  }

  async getEvidenceAnalysisSuggestions(args) {
    const session = this.driver.session();
    
    try {
      const query = `
        // Get evidence types and their analysis patterns
        UNWIND $evidenceIds as evidenceId
        OPTIONAL MATCH (e:Evidence {id: evidenceId})-[:IS_TYPE]->(et:EvidenceType)
        
        WITH collect(DISTINCT et) as evidenceTypes
        UNWIND evidenceTypes as et
        WHERE et IS NOT NULL
        
        // Find analysis patterns for these evidence types
        OPTIONAL MATCH (et)-[:SUGGESTS_ANALYSIS]->(ap:AnalysisPattern)
        OPTIONAL MATCH (et)-[:GOVERNED_BY]->(lc:LegalConcept)
        
        RETURN {
          evidenceType: et.name,
          analysisPatterns: collect(DISTINCT {
            id: ap.id,
            name: ap.name,
            description: ap.description,
            steps: ap.steps,
            relevanceScore: 0.8
          }),
          governingConcepts: collect(DISTINCT {
            id: lc.id,
            name: lc.name,
            category: lc.category
          })
        } as suggestion
      `;

      const result = await session.run(query, {
        evidenceIds: args.evidenceIds
      });

      const suggestions = result.records.map(record => record.get('suggestion'));

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              caseId: args.caseId,
              evidenceIds: args.evidenceIds,
              suggestions,
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    } finally {
      await session.close();
    }
  }

  async trackUserBehavior(args) {
    const session = this.driver.session();
    
    try {
      const query = `
        MERGE (u:User {id: $userId})
        CREATE (u)-[:PERFORMED {
          action: $action,
          timestamp: datetime(),
          context: $context
        }]->(a:UserAction {
          id: randomUUID(),
          action: $action,
          timestamp: datetime(),
          sessionId: $sessionId
        })
        RETURN a.id as actionId
      `;

      const result = await session.run(query, {
        userId: args.userId,
        action: args.action,
        context: JSON.stringify(args.context || {}),
        sessionId: crypto.randomUUID()
      });

      const actionId = result.records[0]?.get('actionId');

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              success: true,
              actionId,
              userId: args.userId,
              action: args.action,
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    } finally {
      await session.close();
    }
  }

  async getCaseSimilarity(args) {
    const session = this.driver.session();
    
    try {
      const query = `
        MATCH (targetCase:Case {id: $caseId})-[:OF_TYPE]->(targetType:CaseType)
        MATCH (similarCase:Case)-[:OF_TYPE]->(targetType)
        WHERE similarCase.id <> $caseId
        
        // Calculate similarity based on multiple factors
        WITH similarCase, targetCase,
             // Case type match gives base similarity
             1.0 as baseScore,
             // Additional similarity factors could be added here
             CASE 
               WHEN similarCase.status = targetCase.status THEN 0.2 
               ELSE 0.0 
             END as statusBonus,
             rand() * 0.3 as randomFactor
        
        WITH similarCase, (baseScore + statusBonus + randomFactor) as similarity
        WHERE similarity >= $threshold
        
        RETURN {
          id: similarCase.id,
          title: similarCase.title,
          status: similarCase.status,
          similarity: similarity,
          description: similarCase.description,
          createdAt: similarCase.createdAt
        } as similarCase
        ORDER BY similarity DESC
        LIMIT $limit
      `;

      const result = await session.run(query, {
        caseId: args.caseId,
        threshold: args.similarityThreshold || 0.7,
        limit: args.limit || 5
      });

      const similarCases = result.records.map(record => record.get('similarCase'));

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              targetCaseId: args.caseId,
              similarCases,
              threshold: args.similarityThreshold,
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    } finally {
      await session.close();
    }
  }

  // Resource read methods
  async readLegalConcepts() {
    const session = this.driver.session();
    
    try {
      const query = `
        MATCH (lc:LegalConcept)
        OPTIONAL MATCH (lc)-[r]-(related)
        WHERE labels(related)[0] IN ['CaseType', 'EvidenceType']
        
        RETURN {
          id: lc.id,
          name: lc.name,
          category: lc.category,
          description: lc.description,
          elements: lc.elements,
          complexity: lc.complexity,
          relationships: collect(DISTINCT {
            type: type(r),
            target: related.name
          })
        } as concept
        LIMIT 50
      `;

      const result = await session.run(query);
      const concepts = result.records.map(record => record.get('concept'));

      return {
        contents: [
          {
            uri: 'neo4j://legal-concepts',
            mimeType: 'application/json',
            text: JSON.stringify({
              legalConcepts: concepts,
              total: concepts.length,
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    } finally {
      await session.close();
    }
  }

  async readUserPatterns() {
    const session = this.driver.session();
    
    try {
      const query = `
        MATCH (ubp:UserBehaviorPattern)
        OPTIONAL MATCH (ubp)-[:BENEFITS_FROM]->(ai:AICapability)
        
        RETURN {
          id: ubp.id,
          name: ubp.name,
          description: ubp.description,
          characteristics: ubp.characteristics,
          recommendations: ubp.recommendations,
          aiCapabilities: collect(DISTINCT ai.name)
        } as pattern
      `;

      const result = await session.run(query);
      const patterns = result.records.map(record => record.get('pattern'));

      return {
        contents: [
          {
            uri: 'neo4j://user-patterns',
            mimeType: 'application/json',
            text: JSON.stringify({
              userBehaviorPatterns: patterns,
              total: patterns.length,
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    } finally {
      await session.close();
    }
  }

  async readCaseTypes() {
    const session = this.driver.session();
    
    try {
      const query = `
        MATCH (ct:CaseType)
        OPTIONAL MATCH (ct)-[:INVOLVES]->(lc:LegalConcept)
        OPTIONAL MATCH (ct)-[:USES_PATTERN]->(ip:InvestigationPattern)
        
        RETURN {
          id: ct.id,
          name: ct.name,
          category: ct.category,
          description: ct.description,
          commonClaims: ct.commonClaims,
          averageDuration: ct.averageDuration,
          complexity: ct.complexity,
          legalConcepts: collect(DISTINCT lc.name),
          investigationPatterns: collect(DISTINCT ip.name)
        } as caseType
      `;

      const result = await session.run(query);
      const caseTypes = result.records.map(record => record.get('caseType'));

      return {
        contents: [
          {
            uri: 'neo4j://case-types',
            mimeType: 'application/json',
            text: JSON.stringify({
              caseTypes,
              total: caseTypes.length,
              timestamp: new Date().toISOString()
            }, null, 2)
          }
        ]
      };
    } finally {
      await session.close();
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('🚀 Neo4j Recommendation MCP Server running');
  }

  async close() {
    if (this.driver) {
      await this.driver.close();
    }
  }
}

// Handle process termination
const server = new Neo4jRecommendationMCP();
process.on('SIGINT', async () => {
  console.error('👋 Shutting down Neo4j MCP Server');
  await server.close();
  process.exit(0);
});

// Start the server
server.run().catch(console.error);