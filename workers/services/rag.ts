// workers/services/rag.ts
import fetch from 'node-fetch';
import type { RagResult } from '../../sveltekit-frontend/src/lib/types/progress';
import { searchByText, findSimilarEvidences } from './embeddings.js';
import { QdrantClient } from '@qdrant/js-client-rest';
import neo4j from 'neo4j-driver';

// Neo4j client for knowledge graph queries
const neo4jDriver = neo4j.driver(
  process.env.NEO4J_URL || 'bolt://localhost:7687',
  neo4j.auth.basic(
    process.env.NEO4J_USER || 'neo4j',
    process.env.NEO4J_PASSWORD || 'password'
  )
);

interface RagRequest {
  evidenceId: string;
  topK?: number;
  model?: string;
  contextWindow?: number;
}

export async function runRag(request: RagRequest): Promise<RagResult> {
  const { evidenceId, topK = 5, model = 'llama3.1', contextWindow = 4000 } = request;
  
  console.log(`📚 Running RAG analysis for evidence: ${evidenceId}`);
  
  try {
    // 1. Get the current evidence content
    const currentEvidence = await getEvidenceContent(evidenceId);
    
    if (!currentEvidence || !currentEvidence.text) {
      throw new Error('No content available for evidence');
    }
    
    // 2. Find similar/related evidences using vector search
    const similarEvidences = await findRelevantEvidences(currentEvidence.text, topK);
    
    // 3. Get related entities from knowledge graph
    const relatedEntities = await getRelatedEntities(evidenceId);
    
    // 4. Build context from similar evidences and entities
    const context = await buildRagContext(currentEvidence, similarEvidences, relatedEntities, contextWindow);
    
    // 5. Generate analysis using LLM
    const analysis = await generateLlmAnalysis(currentEvidence.text, context, model);
    
    // 6. Extract and store entities/relationships
    await extractAndStoreEntities(evidenceId, analysis);
    
    const result: RagResult = {
      summary: analysis.summary,
      snippets: analysis.keyPoints,
      relevantDocs: similarEvidences.map(e => ({
        evidenceId: e.evidenceId,
        score: e.score,
        snippet: e.text?.substring(0, 200) + '...'
      })),
      confidence: analysis.confidence
    };
    
    console.log(`✅ RAG analysis completed for evidence: ${evidenceId}`);
    return result;
    
  } catch (error) {
    console.error(`❌ RAG analysis failed for evidence ${evidenceId}:`, error);
    throw error;
  }
}

async function getEvidenceContent(evidenceId: string): Promise<{ text: string; metadata?: any } | null> {
  try {
    const db = await import('../../sveltekit-frontend/src/lib/server/db.js');
    
    // Get OCR text
    const ocrResult = await db.db
      .select()
      .from('evidence_ocr')
      .where('evidence_id', '=', evidenceId)
      .orderBy('created_at', 'desc')
      .limit(1);
    
    if (ocrResult.length === 0) {
      return null;
    }
    
    return {
      text: ocrResult[0].text,
      metadata: {
        confidence: ocrResult[0].confidence,
        method: JSON.parse(ocrResult[0].metadata || '{}').method
      }
    };
    
  } catch (error) {
    console.error('❌ Failed to get evidence content:', error);
    return null;
  }
}

async function findRelevantEvidences(queryText: string, topK: number): Promise<Array<{
  evidenceId: string;
  score: number;
  text?: string;
  metadata?: any;
}>> {
  try {
    // Use vector similarity search
    const similarEvidences = await searchByText(queryText, topK, 0.6);
    
    // Fetch full text for similar evidences
    const evidencesWithText = await Promise.all(
      similarEvidences.map(async (evidence) => {
        const content = await getEvidenceContent(evidence.evidenceId);
        return {
          ...evidence,
          text: content?.text,
          metadata: content?.metadata
        };
      })
    );
    
    return evidencesWithText.filter(e => e.text); // Only return evidences with text
    
  } catch (error) {
    console.error('❌ Failed to find relevant evidences:', error);
    return [];
  }
}

async function getRelatedEntities(evidenceId: string): Promise<Array<{
  type: string;
  name: string;
  properties: any; // TODO-AUTO: Create EntityProperties interface with specific property types
  relationships: any[]; // TODO-AUTO: Define Relationship interface - type { type: string, target: string, properties?: Record<string, unknown> }
}>> {
  const session = neo4jDriver.session();
  
  try {
    // Query Neo4j for entities related to this evidence
    const result = await session.run(`
      MATCH (e:Evidence {id: $evidenceId})-[r]-(entity)
      RETURN entity, type(r) as relationship, properties(entity) as props
      LIMIT 20
    `, { evidenceId });
    
    return result.records.map(record => ({
      type: record.get('entity').labels[0],
      name: record.get('entity').properties.name || record.get('entity').properties.id,
      properties: record.get('props'),
      relationships: [record.get('relationship')]
    }));
    
  } catch (error) {
    console.error('❌ Failed to get related entities:', error);
    return [];
  } finally {
    await session.close();
  }
}

async function buildRagContext(
  currentEvidence: { text: string; metadata?: any },
  similarEvidences: Array<{ evidenceId: string; score: number; text?: string }>,
  relatedEntities: Array<{ type: string; name: string; properties: any }>,
  maxTokens: number
): Promise<string> {
  let context = '';
  let tokenCount = 0;
  
  // Add current evidence summary
  const currentSummary = currentEvidence.text.substring(0, 500);
  context += `Current Evidence:\n${currentSummary}\n\n`;
  tokenCount += estimateTokens(currentSummary);
  
  // Add similar evidences
  if (similarEvidences.length > 0) {
    context += `Related Evidence:\n`;
    
    for (const evidence of similarEvidences) {
      if (tokenCount >= maxTokens * 0.7) break; // Reserve 30% for entities and prompt
      
      if (evidence.text) {
        const snippet = evidence.text.substring(0, 300);
        const evidenceContext = `- Evidence ${evidence.evidenceId} (similarity: ${evidence.score.toFixed(2)}): ${snippet}...\n`;
        
        if (tokenCount + estimateTokens(evidenceContext) < maxTokens * 0.7) {
          context += evidenceContext;
          tokenCount += estimateTokens(evidenceContext);
        }
      }
    }
    context += '\n';
  }
  
  // Add related entities
  if (relatedEntities.length > 0) {
    context += `Related Entities:\n`;
    
    for (const entity of relatedEntities) {
      if (tokenCount >= maxTokens * 0.9) break;
      
      const entityContext = `- ${entity.type}: ${entity.name}\n`;
      
      if (tokenCount + estimateTokens(entityContext) < maxTokens * 0.9) {
        context += entityContext;
        tokenCount += estimateTokens(entityContext);
      }
    }
    context += '\n';
  }
  
  return context;
}

function estimateTokens(text: string): number {
  // Rough estimation: 1 token ≈ 4 characters for English text
  return Math.ceil(text.length / 4);
}

async function generateLlmAnalysis(
  evidenceText: string,
  context: string,
  model: string
): Promise<{
  summary: string;
  keyPoints: string[];
  confidence: number;
  entities: Array<{ type: string; name: string; properties: any }>;
}> {
  try {
    const ollamaUrl = process.env.OLLAMA_URL || 'http://localhost:11434';
    
    const prompt = `You are a legal evidence analyst. Analyze the following evidence in the context of related information.

${context}

Evidence to Analyze:
${evidenceText}

Please provide a comprehensive analysis including:
1. A clear summary of the evidence
2. Key points and findings
3. Identified entities (people, organizations, locations, dates)
4. Legal relevance and implications
5. Confidence level in your analysis (0.0 to 1.0)

Format your response as JSON with the following structure:
{
  "summary": "Brief summary of the evidence",
  "keyPoints": ["Key point 1", "Key point 2", ...],
  "entities": [{"type": "PERSON", "name": "John Doe", "properties": {"role": "witness"}}, ...],
  "confidence": 0.85,
  "legalImplications": "Analysis of legal relevance"
}`;

    const response = await fetch(`${ollamaUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt,
        stream: false,
        options: {
          temperature: 0.3,
          top_p: 0.9,
          max_tokens: 2000
        }
      })
    });
    
    if (!response.ok) {
      throw new Error(`LLM request failed: ${response.statusText}`);
    }
    
    const data = await response.json() as any;
    
    try {
      // Try to parse JSON response
      const analysis = JSON.parse(data.response);
      
      return {
        summary: analysis.summary || 'Analysis completed',
        keyPoints: analysis.keyPoints || [],
        confidence: analysis.confidence || 0.8,
        entities: analysis.entities || []
      };
      
    } catch (parseError) {
      // Fallback if JSON parsing fails
      console.warn('⚠️ Failed to parse LLM JSON response, using fallback');
      
      return {
        summary: data.response.substring(0, 500),
        keyPoints: extractKeyPoints(data.response),
        confidence: 0.7,
        entities: []
      };
    }
    
  } catch (error) {
    console.error('❌ LLM analysis failed:', error);
    
    // Fallback analysis
    return {
      summary: `Evidence analysis completed for ${evidenceText.length} characters of content`,
      keyPoints: ['Evidence requires manual review', 'Automated analysis unavailable'],
      confidence: 0.5,
      entities: []
    };
  }
}

function extractKeyPoints(text: string): string[] {
  // Simple extraction of sentences that might be key points
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 20);
  return sentences.slice(0, 5).map(s => s.trim());
}

async function extractAndStoreEntities(
  evidenceId: string,
  analysis: { entities: Array<{ type: string; name: string; properties: any }> }
): Promise<void> {
  if (!analysis.entities || analysis.entities.length === 0) {
    return;
  }
  
  const session = neo4jDriver.session();
  
  try {
    // Create/update entities and relationships in Neo4j
    for (const entity of analysis.entities) {
      await session.run(`
        MERGE (e:Evidence {id: $evidenceId})
        MERGE (entity:${entity.type} {name: $entityName})
        SET entity += $properties
        MERGE (e)-[:MENTIONS]->(entity)
      `, {
        evidenceId,
        entityName: entity.name,
        properties: entity.properties || {}
      });
    }
    
    console.log(`✅ Stored ${analysis.entities.length} entities for evidence: ${evidenceId}`);
    
  } catch (error) {
    console.error('❌ Failed to store entities:', error);
  } finally {
    await session.close();
  }
}

// Advanced RAG functions
export async function runMultiStepRag(evidenceIds: string[]): Promise<{
  consolidatedAnalysis: string;
  crossReferences: Array<{ fromId: string; toId: string; relationship: string; confidence: number }>;
  timeline: Array<{ date: string; event: string; evidenceIds: string[] }>;
}> {
  console.log(`🔍 Running multi-step RAG analysis for ${evidenceIds.length} evidences`);
  
  try {
    // 1. Analyze each evidence individually
    const individualAnalyses = await Promise.all(
      evidenceIds.map(id => runRag({ evidenceId: id }))
    );
    
    // 2. Find cross-references between evidences
    const crossReferences = await findCrossReferences(evidenceIds);
    
    // 3. Build timeline from entities and dates
    const timeline = await buildTimeline(evidenceIds);
    
    // 4. Generate consolidated analysis
    const consolidatedAnalysis = await generateConsolidatedAnalysis(
      individualAnalyses,
      crossReferences,
      timeline
    );
    
    return {
      consolidatedAnalysis,
      crossReferences,
      timeline
    };
    
  } catch (error) {
    console.error('❌ Multi-step RAG analysis failed:', error);
    throw error;
  }
}

async function findCrossReferences(evidenceIds: string[]): Promise<Array<{
  fromId: string;
  toId: string;
  relationship: string;
  confidence: number;
}>> {
  const session = neo4jDriver.session();
  
  try {
    const result = await session.run(`
      MATCH (e1:Evidence)-[r1]->(entity)<-[r2]-(e2:Evidence)
      WHERE e1.id IN $evidenceIds AND e2.id IN $evidenceIds AND e1.id <> e2.id
      RETURN e1.id as fromId, e2.id as toId, 
             type(r1) + '_' + type(r2) as relationship,
             entity.name as sharedEntity
    `, { evidenceIds });
    
    const crossRefs = result.records.map(record => ({
      fromId: record.get('fromId'),
      toId: record.get('toId'),
      relationship: record.get('relationship'),
      confidence: 0.8 // TODO: Calculate actual confidence based on entity importance
    }));
    
    return crossRefs;
    
  } catch (error) {
    console.error('❌ Failed to find cross-references:', error);
    return [];
  } finally {
    await session.close();
  }
}

async function buildTimeline(evidenceIds: string[]): Promise<Array<{
  date: string;
  event: string;
  evidenceIds: string[];
}>> {
  const session = neo4jDriver.session();
  
  try {
    const result = await session.run(`
      MATCH (e:Evidence)-[:MENTIONS]->(date:DATE)
      WHERE e.id IN $evidenceIds
      RETURN date.value as dateValue, date.event as event, collect(e.id) as evidenceIds
      ORDER BY date.value
    `, { evidenceIds });
    
    return result.records.map(record => ({
      date: record.get('dateValue'),
      event: record.get('event') || 'Event mentioned in evidence',
      evidenceIds: record.get('evidenceIds')
    }));
    
  } catch (error) {
    console.error('❌ Failed to build timeline:', error);
    return [];
  } finally {
    await session.close();
  }
}

async function generateConsolidatedAnalysis(
  analyses: RagResult[],
  crossReferences: Array<{ fromId: string; toId: string; relationship: string }>,
  timeline: Array<{ date: string; event: string; evidenceIds: string[] }>
): Promise<string> {
  try {
    const ollamaUrl = process.env.OLLAMA_URL || 'http://localhost:11434';
    
    const consolidatedContext = `
Individual Analyses:
${analyses.map((a, i) => `Evidence ${i + 1}: ${a.summary}`).join('\n')}

Cross-References:
${crossReferences.map(cr => `${cr.fromId} -> ${cr.toId}: ${cr.relationship}`).join('\n')}

Timeline:
${timeline.map(t => `${t.date}: ${t.event}`).join('\n')}
`;
    
    const prompt = `As a legal analyst, provide a consolidated analysis of multiple pieces of evidence. Consider the individual analyses, cross-references between evidences, and timeline of events.

${consolidatedContext}

Provide a comprehensive consolidated analysis including:
1. Overall case narrative
2. Key connections between evidences
3. Timeline significance
4. Potential legal implications
5. Areas requiring further investigation`;

    const response = await fetch(`${ollamaUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'llama3.1',
        prompt,
        stream: false,
        options: {
          temperature: 0.3,
          max_tokens: 1500
        }
      })
    });
    
    if (!response.ok) {
      throw new Error(`Consolidated analysis failed: ${response.statusText}`);
    }
    
    const data = await response.json() as any;
    return data.response;
    
  } catch (error) {
    console.error('❌ Consolidated analysis failed:', error);
    return 'Consolidated analysis could not be generated. Please review individual evidence analyses.';
  }
}

// Health check for RAG service
export async function checkRagHealth(): Promise<{
  ollama: boolean;
  neo4j: boolean;
  qdrant: boolean;
}> {
  const health = {
    ollama: false,
    neo4j: false,
    qdrant: false
  };
  
  // Check Ollama
  try {
    const ollamaUrl = process.env.OLLAMA_URL || 'http://localhost:11434';
    const response = await fetch(`${ollamaUrl}/api/tags`);
    health.ollama = response.ok;
  } catch (error) {
    health.ollama = false;
  }
  
  // Check Neo4j
  const session = neo4jDriver.session();
  try {
    await session.run('RETURN 1');
    health.neo4j = true;
  } catch (error) {
    health.neo4j = false;
  } finally {
    await session.close();
  }
  
  // Check Qdrant
  try {
    const qdrantClient = new QdrantClient({
      url: process.env.QDRANT_URL || 'http://localhost:6333'
    });
    await qdrantClient.getCollections();
    health.qdrant = true;
  } catch (error) {
    health.qdrant = false;
  }
  
  return health;
}

// Graceful shutdown
export async function closeRagService(): Promise<void> {
  try {
    await neo4jDriver.close();
    console.log('✅ RAG service closed gracefully');
  } catch (error) {
    console.error('❌ Error closing RAG service:', error);
  }
}
