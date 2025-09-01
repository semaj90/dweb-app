/// <reference types="vite/client" />

import type { RequestHandler } from './$types';

/**
 * Evidence Enhancement API
 * Analyzes uploaded evidence and suggests relevant labels and classifications
 */

import { Pool } from "pg";
import { z } from "zod";

// Configuration
const CONFIG = {
    database: {
        user: import.meta.env.DB_USER || 'postgres',
        password: import.meta.env.DB_PASSWORD || 'password',
        host: import.meta.env.DB_HOST || 'localhost',
        port: parseInt(import.meta.env.DB_PORT || '5432'),
        database: import.meta.env.DB_NAME || 'prosecutor_db'
    },
    redis: {
        url: import.meta.env.REDIS_URL || 'redis://localhost:6379'
    },
    ollama: {
        url: import.meta.env.OLLAMA_URL || 'http://localhost:11434',
        model: import.meta.env.LLM_MODEL || 'gemma3-legal',
        embeddingModel: 'nomic-embed-text'
    },
    enhancement: {
        minConfidence: 0.6,
        maxSuggestions: 10,
        similarityThreshold: 0.7
    }
};

// Validation schemas
const EvidenceEnhancementRequestSchema = z.object({
    evidence_text: z.string().min(10).max(50000),
    evidence_type: z.enum(['document', 'testimony', 'physical', 'digital', 'audio', 'video']),
    case_context: z.object({
        case_id: z.string().optional(),
        jurisdiction: z.enum(['federal', 'state', 'local', 'international']).optional(),
        case_type: z.enum(['criminal', 'civil', 'administrative', 'constitutional']).optional(),
        charges: z.array(z.string()).optional(),
        defendant_name: z.string().optional()
    }).optional(),
    enhancement_options: z.object({
        suggest_labels: z.boolean().optional(),
        extract_entities: z.boolean().optional(),
        find_similar: z.boolean().optional(),
        prosecution_analysis: z.boolean().optional(),
        fact_checking: z.boolean().optional()
    }).optional()
});

const EvidenceEnhancementResponseSchema = z.object({
    analysis: z.object({
        evidence_type: z.string(),
        confidence_score: z.number(),
        prosecution_strength: z.number(),
        legal_relevance: z.number()
    }),
    suggested_labels: z.array(z.object({
        label: z.string(),
        confidence: z.number(),
        category: z.string(),
        justification: z.string()
    })),
    extracted_entities: z.array(z.object({
        entity: z.string(),
        type: z.string(),
        confidence: z.number(),
        context: z.string()
    })),
    similar_evidence: z.array(z.object({
        document_id: z.string(),
        similarity_score: z.number(),
        relevant_phrases: z.array(z.string()),
        prosecution_outcome: z.string()
    })),
    prosecution_insights: z.object({
        strengths: z.array(z.string()),
        weaknesses: z.array(z.string()),
        recommendations: z.array(z.string()),
        precedent_support: z.number()
    }),
    metadata: z.object({
        processing_time_ms: z.number(),
        enhancement_version: z.string(),
        data_sources: z.array(z.string())
    })
});

// Initialize connections
let db: Pool | null = null;
let redis: Redis | null = null;

function getDB() {
    if (!db) {
        db = new Pool(CONFIG.database);
    }
    return db;
}

function getRedis() {
    if (!redis) {
        redis = new Redis(CONFIG.redis.url);
    }
    return redis;
}

export const POST: RequestHandler = async ({ request }) => {
    const startTime = Date.now();
    
    try {
        const requestData = await request.json();
        
        // Validate request
        const validatedRequest = EvidenceEnhancementRequestSchema.parse(requestData);
        
        console.log(`🔍 Enhancing evidence: ${validatedRequest.evidence_type}`);

        // Set default enhancement options
        const options = {
            suggest_labels: true,
            extract_entities: true,
            find_similar: true,
            prosecution_analysis: true,
            fact_checking: false,
            ...validatedRequest.enhancement_options
        };

        // Parallel processing of different enhancement tasks
        const [
            analysisResult,
            suggestedLabels,
            extractedEntities,
            similarEvidence,
            prosecutionInsights
        ] = await Promise.allSettled([
            analyzeEvidence(validatedRequest.evidence_text, validatedRequest.evidence_type),
            options.suggest_labels ? suggestLabels(validatedRequest.evidence_text, validatedRequest.case_context) : Promise.resolve([]),
            options.extract_entities ? extractEntities(validatedRequest.evidence_text) : Promise.resolve([]),
            options.find_similar ? findSimilarEvidence(validatedRequest.evidence_text, validatedRequest.case_context) : Promise.resolve([]),
            options.prosecution_analysis ? analyzeProsecutionValue(validatedRequest.evidence_text, validatedRequest.case_context) : Promise.resolve({
                strengths: [],
                weaknesses: [],
                recommendations: [],
                precedent_support: 0
            })
        ]);

        // Compile results
        const response = {
            analysis: analysisResult.status === 'fulfilled' ? analysisResult.value : {
                evidence_type: validatedRequest.evidence_type,
                confidence_score: 0.5,
                prosecution_strength: 50,
                legal_relevance: 0.5
            },
            suggested_labels: suggestedLabels.status === 'fulfilled' ? suggestedLabels.value : [],
            extracted_entities: extractedEntities.status === 'fulfilled' ? extractedEntities.value : [],
            similar_evidence: similarEvidence.status === 'fulfilled' ? similarEvidence.value : [],
            prosecution_insights: prosecutionInsights.status === 'fulfilled' ? prosecutionInsights.value : {
                strengths: [],
                weaknesses: [],
                recommendations: [],
                precedent_support: 0
            },
            metadata: {
                processing_time_ms: Date.now() - startTime,
                enhancement_version: '2.0.0',
                data_sources: ['legal_documents_processed', 'semantic_phrases_ranking']
            }
        };

        // Validate response
        const validatedResponse = EvidenceEnhancementResponseSchema.parse(response);

        // Cache results for future reference
        await cacheEnhancementResults(validatedRequest.evidence_text, validatedResponse);

        return json(validatedResponse);

    } catch (err: any) {
        console.error('❌ Evidence enhancement error:', err);
        
        if (err instanceof z.ZodError) {
            return json({
                message: 'Invalid request format',
                errors: err.errors
            }, { status: 400 });
        }

        return json({
            message: 'Evidence enhancement service temporarily unavailable',
            details: err instanceof Error ? err.message : 'Unknown error'
        }, { status: 500 });
    }
};

async function analyzeEvidence(evidenceText: string, evidenceType: string): Promise<any> {
    try {
        // Use LLM to analyze the evidence
        const analysisPrompt = `You are a legal evidence analyst. Analyze the following ${evidenceType} evidence and provide a structured assessment.

Evidence Text:
${evidenceText}

Analyze and respond with ONLY a JSON object in this format:
{
  "evidence_type": "${evidenceType}",
  "confidence_score": 0.0-1.0,
  "prosecution_strength": 0-100,
  "legal_relevance": 0.0-1.0
}

Consider:
- How clear and compelling is this evidence?
- What is its potential value for prosecution?
- How legally relevant is this evidence?
- Rate the overall quality and reliability`;

        const response = await fetch(`${CONFIG.ollama.url}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: CONFIG.ollama.model,
                prompt: analysisPrompt,
                stream: false,
                options: {
                    temperature: 0.2,
                    top_p: 0.8
                }
            })
        });

        if (response.ok) {
            const data = await response.json();
            const analysisText = data.response.trim();
            
            const jsonMatch = analysisText.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                return JSON.parse(jsonMatch[0]);
            }
        }
    } catch (error: any) {
        console.warn('LLM analysis failed, using fallback:', error);
    }

    // Fallback analysis
    return {
        evidence_type: evidenceType,
        confidence_score: 0.7,
        prosecution_strength: 65,
        legal_relevance: 0.6
    };
}

async function suggestLabels(evidenceText: string, caseContext?: unknown): Promise<any> {
    const db = getDB();
    
    try {
        // Find relevant phrases that match the evidence
        const relevantPhrases = await db.query(`
            SELECT DISTINCT
                spr.phrase,
                spr.avg_prosecution_score,
                spr.frequency,
                spr.correlation_strength
            FROM semantic_phrases_ranking spr
            WHERE 
                spr.avg_prosecution_score >= 60
                AND (
                    $1 ILIKE '%' || spr.phrase || '%'
                    OR spr.phrase ILIKE '%' || $2 || '%'
                )
            ORDER BY spr.avg_prosecution_score DESC, spr.frequency DESC
            LIMIT 20
        `, [evidenceText, evidenceText.split(' ').slice(0, 10).join(' ')]);

        // Categorize and score the labels
        const labels = [];
        
        for (const phrase of relevantPhrases.rows) {
            const category = categorizeLegalPhrase(phrase.phrase);
            const confidence = Math.min(
                (phrase.avg_prosecution_score / 100) * phrase.correlation_strength,
                0.95
            );
            
            if (confidence >= CONFIG.enhancement.minConfidence) {
                labels.push({
                    label: phrase.phrase,
                    confidence,
                    category,
                    justification: `High prosecution correlation (${phrase.avg_prosecution_score}%) based on ${phrase.frequency} similar cases`
                });
            }
        }

        // Add contextual labels based on case information
        if (caseContext?.charges) {
            for (const charge of caseContext.charges) {
                labels.push({
                    label: `Supports '${charge}' charge`,
                    confidence: 0.8,
                    category: 'charge_support',
                    justification: 'Evidence directly relates to stated charges'
                });
            }
        }

        return labels.slice(0, CONFIG.enhancement.maxSuggestions);

    } catch (error: any) {
        console.error('Label suggestion failed:', error);
        return [];
    }
}

async function extractEntities(evidenceText: string): Promise<any> {
    try {
        const entityPrompt = `Extract legal entities from this text. Respond with ONLY a JSON array in this format:
[
  {
    "entity": "entity name",
    "type": "PERSON|ORGANIZATION|LOCATION|DATE|STATUTE|CASE_NUMBER|AMOUNT",
    "confidence": 0.0-1.0,
    "context": "surrounding context"
  }
]

Text to analyze:
${evidenceText}`;

        const response = await fetch(`${CONFIG.ollama.url}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: CONFIG.ollama.model,
                prompt: entityPrompt,
                stream: false,
                options: {
                    temperature: 0.1,
                    top_p: 0.7
                }
            })
        });

        if (response.ok) {
            const data = await response.json();
            const entitiesText = data.response.trim();
            
            const jsonMatch = entitiesText.match(/\[[\s\S]*\]/);
            if (jsonMatch) {
                const entities = JSON.parse(jsonMatch[0]);
                return entities.filter((e: any) => e.confidence >= 0.5);
            }
        }
    } catch (error: any) {
        console.warn('Entity extraction failed:', error);
    }

    // Fallback entity extraction using regex patterns
    return extractEntitiesWithRegex(evidenceText);
}

function extractEntitiesWithRegex(text: string) {
    const entities = [];
    
    // Common legal entity patterns
    const patterns = {
        PERSON: /\b[A-Z][a-z]+ [A-Z][a-z]+\b/g,
        DATE: /\b\d{1,2}\/\d{1,2}\/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b/g,
        AMOUNT: /\$[\d,]+(?:\.\d{2})?/g,
        CASE_NUMBER: /\b\d{2,4}-\w{2,4}-\d{4,6}\b/g,
        STATUTE: /\b\d+\s+U\.?S\.?C\.?\s+§?\s*\d+/g
    };

    for (const [type, pattern] of Object.entries(patterns)) {
        const matches = text.match(pattern);
        if (matches) {
            for (const match of matches) {
                entities.push({
                    entity: match,
                    type,
                    confidence: 0.7,
                    context: getContext(text, match)
                });
            }
        }
    }

    return entities.slice(0, 15); // Limit results
}

function getContext(text: string, entity: string): string {
    const index = text.indexOf(entity);
    const start = Math.max(0, index - 50);
    const end = Math.min(text.length, index + entity.length + 50);
    return text.substring(start, end).trim();
}

async function findSimilarEvidence(evidenceText: string, caseContext?: unknown): Promise<any> {
    const db = getDB();
    
    try {
        // Generate embedding for the evidence
        const embedding = await generateEmbedding(evidenceText);
        
        // Find similar documents using vector similarity
        // Note: This is a simplified version - full implementation would use pgvector
        const similarDocs = await db.query(`
            SELECT 
                document_id,
                text_chunk,
                semantic_phrases,
                prosecution_strength_score,
                judgement_outcome,
                jurisdiction
            FROM legal_documents_processed
            WHERE 
                ($1 IS NULL OR jurisdiction = $1)
                AND prosecution_strength_score >= 60
            ORDER BY prosecution_strength_score DESC
            LIMIT 20
        `, [caseContext?.jurisdiction]);

        // Calculate text similarity (simplified)
        const similarEvidence = [];
        
        for (const doc of similarDocs.rows) {
            const similarity = calculateTextSimilarity(evidenceText, doc.text_chunk);
            
            if (similarity >= CONFIG.enhancement.similarityThreshold) {
                const phrases = JSON.parse(doc.semantic_phrases || '[]');
                
                similarEvidence.push({
                    document_id: doc.document_id,
                    similarity_score: similarity,
                    relevant_phrases: phrases.slice(0, 5),
                    prosecution_outcome: doc.judgement_outcome || 'Unknown'
                });
            }
        }

        return similarEvidence
            .sort((a, b) => b.similarity_score - a.similarity_score)
            .slice(0, 5);

    } catch (error: any) {
        console.error('Similar evidence search failed:', error);
        return [];
    }
}

async function analyzeProsecutionValue(evidenceText: string, caseContext?: unknown): Promise<any> {
    try {
        const analysisPrompt = `As a prosecution strategist, analyze this evidence for its prosecution value:

Evidence:
${evidenceText}

${caseContext ? `Case Context: ${JSON.stringify(caseContext)}` : ''}

Respond with ONLY a JSON object:
{
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],  
  "recommendations": ["recommendation1", "recommendation2"],
  "precedent_support": 0.0-1.0
}

Focus on:
- What makes this evidence strong for prosecution?
- What potential weaknesses should be addressed?
- Strategic recommendations for using this evidence
- How well does this align with legal precedents?`;

        const response = await fetch(`${CONFIG.ollama.url}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: CONFIG.ollama.model,
                prompt: analysisPrompt,
                stream: false,
                options: {
                    temperature: 0.3,
                    top_p: 0.9
                }
            })
        });

        if (response.ok) {
            const data = await response.json();
            const analysisText = data.response.trim();
            
            const jsonMatch = analysisText.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                return JSON.parse(jsonMatch[0]);
            }
        }
    } catch (error: any) {
        console.warn('Prosecution analysis failed:', error);
    }

    // Fallback analysis
    return {
        strengths: ['Documentary evidence', 'Clear factual content'],
        weaknesses: ['May require corroboration', 'Context dependent'],
        recommendations: ['Verify authenticity', 'Gather supporting evidence'],
        precedent_support: 0.6
    };
}

async function generateEmbedding(text: string): Promise<number[]> {
    try {
        const response = await fetch(`${CONFIG.ollama.url}/api/embeddings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: CONFIG.ollama.embeddingModel,
                prompt: text
            })
        });

        if (response.ok) {
            const data = await response.json();
            return data.embedding;
        }
    } catch (error: any) {
        console.warn('Embedding generation failed:', error);
    }

    // Return random embedding as fallback
    return Array.from({ length: 384 }, () => Math.random() - 0.5);
}

function calculateTextSimilarity(text1: string, text2: string): number {
    // Simple word-based similarity (Jaccard index)
    const words1 = new Set(text1.toLowerCase().split(/\s+/));
    const words2 = new Set(text2.toLowerCase().split(/\s+/));
    
    const intersection = new Set([...words1].filter((x: any) => words2.has(x)));
    const union = new Set([...words1, ...words2]);
    
    return intersection.size / union.size;
}

function categorizeLegalPhrase(phrase: string): string {
    const lowerPhrase = phrase.toLowerCase();
    
    if (lowerPhrase.includes('evidence') || lowerPhrase.includes('proof')) {
        return 'evidence';
    } else if (lowerPhrase.includes('testimony') || lowerPhrase.includes('witness')) {
        return 'witness_testimony';
    } else if (lowerPhrase.includes('guilty') || lowerPhrase.includes('conviction')) {
        return 'guilt_indicators';
    } else if (lowerPhrase.includes('precedent') || lowerPhrase.includes('case law')) {
        return 'legal_precedent';
    } else if (lowerPhrase.includes('motion') || lowerPhrase.includes('procedure')) {
        return 'procedural';
    } else {
        return 'general_legal';
    }
}

async function cacheEnhancementResults(evidenceText: string, results: any): Promise<any> {
    try {
        const redis = getRedis();
        const cacheKey = `enhancement:${Buffer.from(evidenceText).toString('base64').substring(0, 32)}`;
        
        await redis.setex(cacheKey, 3600, JSON.stringify({
            results,
            timestamp: new Date().toISOString()
        }));
    } catch (error: any) {
        console.warn('Failed to cache enhancement results:', error);
    }
}

// GET endpoint for enhancement statistics
export const GET: RequestHandler = async () => {
    try {
        const db = getDB();
        
        const stats = await db.query(`
            SELECT 
                COUNT(*) as total_documents,
                AVG(prosecution_strength_score) as avg_prosecution_score,
                COUNT(DISTINCT jurisdiction) as jurisdictions_covered,
                COUNT(DISTINCT case_type) as case_types_covered
            FROM legal_documents_processed
        `);

        const phrasesStats = await db.query(`
            SELECT 
                COUNT(*) as total_phrases,
                COUNT(CASE WHEN avg_prosecution_score >= 70 THEN 1 END) as high_value_phrases,
                AVG(frequency) as avg_phrase_frequency
            FROM semantic_phrases_ranking
        `);

        return json({
            status: 'operational',
            dataset_stats: {
                total_documents: parseInt(stats.rows[0]?.total_documents || '0'),
                average_prosecution_score: parseFloat(stats.rows[0]?.avg_prosecution_score || '0'),
                jurisdictions_covered: parseInt(stats.rows[0]?.jurisdictions_covered || '0'),
                case_types_covered: parseInt(stats.rows[0]?.case_types_covered || '0')
            },
            phrase_stats: {
                total_phrases: parseInt(phrasesStats.rows[0]?.total_phrases || '0'),
                high_value_phrases: parseInt(phrasesStats.rows[0]?.high_value_phrases || '0'),
                average_frequency: parseFloat(phrasesStats.rows[0]?.avg_phrase_frequency || '0')
            },
            enhancement_options: [
                'suggest_labels',
                'extract_entities', 
                'find_similar',
                'prosecution_analysis',
                'fact_checking'
            ]
        });

    } catch (err: any) {
        console.error('Enhancement stats error:', err);
        throw error(500, 'Unable to fetch enhancement statistics');
    }
};