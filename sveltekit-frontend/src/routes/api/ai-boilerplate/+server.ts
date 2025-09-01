
import type { RequestHandler } from './$types.js';

/**
 * AI-Assisted Boilerplate Generation API
 * Generates legal boilerplate text based on high-performing phrase patterns
 */

import { Pool } from "pg";
import { z } from 'zod';

// Configuration
const CONFIG = {
    database: {
        user: import.meta.env.DB_USER || 'postgres',
        password: import.meta.env.DB_PASSWORD || 'password',
        host: import.meta.env.DB_HOST || 'localhost',
        port: parseInt(import.meta.env.DB_PORT || '5432'),
        database: import.meta.env.DB_NAME || 'prosecutor_db'
    },
    ollama: {
        url: import.meta.env.OLLAMA_URL || 'http://localhost:11434',
        model: import.meta.env.LLM_MODEL || 'gemma3-legal'
    },
    boilerplate: {
        minProsecutionScore: 70,
        maxTemplates: 5,
        templateLength: 300
    }
};

// Validation schemas
const BoilerplateRequestSchema = z.object({
    type: z.enum([
        'prosecution_argument',
        'evidence_summary',
        'legal_motion',
        'case_analysis',
        'sentencing_memo',
        'plea_agreement',
        'discovery_request'
    ]),
    jurisdiction: z.enum(['federal', 'state', 'local', 'international']).optional(),
    context: z.object({
        case_type: z.enum(['criminal', 'civil', 'administrative', 'constitutional']).optional(),
        defendant_name: z.string().optional(),
        charges: z.array(z.string()).optional(),
        evidence_types: z.array(z.string()).optional(),
        precedents: z.array(z.string()).optional(),
        custom_context: z.string().optional()
    }).optional(),
    tone: z.enum(['formal', 'aggressive', 'neutral', 'persuasive']).optional(),
    length: z.enum(['brief', 'standard', 'detailed']).optional()
});

const BoilerplateResponseSchema = z.object({
    boilerplate_text: z.string(),
    source_phrases: z.array(z.string()),
    confidence_score: z.number(),
    prosecution_strength: z.number(),
    suggested_edits: z.array(z.string()),
    metadata: z.object({
        template_type: z.string(),
        jurisdiction: z.string().optional(),
        generation_time_ms: z.number()
    })
});

// Initialize database connection
let db: Pool | null = null;

function getDB() {
    if (!db) {
        db = new Pool(CONFIG.database);
    }
    return db;
}

export const POST: RequestHandler = async ({ request }) => {
    const startTime = Date.now();
    
    try {
        const requestData = await request.json();
        
        // Validate request
        const validatedRequest = BoilerplateRequestSchema.parse(requestData);
        
        console.log(`📝 Generating boilerplate: ${validatedRequest.type}`);

        // Get high-performing phrases for the requested type
        const sourcePhrases = await getHighPerformingPhrases(
            validatedRequest.type,
            validatedRequest.jurisdiction,
            validatedRequest.context
        );

        // Generate boilerplate using LLM
        const boilerplateResult = await generateBoilerplate(
            validatedRequest,
            sourcePhrases
        );

        // Enhance with additional suggestions
        const suggestedEdits = await generateSuggestedEdits(
            boilerplateResult.text,
            validatedRequest.type
        );

        const response = {
            boilerplate_text: boilerplateResult.text,
            source_phrases: sourcePhrases.map((p: any) => p.phrase),
            confidence_score: boilerplateResult.confidence,
            prosecution_strength: boilerplateResult.prosecutionStrength,
            suggested_edits: suggestedEdits,
            metadata: {
                template_type: validatedRequest.type,
                jurisdiction: validatedRequest.jurisdiction,
                generation_time_ms: Date.now() - startTime
            }
        };

        // Validate response
        const validatedResponse = BoilerplateResponseSchema.parse(response);

        return json(validatedResponse);

    } catch (err: any) {
        console.error('❌ AI Boilerplate generation error:', err);
        
        if (err instanceof z.ZodError) {
            return json({
                message: 'Invalid request format',
                errors: err.errors
            }, { status: 400 });
        }

        return json({
            message: 'AI Boilerplate service temporarily unavailable',
            details: err instanceof Error ? err.message : 'Unknown error'
        }, { status: 500 });
    }
};

async function getHighPerformingPhrases(
    type: string,
    jurisdiction?: string,
    context?: { case_type?: string; [key: string]: any }
): Promise<any> {
    const db = getDB();
    
    // Build query based on request parameters
    let sql = `
        SELECT DISTINCT
            spr.phrase,
            spr.avg_prosecution_score,
            spr.frequency,
            spr.correlation_strength,
            COUNT(ldp.id) as usage_count
        FROM semantic_phrases_ranking spr
        JOIN legal_documents_processed ldp ON ldp.semantic_phrases::text LIKE '%' || spr.phrase || '%'
        WHERE spr.avg_prosecution_score >= $1
    `;
    
    const params: any[] = [CONFIG.boilerplate.minProsecutionScore];
    let paramIndex = 2;

    // Filter by jurisdiction if specified
    if (jurisdiction) {
        sql += ` AND ldp.jurisdiction = $${paramIndex}`;
        params.push(jurisdiction);
        paramIndex++;
    }

    // Filter by case type if specified
    if (context?.case_type) {
        sql += ` AND ldp.case_type = $${paramIndex}`;
        params.push(context.case_type);
        paramIndex++;
    }

    // Add type-specific filtering
    sql += getTypeSpecificFilter(type, paramIndex, params);

    sql += `
        GROUP BY spr.phrase, spr.avg_prosecution_score, spr.frequency, spr.correlation_strength
        ORDER BY 
            spr.avg_prosecution_score DESC,
            usage_count DESC,
            spr.correlation_strength DESC
        LIMIT 20
    `;

    const result = await db.query(sql, params);
    return result.rows;
}

function getTypeSpecificFilter(type: string, paramIndex: number, params: any[]): string {
    const typeFilters = {
        'prosecution_argument': " AND (ldp.semantic_phrases::text ILIKE '%prosecution%' OR ldp.semantic_phrases::text ILIKE '%argument%' OR ldp.semantic_phrases::text ILIKE '%evidence%')",
        'evidence_summary': " AND (ldp.semantic_phrases::text ILIKE '%evidence%' OR ldp.semantic_phrases::text ILIKE '%testimony%' OR ldp.semantic_phrases::text ILIKE '%proof%')",
        'legal_motion': " AND (ldp.semantic_phrases::text ILIKE '%motion%' OR ldp.semantic_phrases::text ILIKE '%request%' OR ldp.semantic_phrases::text ILIKE '%order%')",
        'case_analysis': " AND (ldp.semantic_phrases::text ILIKE '%analysis%' OR ldp.semantic_phrases::text ILIKE '%precedent%' OR ldp.semantic_phrases::text ILIKE '%ruling%')",
        'sentencing_memo': " AND (ldp.semantic_phrases::text ILIKE '%sentencing%' OR ldp.semantic_phrases::text ILIKE '%punishment%' OR ldp.semantic_phrases::text ILIKE '%mitigation%')",
        'plea_agreement': " AND (ldp.semantic_phrases::text ILIKE '%plea%' OR ldp.semantic_phrases::text ILIKE '%agreement%' OR ldp.semantic_phrases::text ILIKE '%guilty%')",
        'discovery_request': " AND (ldp.semantic_phrases::text ILIKE '%discovery%' OR ldp.semantic_phrases::text ILIKE '%documents%' OR ldp.semantic_phrases::text ILIKE '%disclosure%')"
    };

    return typeFilters[type] || '';
}

async function generateBoilerplate(request: any, sourcePhrases: any[]): Promise<any> {
    const phraseText = sourcePhrases.map((p: any) => p.phrase).join(', ');
    const avgProsecutionScore = sourcePhrases.reduce((sum, p) => sum + p.avg_prosecution_score, 0) / sourcePhrases.length;

    const systemPrompt = buildSystemPrompt(request.type, request.tone || 'formal');
    const contextPrompt = buildContextPrompt(request.context);
    
    const fullPrompt = `${systemPrompt}

${contextPrompt}

Based on these high-performing legal phrases that have shown strong prosecution correlation:
${phraseText}

Generate a ${request.length || 'standard'} length ${request.type} that incorporates these proven effective phrases while maintaining legal accuracy and ${request.tone || 'formal'} tone.

Requirements:
- Use clear, persuasive legal language
- Incorporate the provided high-scoring phrases naturally
- Maintain professional legal writing standards
- Focus on strength of argument and evidence
- Length: ${getLengthGuidance(request.length)}

Generate the boilerplate text:`;

    try {
        const response = await fetch(`${CONFIG.ollama.url}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: CONFIG.ollama.model,
                prompt: fullPrompt,
                stream: false,
                options: {
                    temperature: 0.4,
                    top_p: 0.9,
                    repeat_penalty: 1.1,
                    num_predict: getLengthTokens(request.length)
                }
            })
        });

        if (!response.ok) {
            throw new Error(`LLM request failed: ${response.status}`);
        }

        const data = await response.json();
        const generatedText = data.response.trim();

        // Calculate confidence based on phrase usage
        const phrasesUsed = sourcePhrases.filter((p: any) => generatedText.toLowerCase().includes(p.phrase.toLowerCase())
        ).length;
        
        const confidence = Math.min((phrasesUsed / sourcePhrases.length) * 0.8 + 0.2, 1.0);

        return {
            text: generatedText,
            confidence,
            prosecutionStrength: avgProsecutionScore
        };

    } catch (error: any) {
        console.error('LLM generation failed:', error);
        
        // Fallback to template-based generation
        return generateFallbackBoilerplate(request.type, sourcePhrases);
    }
}

function buildSystemPrompt(type: string, tone: string): string {
    const basePrompt = "You are an expert legal writer specializing in prosecution documents. ";
    
    const typePrompts = {
        'prosecution_argument': "Generate a compelling prosecution argument that clearly establishes the defendant's guilt and the strength of the evidence.",
        'evidence_summary': "Create a comprehensive evidence summary that highlights the most compelling facts and their legal significance.",
        'legal_motion': "Draft a professional legal motion with proper formatting and persuasive legal reasoning.",
        'case_analysis': "Provide a thorough case analysis examining legal precedents, evidence, and potential outcomes.",
        'sentencing_memo': "Write a sentencing memorandum that effectively argues for appropriate punishment based on the facts and law.",
        'plea_agreement': "Draft a plea agreement that protects the prosecution's interests while following legal requirements.",
        'discovery_request': "Create a comprehensive discovery request that will uncover all relevant evidence for the prosecution."
    };

    const toneAdjustments = {
        'formal': "Use formal legal language and maintain a professional, authoritative tone.",
        'aggressive': "Use strong, assertive language that emphasizes the strength of the prosecution's case.",
        'neutral': "Use objective, fact-based language that presents information clearly and impartially.",
        'persuasive': "Use compelling, convincing language that builds a strong case for the prosecution."
    };

    return `${basePrompt}${typePrompts[type]} ${toneAdjustments[tone]}`;
}

function buildContextPrompt(context?: { 
    defendant_name?: string;
    charges?: string[];
    evidence_types?: string[];
    precedents?: string[];
    custom_context?: string;
    [key: string]: any;
}): string {
    if (!context) return '';

    let contextPrompt = 'Context for this document:\n';
    
    if (context.defendant_name) {
        contextPrompt += `- Defendant: ${context.defendant_name}\n`;
    }
    
    if (context.charges?.length > 0) {
        contextPrompt += `- Charges: ${context.charges.join(', ')}\n`;
    }
    
    if (context.evidence_types?.length > 0) {
        contextPrompt += `- Evidence Types: ${context.evidence_types.join(', ')}\n`;
    }
    
    if (context.precedents?.length > 0) {
        contextPrompt += `- Relevant Precedents: ${context.precedents.join(', ')}\n`;
    }
    
    if (context.custom_context) {
        contextPrompt += `- Additional Context: ${context.custom_context}\n`;
    }

    return contextPrompt + '\n';
}

function getLengthGuidance(length?: string): string {
    switch (length) {
        case 'brief': return '1-2 paragraphs (100-200 words)';
        case 'detailed': return '4-6 paragraphs (400-600 words)';
        default: return '2-4 paragraphs (200-400 words)';
    }
}

function getLengthTokens(length?: string): number {
    switch (length) {
        case 'brief': return 300;
        case 'detailed': return 800;
        default: return 500;
    }
}

function generateFallbackBoilerplate(type: string, sourcePhrases: any[]) {
    const templates = {
        'prosecution_argument': `Based on the compelling evidence presented, the prosecution has demonstrated beyond a reasonable doubt that the defendant is guilty of the charges. The evidence includes ${sourcePhrases.slice(0, 3).map((p: any) => p.phrase).join(', ')}, which clearly establishes the defendant's culpability.`,
        'evidence_summary': `The following evidence strongly supports the prosecution's case: ${sourcePhrases.slice(0, 5).map((p: any) => p.phrase).join(', ')}. This evidence demonstrates a clear pattern of behavior and establishes the necessary elements of the charges.`,
        'legal_motion': `The prosecution respectfully moves the court for relief based on the following grounds: ${sourcePhrases.slice(0, 3).map((p: any) => p.phrase).join(', ')}. The motion is supported by applicable law and compelling evidence.`
    };

    const fallbackText = templates[type] || `The prosecution presents the following legal argument incorporating proven effective elements: ${sourcePhrases.slice(0, 5).map((p: any) => p.phrase).join(', ')}.`;
    
    return {
        text: fallbackText,
        confidence: 0.6,
        prosecutionStrength: 75
    };
}

async function generateSuggestedEdits(text: string, type: string): Promise<string[]> {
    const suggestions = [];
    
    // Basic suggestions based on text analysis
    if (!text.includes('evidence')) {
        suggestions.push('Consider adding specific evidence references');
    }
    
    if (!text.includes('precedent') && type === 'legal_motion') {
        suggestions.push('Include relevant legal precedents');
    }
    
    if (text.length < 200) {
        suggestions.push('Consider expanding with additional supporting arguments');
    }
    
    if (!text.includes('defendant')) {
        suggestions.push('Make sure to clearly identify the defendant');
    }

    // Type-specific suggestions
    const typeSpecificSuggestions = {
        'prosecution_argument': ['Add specific statutory citations', 'Include burden of proof language'],
        'evidence_summary': ['Organize evidence chronologically', 'Highlight most compelling evidence first'],
        'sentencing_memo': ['Include sentencing guidelines reference', 'Address mitigating factors']
    };

    if (typeSpecificSuggestions[type]) {
        suggestions.push(...typeSpecificSuggestions[type]);
    }

    return suggestions.slice(0, 5); // Limit to 5 suggestions
}

// GET endpoint for available templates
export const GET: RequestHandler = async () => {
    try {
        const db = getDB();
        
        // Get statistics about available templates
        const stats = await db.query(`
            SELECT 
                COUNT(*) as total_phrases,
                AVG(avg_prosecution_score) as avg_score,
                COUNT(DISTINCT CASE WHEN avg_prosecution_score >= 80 THEN phrase END) as high_performing_phrases
            FROM semantic_phrases_ranking
        `);

        const templates = [
            {
                type: 'prosecution_argument',
                name: 'Prosecution Argument',
                description: 'Compelling arguments for establishing guilt',
                available_phrases: stats.rows[0]?.high_performing_phrases || 0
            },
            {
                type: 'evidence_summary',
                name: 'Evidence Summary',
                description: 'Comprehensive overview of case evidence',
                available_phrases: stats.rows[0]?.high_performing_phrases || 0
            },
            {
                type: 'legal_motion',
                name: 'Legal Motion',
                description: 'Professional legal motions and requests',
                available_phrases: stats.rows[0]?.high_performing_phrases || 0
            },
            {
                type: 'case_analysis',
                name: 'Case Analysis',
                description: 'Detailed legal case analysis',
                available_phrases: stats.rows[0]?.high_performing_phrases || 0
            },
            {
                type: 'sentencing_memo',
                name: 'Sentencing Memorandum',
                description: 'Arguments for appropriate sentencing',
                available_phrases: stats.rows[0]?.high_performing_phrases || 0
            }
        ];

        return json({
            templates,
            statistics: {
                total_phrases: parseInt(String(stats.rows[0]?.total_phrases || '0')),
                average_score: parseFloat(String(stats.rows[0]?.avg_score || '0')),
                high_performing_count: parseInt(String(stats.rows[0]?.high_performing_phrases || '0'))
            }
        });

    } catch (err: any) {
        console.error('Template listing error:', err);
        throw error(500, 'Unable to fetch template information');
    }
};