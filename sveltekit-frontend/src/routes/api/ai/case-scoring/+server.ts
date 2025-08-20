// @ts-nocheck
import { type RequestHandler,  json } from '@sveltejs/kit';
import { qdrantService } from "$lib/server/services/qdrant-service";

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { content, evidenceType, caseId, evidenceId, metadata = {} } = await request.json();
    
    const score = await qdrantService.calculateAISummaryScore(content, evidenceType, metadata);
    
    return json({
      score,
      breakdown: {
        admissibility: Math.round(score * 0.25),
        relevance: Math.round(score * 0.25), 
        quality: Math.round(score * 0.25),
        strategic: Math.round(score * 0.25)
      },
      reasoning: await generateScoreReasoning(score, evidenceType),
      confidence: score > 70 ? 0.85 : score > 50 ? 0.75 : 0.65,
      lastUpdated: new Date().toISOString()
    });
  } catch (error) {
    return json({ error: 'Scoring failed', details: error.message }, { status: 500 });
  }
};

async function generateScoreReasoning(score: number, evidenceType: string): Promise<string> {
  if (score >= 80) return `High-value ${evidenceType} with strong legal admissibility and strategic relevance.`;
  if (score >= 60) return `Solid ${evidenceType} evidence with moderate legal value and clear procedural compliance.`;
  if (score >= 40) return `Basic ${evidenceType} evidence requiring additional corroboration for optimal case strength.`;
  return `Limited ${evidenceType} evidence with significant admissibility concerns requiring review.`;
}