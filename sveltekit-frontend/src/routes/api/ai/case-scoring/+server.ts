
import { json } from '@sveltejs/kit';
import { qdrantService } from "$lib/server/services/qdrant-service";
import type { RequestHandler } from './$types';


export const POST: RequestHandler = async ({ request }) => {
  try {
    const { content, evidenceType = 'evidence', metadata = {} } = await request.json();

    const rawScore = await qdrantService.calculateAISummaryScore(content, evidenceType, metadata);
    const score = Number(rawScore);

    if (Number.isNaN(score)) {
      throw new Error('Invalid score returned from qdrantService');
    }

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
  } catch (error: any) {
    return json({ error: 'Scoring failed', details: error?.message ?? String(error) }, { status: 500 });
  }
};

async function generateScoreReasoning(score: number, evidenceType: string): Promise<string> {
  if (score >= 80) return `High-value ${evidenceType} with strong legal admissibility and strategic relevance.`;
  if (score >= 60) return `Solid ${evidenceType} evidence with moderate legal value and clear procedural compliance.`;
  if (score >= 40) return `Basic ${evidenceType} evidence requiring additional corroboration for optimal case strength.`;
  return `Limited ${evidenceType} evidence with significant admissibility concerns requiring review.`;
}