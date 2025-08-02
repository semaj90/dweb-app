// CaseScoringService.ts - AI-Powered Case Scoring System
// Implements 0-100 scoring with multi-criteria analysis

import { ollamaService } from './OllamaService';
import { db } from '../db';
import { caseScores } from '../db/schema';
import { logger } from '../logger';
import { eq } from 'drizzle-orm';
import type { CaseScoringRequest, CaseScoringResult, ScoringCriteria } from '$lib/types';

export class CaseScoringService {
    private readonly DEFAULT_TEMPERATURE = 0.7;
    private readonly SCORING_MODEL = 'gemma3-legal';
    
    // Scoring weights for different criteria
    private readonly CRITERIA_WEIGHTS = {
        evidence_strength: 0.25,
        witness_reliability: 0.20,
        legal_precedent: 0.20,
        public_interest: 0.15,
        case_complexity: 0.10,
        resource_requirements: 0.10
    };

    /**
     * Score a case using AI analysis
     */
    async scoreCase(request: CaseScoringRequest): Promise<CaseScoringResult> {
        try {
            const startTime = Date.now();
            
            // Validate request
            this.validateRequest(request);
            
            // Generate AI analysis
            const aiAnalysis = await this.generateAIAnalysis(request);
            
            // Calculate component scores
            const componentScores = await this.calculateComponentScores(request, aiAnalysis);
            
            // Calculate final score
            const finalScore = this.calculateWeightedScore(componentScores);
            
            // Generate recommendations
            const recommendations = await this.generateRecommendations(
                request, 
                componentScores, 
                finalScore
            );
            
            // Store scoring result
            const scoringResult: CaseScoringResult = {
                case_id: request.case_id,
                score: finalScore,
                scoring_criteria: componentScores,
                ai_analysis: aiAnalysis,
                recommendations,
                confidence: this.calculateConfidence(componentScores),
                processing_time: Date.now() - startTime,
                timestamp: new Date().toISOString()
            };
            
            // Save to database
            await this.saveScoring(scoringResult, request.temperature);
            
            logger.info('Case scored successfully', { 
                case_id: request.case_id, 
                score: finalScore 
            });
            
            return scoringResult;
        } catch (error) {
            logger.error('Failed to score case', error);
            throw error;
        }
    }

    /**
     * Generate AI analysis of the case
     */
    private async generateAIAnalysis(request: CaseScoringRequest): Promise<string> {
        const prompt = `Analyze this legal case for prosecution viability:

Case Title: ${request.case_data.title}
Description: ${request.case_data.description}
Evidence Count: ${request.case_data.evidence_count || 0}
Witness Count: ${request.case_data.witness_count || 0}
Complexity: ${request.case_data.complexity || 'medium'}

Scoring Criteria Provided:
${JSON.stringify(request.scoring_criteria, null, 2)}

Provide a comprehensive analysis covering:
1. Strength of evidence and its admissibility
2. Reliability and credibility of witnesses
3. Relevant legal precedents and their applicability
4. Public interest and societal impact
5. Resource requirements and case complexity
6. Likelihood of successful prosecution
7. Potential challenges and weaknesses
8. Strategic recommendations

Be objective, thorough, and consider both strengths and weaknesses.`;

        const analysis = await ollamaService.generateCompletion(
            prompt,
            {
                temperature: request.temperature || this.DEFAULT_TEMPERATURE,
                max_tokens: 1000,
                model: this.SCORING_MODEL
            }
        );

        return analysis;
    }

    /**
     * Calculate component scores based on criteria
     */
    private async calculateComponentScores(
        request: CaseScoringRequest, 
        aiAnalysis: string
    ): Promise<ScoringCriteria> {
        // Extract scores from provided criteria
        const provided = request.scoring_criteria;
        
        // Use AI to analyze additional factors
        const aiScorePrompt = `Based on this case analysis, provide numerical scores (0-1) for each criterion:

Analysis: ${aiAnalysis}

Rate the following on a scale of 0 to 1:
1. Evidence Strength (considering admissibility and weight)
2. Witness Reliability (considering credibility and consistency)
3. Legal Precedent Support (considering applicable case law)
4. Public Interest (considering societal impact and deterrence)
5. Case Complexity (inverse - lower score for more complex)
6. Resource Requirements (inverse - lower score for more resources needed)

Respond in JSON format with keys: evidence_strength, witness_reliability, legal_precedent, public_interest, case_complexity, resource_requirements`;

        try {
            const aiScoresRaw = await ollamaService.generateCompletion(
                aiScorePrompt,
                {
                    temperature: 0.3, // Lower temperature for more consistent scoring
                    max_tokens: 200,
                    model: this.SCORING_MODEL
                }
            );

            // Parse AI scores
            const aiScores = this.parseAIScores(aiScoresRaw);
            
            // Combine provided scores with AI analysis
            return {
                evidence_strength: provided.evidence_strength ?? aiScores.evidence_strength ?? 0.5,
                witness_reliability: provided.witness_reliability ?? aiScores.witness_reliability ?? 0.5,
                legal_precedent: provided.legal_precedent ?? aiScores.legal_precedent ?? 0.5,
                public_interest: provided.public_interest ?? aiScores.public_interest ?? 0.5,
                case_complexity: aiScores.case_complexity ?? 0.5,
                resource_requirements: aiScores.resource_requirements ?? 0.5,
                ...provided // Include any additional criteria
            };
        } catch (error) {
            logger.warn('Failed to get AI scores, using defaults', error);
            // Fallback to provided scores or defaults
            return {
                evidence_strength: provided.evidence_strength ?? 0.5,
                witness_reliability: provided.witness_reliability ?? 0.5,
                legal_precedent: provided.legal_precedent ?? 0.5,
                public_interest: provided.public_interest ?? 0.5,
                case_complexity: 0.5,
                resource_requirements: 0.5,
                ...provided
            };
        }
    }

    /**
     * Calculate weighted final score
     */
    private calculateWeightedScore(criteria: ScoringCriteria): number {
        let weightedSum = 0;
        let totalWeight = 0;

        for (const [key, weight] of Object.entries(this.CRITERIA_WEIGHTS)) {
            const value = criteria[key as keyof ScoringCriteria];
            if (typeof value === 'number') {
                weightedSum += value * weight;
                totalWeight += weight;
            }
        }

        // Normalize to 0-100 scale
        const normalizedScore = totalWeight > 0 ? (weightedSum / totalWeight) * 100 : 50;
        
        // Round to integer
        return Math.round(Math.max(0, Math.min(100, normalizedScore)));
    }

    /**
     * Generate actionable recommendations
     */
    private async generateRecommendations(
        request: CaseScoringRequest,
        scores: ScoringCriteria,
        finalScore: number
    ): Promise<string[]> {
        const recommendations: string[] = [];

        // Score-based recommendations
        if (finalScore >= 80) {
            recommendations.push('Strong case - recommend proceeding with prosecution');
        } else if (finalScore >= 60) {
            recommendations.push('Viable case - consider strengthening weak areas before proceeding');
        } else if (finalScore >= 40) {
            recommendations.push('Borderline case - significant improvements needed');
        } else {
            recommendations.push('Weak case - recommend further investigation or declining prosecution');
        }

        // Component-specific recommendations
        if (scores.evidence_strength < 0.6) {
            recommendations.push('Strengthen evidence collection and chain of custody');
        }
        if (scores.witness_reliability < 0.6) {
            recommendations.push('Assess witness credibility and consider additional witnesses');
        }
        if (scores.legal_precedent < 0.6) {
            recommendations.push('Research additional supporting case law and precedents');
        }
        if (scores.public_interest < 0.5) {
            recommendations.push('Consider public interest factors and potential community impact');
        }
        if (scores.case_complexity > 0.7) {
            recommendations.push('Case complexity is manageable - allocate appropriate resources');
        }
        if (scores.resource_requirements > 0.7) {
            recommendations.push('Resource requirements are reasonable - proceed with standard allocation');
        }

        // AI-generated strategic recommendations
        const strategyPrompt = `Based on a case score of ${finalScore}/100 and the following analysis:
${request.case_data.description}

Provide 2-3 specific strategic recommendations for the prosecution team.`;

        try {
            const aiRecommendations = await ollamaService.generateCompletion(
                strategyPrompt,
                {
                    temperature: 0.5,
                    max_tokens: 200,
                    model: this.SCORING_MODEL
                }
            );

            // Parse and add AI recommendations
            const parsed = aiRecommendations.split('\n')
                .filter(line => line.trim().length > 10)
                .slice(0, 3);
            recommendations.push(...parsed);
        } catch (error) {
            logger.warn('Failed to generate AI recommendations', error);
        }

        return recommendations;
    }

    /**
     * Calculate confidence level of the scoring
     */
    private calculateConfidence(scores: ScoringCriteria): number {
        // Calculate variance in scores
        const values = Object.values(scores).filter(v => typeof v === 'number') as number[];
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        
        // Lower variance = higher confidence
        const confidence = Math.max(0.5, 1 - (variance * 2));
        return Math.round(confidence * 100) / 100;
    }

    /**
     * Parse AI-generated scores from text
     */
    private parseAIScores(aiResponse: string): Partial<ScoringCriteria> {
        try {
            // Try to extract JSON from response
            const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                const parsed = JSON.parse(jsonMatch[0]);
                
                // Validate and normalize scores
                const scores: Partial<ScoringCriteria> = {};
                for (const [key, value] of Object.entries(parsed)) {
                    if (typeof value === 'number') {
                        scores[key as keyof ScoringCriteria] = Math.max(0, Math.min(1, value));
                    }
                }
                return scores;
            }
        } catch (error) {
            logger.warn('Failed to parse AI scores', error);
        }
        
        return {};
    }

    /**
     * Validate scoring request
     */
    private validateRequest(request: CaseScoringRequest): void {
        if (!request.case_id) {
            throw new Error('Case ID is required');
        }
        if (!request.case_data) {
            throw new Error('Case data is required');
        }
        if (!request.scoring_criteria) {
            throw new Error('Scoring criteria is required');
        }
        
        // Validate temperature if provided
        if (request.temperature !== undefined) {
            if (request.temperature < 0 || request.temperature > 1) {
                throw new Error('Temperature must be between 0 and 1');
            }
        }
    }

    /**
     * Save scoring result to database
     */
    private async saveScoring(result: CaseScoringResult, temperature?: number): Promise<void> {
        try {
            await db.insert(caseScores).values({
                case_id: result.case_id,
                score: result.score,
                scoring_criteria: result.scoring_criteria,
                ai_analysis: result.ai_analysis,
                temperature: temperature || this.DEFAULT_TEMPERATURE,
                created_at: new Date(),
                updated_at: new Date()
            });
        } catch (error) {
            logger.error('Failed to save case scoring', error);
            // Don't throw - scoring can still be returned even if save fails
        }
    }

    /**
     * Get historical scores for a case
     */
    async getCaseScoreHistory(caseId: string): Promise<CaseScoringResult[]> {
        try {
            const scores = await db
                .select()
                .from(caseScores)
                .where(eq(caseScores.case_id, caseId))
                .orderBy(caseScores.created_at);

            return scores.map(score => ({
                case_id: score.case_id,
                score: score.score,
                scoring_criteria: score.scoring_criteria as ScoringCriteria,
                ai_analysis: score.ai_analysis || '',
                recommendations: [], // Not stored historically
                confidence: 0, // Not stored historically
                processing_time: 0, // Not stored historically
                timestamp: score.created_at.toISOString()
            }));
        } catch (error) {
            logger.error('Failed to get case score history', error);
            throw error;
        }
    }
}

// Export singleton instance
export const caseScoringService = new CaseScoringService();
