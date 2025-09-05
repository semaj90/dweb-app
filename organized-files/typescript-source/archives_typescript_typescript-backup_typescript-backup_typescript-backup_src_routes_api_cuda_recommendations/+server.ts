import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export interface RecommendationRequest {
    unlockedAchievements: string[];
    userContext?: {
        currentPath?: string;
        recentActions?: string[];
        preferences?: Record<string, any>;
    };
    canvasState?: {
        viewportPosition?: [number, number];
        zoomLevel?: number;
        selectedObjects?: string[];
    };
}

export interface CudaRecommendationResult {
    recommendation: string;
    confidence: number;
    reasoning: string;
    suggestedActions: Array<{
        type: 'navigate' | 'action' | 'explore';
        target: string;
        description: string;
    }>;
    priority: 'low' | 'medium' | 'high' | 'urgent';
}

async function callCudaRecommendationService(data: RecommendationRequest): Promise<CudaRecommendationResult> {
    try {
        // Call your existing Go microservice on port 8094 (Enhanced RAG)
        const response = await fetch('http://localhost:8094/api/recommendations/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'SvelteKit-Frontend/1.0',
            },
            body: JSON.stringify({
                achievements: data.unlockedAchievements,
                context: data.userContext,
                canvas_state: data.canvasState,
                request_type: 'cuda_accelerated_recommendation',
            }),
        });

        if (!response.ok) {
            throw new Error(`CUDA service responded with ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        return result as CudaRecommendationResult;

    } catch (error: any) {
        console.error('Error calling CUDA recommendation service:', error);
        
        // Fallback recommendation based on achievements
        const fallbackRecommendation = generateFallbackRecommendation(data.unlockedAchievements);
        return fallbackRecommendation;
    }
}

function generateFallbackRecommendation(achievements: string[]): CudaRecommendationResult {
    const achievementCount = achievements.length;
    
    let recommendation: CudaRecommendationResult;
    
    if (achievementCount === 0) {
        recommendation = {
            recommendation: "Start with the 'First Steps' tutorial to unlock your initial achievements",
            confidence: 0.9,
            reasoning: "New user with no achievements - guide them through onboarding",
            suggestedActions: [
                {
                    type: 'navigate',
                    target: '/tutorial/first-steps',
                    description: 'Begin interactive tutorial'
                },
                {
                    type: 'explore',
                    target: 'evidence-collection',
                    description: 'Learn evidence management basics'
                }
            ],
            priority: 'high'
        };
    } else if (achievementCount < 5) {
        recommendation = {
            recommendation: "Explore case file management - you're ready for intermediate features",
            confidence: 0.8,
            reasoning: "User has basic achievements, ready to learn case organization",
            suggestedActions: [
                {
                    type: 'navigate',
                    target: '/cases/create',
                    description: 'Create your first case file'
                },
                {
                    type: 'action',
                    target: 'upload-evidence',
                    description: 'Upload evidence documents'
                }
            ],
            priority: 'medium'
        };
    } else {
        recommendation = {
            recommendation: "Advanced AI-powered legal analysis tools are now available to you",
            confidence: 0.95,
            reasoning: "Experienced user with multiple achievements - offer advanced features",
            suggestedActions: [
                {
                    type: 'explore',
                    target: 'ai-legal-analysis',
                    description: 'Use AI for document analysis'
                },
                {
                    type: 'navigate',
                    target: '/reports/advanced',
                    description: 'Generate comprehensive case reports'
                }
            ],
            priority: 'low'
        };
    }
    
    return recommendation;
}

export const POST: RequestHandler = async ({ request }): Promise<any> => {
    try {
        const requestData: RecommendationRequest = await request.json();

        // Validate required fields
        if (!requestData.unlockedAchievements || !Array.isArray(requestData.unlockedAchievements)) {
            return json({ 
                error: 'Invalid request: unlockedAchievements array is required' 
            }, { status: 400 });
        }

        console.log('Processing CUDA recommendation request:', {
            achievementCount: requestData.unlockedAchievements.length,
            hasContext: !!requestData.userContext,
            hasCanvasState: !!requestData.canvasState
        });

        // Call the CUDA-accelerated recommendation service
        const result = await callCudaRecommendationService(requestData);

        return json(result, { status: 200 });

    } catch (error: any) {
        console.error('API Error in recommendations endpoint:', error);
        return json({ 
            error: 'Failed to generate recommendations',
            details: error instanceof Error ? error.message : 'Unknown error'
        }, { status: 500 });
    }
};

export const GET: RequestHandler = async (): Promise<any> => {
    return json({
        endpoint: 'CUDA Recommendations API',
        version: '1.0.0',
        methods: ['POST'],
        description: 'Generates AI-powered recommendations using CUDA acceleration',
        status: 'active'
    });
};