import type { Writable } from 'svelte/store';

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

export class CudaRecommendationClient {
    private static instance: CudaRecommendationClient;
    private cache = new Map<string, { result: CudaRecommendationResult; timestamp: number }>();
    private readonly cacheTimeout = 5 * 60 * 1000; // 5 minutes

    static getInstance(): CudaRecommendationClient {
        if (!CudaRecommendationClient.instance) {
            CudaRecommendationClient.instance = new CudaRecommendationClient();
        }
        return CudaRecommendationClient.instance;
    }

    private constructor() {}

    /**
     * Get CUDA-accelerated recommendations based on user achievements and context
     */
    async getCudaRecommendations(request: RecommendationRequest): Promise<CudaRecommendationResult> {
        const cacheKey = this.generateCacheKey(request);
        const cached = this.cache.get(cacheKey);

        // Return cached result if still valid
        if (cached && (Date.now() - cached.timestamp) < this.cacheTimeout) {
            console.log('Using cached recommendation');
            return cached.result;
        }

        try {
            const response = await fetch('/api/cuda/recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`API request failed: ${response.status} ${response.statusText}. ${errorData.error || ''}`);
            }

            const result: CudaRecommendationResult = await response.json();

            // Cache the successful result
            this.cache.set(cacheKey, {
                result,
                timestamp: Date.now()
            });

            return result;

        } catch (error: any) {
            console.error('Failed to fetch CUDA recommendations:', error);
            throw error;
        }
    }

    /**
     * Convenience method for Canvas components to get recommendations
     * This method extracts achievements from your Canvas context
     */
    async getCanvasRecommendations(achievements: Record<string, any>, canvasContext?: {
        viewportPosition?: [number, number];
        zoomLevel?: number;
        selectedObjects?: string[];
    }): Promise<CudaRecommendationResult> {
        const request: RecommendationRequest = {
            unlockedAchievements: Object.keys(achievements).filter(key => achievements[key]),
            userContext: {
                currentPath: window.location.pathname,
                recentActions: this.getRecentActions(),
            },
            canvasState: canvasContext
        };

        return this.getCudaRecommendations(request);
    }

    /**
     * Clear the recommendation cache (useful for testing or when user context changes significantly)
     */
    clearCache(): void {
        this.cache.clear();
    }

    /**
     * Helper to generate cache keys
     */
    private generateCacheKey(request: RecommendationRequest): string {
        const key = {
            achievements: request.unlockedAchievements.sort(),
            path: request.userContext?.currentPath,
            canvasHash: request.canvasState ? 
                `${request.canvasState.zoomLevel}_${request.canvasState.viewportPosition?.[0]}_${request.canvasState.viewportPosition?.[1]}` 
                : 'no_canvas'
        };
        return JSON.stringify(key);
    }

    /**
     * Extract recent user actions from browser history/session storage
     */
    private getRecentActions(): string[] {
        try {
            const stored = sessionStorage.getItem('recent_user_actions');
            return stored ? JSON.parse(stored) : [];
        } catch {
            return [];
        }
    }

    /**
     * Track user actions for better recommendations
     */
    trackAction(action: string): void {
        try {
            const recentActions = this.getRecentActions();
            recentActions.unshift(action);
            
            // Keep only the last 10 actions
            const trimmed = recentActions.slice(0, 10);
            
            sessionStorage.setItem('recent_user_actions', JSON.stringify(trimmed));
        } catch (error: any) {
            console.warn('Could not track user action:', error);
        }
    }
}

// Export a singleton instance for easy use
export const cudaRecommendationClient = CudaRecommendationClient.getInstance();

/**
 * Svelte store-compatible reactive recommendation service
 * Use this in your Canvas components for reactive recommendations
 */
export function createRecommendationStore(achievements: Writable<Record<string, any>>) {
    let currentRecommendation: CudaRecommendationResult | null = null;
    let isLoading = false;
    let error: string | null = null;

    const updateRecommendation = async (achievementData: Record<string, any>): Promise<any> => {
        if (isLoading) return;
        
        isLoading = true;
        error = null;

        try {
            const recommendation = await cudaRecommendationClient.getCanvasRecommendations(achievementData);
            currentRecommendation = recommendation;
        } catch (err: any) {
            error = err instanceof Error ? err.message : 'Unknown error occurred';
            currentRecommendation = null;
        } finally {
            isLoading = false;
        }
    };

    return {
        getCurrentRecommendation: () => currentRecommendation,
        isLoading: () => isLoading,
        getError: () => error,
        refresh: updateRecommendation,
        trackAction: (action: string) => cudaRecommendationClient.trackAction(action)
    };
}