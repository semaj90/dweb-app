<!-- 
  Example integration of CUDA recommendations with your Canvas component
  Add these functions and reactive statements to your existing Canvas component
-->
<script lang="ts">
    import { onMount } from 'svelte';
    import { cudaRecommendationClient, type CudaRecommendationResult } from '$lib/services/cuda-recommendation-client';

    // Your existing achievement system (from your Canvas component)
    let achievements = $state({
        firstLoad: false,
        exploredUI: false,
        interactedWithCanvas: false,
        // ... your other achievements
    });

    // CUDA recommendation state
    let currentRecommendation: CudaRecommendationResult | null = $state(null);
    let recommendationLoading = $state(false);
    let recommendationError: string | null = $state(null);

    // Canvas state to send to CUDA backend
    let canvasViewport = $state({ x: 0, y: 0 });
    let canvasZoom = $state(1.0);
    let selectedCanvasObjects = $state<string[]>([]);

    /**
     * Call this function to get CUDA-powered recommendations
     * This sends your achievements and canvas state to your Go backend
     */
    async function getCudaRecommendations() {
        if (recommendationLoading) return;

        recommendationLoading = true;
        recommendationError = null;

        try {
            // Track this action for future recommendations
            cudaRecommendationClient.trackAction('requested_cuda_recommendation');

            // Send your Canvas data to the CUDA backend
            const recommendation = await cudaRecommendationClient.getCanvasRecommendations(
                achievements,
                {
                    viewportPosition: [canvasViewport.x, canvasViewport.y],
                    zoomLevel: canvasZoom,
                    selectedObjects: selectedCanvasObjects
                }
            );

            currentRecommendation = recommendation;

            // Show the recommendation in your UI (you can customize this)
            showRecommendationNotification(recommendation);

        } catch (error) {
            recommendationError = error instanceof Error ? error.message : 'Failed to get recommendations';
            console.error('CUDA recommendation error:', error);
        } finally {
            recommendationLoading = false;
        }
    }

    /**
     * Automatically get new recommendations when achievements change
     * This creates a reactive system that updates recommendations based on user progress
     */
    $effect(() => {
        const achievementCount = Object.keys(achievements).filter(key => achievements[key]).length;
        
        // Only auto-update if user has at least one achievement
        if (achievementCount > 0) {
            // Debounce the recommendation updates
            const timeoutId = setTimeout(() => {
                getCudaRecommendations();
            }, 1000);

            return () => clearTimeout(timeoutId);
        }
    });

    /**
     * Display the recommendation in your notification system
     */
    function showRecommendationNotification(rec: CudaRecommendationResult) {
        // Use your existing notification system
        // Replace this with your actual notification function
        console.log('🎯 CUDA Recommendation:', rec.recommendation);
        
        // Example: You might have a notification store or function
        // notificationStore.add({
        //     type: rec.priority === 'high' ? 'info' : 'suggestion',
        //     title: 'AI Recommendation',
        //     message: rec.recommendation,
        //     actions: rec.suggestedActions
        // });
    }

    /**
     * Handle recommended actions from the CUDA backend
     */
    function executeRecommendedAction(action: CudaRecommendationResult['suggestedActions'][0]) {
        cudaRecommendationClient.trackAction(`executed_${action.type}_${action.target}`);

        switch (action.type) {
            case 'navigate':
                // Navigate to the suggested page
                window.location.href = action.target;
                break;
                
            case 'action':
                // Execute a specific action (customize based on your app's actions)
                if (action.target === 'upload-evidence') {
                    // Trigger file upload dialog
                    // uploadEvidence();
                } else if (action.target === 'create-case') {
                    // Open case creation modal
                    // openCaseCreationModal();
                }
                break;
                
            case 'explore':
                // Guide user to explore a feature
                // highlightFeature(action.target);
                break;
        }
    }

    // Example: Update canvas state that gets sent to CUDA backend
    function updateCanvasViewport(x: number, y: number) {
        canvasViewport = { x, y };
        // The $effect above will automatically trigger a new recommendation
    }

    function updateCanvasZoom(zoom: number) {
        canvasZoom = zoom;
    }

    function selectCanvasObject(objectId: string) {
        if (!selectedCanvasObjects.includes(objectId)) {
            selectedCanvasObjects = [...selectedCanvasObjects, objectId];
        }
    }

    onMount(() => {
        // Initial recommendation request when component loads
        if (Object.keys(achievements).some(key => achievements[key])) {
            getCudaRecommendations();
        }
    });
</script>

<!-- 
  Add this UI to your existing Canvas component to show recommendations
-->
{#if currentRecommendation}
    <div class="cuda-recommendation-panel">
        <div class="recommendation-header">
            <h3>🎯 AI Recommendation</h3>
            <span class="confidence">Confidence: {(currentRecommendation.confidence * 100).toFixed(0)}%</span>
        </div>
        
        <p class="recommendation-text">{currentRecommendation.recommendation}</p>
        
        {#if currentRecommendation.reasoning}
            <details class="reasoning">
                <summary>Why this recommendation?</summary>
                <p>{currentRecommendation.reasoning}</p>
            </details>
        {/if}

        {#if currentRecommendation.suggestedActions.length > 0}
            <div class="suggested-actions">
                <h4>Suggested Actions:</h4>
                {#each currentRecommendation.suggestedActions as action}
                    <button 
                        class="action-button" 
                        onclick={() => executeRecommendedAction(action)}
                    >
                        {action.description}
                    </button>
                {/each}
            </div>
        {/if}
    </div>
{/if}

<!-- Loading and error states -->
{#if recommendationLoading}
    <div class="recommendation-loading">
        <span>🧠 Getting AI recommendations...</span>
    </div>
{/if}

{#if recommendationError}
    <div class="recommendation-error">
        <p>❌ {recommendationError}</p>
        <button onclick={getCudaRecommendations}>Try Again</button>
    </div>
{/if}

<!-- Manual recommendation request button -->
<button 
    class="get-recommendations-btn" 
    onclick={getCudaRecommendations}
    disabled={recommendationLoading}
>
    {recommendationLoading ? 'Getting Recommendations...' : '🎯 Get AI Recommendations'}
</button>

<style>
    .cuda-recommendation-panel {
        position: fixed;
        top: 20px;
        right: 20px;
        max-width: 300px;
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 1000;
    }

    .recommendation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .confidence {
        font-size: 0.8rem;
        opacity: 0.7;
    }

    .recommendation-text {
        margin-bottom: 1rem;
        line-height: 1.4;
    }

    .reasoning {
        margin-bottom: 1rem;
        font-size: 0.9rem;
        opacity: 0.8;
    }

    .suggested-actions {
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        padding-top: 0.5rem;
    }

    .action-button {
        display: block;
        width: 100%;
        margin: 0.25rem 0;
        padding: 0.5rem;
        background: #007acc;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.8rem;
    }

    .action-button:hover {
        background: #005a99;
    }

    .recommendation-loading, .recommendation-error {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem;
        border-radius: 4px;
        z-index: 1001;
    }

    .recommendation-loading {
        background: rgba(0, 123, 255, 0.9);
        color: white;
    }

    .recommendation-error {
        background: rgba(220, 53, 69, 0.9);
        color: white;
    }

    .get-recommendations-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 0.75rem 1rem;
        background: #28a745;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.9rem;
        z-index: 1000;
    }

    .get-recommendations-btn:hover:not(:disabled) {
        background: #218838;
    }

    .get-recommendations-btn:disabled {
        background: #6c757d;
        cursor: not-allowed;
    }
</style>