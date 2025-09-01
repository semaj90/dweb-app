<!--
  Global Feedback Integration Component
  Provides feedback hooks for any component or interaction
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { getFeedbackStore } from '$lib/stores/feedback-store.svelte';
  import { aiRecommendationEngine } from '$lib/services/ai-recommendation-engine';

  // Props
  let { 
    interactionType = $bindable(),
    context = $bindable(),
    autoTrigger = $bindable(),
    delay = $bindable(),
    priority = $bindable(),
    trackOnMount = $bindable(),
    trackOnVisible = $bindable(),
    ratingType = $bindable()
  } = $props();

  // Get feedback store
  const store = getFeedbackStore();
  
  let mounted = $state(false);
  let visible = $state(false);
  let interactionId: string | null = $state(null);
let element = $state<HTMLElement;

  onMount(() >(> {
    mounted = true);
    
    if (trackOnMount) {
      triggerFeedback();
    }

    if (trackOnVisible) {
      setupVisibilityTracking();
    }

    return () => {
      // Cleanup
    };
  });

  /**
   * Trigger feedback collection for this interaction
   */
  export function triggerFeedback(customContext: Record<string, any> = {}) {
    const finalContext = { ...context, ...customContext };
    
    interactionId = store.trackInteraction(interactionType, finalContext, {
      autoTrigger,
      priority,
      delay
    });

    // Generate enhanced recommendations if AI interaction
    if (interactionType.includes('ai_') || interactionType.includes('search_')) {
      generateRecommendations(finalContext);
    }

    return interactionId;
  }

  /**
   * Update interaction context
   */
  export function updateContext(newContext: Record<string, any>) {
    context = { ...context, ...newContext };
  }

  /**
   * Mark interaction as completed successfully
   */
  export function markCompleted(result: any = {}) {
    if (interactionId) {
      updateContext({ 
        completed: true, 
        result,
        completedAt: new Date().toISOString()
      });
    }
  }

  /**
   * Mark interaction as failed
   */
  export function markFailed(error: any = {}) {
    if (interactionId) {
      updateContext({ 
        failed: true, 
        error,
        failedAt: new Date().toISOString()
      });
      
      // Higher priority for failed interactions
      triggerFeedback({ priority: 'high', error });
    }
  }

  /**
   * Setup visibility tracking using Intersection Observer
   */
  function setupVisibilityTracking() {
    if (!element || typeof IntersectionObserver === 'undefined') return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !visible) {
            visible = true;
            triggerFeedback({ viewedAt: new Date().toISOString() });
          }
        });
      },
      { threshold: 0.5, rootMargin: '0px 0px -10% 0px' }
    );

    observer.observe(element);

    return () => observer.disconnect();
  }

  /**
   * Generate AI recommendations based on interaction
   */
  async function generateRecommendations(interactionContext: Record<string, any>) {
    try {
      const userContext = store.userContext;
      const legalDomain = interactionContext.legalDomain || 'general';
      const query = interactionContext.query || interactionType;

      await aiRecommendationEngine.generateEnhancedRecommendations(
        userContext,
        query,
        legalDomain
      );
    } catch (error) {
      console.error('❌ Failed to generate recommendations:', error);
    }
  }

  /**
   * Quick feedback methods for common scenarios
   */
  export const feedback = {
    // AI Response feedback
    aiResponse: (query: string, response: string, confidence: number = 0) => {
      triggerFeedback({
        query,
        response: response.substring(0, 200) + '...',
        confidence,
        aiModel: 'gemma3-legal'
      });
    },

    // Search results feedback
    searchResults: (query: string, resultCount: number, relevance: number = 0) => {
      triggerFeedback({
        query,
        resultCount,
        relevance,
        searchType: 'legal'
      });
    },

    // Document processing feedback
    documentProcessed: (filename: string, processingTime: number, success: boolean) => {
      triggerFeedback({
        filename,
        processingTime,
        success,
        documentType: 'legal'
      });
    },

    // Feature usage feedback
    featureUsed: (featureName: string, usageContext: Record<string, any> = {}) => {
      triggerFeedback({
        featureName,
        ...usageContext,
        featureCategory: 'legal_ai'
      });
    },

    // Error feedback
    error: (errorType: string, errorMessage: string, stack?: string) => {
      markFailed({
        errorType,
        errorMessage,
        stack: stack?.substring(0, 500)
      });
    }
  };
</script>

<!-- Invisible tracking element -->
<div bind:this={element} class="feedback-tracker" data-interaction={interactionType}>
  <slot {triggerFeedback} {updateContext} {markCompleted} {markFailed} {feedback} {interactionId} />
</div>

<style>
  .feedback-tracker {
    display: contents;
  }
</style>