<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  import { performanceMonitor, errorHandler, optimizeComponent } from '$lib/utils/browser-performance';
  import type { StartupStatus } from '$lib/services/multi-library-startup';
  import { feedbackStore, createFeedbackStore, setFeedbackStore } from '$lib/stores/feedback-store.svelte';
  import { aiRecommendationEngine } from '$lib/services/ai-recommendation-engine';
  import FeedbackWidget from '$lib/components/feedback/FeedbackWidget.svelte';
  import type { FeedbackTrigger } from '$lib/types/feedback';

  let startupStatus: StartupStatus | null = null;
  let showStartupLog = false;
  let currentFeedbackTrigger: FeedbackTrigger | null = null;
  let showFeedback = false;

  // Create and set feedback store context
  const store = createFeedbackStore();
  setFeedbackStore(store);

  onMount(async () => {
    if (!browser) return;

    console.log('🚀 Initializing YoRHa Legal AI Platform...');

    try {
      // Initialize multi-library integration on app startup (browser-only)
      const { multiLibraryStartup } = await import('$lib/services/multi-library-startup');
      startupStatus = await multiLibraryStartup.initialize();

      // Initialize feedback system
      const userId = 'user_' + Date.now(); // In production, get from auth
      const session = store.initializeSession(userId);

      // Track platform initialization
      store.trackInteraction('platform_initialization', {
        services: startupStatus?.services || {},
        initTime: startupStatus?.initTime || 0
      });

      if (startupStatus?.initialized) {
        console.log('✅ YoRHa Legal AI Platform Ready');

        // Show brief startup notification
        showStartupLog = true;
        setTimeout(() => {
          showStartupLog = false;
        }, 4000);

        // Generate initial recommendations
        await aiRecommendationEngine.generateEnhancedRecommendations(
          {
            userId: session.userId,
            sessionId: session.id,
            deviceType: store.userContext.deviceType,
            userType: 'attorney' // In production, get from user profile
          },
          'platform startup',
          'general'
        );

        // Log Chrome Windows optimization status
        const compatibilityReport = errorHandler.getCompatibilityReport();
        console.log('🎯 Browser Performance Report:', compatibilityReport);
      }
    } catch (error) {
      console.error('❌ Platform initialization failed:', error);
      store.trackInteraction('platform_error', { error: (error as Error)?.message ?? String(error) });
    }

    // Listen for feedback triggers
    const feedbackInterval = setInterval(() => {
      if (!store.isCollecting && !showFeedback) {
        const trigger = store.showNextFeedback();
        if (trigger) {
          currentFeedbackTrigger = trigger;
          showFeedback = true;
        }
      }
    }, 1000);

    return () => {
      clearInterval(feedbackInterval);
      store.clearSession();
    };
  });

  // Feedback handlers (use Svelte event handlers with e.detail)
  async function handleFeedbackSubmitted(event: CustomEvent) {
    const data: any = event.detail;
    const success = await store.submitFeedback(
      data.interactionId,
      data.rating,
      data.feedback,
      currentFeedbackTrigger?.type || 'response_quality'
    );

    if (success) {
      console.log('✅ Feedback submitted successfully');
      // Generate updated recommendations based on feedback
      await aiRecommendationEngine.generateEnhancedRecommendations(
        store.userContext,
        'feedback provided',
        'user_experience'
      );
    }

    showFeedback = false;
    currentFeedbackTrigger = null;
  }

  function handleFeedbackError(event: CustomEvent) {
    console.error('❌ Feedback submission failed:', event.detail ?? event);
    showFeedback = false;
    currentFeedbackTrigger = null;
  }

  function handleFeedbackClosed() {
    showFeedback = false;
    currentFeedbackTrigger = null;
    store.cancelFeedback();
  }
</script>

<!-- Multi-Library Startup Notification -->
{#if showStartupLog && startupStatus}
  <div class="startup-notification">
    <div class="startup-content">
      <h3>🚀 YoRHa Legal AI Platform</h3>
      <p>Multi-Library Integration Complete</p>
      <div class="startup-services">
        {#each Object.entries(startupStatus.services) as [service, status]}
          <span class="service-status" class:ready={status} class:failed={!status}>
            {status ? '✅' : '❌'} {service.toUpperCase()}
          </span>
        {/each}
      </div>
      <p class="startup-time">Initialized in {startupStatus.initTime}ms</p>
    </div>
  </div>
{/if}

<div class="yorha-3d-panel nes-legal-container gpu-accelerated transform-3d">
  <header class="nes-legal-header">
    <h1 class="nes-legal-title neural-sprite-active">YoRHa Legal AI</h1>
    <nav class="nes-nav-main">
      <a href="/" class="nes-legal-priority-medium yorha-3d-button">Home</a>
      <a href="/yorha-command-center" class="nes-legal-priority-high yorha-3d-button">YoRHa Command Center</a>
      <a href="/demo/enhanced-rag-semantic" class="nes-legal-priority-medium yorha-3d-button">Enhanced RAG Demo</a>
      <a href="/endpoints" class="nes-legal-priority-low yorha-3d-button">Endpoints</a>
      {#if startupStatus?.initialized}
        <span class="nes-legal-priority-critical neural-sprite-active">🟢 INTEGRATED</span>
      {:else}
        <span class="nes-legal-priority-low neural-sprite-loading">🟡 LOADING</span>
      {/if}
    </nav>
  </header>
  <main class="nes-main-content">
    <slot />
  </main>
</div>

{#if currentFeedbackTrigger}
  <FeedbackWidget
    interactionId={currentFeedbackTrigger.interactionId}
    sessionId={store.userContext.sessionId}
    userId={store.userContext.userId}
    context={currentFeedbackTrigger.context}
    show={showFeedback}
    ratingType={currentFeedbackTrigger.type}
    on:submitted={handleFeedbackSubmitted}
    on:error={handleFeedbackError}
    on:closed={handleFeedbackClosed}
  />
{/if}

<style>
  /* Startup Notification Styles */
  .startup-notification {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
    border: 2px solid #ffd700;
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
    animation: slideIn 0.5s ease-out;
    max-width: 400px;
  }

  .startup-content h3 {
    margin: 0 0 0.5rem 0;
    color: #ffd700;
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .startup-content p {
    margin: 0 0 1rem 0;
    color: #e0e0e0;
    font-size: 0.9rem;
  }

  .startup-services {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
    margin: 1rem 0;
  }

  .service-status {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    border: 1px solid;
    font-family: 'JetBrains Mono', monospace;
  }

  .service-status.ready {
    color: #00ff41;
    border-color: #00ff41;
    background: rgba(0, 255, 65, 0.1);
  }

  .service-status.failed {
    color: #ff0041;
    border-color: #ff0041;
    background: rgba(255, 0, 65, 0.1);
  }

  .startup-time {
    font-size: 0.8rem !important;
    color: #b0b0b0 !important;
    text-align: right;
    margin: 0.5rem 0 0 0 !important;
  }

  @keyframes slideIn {
    from {
      transform: translateX(100%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }

  /* These styles are handled by global CSS classes that are actually used in the template */
</style>