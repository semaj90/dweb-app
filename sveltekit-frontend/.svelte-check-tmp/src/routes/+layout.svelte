<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  import { errorHandler } from '$lib/utils/browser-performance';
  import type { StartupStatus } from '$lib/services/multi-library-startup';
  import { createFeedbackStore, setFeedbackStore } from '$lib/stores/feedback-store.svelte';
  import { aiRecommendationEngine } from '$lib/services/ai-recommendation-engine';
  import FeedbackWidget from '$lib/components/feedback/FeedbackWidget.svelte';
  import type { FeedbackTrigger } from '$lib/types/feedback';

  // Modern button component
  import ModernButton from '$lib/components/ui/button/Button.svelte';

  let startupStatus: StartupStatus | null = $state(null);
  let showStartupLog = $state(false);
  let currentFeedbackTrigger: FeedbackTrigger | null = $state(null);
  let showFeedback = $state(false);

  // Create store only in browser (inside onMount) to avoid hydration issues
  let store: ReturnType<typeof createFeedbackStore> | null = $state(null);

  onMount(async () => {
    if (!browser) return;

    // create and set feedback store in browser
    store = createFeedbackStore();
    setFeedbackStore(store);

    console.log('🚀 Initializing YoRHa Legal AI Platform...');

    try {
      // Initialize multi-library integration on app startup (browser-only)
      const { multiLibraryStartup } = await import('$lib/services/multi-library-startup');
      startupStatus = await multiLibraryStartup.initialize();

      // Initialize feedback system (ensure store exists)
      let session = null;
      if (store) {
        const userId = 'user_' + Date.now(); // In production, get from auth
        session = store.initializeSession(userId);

        // Track platform initialization
        store.trackInteraction('platform_initialization', {
          services: startupStatus?.services || {},
          initTime: startupStatus?.initTime || 0
        });
      }

      if (startupStatus?.initialized) {
        console.log('✅ YoRHa Legal AI Platform Ready');

        // Show brief startup notification
        showStartupLog = true;
        setTimeout(() => {
          showStartupLog = false;
        }, 4000);

        // Generate initial recommendations
        if (session) {
          await aiRecommendationEngine.generateEnhancedRecommendations(
            {
              userId: session.userId,
              sessionId: session.id,
              deviceType: store?.userContext?.deviceType || 'desktop',
              userType: 'attorney' // In production, get from user profile
            },
            'platform startup',
            'general'
          );
        }

        // Log Chrome Windows optimization status
        const compatibilityReport = errorHandler.getCompatibilityReport();
        console.log('🎯 Browser Performance Report:', compatibilityReport);
      }
    } catch (error) {
      console.error('❌ Platform initialization failed:', error);
      store?.trackInteraction('platform_error', { error: (error as Error)?.message ?? String(error) });
    }

    return () => {
      store?.clearSession();
    };
  });

  // Feedback handlers (use Svelte event handlers with e.detail)
  async function handleFeedbackSubmitted(event: CustomEvent) {
    const data: any = event.detail;
    const success = await store?.submitFeedback(
      data.interactionId,
      data.rating,
      data.feedback,
      currentFeedbackTrigger?.type || 'response_quality'
    );

    if (success) {
      console.log('✅ Feedback submitted successfully');
      // Generate updated recommendations based on feedback
      await aiRecommendationEngine.generateEnhancedRecommendations(
        store?.userContext || { userId: '', sessionId: '', deviceType: 'desktop' },
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
    store?.cancelFeedback();
  }
</script>

<!-- Modern Startup Toast Notification -->
{#if showStartupLog && startupStatus}
  <div class="fixed top-5 right-5 z-50 max-w-sm">
    <div class="bg-nier-bg-secondary border border-nier-border-primary rounded-lg p-golden-lg shadow-lg">
      <h3 class="text-nier-accent-warm font-bold text-lg uppercase tracking-wide mb-golden-sm">
        🚀 YoRHa Legal AI Platform
      </h3>
      <p class="text-nier-text-secondary mb-golden-md text-sm">
        Multi-Library Integration Complete
      </p>
      <div class="grid grid-cols-2 gap-golden-xs mb-golden-md">
        {#each Object.entries(startupStatus.services) as [service, status]}
          <span
            class="text-xs font-mono px-golden-xs py-1 border rounded {status ? 'border-green-500 bg-green-500/10 text-green-400' : 'border-red-500 bg-red-500/10 text-red-400'}"
          >
            {status ? '✅' : '❌'} {service.toUpperCase()}
          </span>
        {/each}
      </div>
      <p class="text-nier-text-muted text-xs text-right font-mono">
        Initialized in {startupStatus.initTime}ms
      </p>
    </div>
  </div>
{/if}

<div class="min-h-screen bg-nier-bg-primary text-nier-text-primary font-mono">
  <!-- Modern Header with Golden Ratio Grid -->
  <header class="border-b border-nier-border-muted bg-nier-bg-secondary/50 backdrop-blur-sm sticky top-0 z-40">
    <div class="container mx-auto px-golden-lg py-golden-md">
      <div class="flex items-center justify-between gap-golden-lg">
        <!-- Logo Section -->
        <div class="flex items-center gap-golden-sm">
          <h1 class="text-nier-accent-warm font-bold text-2xl tracking-wider uppercase">
            YoRHa Legal AI
          </h1>
          {#if startupStatus?.initialized}
            <span class="bg-green-500/20 text-green-400 border-green-500/30 border text-xs px-2 py-1 rounded">
              🟢 INTEGRATED
            </span>
          {:else}
            <span class="bg-yellow-500/20 text-yellow-400 border-yellow-500/30 border text-xs px-2 py-1 animate-pulse rounded">
              🟡 LOADING
            </span>
          {/if}
        </div>

        <!-- Navigation -->
        <nav class="hidden md:flex items-center gap-golden-sm">
          <ModernButton
            href="/"
            variant="ghost"
            size="sm"
            class="text-nier-text-secondary hover:text-nier-accent-warm hover:bg-nier-bg-tertiary"
          >
            Home
          </ModernButton>
          <ModernButton
            href="/yorha-command-center"
            variant="ghost"
            size="sm"
            class="text-nier-text-secondary hover:text-nier-accent-warm hover:bg-nier-bg-tertiary"
          >
            Command Center
          </ModernButton>
          <ModernButton
            href="/evidenceboard"
            variant="ghost"
            size="sm"
            class="text-nier-text-secondary hover:text-nier-accent-warm hover:bg-nier-bg-tertiary"
          >
            Evidence Board
          </ModernButton>
          <ModernButton
            href="/demo/enhanced-rag-semantic"
            variant="ghost"
            size="sm"
            class="text-nier-text-secondary hover:text-nier-accent-warm hover:bg-nier-bg-tertiary"
          >
            RAG Demo
          </ModernButton>
        </nav>

        <!-- Auth Buttons -->
        <div class="flex items-center gap-golden-sm">
          <ModernButton
            href="/auth/login"
            variant="outline"
            size="sm"
            class="border-nier-accent-warm text-nier-accent-warm hover:bg-nier-accent-warm hover:text-nier-bg-primary"
          >
            Login
          </ModernButton>
          <ModernButton
            href="/auth/register"
            variant="primary"
            size="sm"
            class="bg-gradient-to-r from-nier-accent-warm to-nier-accent-cool text-nier-bg-primary font-bold"
          >
            Register
          </ModernButton>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Content with Golden Ratio Spacing -->
  <main class="container mx-auto px-golden-lg py-golden-xl min-h-[calc(100vh-theme(spacing.16))]">
    <slot />
  </main>
</div>

{#if currentFeedbackTrigger}
  <FeedbackWidget
    interactionId={currentFeedbackTrigger.interactionId}
    sessionId={store?.userContext?.sessionId || ''}
    userId={store?.userContext?.userId || ''}
    context={currentFeedbackTrigger.context}
    show={showFeedback}
    ratingType={currentFeedbackTrigger.type}
    submitted={handleFeedbackSubmitted}
    error={handleFeedbackError}
    closed={handleFeedbackClosed}
  />
{/if}

<style>
  /* Golden Ratio Custom CSS Properties */
  :global(:root) {
    --golden-base: 1rem;
    --golden-xs: calc(var(--golden-base) / 2.618); /* ~0.382rem */
    --golden-sm: calc(var(--golden-base) / 1.618); /* ~0.618rem */
    --golden-md: var(--golden-base); /* 1rem */
    --golden-lg: calc(var(--golden-base) * 1.618); /* ~1.618rem */
    --golden-xl: calc(var(--golden-base) * 2.618); /* ~2.618rem */
    --golden-2xl: calc(var(--golden-base) * 4.236); /* ~4.236rem */
  }

  /* UnoCSS Golden Ratio Utilities */
  :global(.p-golden-xs) { padding: var(--golden-xs); }
  :global(.p-golden-sm) { padding: var(--golden-sm); }
  :global(.p-golden-md) { padding: var(--golden-md); }
  :global(.p-golden-lg) { padding: var(--golden-lg); }
  :global(.p-golden-xl) { padding: var(--golden-xl); }
  :global(.p-golden-2xl) { padding: var(--golden-2xl); }

  :global(.px-golden-xs) { padding-left: var(--golden-xs); padding-right: var(--golden-xs); }
  :global(.px-golden-sm) { padding-left: var(--golden-sm); padding-right: var(--golden-sm); }
  :global(.px-golden-md) { padding-left: var(--golden-md); padding-right: var(--golden-md); }
  :global(.px-golden-lg) { padding-left: var(--golden-lg); padding-right: var(--golden-lg); }
  :global(.px-golden-xl) { padding-left: var(--golden-xl); padding-right: var(--golden-xl); }

  :global(.py-golden-xs) { padding-top: var(--golden-xs); padding-bottom: var(--golden-xs); }
  :global(.py-golden-sm) { padding-top: var(--golden-sm); padding-bottom: var(--golden-sm); }
  :global(.py-golden-md) { padding-top: var(--golden-md); padding-bottom: var(--golden-md); }
  :global(.py-golden-lg) { padding-top: var(--golden-lg); padding-bottom: var(--golden-lg); }
  :global(.py-golden-xl) { padding-top: var(--golden-xl); padding-bottom: var(--golden-xl); }

  :global(.m-golden-xs) { margin: var(--golden-xs); }
  :global(.m-golden-sm) { margin: var(--golden-sm); }
  :global(.m-golden-md) { margin: var(--golden-md); }
  :global(.m-golden-lg) { margin: var(--golden-lg); }
  :global(.m-golden-xl) { margin: var(--golden-xl); }
  :global(.m-golden-2xl) { margin: var(--golden-2xl); }

  :global(.mb-golden-xs) { margin-bottom: var(--golden-xs); }
  :global(.mb-golden-sm) { margin-bottom: var(--golden-sm); }
  :global(.mb-golden-md) { margin-bottom: var(--golden-md); }
  :global(.mb-golden-lg) { margin-bottom: var(--golden-lg); }
  :global(.mb-golden-xl) { margin-bottom: var(--golden-xl); }

  :global(.gap-golden-xs) { gap: var(--golden-xs); }
  :global(.gap-golden-sm) { gap: var(--golden-sm); }
  :global(.gap-golden-md) { gap: var(--golden-md); }
  :global(.gap-golden-lg) { gap: var(--golden-lg); }
  :global(.gap-golden-xl) { gap: var(--golden-xl); }

  /* Container responsive spacing with golden ratio */
  :global(.container) {
    max-width: 90rem;
    margin: 0 auto;
  }

  /* Smooth transitions for YoRHa theme */
  :global(*) {
    transition: color 0.2s ease, background-color 0.2s ease, border-color 0.2s ease;
  }

  /* Typography enhancements */
  :global(.font-mono) {
    font-family: 'JetBrains Mono', 'Roboto Mono', 'SF Mono', monospace;
  }
</style>