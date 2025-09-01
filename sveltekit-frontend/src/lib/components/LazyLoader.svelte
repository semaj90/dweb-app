<!-- LazyLoader.svelte - Universal lazy loading wrapper component -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { 
    lazyLoad, 
    createLazyStore, 
    LAZY_LOAD_PRESETS, 
    lazyLoadProfiler,
    type LazyLoadOptions,
    type LazyLoadPreset,
    type LazyComponentState
  } from '$lib/utils/intersection-observer.js';

  // Component props
  let {
    // Lazy loading configuration
    preset = 'NORMAL' as LazyLoadPreset,
    customOptions = {} as LazyLoadOptions,
    
    // Loading states
    showPlaceholder = true,
    placeholderHeight = '200px',
    placeholderClass = '',
    
    // Content
    loadingText = 'Loading...',
    errorText = 'Failed to load content',
    
    // Behavior
    unloadWhenHidden = false,
    enableProfiling = false,
    
    // Callbacks
    onLoad = undefined as (() => void) | undefined,
    onError = undefined as ((error: Error) => void) | undefined,
    
    // Style
    class: className = '',
    style = '',
    
    // Accessibility
    'aria-label': ariaLabel = 'Lazy loaded content',
    
    // Component state binding
    lazyState = $bindable() as LazyComponentState | undefined
  } = $props();

  // Internal state
  let containerElement: HTMLElement;
  let loadError: Error | null = $state(null);
  let isLoading = $state(false);

  // Create lazy loading store
  const lazyStore = createLazyStore();
  
  // Update bound state when internal state changes
  $effect(() => {
    if (lazyState !== undefined) {
      lazyState = $lazyStore;
    }
  });

  // Derived states
  const shouldRender = $derived($lazyStore.hasBeenVisible && !loadError);
  const shouldShowPlaceholder = $derived(showPlaceholder && (!$lazyStore.isVisible || isLoading));

  // Determine options from preset or custom
  const options = $derived.by(() => {
    const baseOptions = LAZY_LOAD_PRESETS[preset] || LAZY_LOAD_PRESETS.NORMAL;
    return { ...baseOptions, ...customOptions };
  });

  // Handle intersection
  function handleIntersection(entry: any) {
    lazyStore.setVisible(entry.isIntersecting, entry.intersectionRatio);
    
    if (entry.isIntersecting && !$lazyStore.hasBeenVisible) {
      if (enableProfiling) {
        lazyLoadProfiler.recordLoad(entry.target);
      }
      
      // Trigger loading callback
      if (onLoad) {
        try {
          onLoad();
        } catch (error) {
          handleError(error instanceof Error ? error : new Error('Loading callback failed'));
        }
      }
    }

    // Handle unloading if configured
    if (unloadWhenHidden && !entry.isIntersecting && $lazyStore.hasBeenVisible) {
      lazyStore.reset();
    }
  }

  function handleError(error: Error) {
    loadError = error;
    if (onError) {
      onError(error);
    }
    console.error('LazyLoader error:', error);
  }

  // Setup intersection observer
  onMount(() => {
    if (containerElement) {
      if (enableProfiling) {
        lazyLoadProfiler.startObserving(containerElement);
      }
    }
  });

  // Cleanup
  onDestroy(() => {
    lazyStore.reset();
  });
</script>

<!-- 
  Container element with intersection observer
  The use:lazyLoad action handles all the intersection logic
-->
<div
  bind:this={containerElement}
  use:lazyLoad={{
    ...options,
    onIntersect: handleIntersection
  }}
  class="lazy-loader-container {className}"
  {style}
  aria-label={ariaLabel}
  role="region"
>
  {#if loadError}
    <!-- Error state -->
    <div class="lazy-loader-error" role="alert">
      <div class="error-icon">⚠️</div>
      <p class="error-message">{errorText}</p>
      <button
        class="retry-button"
        onclick={() => {
          loadError = null;
          lazyStore.reset();
        }}
      >
        Retry
      </button>
    </div>
  
  {:else if shouldShowPlaceholder}
    <!-- Loading placeholder -->
    <div 
      class="lazy-loader-placeholder {placeholderClass}"
      style="min-height: {placeholderHeight}"
      aria-label="Loading content"
    >
      <div class="placeholder-content">
        <div class="loading-spinner" aria-hidden="true"></div>
        <p class="loading-text">{loadingText}</p>
        
        {#if enableProfiling && $lazyStore.intersectionRatio > 0}
          <div class="debug-info">
            Intersection: {Math.round($lazyStore.intersectionRatio * 100)}%
          </div>
        {/if}
      </div>
    </div>
  
  {:else if shouldRender}
    <!-- Actual content - only render when visible -->
    <div class="lazy-loader-content" data-lazy-loaded="true">
      <slot 
        isVisible={$lazyStore.isVisible}
        hasBeenVisible={$lazyStore.hasBeenVisible}
        intersectionRatio={$lazyStore.intersectionRatio}
      />
    </div>
  {/if}
</div>

<style>
  .lazy-loader-container {
    position: relative;
    width: 100%;
  }

  /* Placeholder styles */
  .lazy-loader-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(
      90deg,
      rgba(255, 255, 255, 0.1) 25%,
      rgba(255, 255, 255, 0.2) 50%,
      rgba(255, 255, 255, 0.1) 75%
    );
    background-size: 200% 100%;
    animation: loading-shimmer 2s infinite;
    border-radius: 4px;
    min-height: 200px;
  }

  .placeholder-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: rgba(255, 255, 255, 0.7);
  }

  /* Loading spinner */
  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid rgba(255, 255, 255, 0.2);
    border-top: 3px solid rgba(255, 255, 255, 0.8);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .loading-text {
    margin: 0;
    font-size: 14px;
    text-align: center;
  }

  .debug-info {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.5);
    font-family: monospace;
  }

  /* Error styles */
  .lazy-loader-error {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 24px;
    background: rgba(255, 0, 0, 0.1);
    border: 1px solid rgba(255, 0, 0, 0.3);
    border-radius: 4px;
    color: #ff6b6b;
  }

  .error-icon {
    font-size: 32px;
  }

  .error-message {
    margin: 0;
    text-align: center;
    font-size: 14px;
  }

  .retry-button {
    padding: 8px 16px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 4px;
    color: #ffffff;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.2s ease;
  }

  .retry-button:hover {
    background: rgba(255, 255, 255, 0.2);
  }

  /* Content styles */
  .lazy-loader-content {
    width: 100%;
  }

  /* Animations */
  @keyframes loading-shimmer {
    0% {
      background-position: -200% 0;
    }
    100% {
      background-position: 200% 0;
    }
  }

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }

  /* Responsive adjustments */
  @media (max-width: 768px) {
    .placeholder-content {
      gap: 8px;
    }
    
    .loading-spinner {
      width: 24px;
      height: 24px;
      border-width: 2px;
    }
    
    .loading-text {
      font-size: 12px;
    }
  }

  /* Dark theme optimizations */
  @media (prefers-color-scheme: dark) {
    .lazy-loader-placeholder {
      background: linear-gradient(
        90deg,
        rgba(255, 255, 255, 0.05) 25%,
        rgba(255, 255, 255, 0.1) 50%,
        rgba(255, 255, 255, 0.05) 75%
      );
    }
  }

  /* High contrast mode */
  @media (prefers-contrast: high) {
    .lazy-loader-placeholder {
      border: 2px solid currentColor;
    }
    
    .loading-spinner {
      border-color: currentColor;
      border-top-color: transparent;
    }
  }

  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    .loading-spinner {
      animation: none;
    }
    
    .lazy-loader-placeholder {
      animation: none;
    }
  }
</style>