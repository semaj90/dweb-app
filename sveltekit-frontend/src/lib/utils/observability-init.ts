/**
 * Observability system initialization
 * Automatically integrates with SvelteKit lifecycle and server-side metrics
 */

import { browser } from '$app/environment';
import { page } from '$app/stores';
import { observabilityClient, trackPageLoad } from './observability-client.js';
import { timingMetrics } from './timing-metrics.js';

// Global observability state
let isInitialized = false;
let currentRouteId: string | null = null;

/**
 * Initialize observability system with SvelteKit integration
 */
export function initializeObservability() {
  if (!browser || isInitialized) return;

  console.log('🔍 Initializing observability system...');

  // Configure observability client with development-friendly settings
  observabilityClient.initialize({
    enableMetrics: true,
    enablePerformanceTracking: true,
    enableWebVitals: true,
    metricsEndpoint: '/api/v1/observability/client',
    batchSize: 5, // Smaller batch for development visibility
    flushInterval: 15000, // 15 seconds for development
    debugMode: window.location.hostname === 'localhost' || 
               window.location.search.includes('debug-observability')
  });

  // Track initial page load
  if (typeof document !== 'undefined' && document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      trackInitialPageLoad();
    });
  } else {
    trackInitialPageLoad();
  }

  // Set up page navigation tracking
  setupNavigationTracking();

  // Set up performance monitoring
  setupPerformanceMonitoring();

  // Track visibility changes for metrics flushing
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      observabilityClient.flushMetrics();
    }
  });

  // Track before unload for final metrics flush
  window.addEventListener('beforeunload', () => {
    observabilityClient.flushMetrics();
  });

  isInitialized = true;
  console.log('✅ Observability system initialized');
}

function trackInitialPageLoad() {
  const routeId = extractRouteId();
  currentRouteId = routeId;
  trackPageLoad(routeId);
  
  // Track initial performance metrics
  setTimeout(() => {
    const snapshot = observabilityClient.getPerformanceSnapshot();
    console.log('📊 Initial performance snapshot:', snapshot);
  }, 1000);
}

function setupNavigationTracking() {
  // Listen to page store changes for SvelteKit navigation
  if (typeof page !== 'undefined' && page.subscribe) {
    page.subscribe(($page) => {
      if ($page.route?.id && $page.route.id !== currentRouteId) {
        const newRouteId = $page.route.id;
        currentRouteId = newRouteId;
        trackPageLoad(newRouteId);
      }
    });
  }

  // Fallback for history API navigation
  let lastPath = window.location.pathname;
  
  const trackNavigation = () => {
    const currentPath = window.location.pathname;
    if (currentPath !== lastPath) {
      lastPath = currentPath;
      const routeId = extractRouteId();
      currentRouteId = routeId;
      trackPageLoad(routeId);
    }
  };

  window.addEventListener('popstate', trackNavigation);
  
  // Override pushState and replaceState to track programmatic navigation
  const originalPushState = history.pushState;
  const originalReplaceState = history.replaceState;
  
  history.pushState = function(...args) {
    originalPushState.apply(history, args);
    setTimeout(trackNavigation, 0);
  };
  
  history.replaceState = function(...args) {
    originalReplaceState.apply(history, args);
    setTimeout(trackNavigation, 0);
  };
}

function setupPerformanceMonitoring() {
  // Monitor long tasks
  if ('PerformanceObserver' in window) {
    try {
      const longTaskObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.duration > 50) { // Tasks longer than 50ms
            console.warn(`⚡ Long task detected: ${Math.round(entry.duration)}ms`, {
              name: entry.name,
              startTime: entry.startTime,
              duration: entry.duration
            });
          }
        }
      });
      
      longTaskObserver.observe({ entryTypes: ['longtask'] });
    } catch (e) {
      console.warn('Long task observer not supported:', e);
    }
  }

  // Monitor resource loading
  if (window.performance && window.performance.getEntriesByType) {
    setTimeout(() => {
      const resources = performance.getEntriesByType('resource');
      const slowResources = resources.filter((resource: any) => resource.duration > 1000);
      
      if (slowResources.length > 0) {
        console.warn('🐌 Slow resources detected:', slowResources.map((r: any) => ({
          name: r.name.split('/').pop(),
          duration: `${Math.round(r.duration)}ms`,
          size: r.transferSize ? `${Math.round(r.transferSize / 1024)}KB` : 'unknown'
        })));
      }
    }, 2000);
  }
}

function extractRouteId(): string {
  // Try to extract route ID from current location
  if (browser && typeof page !== 'undefined') {
    try {
      // Access page store synchronously (may not work in all contexts)
      const currentPage = get(page);
      if (currentPage?.route?.id) {
        return currentPage.route.id;
      }
    } catch (e) {
      // Fall back to pathname-based routing
    }
  }

  // Fallback: use pathname as route ID
  const pathname = window.location.pathname;
  
  // Convert common SvelteKit route patterns
  if (pathname === '/') return '/(app)';
  if (pathname.startsWith('/api/')) return pathname;
  if (pathname.includes('[')) return pathname; // Already parameterized
  
  // Convert pathname to route-like format
  return pathname.replace(/\/\d+/g, '/[id]')
                 .replace(/\/[^\/]+\.(json|html|xml)$/, '/[file]');
}

// Svelte store getter fallback
function get(store: any) {
  let value: any;
  const unsubscribe = store.subscribe((v: any) => { value = v; });
  unsubscribe();
  return value;
}

/**
 * Create an enhanced fetch function with observability
 */
export function createObservableFetch() {
  if (!browser) return fetch;
  
  return observabilityClient.createObservableFetch();
}

/**
 * Track a custom performance event
 */
export function trackCustomEvent(name: string, data?: any) {
  if (!browser || !isInitialized) return;
  
  timingMetrics.mark(`custom-${name}`);
  
  if (observabilityClient.getCapabilities()?.debugMode) {
    console.log(`📊 Custom event tracked: ${name}`, data);
  }
}

/**
 * Get current observability status
 */
export function getObservabilityStatus() {
  if (!browser) {
    return { initialized: false, browser: false };
  }
  
  return {
    initialized: isInitialized,
    browser: true,
    currentRoute: currentRouteId,
    capabilities: observabilityClient.getCapabilities(),
    performanceSnapshot: isInitialized ? observabilityClient.getPerformanceSnapshot() : null
  };
}

// Auto-initialize if in browser
if (browser) {
  // Wait for DOM to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeObservability);
  } else {
    initializeObservability();
  }
}