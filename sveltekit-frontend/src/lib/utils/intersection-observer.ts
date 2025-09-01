// Intersection Observer Utility for Lazy Loading Heavy Components
// Provides performance optimization by deferring component rendering until needed

export interface LazyLoadOptions {
  root?: Element | null;
  rootMargin?: string;
  threshold?: number | number[];
  once?: boolean; // Only trigger once and then disconnect
  fallbackDelay?: number; // Fallback timeout for older browsers
}

export interface LazyLoadEntry {
  element: Element;
  isIntersecting: boolean;
  intersectionRatio: number;
  target: Element;
}

export type LazyLoadCallback = (entry: LazyLoadEntry) => void;

class LazyLoadManager {
  private observer: IntersectionObserver | null = null;
  private callbacks = new Map<Element, LazyLoadCallback>();
  private options: LazyLoadOptions;
  private fallbackTimeouts = new Map<Element, number>();

  constructor(options: LazyLoadOptions = {}) {
    this.options = {
      root: null,
      rootMargin: '50px',
      threshold: 0.1,
      once: true,
      fallbackDelay: 2000,
      ...options
    };

    this.createObserver();
  }

  private createObserver() {
    if (!this.supportsIntersectionObserver()) {
      console.warn('IntersectionObserver not supported, using fallback');
      return;
    }

    this.observer = new IntersectionObserver(
      (entries) => this.handleIntersection(entries),
      {
        root: this.options.root,
        rootMargin: this.options.rootMargin,
        threshold: this.options.threshold
      }
    );
  }

  private supportsIntersectionObserver(): boolean {
    return typeof window !== 'undefined' && 'IntersectionObserver' in window;
  }

  private handleIntersection(entries: IntersectionObserverEntry[]) {
    entries.forEach((entry) => {
      const callback = this.callbacks.get(entry.target);
      if (callback) {
        const lazyEntry: LazyLoadEntry = {
          element: entry.target,
          isIntersecting: entry.isIntersecting,
          intersectionRatio: entry.intersectionRatio,
          target: entry.target
        };

        callback(lazyEntry);

        // If it's a one-time observation and element is intersecting, stop observing
        if (this.options.once && entry.isIntersecting) {
          this.unobserve(entry.target);
        }
      }
    });
  }

  observe(element: Element, callback: LazyLoadCallback): void {
    if (!element) {
      console.warn('LazyLoadManager: Cannot observe null element');
      return;
    }

    this.callbacks.set(element, callback);

    if (this.observer) {
      this.observer.observe(element);
    } else {
      // Fallback for browsers without IntersectionObserver
      this.setupFallback(element, callback);
    }
  }

  private setupFallback(element: Element, callback: LazyLoadCallback): void {
    const timeoutId = window.setTimeout(() => {
      // For fallback, assume element is always intersecting after delay
      callback({
        element,
        isIntersecting: true,
        intersectionRatio: 1,
        target: element
      });
    }, this.options.fallbackDelay);

    this.fallbackTimeouts.set(element, timeoutId);
  }

  unobserve(element: Element): void {
    if (this.observer) {
      this.observer.unobserve(element);
    }

    // Clear fallback timeout if exists
    const timeoutId = this.fallbackTimeouts.get(element);
    if (timeoutId) {
      clearTimeout(timeoutId);
      this.fallbackTimeouts.delete(element);
    }

    this.callbacks.delete(element);
  }

  disconnect(): void {
    if (this.observer) {
      this.observer.disconnect();
    }

    // Clear all fallback timeouts
    this.fallbackTimeouts.forEach((timeoutId) => clearTimeout(timeoutId));
    this.fallbackTimeouts.clear();
    this.callbacks.clear();
  }
}

// Global instance for the app
let globalLazyLoader: LazyLoadManager | null = null;

export function getLazyLoader(options?: LazyLoadOptions): LazyLoadManager {
  if (!globalLazyLoader) {
    globalLazyLoader = new LazyLoadManager(options);
  }
  return globalLazyLoader;
}

// Svelte action for easy component integration
export function lazyLoad(
  element: Element,
  options: LazyLoadOptions & { 
    onIntersect: LazyLoadCallback;
  }
) {
  const { onIntersect, ...loaderOptions } = options;
  const loader = getLazyLoader(loaderOptions);

  loader.observe(element, onIntersect);

  return {
    destroy() {
      loader.unobserve(element);
    }
  };
}

// Svelte store for reactive lazy loading state
import { writable } from 'svelte/store';

export interface LazyComponentState {
  isVisible: boolean;
  hasBeenVisible: boolean;
  intersectionRatio: number;
}

export function createLazyStore(initialState: Partial<LazyComponentState> = {}) {
  const { subscribe, set, update } = writable<LazyComponentState>({
    isVisible: false,
    hasBeenVisible: false,
    intersectionRatio: 0,
    ...initialState
  });

  return {
    subscribe,
    setVisible: (isVisible: boolean, intersectionRatio: number = 1) => {
      update(state => ({
        ...state,
        isVisible,
        hasBeenVisible: state.hasBeenVisible || isVisible,
        intersectionRatio
      }));
    },
    reset: () => {
      set({
        isVisible: false,
        hasBeenVisible: false,
        intersectionRatio: 0
      });
    }
  };
}

// Utility functions for common lazy loading patterns

export function createComponentLazyLoader(
  element: Element,
  options: LazyLoadOptions = {}
): Promise<LazyLoadEntry> {
  return new Promise((resolve) => {
    const loader = getLazyLoader({ once: true, ...options });
    
    loader.observe(element, (entry) => {
      if (entry.isIntersecting) {
        resolve(entry);
      }
    });
  });
}

export function lazyLoadImage(img: HTMLImageElement, src: string, options: LazyLoadOptions = {}): void {
  const loader = getLazyLoader({ once: true, ...options });
  
  loader.observe(img, (entry) => {
    if (entry.isIntersecting) {
      img.src = src;
      img.classList.add('lazy-loaded');
    }
  });
}

// Performance monitoring helpers
export interface LazyLoadMetrics {
  totalObserved: number;
  totalLoaded: number;
  averageLoadTime: number;
  loadTimes: number[];
}

class LazyLoadProfiler {
  private metrics: LazyLoadMetrics = {
    totalObserved: 0,
    totalLoaded: 0,
    averageLoadTime: 0,
    loadTimes: []
  };

  private loadStartTimes = new Map<Element, number>();

  startObserving(element: Element): void {
    this.metrics.totalObserved++;
    this.loadStartTimes.set(element, Date.now());
  }

  recordLoad(element: Element): void {
    const startTime = this.loadStartTimes.get(element);
    if (startTime) {
      const loadTime = Date.now() - startTime;
      this.metrics.loadTimes.push(loadTime);
      this.metrics.totalLoaded++;
      
      // Update average
      this.metrics.averageLoadTime = 
        this.metrics.loadTimes.reduce((sum, time) => sum + time, 0) / this.metrics.loadTimes.length;

      this.loadStartTimes.delete(element);
    }
  }

  getMetrics(): LazyLoadMetrics {
    return { ...this.metrics };
  }

  reset(): void {
    this.metrics = {
      totalObserved: 0,
      totalLoaded: 0,
      averageLoadTime: 0,
      loadTimes: []
    };
    this.loadStartTimes.clear();
  }
}

export const lazyLoadProfiler = new LazyLoadProfiler();

// Presets for common scenarios

export const LAZY_LOAD_PRESETS = {
  // Load just before entering viewport
  EAGER: {
    rootMargin: '100px',
    threshold: 0,
    once: true
  },
  
  // Load when partially visible
  NORMAL: {
    rootMargin: '50px',
    threshold: 0.1,
    once: true
  },
  
  // Load only when mostly visible
  LAZY: {
    rootMargin: '0px',
    threshold: 0.5,
    once: true
  },
  
  // For heavy components that should load early
  HEAVY_COMPONENT: {
    rootMargin: '200px',
    threshold: 0,
    once: true,
    fallbackDelay: 1000
  },
  
  // For images and media
  MEDIA: {
    rootMargin: '50px',
    threshold: 0,
    once: true
  },
  
  // For continuous monitoring (animations, etc.)
  CONTINUOUS: {
    rootMargin: '0px',
    threshold: [0, 0.25, 0.5, 0.75, 1],
    once: false
  }
} as const;

export type LazyLoadPreset = keyof typeof LAZY_LOAD_PRESETS;