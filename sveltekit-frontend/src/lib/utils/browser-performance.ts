/// <reference types="vite/client" />
/**
 * Chrome Windows 3D Acceleration & Browser Performance Utilities
 * Optimizes performance for Chrome on Windows with hardware acceleration
 */

// Check if browser supports GPU acceleration
export function supportsGPUAcceleration(): boolean {
  if (typeof window === 'undefined') return false;
  
  // Check for WebGL support (indicates GPU availability)
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
  return !!gl;
}

// Check if running on Chrome Windows
export function isChromeWindows(): boolean {
  if (typeof window === 'undefined') return false;
  
  const userAgent = navigator.userAgent;
  const isChrome = /Chrome/.test(userAgent) && !/Edge/.test(userAgent);
  const isWindows = /Windows/.test(userAgent);
  
  return isChrome && isWindows;
}

// Apply GPU acceleration classes to element
export function enableGPUAcceleration(element: HTMLElement): void {
  if (!element || !supportsGPUAcceleration()) return;
  
  element.classList.add('gpu-accelerated');
  
  // Apply Chrome-specific optimizations
  if (isChromeWindows()) {
    element.classList.add('chrome-optimized');
  }
}

// Optimize element for smooth animations
export function optimizeForAnimations(element: HTMLElement): void {
  if (!element) return;
  
  element.classList.add('animate-gpu');
  element.style.willChange = 'transform, opacity';
}

// Performance monitoring for Chrome Windows
export class BrowserPerformanceMonitor {
  private performanceObserver?: PerformanceObserver;
  private metrics: Map<string, number> = new Map();

  constructor() {
    this.initializePerformanceMonitoring();
  }

  private initializePerformanceMonitoring(): void {
    if (typeof window === 'undefined' || !window.PerformanceObserver) return;

    this.performanceObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        // Track paint metrics for Chrome Windows
        if (entry.entryType === 'paint') {
          this.metrics.set(entry.name, entry.startTime);
        }
        
        // Track layout shifts (important for 3D acceleration)
        if (entry.entryType === 'layout-shift') {
          const value = (entry as any).value;
          if (value) {
            this.metrics.set('cumulative-layout-shift', 
              (this.metrics.get('cumulative-layout-shift') || 0) + value
            );
          }
        }
      }
    });

    // Observe paint and layout metrics
    try {
      this.performanceObserver.observe({ 
        entryTypes: ['paint', 'layout-shift'] 
      });
    } catch (error: any) {
      console.warn('Performance monitoring not available:', error);
    }
  }

  getMetrics(): Record<string, number> {
    return Object.fromEntries(this.metrics);
  }

  // Check if hardware acceleration is working effectively
  isHardwareAccelerated(): boolean {
    const firstPaint = this.metrics.get('first-paint');
    const firstContentfulPaint = this.metrics.get('first-contentful-paint');
    
    // Good hardware acceleration should have fast paint times
    return (firstPaint && firstPaint < 100) || 
           (firstContentfulPaint && firstContentfulPaint < 300) ||
           false;
  }

  dispose(): void {
    this.performanceObserver?.disconnect();
    this.metrics.clear();
  }
}

// Browser error handling and reporting
export class BrowserErrorHandler {
  private errors: Array<{
    message: string;
    source: string;
    line: number;
    column: number;
    timestamp: number;
  }> = [];

  constructor() {
    this.initializeErrorHandling();
  }

  private initializeErrorHandling(): void {
    if (typeof window === 'undefined') return;

    // Global error handler
    window.addEventListener('error', (event: any) => {
      this.logError({
        message: event.message,
        source: event.filename || 'unknown',
        line: event.lineno || 0,
        column: event.colno || 0,
        timestamp: Date.now()
      });
    });

    // Promise rejection handler
    window.addEventListener('unhandledrejection', (event: any) => {
      this.logError({
        message: `Unhandled Promise Rejection: ${event.reason}`,
        source: 'promise',
        line: 0,
        column: 0,
        timestamp: Date.now()
      });
    });
  }

  private logError(error: typeof this.errors[0]): void {
    this.errors.push(error);
    
    // Keep only last 10 errors
    if (this.errors.length > 10) {
      this.errors.shift();
    }

    // Log to console in development
    if (import.meta.env.DEV) {
      console.error('Browser Error:', error);
    }
  }

  getErrors(): typeof this.errors {
    return [...this.errors];
  }

  clearErrors(): void {
    this.errors = [];
  }

  // Report Chrome Windows specific issues
  getCompatibilityReport(): {
    browser: string;
    gpuSupport: boolean;
    hardwareAcceleration: boolean;
    errors: number;
    recommendations: string[];
  } {
    const recommendations: string[] = [];
    
    if (!supportsGPUAcceleration()) {
      recommendations.push('Enable hardware acceleration in Chrome settings');
    }
    
    if (this.errors.length > 5) {
      recommendations.push('Multiple JavaScript errors detected - check console');
    }
    
    if (!isChromeWindows()) {
      recommendations.push('For best performance, use Chrome on Windows');
    }

    return {
      browser: navigator.userAgent,
      gpuSupport: supportsGPUAcceleration(),
      hardwareAcceleration: isChromeWindows(),
      errors: this.errors.length,
      recommendations
    };
  }
}

// Singleton instances
export const performanceMonitor = new BrowserPerformanceMonitor();
export const errorHandler = new BrowserErrorHandler();
;
// Utility functions for components
export function optimizeComponent(element: HTMLElement): void {
  enableGPUAcceleration(element);
  
  // Add smooth scrolling if element is scrollable
  const hasOverflow = getComputedStyle(element).overflow !== 'visible';
  if (hasOverflow) {
    element.classList.add('gpu-smooth-scroll');
  }
}

// Detect and handle Chrome Windows specific features
export function initializeChromeWindowsOptimizations(): void {
  if (typeof window === 'undefined') return;

  // Enable smooth scrolling globally
  document.documentElement.style.scrollBehavior = 'smooth';
  
  // Apply Chrome-specific font rendering
  if (isChromeWindows()) {
    document.body.style.fontFeatureSettings = '"liga"';
    document.body.style.fontKerning = 'auto';
  }

  // Log performance and compatibility info
  setTimeout(() => {
    const report = errorHandler.getCompatibilityReport();
    console.log('🎯 Chrome Windows Optimization Report:', report);
    
    if (report.recommendations.length > 0) {
      console.log('💡 Recommendations:', report.recommendations);
    }
  }, 1000);
}

// Initialize on import
if (typeof window !== 'undefined') {
  initializeChromeWindowsOptimizations();
}