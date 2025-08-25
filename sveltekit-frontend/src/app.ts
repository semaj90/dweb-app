// Main application initialization
// Import polyfills first to ensure browser compatibility
import './lib/polyfills';
import './app.css';

// Initialize global error handling for better debugging
if (typeof window !== 'undefined') {
  // Enhanced error handling for development
  window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    
    // Log component-related errors with more context
    if (event.error?.stack?.includes('bits-ui') || event.error?.stack?.includes('melt')) {
      console.error('UI Component Error:', {
        message: event.error.message,
        component: 'bits-ui/melt-ui',
        stack: event.error.stack,
        filename: event.filename,
        lineno: event.lineno
      });
    }
  });

  // Handle unhandled promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    
    // Prevent default browser behavior for known recoverable errors
    if (event.reason?.message?.includes('fetch') || 
        event.reason?.message?.includes('network')) {
      event.preventDefault();
      console.warn('Network error handled gracefully');
    }
  });

  // Initialize application state
  console.log('Legal AI Platform initialized with browser polyfills');
  
  // Set up performance monitoring if available
  if (typeof PerformanceObserver !== 'undefined') {
    try {
      const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if (entry.entryType === 'navigation') {
            console.log(`Page load time: ${entry.duration}ms`);
          }
        });
      });
      observer.observe({ entryTypes: ['navigation'] });
    } catch (err) {
      console.warn('Performance monitoring not available:', err);
    }
  }
}

// Export app configuration for use by other modules
export const appConfig = {
  version: '1.0.0',
  environment: typeof window !== 'undefined' ? 'browser' : 'server',
  features: {
    vectorSearch: true,
    aiAssistant: true,
    realTimeSearch: true,
    hybridVectorOps: true
  },
  api: {
    baseUrl: typeof window !== 'undefined' ? window.location.origin : 'http://localhost:5173',
    timeout: 30000
  },
  ui: {
    theme: 'yorha-legal',
    prefersDarkMode: typeof window !== 'undefined' ? 
      window.matchMedia('(prefers-color-scheme: dark)').matches : true
  }
};

export default appConfig;