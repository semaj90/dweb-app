// Browser polyfills for Node.js modules and enhanced compatibility
// This ensures Bits UI, Melt UI, and other dependencies work properly in browser context

// Global polyfills for Node.js globals
declare global {
  interface Window {
    global: typeof globalThis;
    process: {
      env: Record<string, string | undefined>;
      browser: boolean;
    };
    Buffer: any;
  }
}

// Polyfill process.env for browser
if (typeof window !== 'undefined') {
  window.global = window.global || globalThis;
  
  if (!window.process) {
    window.process = {
      env: {
        NODE_ENV: import.meta.env.MODE || 'development',
        PUBLIC_ENV: 'browser'
      },
      browser: true
    } as any;
  }
}

// Polyfill Buffer for browser if needed
if (typeof window !== 'undefined' && !window.Buffer) {
  // Lightweight Buffer polyfill for basic operations
  window.Buffer = {
    from: (str: string, encoding?: string) => new TextEncoder().encode(str),
    isBuffer: (obj: any) => obj instanceof Uint8Array,
    alloc: (size: number) => new Uint8Array(size)
  } as any;
}

// Ensure crypto is available (for Node.js crypto module usage)
if (typeof window !== 'undefined' && !window.crypto) {
  // Modern browsers have crypto.subtle, but older browsers might need this
  console.warn('Crypto API not available, some features may be limited');
}

// Stream polyfill for browser (if needed by dependencies)
if (typeof window !== 'undefined' && typeof ReadableStream === 'undefined') {
  // Import streams-polyfill if available, or provide basic implementation
  console.warn('ReadableStream not available, some streaming features may be limited');
}

// Path manipulation utilities for browser
export const pathUtils = {
  join: (...parts: string[]) => {
    return parts
      .filter(Boolean)
      .join('/')
      .replace(/\/+/g, '/') // Remove double slashes
      .replace(/\/$/g, ''); // Remove trailing slash
  },
  
  resolve: (...parts: string[]) => {
    let resolvedPath = '';
    for (const part of parts) {
      if (part && part !== '.') {
        resolvedPath = resolvedPath ? `${resolvedPath}/${part}` : part;
      }
    }
    return resolvedPath.startsWith('/') ? resolvedPath : `/${resolvedPath}`;
  },
  
  dirname: (path: string) => {
    const lastSlash = path.lastIndexOf('/');
    return lastSlash === -1 ? '.' : path.slice(0, lastSlash);
  },
  
  basename: (path: string, ext?: string) => {
    let base = path.split('/').pop() || '';
    if (ext && base.endsWith(ext)) {
      base = base.slice(0, -ext.length);
    }
    return base;
  }
};

// URL utilities for better compatibility
export const urlUtils = {
  isAbsolute: (url: string) => {
    return /^https?:\/\//.test(url) || url.startsWith('/');
  },
  
  resolve: (base: string, relative: string) => {
    if (urlUtils.isAbsolute(relative)) return relative;
    return new URL(relative, base).href;
  },
  
  parse: (url: string) => {
    try {
      return new URL(url);
    } catch {
      // Fallback for relative URLs
      return new URL(url, window?.location?.href || 'http://localhost/');
    }
  }
};

// Enhanced fetch with timeout and better error handling
export const enhancedFetch = async (
  url: string, 
  options: RequestInit & { timeout?: number } = {}
): Promise<Response> => {
  const { timeout = 30000, ...fetchOptions } = options;
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(`Request timeout after ${timeout}ms`);
    }
    
    throw error;
  }
};

// Debounce utility for search and other operations
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: ReturnType<typeof setTimeout>;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

// Throttle utility for performance-sensitive operations
export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean;
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
};

// Local storage utilities with error handling
export const storage = {
  get: <T = any>(key: string, defaultValue?: T): T | null => {
    try {
      if (typeof window === 'undefined') return defaultValue || null;
      
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue || null;
    } catch (error) {
      console.warn(`Failed to get localStorage item "${key}":`, error);
      return defaultValue || null;
    }
  },
  
  set: (key: string, value: any): boolean => {
    try {
      if (typeof window === 'undefined') return false;
      
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (error) {
      console.warn(`Failed to set localStorage item "${key}":`, error);
      return false;
    }
  },
  
  remove: (key: string): boolean => {
    try {
      if (typeof window === 'undefined') return false;
      
      localStorage.removeItem(key);
      return true;
    } catch (error) {
      console.warn(`Failed to remove localStorage item "${key}":`, error);
      return false;
    }
  },
  
  clear: (): boolean => {
    try {
      if (typeof window === 'undefined') return false;
      
      localStorage.clear();
      return true;
    } catch (error) {
      console.warn('Failed to clear localStorage:', error);
      return false;
    }
  }
};

// Initialize polyfills
if (typeof window !== 'undefined') {
  // Ensure all polyfills are applied
  console.log('Browser polyfills initialized for Legal AI Platform');
}

export default {
  pathUtils,
  urlUtils,
  enhancedFetch,
  debounce,
  throttle,
  storage
};