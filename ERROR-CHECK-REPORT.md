// TIMEOUT FIX FOR FETCH OPERATIONS
// Add this helper function to enhanced-orchestrator-fixed.mjs

async function fetchWithTimeout(url, options = {}, timeout = 10000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeout}ms`);
    }
    throw error;
  }
}

// Usage in verifyModels():
const response = await fetchWithTimeout('http://localhost:11434/api/tags', {
  method: 'GET'
}, 10000);

// Usage in validateAllAPIs():
const response = await fetchWithTimeout(endpoint.url, { 
  method: 'GET'
}, 10000);
