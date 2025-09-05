/**
 * Environment Service - SvelteKit 2 + Svelte 5 Compatible
 * Provides environment detection, LLM endpoint health, and client utilities
 */

import { browser } from '$app/environment';
import { env } from '$env/dynamic/public';
import type { LLMEndpoint, ClientEnvironment } from '../types/service-types';

/**
 * Client Environment Detection
 */
export const CLIENT_ENV: ClientEnvironment = {
  dev: import.meta.env.DEV,
  prod: import.meta.env.PROD, 
  preview: import.meta.env.MODE === 'preview',
  browser: browser
};

/**
 * LLM Endpoint Health Checker
 */
class LLMHealthChecker {
  private endpoints: Map<string, LLMEndpoint> = new Map();
  private checkInterval: number = 30000; // 30 seconds
  private intervalId?: NodeJS.Timeout;

  constructor() {
    // Initialize with default endpoints from your MCP config
    this.addEndpoint({
      url: 'http://localhost:11434',
      model: 'gemma3-legal',
      healthy: false,
      latency: 0,
      lastCheck: 0
    });

    this.addEndpoint({
      url: 'http://localhost:11434',
      model: 'nomic-embed-text',
      healthy: false,
      latency: 0,
      lastCheck: 0
    });

    // Only start health checking in browser
    if (CLIENT_ENV.browser) {
      this.startHealthChecking();
    }
  }

  addEndpoint(endpoint: LLMEndpoint): void {
    const key = `${endpoint.url}/${endpoint.model}`;
    this.endpoints.set(key, endpoint);
  }

  async checkEndpointHealth(endpoint: LLMEndpoint): Promise<LLMEndpoint> {
    const startTime = Date.now();
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${endpoint.url}/api/tags`, {
        signal: controller.signal,
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      clearTimeout(timeoutId);
      
      if (response.ok) {
        const data = await response.json();
        const hasModel = data.models?.some((m: any) => 
          m.name?.includes(endpoint.model) || m.model?.includes(endpoint.model)
        );

        return {
          ...endpoint,
          healthy: hasModel,
          latency: Date.now() - startTime,
          lastCheck: Date.now()
        };
      }
    } catch (error: any) {
      console.warn(`Health check failed for ${endpoint.url}/${endpoint.model}:`, error);
    }

    return {
      ...endpoint,
      healthy: false,
      latency: Date.now() - startTime,
      lastCheck: Date.now()
    };
  }

  async getHealthyLlmEndpoint(model?: string): Promise<LLMEndpoint | null> {
    // Update health status first
    await this.checkAllEndpoints();

    // Find healthy endpoint for specific model
    if (model) {
      for (const endpoint of this.endpoints.values()) {
        if (endpoint.model === model && endpoint.healthy) {
          return endpoint;
        }
      }
    }

    // Find any healthy endpoint
    for (const endpoint of this.endpoints.values()) {
      if (endpoint.healthy) {
        return endpoint;
      }
    }

    return null;
  }

  async checkAllEndpoints(): Promise<void> {
    const promises = Array.from(this.endpoints.entries()).map(async ([key, endpoint]) => {
      const updated = await this.checkEndpointHealth(endpoint);
      this.endpoints.set(key, updated);
    });

    await Promise.allSettled(promises);
  }

  startHealthChecking(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }

    this.intervalId = setInterval(() => {
      this.checkAllEndpoints();
    }, this.checkInterval);
  }

  stopHealthChecking(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
  }

  getEndpointStats(): Array<LLMEndpoint> {
    return Array.from(this.endpoints.values());
  }
}

// Global instance
const healthChecker = new LLMHealthChecker();

/**
 * Get healthy LLM endpoint - compatible with existing code
 */
export async function getHealthyLlmEndpoint(model?: string): Promise<string | null> {
  const endpoint = await healthChecker.getHealthyLlmEndpoint(model);
  return endpoint ? endpoint.url : null;
}

/**
 * Environment variables helper
 */
export function getEnvVar(key: string, defaultValue?: string): string | undefined {
  // Try public env first
  const publicValue = env[`PUBLIC_${key}`] || env[key];
  if (publicValue) return publicValue;

  // Try import.meta.env
  const metaValue = import.meta.env[key] || import.meta.env[`VITE_${key}`];
  if (metaValue) return metaValue;

  return defaultValue;
}

/**
 * Service configuration helper
 */
export function getServiceConfig(serviceName: string) {
  return {
    baseUrl: getEnvVar(`${serviceName.toUpperCase()}_URL`, `http://localhost:8094`),
    enabled: getEnvVar(`${serviceName.toUpperCase()}_ENABLED`, 'true') === 'true',
    timeout: parseInt(getEnvVar(`${serviceName.toUpperCase()}_TIMEOUT`, '30000'), 10),
    retryAttempts: parseInt(getEnvVar(`${serviceName.toUpperCase()}_RETRIES`, '3'), 10),
  };
}

/**
 * Check if we're in development mode
 */
export const isDev = (): boolean => CLIENT_ENV.dev;

/**
 * Check if we're running in browser
 */
export const isBrowser = (): boolean => CLIENT_ENV.browser;

/**
 * Get current environment string
 */
export const getEnvironment = (): string => {
  if (CLIENT_ENV.dev) return 'development';
  if (CLIENT_ENV.preview) return 'preview';
  if (CLIENT_ENV.prod) return 'production';
  return 'unknown';
};

/**
 * Cleanup function for when component unmounts
 */
export function cleanup(): void {
  healthChecker.stopHealthChecking();
}

/**
 * Export health checker instance for advanced usage
 */
export { healthChecker };