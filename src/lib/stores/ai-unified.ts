// AI Unified Store - Centralized AI system management
import { writable, derived, type Readable } from 'svelte/store';
import { browser } from '$app/environment';
import { createAISystemStore } from './ai-system-store';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  metadata?: {
    model?: string;
    processingTime?: number;
    confidence?: number;
  };
}

interface AIUnifiedState {
  // System integration
  systemStore: ReturnType<typeof createAISystemStore> | null;
  
  // Chat state
  chatState: {
    messages: ChatMessage[];
    isProcessing: boolean;
    currentModel: string;
    sessionId: string;
    context: string;
  };
  
  // Performance metrics
  performance: {
    responseTime: number;
    tokensPerSecond: number;
    memoryUsage: number;
    cacheHitRate: number;
    errorRate: number;
  };
  
  // Service health
  services: {
    ollama: 'healthy' | 'degraded' | 'offline';
    goMicroservice: 'healthy' | 'degraded' | 'offline';
    postgres: 'healthy' | 'degraded' | 'offline';
    redis: 'healthy' | 'degraded' | 'offline';
  };
  
  // Configuration
  config: {
    enableAdaptiveQuality: boolean;
    enablePerformanceMonitoring: boolean;
    maxConcurrentRequests: number;
    defaultModel: string;
  };
}

const initialState: AIUnifiedState = {
  systemStore: null,
  chatState: {
    messages: [],
    isProcessing: false,
    currentModel: 'gemma3-legal',
    sessionId: '',
    context: 'legal-analysis'
  },
  performance: {
    responseTime: 0,
    tokensPerSecond: 0,
    memoryUsage: 0,
    cacheHitRate: 0,
    errorRate: 0
  },
  services: {
    ollama: 'offline',
    goMicroservice: 'offline',
    postgres: 'offline',
    redis: 'offline'
  },
  config: {
    enableAdaptiveQuality: true,
    enablePerformanceMonitoring: true,
    maxConcurrentRequests: 5,
    defaultModel: 'gemma3-legal'
  }
};

// Main unified store
export const aiUnified = writable<AIUnifiedState>(initialState);

// Derived stores for specific aspects
export const chatState = derived(aiUnified, ($ai) => $ai.chatState);
export const performanceMetrics = derived(aiUnified, ($ai) => $ai.performance);
export const serviceHealth = derived(aiUnified, ($ai) => $ai.services);
export const aiConfig = derived(aiUnified, ($ai) => $ai.config);

// Store actions
export const aiActions = {
  // Initialize the AI system
  async initialize() {
    if (!browser) return;
    
    try {
      const systemStore = createAISystemStore();
      
      aiUnified.update(state => ({
        ...state,
        systemStore,
        chatState: {
          ...state.chatState,
          sessionId: generateSessionId()
        }
      }));
      
      // Initialize system store with config
      await systemStore.initialize({
        windowsOptimizations: {
          enableGPUAcceleration: true,
          enableSIMD: true,
          maxWorkerThreads: 8,
          enableWebAssembly: true
        },
        performance: {
          enableJITCompilation: true,
          cacheStrategy: 'hybrid',
          enableRealTimeMetrics: true
        }
      });
      
      // Start health monitoring
      this.startHealthMonitoring();
      
      console.log('✅ AI Unified System initialized');
    } catch (error) {
      console.error('❌ AI System initialization failed:', error);
    }
  },
  
  // Send a chat message
  async sendMessage(content: string, context?: string) {
    if (!browser) return;
    
    const messageId = generateMessageId();
    const timestamp = Date.now();
    
    // Add user message
    aiUnified.update(state => ({
      ...state,
      chatState: {
        ...state.chatState,
        isProcessing: true,
        messages: [...state.chatState.messages, {
          id: messageId,
          role: 'user',
          content,
          timestamp
        }]
      }
    }));
    
    try {
      const startTime = performance.now();
      
      // Call AI service (Go microservice or direct Ollama)
      const response = await this.callAIService(content, context);
      
      const processingTime = performance.now() - startTime;
      
      // Add assistant response
      aiUnified.update(state => ({
        ...state,
        chatState: {
          ...state.chatState,
          isProcessing: false,
          messages: [...state.chatState.messages, {
            id: generateMessageId(),
            role: 'assistant',
            content: response.response,
            timestamp: Date.now(),
            metadata: {
              model: response.model || state.chatState.currentModel,
              processingTime,
              confidence: response.confidence
            }
          }]
        },
        performance: {
          ...state.performance,
          responseTime: processingTime,
          tokensPerSecond: response.tokensPerSecond || 0
        }
      }));
      
      // Update analytics
      const currentState = await new Promise(resolve => {
        const unsubscribe = aiUnified.subscribe(state => {
          unsubscribe();
          resolve(state);
        });
      });
      
      if (currentState.systemStore) {
        currentState.systemStore.logInteraction({
          type: 'chat-message',
          content,
          response: response.response,
          processingTime,
          timestamp
        });
      }
      
    } catch (error) {
      console.error('Chat message error:', error);
      
      aiUnified.update(state => ({
        ...state,
        chatState: {
          ...state.chatState,
          isProcessing: false,
          messages: [...state.chatState.messages, {
            id: generateMessageId(),
            role: 'assistant',
            content: 'I apologize, but I encountered an error processing your request. Please try again.',
            timestamp: Date.now(),
            metadata: { model: 'error-fallback' }
          }]
        },
        performance: {
          ...state.performance,
          errorRate: state.performance.errorRate + 0.01
        }
      }));
    }
  },
  
  // Call AI service with fallback
  async callAIService(content: string, context?: string) {
    const endpoints = [
      'http://localhost:8094/api/rag', // Go microservice
      'http://localhost:11434/api/generate' // Direct Ollama
    ];
    
    for (const endpoint of endpoints) {
      try {
        if (endpoint.includes('8094')) {
          // Enhanced RAG service
          const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              query: content,
              context: context || 'legal-analysis',
              model: 'gemma3-legal',
              max_tokens: 1000
            })
          });
          
          if (response.ok) {
            const data = await response.json();
            return {
              response: data.response || data.answer,
              model: data.model,
              confidence: data.confidence,
              tokensPerSecond: data.performance?.tokens_per_second
            };
          }
        } else {
          // Direct Ollama
          const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: 'gemma3-legal',
              prompt: content,
              stream: false
            })
          });
          
          if (response.ok) {
            const data = await response.json();
            return {
              response: data.response,
              model: 'gemma3-legal',
              confidence: 0.8
            };
          }
        }
      } catch (error) {
        console.warn(`Failed to call ${endpoint}:`, error);
      }
    }
    
    throw new Error('All AI services unavailable');
  },
  
  // Start health monitoring
  startHealthMonitoring() {
    if (!browser) return;
    
    const checkHealth = async () => {
      const services = {
        ollama: await this.checkServiceHealth('http://localhost:11434/api/tags'),
        goMicroservice: await this.checkServiceHealth('http://localhost:8094/health'),
        postgres: await this.checkServiceHealth('/api/health/postgres'),
        redis: await this.checkServiceHealth('/api/health/redis')
      };
      
      aiUnified.update(state => ({
        ...state,
        services
      }));
    };
    
    // Check immediately and then every 30 seconds
    checkHealth();
    setInterval(checkHealth, 30000);
  },
  
  // Check individual service health
  async checkServiceHealth(endpoint: string): Promise<'healthy' | 'degraded' | 'offline'> {
    try {
      const response = await fetch(endpoint, {
        method: 'GET',
        timeout: 5000
      });
      
      if (response.ok) {
        return 'healthy';
      } else if (response.status >= 500) {
        return 'degraded';
      }
    } catch (error) {
      // Service is offline or unreachable
    }
    
    return 'offline';
  },
  
  // Clear chat history
  clearChat() {
    aiUnified.update(state => ({
      ...state,
      chatState: {
        ...state.chatState,
        messages: [],
        sessionId: generateSessionId()
      }
    }));
  },
  
  // Update configuration
  updateConfig(newConfig: Partial<AIUnifiedState['config']>) {
    aiUnified.update(state => ({
      ...state,
      config: {
        ...state.config,
        ...newConfig
      }
    }));
  },
  
  // Get performance recommendations
  getRecommendations() {
    let recommendations: Array<{type: string, priority: string, message: string}> = [];
    
    const unsubscribe = aiUnified.subscribe(state => {
      if (state.performance.responseTime > 5000) {
        recommendations.push({
          type: 'performance',
          priority: 'high',
          message: 'AI response time is slow. Consider enabling adaptive quality control.'
        });
      }
      
      if (state.performance.errorRate > 0.1) {
        recommendations.push({
          type: 'reliability',
          priority: 'high',
          message: 'High error rate detected. Check service health.'
        });
      }
      
      if (state.services.ollama === 'offline' && state.services.goMicroservice === 'offline') {
        recommendations.push({
          type: 'availability',
          priority: 'critical',
          message: 'All AI services are offline. Check service status.'
        });
      }
    });
    
    unsubscribe();
    return recommendations;
  }
};

// Utility functions
function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
}

function generateMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
}

// Auto-initialize if in browser
if (browser) {
  aiActions.initialize();
}

// Export types
export type { ChatMessage, AIUnifiedState };