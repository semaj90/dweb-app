// AI Assistant Store with XState + Svelte 5 Integration
// Bridges AI Assistant machine with reactive Svelte components
import { browser } from '$app/environment';
import { createActor } from 'xstate';
import { aiAssistantMachine, aiAssistantServices, aiAssistantActions } from '$lib/machines/aiAssistantMachine.js';

// AI Assistant reactive state interface
export interface AIAssistantState {
  isActive: boolean;
  isProcessing: boolean;
  currentQuery: string;
  response: string;
  conversationHistory: ConversationEntry[];
  model: string;
  temperature: number;
  maxTokens: number;
  error: string | null;
  ollamaClusterHealth: {
    primary: boolean;
    secondary: boolean;
    embeddings: boolean;
  };
  context7Analysis?: Context7Analysis;
  usage: {
    totalQueries: number;
    totalTokens: number;
    averageResponseTime: number;
  };
  streamingActive: boolean;
  streamBuffer: string;
}

export interface ConversationEntry {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    model: string;
    temperature: number;
    responseTime: number;
    tokenCount: number;
    context7Used: boolean;
  };
}

export interface Context7Analysis {
  suggestions: string[];
  codeExamples: any[];
  documentation: string;
  confidence: number;
}

// Create reactive AI assistant state using $state rune
const aiAssistantState = $state<AIAssistantState>({
  isActive: false,
  isProcessing: false,
  currentQuery: '',
  response: '',
  conversationHistory: [],
  model: 'gemma3-legal',
  temperature: 0.7,
  maxTokens: 2048,
  error: null,
  ollamaClusterHealth: {
    primary: false,
    secondary: false,
    embeddings: false
  },
  usage: {
    totalQueries: 0,
    totalTokens: 0,
    averageResponseTime: 0
  },
  streamingActive: false,
  streamBuffer: ''
});

// Create XState actor for AI assistant
const aiAssistantActor = createActor(aiAssistantMachine, {
  services: aiAssistantServices,
  actions: aiAssistantActions
});

export class AIAssistantManager {
  private actor = aiAssistantActor;
  private healthCheckInterval: number | null = null;

  constructor() {
    if (browser) {
      this.initialize();
    }
  }

  // Get current AI assistant state (reactive)
  get state() {
    return aiAssistantState;
  }

  // Initialize AI assistant manager
  private initialize() {
    // Start the XState actor
    this.actor.start();

    // Subscribe to state changes
    this.actor.subscribe((state) => {
      this.updateAIAssistantState(state);
    });

    // Start periodic health checks
    this.startHealthChecks();

    // Initial cluster health check
    this.checkClusterHealth();
  }

  // Update reactive state from XState machine
  private updateAIAssistantState(machineState: any) {
    const { context } = machineState;
    
    aiAssistantState.isActive = !machineState.matches('idle');
    aiAssistantState.isProcessing = context.isProcessing;
    aiAssistantState.currentQuery = context.currentQuery;
    aiAssistantState.response = context.response;
    aiAssistantState.conversationHistory = context.conversationHistory;
    aiAssistantState.model = context.model;
    aiAssistantState.temperature = context.temperature;
    aiAssistantState.maxTokens = context.maxTokens;
    aiAssistantState.error = context.error;
    aiAssistantState.ollamaClusterHealth = context.ollamaClusterHealth;
    aiAssistantState.context7Analysis = context.context7Analysis;
    aiAssistantState.usage = context.usage;
    aiAssistantState.streamingActive = context.activeStreaming;
    aiAssistantState.streamBuffer = context.streamBuffer;
  }

  // Send a message to the AI assistant
  async sendMessage(message: string, options?: {
    useContext7?: boolean;
    model?: string;
    temperature?: number;
  }) {
    if (!message.trim()) {
      throw new Error('Message cannot be empty');
    }

    try {
      // Set model and temperature if provided
      if (options?.model) {
        this.setModel(options.model);
      }
      if (options?.temperature !== undefined) {
        this.setTemperature(options.temperature);
      }

      // Send message to AI assistant
      this.actor.send({
        type: 'SEND_MESSAGE',
        message: message.trim(),
        useContext7: options?.useContext7 || false
      });

      console.log('Message sent to AI assistant:', message);
    } catch (error) {
      console.error('Failed to send message:', error);
      throw error;
    }
  }

  // Set the AI model
  setModel(model: string) {
    const validModels = [
      'gemma3-legal',
      'nomic-embed-text',
      'deeds-web',
      'llama2',
      'mistral'
    ];

    if (!validModels.includes(model)) {
      throw new Error(`Invalid model: ${model}. Valid models: ${validModels.join(', ')}`);
    }

    this.actor.send({
      type: 'SET_MODEL',
      model
    });
  }

  // Set temperature for response generation
  setTemperature(temperature: number) {
    if (temperature < 0 || temperature > 2) {
      throw new Error('Temperature must be between 0 and 2');
    }

    this.actor.send({
      type: 'SET_TEMPERATURE',
      temperature
    });
  }

  // Clear conversation history
  clearConversation() {
    this.actor.send({ type: 'CLEAR_CONVERSATION' });
    console.log('Conversation cleared');
  }

  // Retry the last failed query
  retryLast() {
    if (aiAssistantState.error) {
      this.actor.send({ type: 'RETRY_LAST' });
      console.log('Retrying last query');
    }
  }

  // Stop current generation
  stopGeneration() {
    if (aiAssistantState.isProcessing) {
      this.actor.send({ type: 'STOP_GENERATION' });
      console.log('Generation stopped');
    }
  }

  // Start streaming mode
  startStreaming(message: string) {
    if (!message.trim()) {
      throw new Error('Message cannot be empty');
    }

    this.actor.send({
      type: 'START_STREAMING'
    });

    // Then send the message
    this.sendMessage(message);
  }

  // Check Ollama cluster health
  async checkClusterHealth() {
    try {
      this.actor.send({ type: 'CHECK_CLUSTER_HEALTH' });
      console.log('Cluster health check initiated');
    } catch (error) {
      console.error('Failed to check cluster health:', error);
    }
  }

  // Analyze query with Context7
  async analyzeWithContext7(topic: string) {
    try {
      this.actor.send({
        type: 'ANALYZE_WITH_CONTEXT7',
        topic
      });
      console.log('Context7 analysis initiated for topic:', topic);
    } catch (error) {
      console.error('Failed to analyze with Context7:', error);
      throw error;
    }
  }

  // Get conversation statistics
  getConversationStats() {
    const history = aiAssistantState.conversationHistory;
    const userMessages = history.filter(entry => entry.type === 'user');
    const assistantMessages = history.filter(entry => entry.type === 'assistant');

    return {
      totalMessages: history.length,
      userMessages: userMessages.length,
      assistantMessages: assistantMessages.length,
      averageResponseTime: aiAssistantState.usage.averageResponseTime,
      totalTokens: aiAssistantState.usage.totalTokens,
      conversationDuration: history.length > 0 ? 
        Date.now() - history[0].timestamp.getTime() : 0
    };
  }

  // Export conversation to JSON
  exportConversation() {
    const stats = this.getConversationStats();
    const exportData = {
      timestamp: new Date().toISOString(),
      model: aiAssistantState.model,
      temperature: aiAssistantState.temperature,
      conversation: aiAssistantState.conversationHistory,
      statistics: stats,
      usage: aiAssistantState.usage
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ai_conversation_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log('Conversation exported');
  }

  // Import conversation from JSON
  async importConversation(file: File) {
    try {
      const text = await file.text();
      const data = JSON.parse(text);

      if (data.conversation && Array.isArray(data.conversation)) {
        // Clear current conversation
        this.clearConversation();

        // Restore conversation history
        aiAssistantState.conversationHistory = data.conversation.map((entry: any) => ({
          ...entry,
          timestamp: new Date(entry.timestamp)
        }));

        // Restore settings if available
        if (data.model) {
          this.setModel(data.model);
        }
        if (data.temperature !== undefined) {
          this.setTemperature(data.temperature);
        }

        console.log('Conversation imported successfully');
      } else {
        throw new Error('Invalid conversation file format');
      }
    } catch (error) {
      console.error('Failed to import conversation:', error);
      throw error;
    }
  }

  // Get cluster status summary
  getClusterStatus() {
    const health = aiAssistantState.ollamaClusterHealth;
    const healthyCount = Object.values(health).filter(Boolean).length;
    const totalCount = Object.keys(health).length;

    return {
      healthy: healthyCount === totalCount,
      healthyCount,
      totalCount,
      status: healthyCount === totalCount ? 'all_healthy' : 
              healthyCount > 0 ? 'partial' : 'all_down',
      details: health
    };
  }

  // Get available models from cluster
  async getAvailableModels() {
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      if (response.ok) {
        const data = await response.json();
        return data.models?.map((model: any) => model.name) || [];
      }
      return ['gemma3-legal', 'nomic-embed-text', 'deeds-web']; // Fallback
    } catch (error) {
      console.error('Failed to fetch available models:', error);
      return ['gemma3-legal', 'nomic-embed-text', 'deeds-web']; // Fallback
    }
  }

  // Start periodic health checks
  private startHealthChecks() {
    // Health check every 30 seconds
    this.healthCheckInterval = window.setInterval(() => {
      this.checkClusterHealth();
    }, 30000);
  }

  // Stop health checks
  private stopHealthChecks() {
    if (this.healthCheckInterval) {
      window.clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
  }

  // Clean up on destroy
  destroy() {
    this.actor.stop();
    this.stopHealthChecks();
  }
}

// Create singleton AI assistant manager
export const aiAssistantManager = new AIAssistantManager();

// Reactive getters for use in components
export const isAIActive = () => aiAssistantState.isActive;
export const isProcessing = () => aiAssistantState.isProcessing;
export const currentResponse = () => aiAssistantState.response;
export const conversationHistory = () => aiAssistantState.conversationHistory;
export const currentModel = () => aiAssistantState.model;
export const currentTemperature = () => aiAssistantState.temperature;
export const aiError = () => aiAssistantState.error;
export const clusterHealth = () => aiAssistantState.ollamaClusterHealth;
export const context7Analysis = () => aiAssistantState.context7Analysis;
export const aiUsage = () => aiAssistantState.usage;

// Convenience functions
export const sendAIMessage = (message: string, options?: unknown) => 
  aiAssistantManager.sendMessage(message, options);
export const setAIModel = (model: string) => aiAssistantManager.setModel(model);
export const setAITemperature = (temp: number) => aiAssistantManager.setTemperature(temp);
export const clearAIConversation = () => aiAssistantManager.clearConversation();
export const checkAIClusterHealth = () => aiAssistantManager.checkClusterHealth();

// Initialize AI assistant manager when module loads
if (browser) {
  // Auto-cleanup on page unload
  window.addEventListener('beforeunload', () => {
    aiAssistantManager.destroy();
  });
}