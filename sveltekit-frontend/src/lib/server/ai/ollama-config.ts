/**
 * Ollama Configuration for High-Performance AI Assistant
 * Using local gemma3:legal-latest model with legal-bert fallback
 */

import type { OllamaConfig, ModelConfig } from './types';

// Model configurations aligned with the blueprint architecture
export const MODELS: Record<string, ModelConfig> = {
  'gemma3:legal-latest': {
    name: 'gemma3:legal-latest',
    type: 'local',
    capabilities: ['text-generation', 'embeddings', 'legal-analysis'],
    contextWindow: 8192,
    embeddingDimension: 768,
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    systemPrompt: `You are a sophisticated legal AI assistant powered by Gemma3, specialized in legal document analysis, contract review, and case law research. 
    You provide accurate, context-aware legal insights while maintaining strict confidentiality and professional standards.
    Your responses are based on deep understanding of legal terminology, precedents, and regulatory frameworks.`,
    options: {
      num_gpu: 1, // Use GPU acceleration
      num_thread: 8, // Parallel processing threads
      repeat_penalty: 1.1,
      seed: 42,
      stop: ['User:', 'Human:', '\n\n\n']
    }
  },
  'legal-bert': {
    name: 'legal-bert',
    type: 'fallback',
    capabilities: ['text-generation', 'legal-analysis', 'embeddings'],
    contextWindow: 512, // BERT models have smaller context
    embeddingDimension: 768,
    temperature: 0.5, // Lower temperature for more precise legal analysis
    topP: 0.85,
    topK: 30,
    systemPrompt: `You are Legal-BERT, a specialized language model trained on legal texts including legislation, court cases, and contracts.
    You excel at understanding legal terminology, identifying legal concepts, and providing accurate analysis of legal documents.
    Focus on precision and cite relevant legal principles when applicable.`,
    options: {
      num_gpu: 1,
      num_thread: 4,
      repeat_penalty: 1.05,
      stop: ['\n\n', 'END', 'User:']
    }
  },
  'nomic-embed-text': {
    name: 'nomic-embed-text',
    type: 'embedding',
    capabilities: ['embeddings'],
    embeddingDimension: 768,
    contextWindow: 8192
  },
  'bge-large-en': {
    name: 'bge-large-en',
    type: 'embedding',
    capabilities: ['embeddings'],
    embeddingDimension: 1024,
    contextWindow: 512
  }
};

// Fallback chain configuration - llama3.2 removed
export const FALLBACK_CHAIN = {
  'legal-analysis': [
    'gemma3:legal-latest',  // Primary
    'legal-bert'            // Legal-specific fallback only
  ],
  'text-generation': [
    'gemma3:legal-latest',
    'legal-bert'
  ],
  'embeddings': [
    'nomic-embed-text',
    'bge-large-en'
  ]
};

export const OLLAMA_CONFIG: OllamaConfig = {
  baseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434',
  defaultModel: 'gemma3:legal-latest',
  embeddingModel: 'nomic-embed-text',
  fallbackModel: 'legal-bert',
  fallbackModels: {
    legal: 'legal-bert',
    general: 'legal-bert' // Use legal-bert as general fallback too
  },
  timeout: 60000, // 60 seconds for complex legal analysis
  maxRetries: 3,
  streamEnabled: true,
  
  // GPU acceleration settings
  gpu: {
    enabled: true,
    layers: 35, // Number of layers to offload to GPU
    mainGpu: 0,
    tensorSplit: null
  },
  
  // Performance optimization
  performance: {
    batchSize: 32,
    parallelRequests: 4,
    cacheEnabled: true,
    cacheTTL: 3600 // 1 hour cache
  },
  
  // Advanced features from blueprint
  features: {
    som: true, // Self-Organizing Map for topic modeling
    proactiveCaching: true,
    multiModalIndexing: true,
    reinforcementLearning: false, // Can be enabled later
    webGpuAcceleration: true,
    intelligentFallback: true // Enable smart model selection
  }
};

/**
 * Get model configuration with fallback support
 */
export function getModelConfig(modelName: string = OLLAMA_CONFIG.defaultModel): ModelConfig {
  return MODELS[modelName] || MODELS[OLLAMA_CONFIG.fallbackModels?.legal || 'legal-bert'];
}

/**
 * Check if model supports a specific capability
 */
export function modelSupportsCapability(modelName: string, capability: string): boolean {
  const config = getModelConfig(modelName);
  return config.capabilities.includes(capability);
}

/**
 * Get optimal model for a specific task with fallback chain
 */
export function getOptimalModel(task: 'embedding' | 'generation' | 'legal-analysis'): string[] {
  const taskMap = {
    'embedding': FALLBACK_CHAIN['embeddings'],
    'generation': FALLBACK_CHAIN['text-generation'],
    'legal-analysis': FALLBACK_CHAIN['legal-analysis']
  };
  
  return taskMap[task] || [OLLAMA_CONFIG.defaultModel];
}

/**
 * Get the best available model from a list
 * @param preferredModels Array of model names in order of preference
 * @param availableModels Array of currently available model names
 */
export function selectBestAvailableModel(
  preferredModels: string[],
  availableModels: string[]
): string | null {
  for (const model of preferredModels) {
    // Check exact match
    if (availableModels.includes(model)) {
      return model;
    }
    
    // Check partial match for variants (e.g., legal-bert:latest)
    const matchingModel = availableModels.find(available => 
      available.includes(model.split(':')[0])
    );
    
    if (matchingModel) {
      return matchingModel;
    }
  }
  
  // If no preferred models available, return first available or null
  return availableModels[0] || null;
}

/**
 * Determine if a task should use legal-specific model
 */
export function isLegalTask(prompt: string): boolean {
  const legalKeywords = [
    'contract', 'agreement', 'legal', 'law', 'court', 'case',
    'statute', 'regulation', 'compliance', 'liability', 'clause',
    'jurisdiction', 'plaintiff', 'defendant', 'litigation',
    'intellectual property', 'patent', 'trademark', 'copyright',
    'tort', 'negligence', 'breach', 'damages', 'remedy',
    'arbitration', 'mediation', 'settlement', 'precedent',
    'deed', 'title', 'evidence', 'testimony', 'witness',
    'prosecutor', 'defense', 'attorney', 'counsel', 'judge'
  ];
  
  const lowerPrompt = prompt.toLowerCase();
  return legalKeywords.some(keyword => lowerPrompt.includes(keyword));
}

export default OLLAMA_CONFIG;
