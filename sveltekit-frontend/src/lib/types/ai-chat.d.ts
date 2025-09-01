import stream from "stream";
// AI Chat Type Definitions
export interface AIMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: Date;
  metadata?: Record<string, any>;
}

export interface AIChat {
  id: string;
  messages: AIMessage[];
  model?: string;
  temperature?: number;
  maxTokens?: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface AIStreamResponse {
  text: string;
  isComplete: boolean;
  error?: string;
}

export interface AICompletionOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  stop?: string[];
}

export interface AIProvider {
  complete(prompt: string, options?: AICompletionOptions): Promise<string>;
  stream(prompt: string, options?: AICompletionOptions): AsyncIterable<AIStreamResponse>;
  embed(text: string): Promise<number[]>;
}

export type AIModelType = 'gpt-4' | 'gpt-3.5-turbo' | 'claude' | 'gemma' | 'llama' | 'custom';

export interface AIConfig {
  provider: 'openai' | 'anthropic' | 'ollama' | 'custom';
  model: AIModelType;
  apiKey?: string;
  baseUrl?: string;
  defaultOptions?: AICompletionOptions;
}
