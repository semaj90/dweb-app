// Fix embedding type handling
const normalizeEmbedding = (embedding: number[] | number[][]): number[] => {
  if (Array.isArray(embedding) && Array.isArray(embedding[0])) {
    return embedding[0] as number[];
  }
  return embedding as number[];
};

// Use in the problematic section:
// Enhanced AI Service with Local LLM Integration
// Enhanced AI Service with Local LLM Integration and Vector Database Support
// Combines cloud (OpenAI/Ollama) and local (Tauri/Rust) LLM capabilities
// Supports PostgreSQL with pgvector, Qdrant, Redis, RabbitMQ, and Neo4j
import { browser } from '$app/environment';
// Use environment variable safely
const PUBLIC_OLLAMA_URL = typeof window !== 'undefined' 
  ? 'http://localhost:11434' 
  : process.env.OLLAMA_URL || 'http://localhost:11434';
import { tauriLLM } from "./tauri-llm";

// Client-safe configuration
const AI_CONFIG = {
  OLLAMA_URL: PUBLIC_OLLAMA_URL || 'http://localhost:11434',
  OLLAMA_MODEL: 'llama2',
  // Note: API keys should be handled server-side only
  OPENAI_API_KEY: null, // Will be passed from server
} as const;

export type LLMProvider = "openai" | "ollama" | "tauri-local" | "auto";
export type EmbeddingProvider =
  | "openai"
  | "tauri-legal-bert"
  | "tauri-bert"
  | "auto";

export interface AIServiceConfig {
  preferLocal: boolean;
  fallbackToCloud: boolean;
  legalDomain: boolean;
  maxRetries: number;
  timeoutMs: number;
}
export interface GenerationOptions {
  provider?: LLMProvider;
  temperature?: number;
  maxTokens?: number;
  systemPrompt?: string;
  context?: string[];
  legalContext?: boolean;
}
export interface EmbeddingOptions {
  provider?: EmbeddingProvider;
  batchSize?: number;
  normalize?: boolean;
  legalDomain?: boolean;
}
class EnhancedAIService {
  private config: AIServiceConfig;
  private isInitialized = false;

  constructor(config: Partial<AIServiceConfig> = {}) {
    this.config = {
      preferLocal: true,
      fallbackToCloud: true,
      legalDomain: true,
      maxRetries: 3,
      timeoutMs: 30000,
      ...config,
    };
  }
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    // Initialize Tauri LLM service
    await tauriLLM.initialize();

    this.isInitialized = true;
    console.log("Enhanced AI Service initialized");
  }
  // Generate embeddings with intelligent provider selection
  async generateEmbedding(
    text: string | string[],
    options: EmbeddingOptions = {},
  ): Promise<number[] | number[][]> {
    await this.initialize();

    const provider = this.selectEmbeddingProvider(options);
    const isArray = Array.isArray(text);
    const inputs = isArray ? text : [text];

    try {
      let result: number[][];

      switch (provider) {
        case "tauri-legal-bert":
          if (tauriLLM.isAvailable()) {
            console.log("Using local legal-BERT for embeddings");
            result = (await tauriLLM.generateEmbedding(inputs, {
              batchSize: options.batchSize,
              normalize: options.normalize,
              poolingStrategy: "mean",
            })) as number[][];
          } else {
            throw new Error("Legal-BERT not available locally");
          }
          break;

        case "tauri-bert":
          if (tauriLLM.isAvailable()) {
            console.log("Using local BERT for embeddings");
            result = (await tauriLLM.generateEmbedding(
              inputs,
              options,
            )) as number[][];
          } else {
            throw new Error("Local BERT not available");
          }
          break;

        case "openai":
        default:
          console.log("Using OpenAI for embeddings");
          result = await this.generateOpenAIEmbeddings(inputs);
          break;
      }
      return isArray ? result : result[0];
    } catch (error) {
      console.error(`Embedding generation failed with ${provider}:`, error);

      // Fallback logic
      if (this.config.fallbackToCloud && provider.startsWith("tauri-")) {
        console.log("Falling back to OpenAI embeddings");
        const fallbackResult = await this.generateOpenAIEmbeddings(inputs);
        return isArray ? fallbackResult : fallbackResult[0];
      }
      throw error;
    }
  }
  // Generate AI responses with intelligent provider selection
  async generateResponse(
    prompt: string,
    options: GenerationOptions = {},
  ): Promise<string> {
    await this.initialize();

    const provider = this.selectLLMProvider(options);
    const systemPrompt = this.buildSystemPrompt(options);
    const fullPrompt = this.buildFullPrompt(
      prompt,
      systemPrompt,
      options.context,
    );

    try {
      let response: string;

      switch (provider) {
        case "tauri-local":
          if (tauriLLM.isAvailable()) {
            console.log("Using local Tauri LLM");
            response = await tauriLLM.runInference(fullPrompt, {
              temperature: options.temperature,
              maxTokens: options.maxTokens,
            });
          } else {
            throw new Error("Local LLM not available");
          }
          break;

        case "ollama":
          console.log("Using Ollama LLM");
          response = await this.generateOllamaResponse(fullPrompt, options);
          break;

        case "openai":
        default:
          console.log("Using OpenAI");
          response = await this.generateOpenAIResponse(fullPrompt, options);
          break;
      }
      return response;
    } catch (error) {
      console.error(`LLM generation failed with ${provider}:`, error);

      // Fallback logic
      if (this.config.fallbackToCloud && provider === "tauri-local") {
        console.log("Falling back to cloud LLM");
        return this.generateResponse(prompt, {
          ...options,
          provider: "openai",
        });
      }
      throw error;
    }
  }
  // Legal document analysis using local legal-BERT
  async analyzeLegalDocument(text: string): Promise<{
    classification: any;
    keyEntities: string[];
    similarity: number;
    summary: string;
    riskAssessment: string;
  }> {
    await this.initialize();

    if (!tauriLLM.isAvailable()) {
      throw new Error("Local legal analysis requires Tauri environment");
    }
    try {
      // Run parallel analysis with fallback for missing methods
      const [classification, summary] = await Promise.all([
        this.generateResponse(
          `Classify this legal document type and return only the classification (e.g., "Contract", "Brief", "Motion"): ${text.substring(0, 500)}...`,
          { provider: "tauri-local", legalContext: true },
        ),
        this.generateResponse(
          `Provide a concise summary of this legal document: ${text.substring(0, 1000)}...`,
          { provider: "tauri-local", legalContext: true },
        ),
      ]);

      // Extract key entities (simplified - in real implementation, use NER model)
      const keyEntities = this.extractLegalEntities(text);

      // Risk assessment
      const riskAssessment = await this.generateResponse(
        `Assess the legal risks in this document: ${text.substring(0, 800)}...`,
        { provider: "tauri-local", legalContext: true },
      );

      return {
        classification,
        keyEntities,
        similarity: 0.8, // Default similarity score since classification is a string
        summary,
        riskAssessment,
      };
    } catch (error) {
      console.error("Legal document analysis failed:", error);
      throw error;
    }
  }
  // Batch processing for large document sets
  async batchAnalyzeDocuments(
    documents: Array<{ id: string; text: string; type?: string }>,
  ): Promise<
    Array<{
      id: string;
      embedding: number[];
      classification?: any;
      summary?: string;
      error?: string;
    }>
  > {
    await this.initialize();

    if (tauriLLM.isAvailable()) {
      // Fallback to sequential processing since batchProcessDocuments is not implemented
      const results: Array<{
        id: string;
        embedding: number[];
        classification?: any;
        summary?: string;
        error?: string;
      }> = [];
      for (const doc of documents) {
        try {
          const embedding = await this.generateEmbedding(doc.text);
          const classification = await this.generateResponse(
            `Classify this document type: ${doc.text.substring(0, 200)}...`,
            { provider: "tauri-local" },
          );
          const summary = await this.generateResponse(
            `Summarize: ${doc.text.substring(0, 500)}...`,
            { provider: "tauri-local" },
          );

          results.push({
            id: doc.id,
            embedding: normalizeEmbedding(embedding),
            classification: classification,
            summary: summary,
          });
        } catch (error) {
          results.push({
            id: doc.id,
            embedding: [],
            error: error instanceof Error ? error.message : "Processing failed",
          });
        }
      }
      return results;
    } else {
      // Fallback to sequential processing
      const results: Array<{
        id: string;
        embedding: number[];
        classification?: any;
        summary?: string;
        error?: string;
      }> = [];
      for (const doc of documents) {
        try {
          const embedding = (await this.generateEmbedding(doc.text, {
            legalDomain: true,
          })) as number[];
          const summary = await this.generateResponse(
            `Summarize this document: ${doc.text.substring(0, 500)}...`,
            { maxTokens: 100 },
          );

          results.push({
            id: doc.id,
            embedding,
            summary,
          });
        } catch (error) {
          results.push({
            id: doc.id,
            embedding: [],
            error: error instanceof Error ? error.message : "Unknown error",
          });
        }
      }
      return results;
    }
  }
  // Smart provider selection for embeddings
  private selectEmbeddingProvider(
    options: EmbeddingOptions,
  ): EmbeddingProvider {
    if (options.provider && options.provider !== "auto") {
      return options.provider;
    }
    // Prefer local legal-BERT for legal domain
    if (
      this.config.preferLocal &&
      this.config.legalDomain &&
      tauriLLM.isAvailable()
    ) {
      const models = tauriLLM.getAvailableModels();
      const hasLegalBERT = models.some(
        (m) => m.architecture === "legal-bert" && m.type === "embedding",
      );

      if (hasLegalBERT) {
        return "tauri-legal-bert";
      }
      const hasBERT = models.some(
        (m) => m.architecture === "bert" && m.type === "embedding",
      );
      if (hasBERT) {
        return "tauri-bert";
      }
    }
    return "openai";
  }
  // Smart provider selection for LLM
  private selectLLMProvider(options: GenerationOptions): LLMProvider {
    if (options.provider && options.provider !== "auto") {
      return options.provider;
    }
    // Prefer local for privacy-sensitive legal content
    if (
      this.config.preferLocal &&
      options.legalContext &&
      tauriLLM.isAvailable()
    ) {
      const models = tauriLLM.getAvailableModels();
      const hasLegalLLM = models.some(
        (m) => m.type === "chat" && m.domain === "legal",
      );

      if (hasLegalLLM) {
        return "tauri-local";
      }
    }
    // Check Ollama availability
    if (AI_CONFIG.OLLAMA_URL) {
      return "ollama";
    }
    return "openai";
  }
  // Build system prompt based on context
  private buildSystemPrompt(options: GenerationOptions): string {
    let prompt = options.systemPrompt || "";

    if (this.config.legalDomain || options.legalContext) {
      prompt += `
You are a specialized legal AI assistant with expertise in legal document analysis, case law, and legal procedures.
Provide accurate, professional responses based on legal knowledge and cite relevant legal authorities when possible.
Consider jurisdiction-specific laws and regulations. Always clarify if you need more context about the specific jurisdiction.
`;
    }
    return prompt.trim();
  }
  // Build full prompt with context
  private buildFullPrompt(
    prompt: string,
    systemPrompt: string,
    context?: string[],
  ): string {
    let fullPrompt = "";

    if (systemPrompt) {
      fullPrompt += `${systemPrompt}\n\n`;
    }
    if (context && context.length > 0) {
      fullPrompt += `Context:\n${context.join("\n")}\n\n`;
    }
    fullPrompt += `Query: ${prompt}`;

    return fullPrompt;
  }
  // OpenAI embeddings implementation - should be moved to server-side
  private async generateOpenAIEmbeddings(texts: string[]): Promise<number[][]> {
    // Note: This should be called via API endpoint, not directly in client
    const response = await fetch("/api/ai/embedding", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        texts: texts,
        provider: "openai"
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`);
    }
    const data = await response.json();
    return data.data.map((item: any) => item.embedding);
  }
  // OpenAI chat completion implementation
  private async generateOpenAIResponse(
    prompt: string,
    options: GenerationOptions,
  ): Promise<string> {
    // Use server-side API endpoint for OpenAI calls
    const response = await fetch("/api/ai/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages: [{ role: "user", content: prompt }],
        max_tokens: options.maxTokens || 512,
        temperature: options.temperature || 0.7,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`);
    }
    const data = await response.json();
    return data.choices[0]?.message?.content || "No response generated";
  }
  // Ollama implementation
  private async generateOllamaResponse(
    prompt: string,
    options: GenerationOptions,
  ): Promise<string> {
    const ollamaUrl = AI_CONFIG.OLLAMA_URL;

    const response = await fetch(`${ollamaUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: AI_CONFIG.OLLAMA_MODEL,
        prompt,
        stream: false,
        options: {
          temperature: options.temperature || 0.7,
          max_tokens: options.maxTokens || 512,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }
    const data = await response.json();
    return data.response || "No response generated";
  }
  // Simple legal entity extraction (placeholder for NER model)
  private extractLegalEntities(text: string): string[] {
    const entities: string[] = [];
    const patterns = {
      "Case Citations": /\b\d+\s+[A-Z][a-z.]+\s+\d+\b/g,
      Statutes: /\b\d+\s+U\.?S\.?C\.?\s+§?\s*\d+/gi,
      Courts:
        /\b(Supreme Court|Court of Appeals|District Court|Circuit Court)\b/gi,
      "Legal Terms":
        /\b(plaintiff|defendant|appellant|appellee|damages|injunction|summary judgment)\b/gi,
    };

    for (const [type, pattern] of Object.entries(patterns)) {
      const matches = text.match(pattern);
      if (matches) {
        entities.push(...matches.map((match) => `${type}: ${match}`));
      }
    }
    return entities.slice(0, 10); // Limit to top 10
  }
  // Get service status
  getStatus(): {
    initialized: boolean;
    tauriAvailable: boolean;
    currentModels: any;
    config: AIServiceConfig;
  } {
    return {
      initialized: this.isInitialized,
      tauriAvailable: tauriLLM.isAvailable(),
      currentModels: tauriLLM.getCurrentModels(),
      config: this.config,
    };
  }
  // Update configuration
  updateConfig(newConfig: Partial<AIServiceConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }
}
// Export singleton instance
export const aiService = new EnhancedAIService();

// Export for use in other services
export default aiService;
