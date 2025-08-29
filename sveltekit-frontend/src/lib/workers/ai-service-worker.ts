
/**
 * AI Service Worker for Multi-Threading LLM Processing
 * Handles parallel AI tasks across multiple LLM providers
 */

/// <reference lib="webworker" />

import crypto from "crypto";
import type {
  LLMModel,
  AITask,
  AIResponse,
  WorkerMessage,
} from "$lib/types/ai-worker.js";

declare const self: DedicatedWorkerGlobalScope;

export interface AIProviderConfig {
  id: string;
  type: "ollama" | "llamacpp" | "autogen" | "crewai";
  endpoint: string;
  timeout: number;
  retries: number;
}

class AIServiceWorker {
  private providers: Map<string, AIProviderConfig> = new Map();
  private activeRequests: Map<string, AbortController> = new Map();
  private requestQueue: AITask[] = [];
  private maxConcurrentRequests = 3;
  private activeRequestCount = 0;

  constructor() {
    this.initializeProviders();
    this.setupMessageHandlers();
  }

  private initializeProviders() {
    // Initialize supported AI providers
    this.providers.set("ollama", {
      id: "ollama",
      type: "ollama",
      endpoint: "http://localhost:11434",
      timeout: 30000,
      retries: 2,
    });

    this.providers.set("llamacpp", {
      id: "llamacpp",
      type: "llamacpp",
      endpoint: "http://localhost:8000",
      timeout: 15000,
      retries: 3,
    });

    this.providers.set("autogen", {
      id: "autogen",
      type: "autogen",
      endpoint: "http://localhost:8001",
      timeout: 45000,
      retries: 1,
    });

    this.providers.set("crewai", {
      id: "crewai",
      type: "crewai",
      endpoint: "http://localhost:8002",
      timeout: 60000,
      retries: 1,
    });
  }

  private setupMessageHandlers() {
    self.addEventListener(
      "message",
      async (event: MessageEvent<WorkerMessage>) => {
        const { type, payload, taskId } = event.data;

        try {
          switch (type) {
            case "PROCESS_AI_TASK":
              await this.processAITask(payload as AITask, taskId);
              break;
            case "CANCEL_TASK":
              this.cancelTask(taskId);
              break;
            case "GET_STATUS":
              this.sendStatus();
              break;
            case "UPDATE_PROVIDER_CONFIG":
              this.updateProviderConfig(payload);
              break;
            default:
              console.warn("Unknown message type:", type);
          }
        } catch (error: any) {
          this.sendError(taskId, error as Error);
        }
      },
    );
  }

  private async processAITask(task: AITask, taskId: string) {
    // Add to queue if at max capacity
    if (this.activeRequestCount >= this.maxConcurrentRequests) {
      this.requestQueue.push({ ...task, taskId });
      this.sendMessage({
        type: "TASK_QUEUED",
        taskId,
        payload: { position: this.requestQueue.length },
      });
      return;
    }

    this.activeRequestCount++;
    const abortController = new AbortController();
    this.activeRequests.set(taskId, abortController);

    try {
      this.sendMessage({
        type: "TASK_STARTED",
        taskId,
        payload: { providerId: task.providerId },
      });

      const result = await this.executeAITask(task, abortController.signal);

      this.sendMessage({
        type: "TASK_COMPLETED",
        taskId,
        payload: result,
      });
    } catch (error: any) {
      if (error instanceof Error && error.name === "AbortError") {
        this.sendMessage({
          type: "TASK_CANCELLED",
          taskId,
          payload: null,
        });
      } else {
        this.sendError(taskId, error as Error);
      }
    } finally {
      this.activeRequests.delete(taskId);
      this.activeRequestCount--;
      this.processQueue();
    }
  }

  private async executeAITask(
    task: AITask,
    signal: AbortSignal,
  ): Promise<AIResponse> {
    const provider = this.providers.get(task.providerId);
    if (!provider) {
      throw new Error(`Provider ${task.providerId} not found`);
    }

    let lastError: Error | null = null;

    // Retry logic
    for (let attempt = 0; attempt <= provider.retries; attempt++) {
      try {
        const response = await this.callProvider(provider, task, signal);
        return response;
      } catch (error: any) {
        lastError = error as Error;

        if (signal.aborted || attempt === provider.retries) {
          break;
        }

        // Wait before retry with exponential backoff
        await this.delay(Math.pow(2, attempt) * 1000);
      }
    }

    throw lastError || new Error("Unknown error during AI task execution");
  }

  private async callProvider(
    provider: AIProviderConfig,
    task: AITask,
    signal: AbortSignal,
  ): Promise<AIResponse> {
    const timeoutId = setTimeout(() => {
      if (!signal.aborted) {
        signal.dispatchEvent(new Event("timeout"));
      }
    }, provider.timeout);

    try {
      switch (provider.type) {
        case "ollama":
          return await this.callOllama(provider, task, signal);
        case "autogen":
          return await this.callAutoGen(provider, task, signal);
        case "crewai":
          return await this.callCrewAI(provider, task, signal);
        default:
          throw new Error(`Unsupported provider type: ${provider.type}`);
      }
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private async callOllama(
    provider: AIProviderConfig,
    task: AITask,
    signal: AbortSignal,
  ): Promise<AIResponse> {
    const response = await fetch(`${provider.endpoint}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: task.model,
        prompt: task.prompt,
        system: task.systemPrompt,
        stream: false,
        options: {
          temperature: task.temperature || 0.1,
          top_p: task.topP || 0.9,
          top_k: task.topK || 40,
          repeat_penalty: task.repeatPenalty || 1.05,
        },
      }),
      signal,
    });

    if (!response.ok) {
      throw new Error(
        `Ollama API error: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();

    return {
      id: crypto.randomUUID(),
      content: data.response,
      providerId: provider.id,
      model: task.model,
      tokensUsed: data.eval_count || 0,
      responseTime: data.total_duration
        ? Math.round(data.total_duration / 1000000)
        : 0,
      metadata: {
        evalCount: data.eval_count,
        evalDuration: data.eval_duration,
        loadDuration: data.load_duration,
      },
    };
  }

  

  private async callAutoGen(
    provider: AIProviderConfig,
    task: AITask,
    signal: AbortSignal,
  ): Promise<AIResponse> {
    const response = await fetch(`${provider.endpoint}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agents: task.agents || ["assistant"],
        message: task.prompt,
        max_rounds: task.maxRounds || 5,
        context: task.context || {},
      }),
      signal,
    });

    if (!response.ok) {
      throw new Error(
        `AutoGen API error: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();

    return {
      id: crypto.randomUUID(),
      content: data.final_response,
      providerId: provider.id,
      model: "autogen-agents",
      tokensUsed: data.total_tokens || 0,
      responseTime: Date.now() - task.timestamp,
      metadata: {
        rounds: data.rounds,
        agents: data.agent_responses,
        conversationId: data.conversation_id,
      },
    };
  }

  private async callCrewAI(
    provider: AIProviderConfig,
    task: AITask,
    signal: AbortSignal,
  ): Promise<AIResponse> {
    const response = await fetch(`${provider.endpoint}/api/crew/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        crew_id: task.crewId || "legal-analysis-crew",
        task: task.prompt,
        context: task.context || {},
        agents: task.agents || ["researcher", "analyst", "writer"],
      }),
      signal,
    });

    if (!response.ok) {
      throw new Error(
        `CrewAI API error: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();

    return {
      id: crypto.randomUUID(),
      content: data.final_output,
      providerId: provider.id,
      model: "crewai-agents",
      tokensUsed: data.total_tokens || 0,
      responseTime: Date.now() - task.timestamp,
      metadata: {
        taskId: data.task_id,
        agents: data.agent_outputs,
        executionTime: data.execution_time,
      },
    };
  }

  private processQueue() {
    if (
      this.requestQueue.length > 0 &&
      this.activeRequestCount < this.maxConcurrentRequests
    ) {
      const task = this.requestQueue.shift();
      if (task) {
        this.processAITask(task, task.taskId);
      }
    }
  }

  private cancelTask(taskId: string) {
    const controller = this.activeRequests.get(taskId);
    if (controller) {
      controller.abort();
      this.activeRequests.delete(taskId);
    }

    // Remove from queue if present
    this.requestQueue = this.requestQueue.filter(
      (task) => task.taskId !== taskId,
    );
  }

  private updateProviderConfig(config: Partial<AIProviderConfig>) {
    if (config.id && this.providers.has(config.id)) {
      const existing = this.providers.get(config.id)!;
      this.providers.set(config.id, { ...existing, ...config });
    }
  }

  private sendStatus() {
    this.sendMessage({
      type: "STATUS_UPDATE",
      taskId: "status",
      payload: {
        activeRequests: this.activeRequestCount,
        queueLength: this.requestQueue.length,
        providers: Array.from(this.providers.values()),
        maxConcurrent: this.maxConcurrentRequests,
      },
    });
  }

  private sendMessage(message: WorkerMessage) {
    self.postMessage(message);
  }

  private sendError(taskId: string, error: Error) {
    this.sendMessage({
      type: "TASK_ERROR",
      taskId,
      payload: {
        name: error.name,
        message: error.message,
        stack: error.stack,
      },
    });
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// Initialize the worker
new AIServiceWorker();

export {};
