// Local LLM Configuration for Desktop and Web App Development
// This file manages local model downloads, setup, and integration

import { dev } from "$app/environment";
import { tauriLLM } from "$lib/services/tauri-llm";
import {
  createDir,
  writeTextFile,
  readTextFile,
  exists,
} from "@tauri-apps/api/fs";
import { join, appLocalDataDir } from "@tauri-apps/api/path";

export interface LocalLLMConfig {
  models: {
    embedding: {
      name: string;
      url: string;
      size: string;
      dimensions: number;
      description: string;
    };
    chat: {
      name: string;
      url: string;
      size: string;
      contextLength: number;
      description: string;
    };
    classification: {
      name: string;
      url: string;
      size: string;
      classes: string[];
      description: string;
    };
  };
  paths: {
    modelsDir: string;
    cacheDir: string;
    configFile: string;
  };
  development: {
    autoDownload: boolean;
    preferLocal: boolean;
    fallbackToCloud: boolean;
  };
}
export const localLLMConfig: LocalLLMConfig = {
  models: {
    embedding: {
      name: "legal-bert-base-uncased",
      url: "https://huggingface.co/nlpaueb/legal-bert-base-uncased",
      size: "440MB",
      dimensions: 768,
      description: "Legal domain BERT model for embedding generation",
    },
    chat: {
      name: "llama-2-7b-chat-legal",
      url: "https://huggingface.co/microsoft/DialoGPT-medium",
      size: "2.8GB",
      contextLength: 2048,
      description: "Legal-focused chat model for conversational AI",
    },
    classification: {
      name: "legal-bert-classification",
      url: "https://huggingface.co/nlpaueb/legal-bert-base-uncased",
      size: "440MB",
      classes: ["contract", "evidence", "statute", "case_law", "regulation"],
      description: "Legal document classification model",
    },
  },
  paths: {
    modelsDir: dev ? "./local-llms" : "%LOCALAPPDATA%/LegalRAG/models",
    cacheDir: dev ? "./llm-cache" : "%LOCALAPPDATA%/LegalRAG/cache",
    configFile: dev
      ? "./llm-config.json"
      : "%LOCALAPPDATA%/LegalRAG/config.json",
  },
  development: {
    autoDownload: true, // Enable automatic download for development
    preferLocal: true,
    fallbackToCloud: true,
  },
};

class LocalLLMManager {
  private isInitialized = false;
  private modelsPath = "";
  private cachePath = "";

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Set up paths based on environment
      if (dev) {
        this.modelsPath = "./local-llms";
        this.cachePath = "./llm-cache";
      } else {
        const appDataDir = await appLocalDataDir();
        this.modelsPath = await join(appDataDir, "LegalRAG", "models");
        this.cachePath = await join(appDataDir, "LegalRAG", "cache");
      }
      // Create directories if they don't exist
      await this.ensureDirectories();

      // Initialize Tauri LLM service
      if (typeof window !== "undefined" && window.__TAURI__) {
        await tauriLLM.initialize();
      }
      this.isInitialized = true;
      console.log("Local LLM Manager initialized");
    } catch (error) {
      console.error("Failed to initialize Local LLM Manager:", error);
    }
  }
  private async ensureDirectories(): Promise<void> {
    try {
      if (typeof window !== "undefined" && window.__TAURI__) {
        // Tauri environment
        const modelsExists = await exists(this.modelsPath);
        const cacheExists = await exists(this.cachePath);

        if (!modelsExists) {
          await createDir(this.modelsPath, { recursive: true });
        }
        if (!cacheExists) {
          await createDir(this.cachePath, { recursive: true });
        }
      } else {
        // Web environment - create directories through Node.js backend
        await fetch("/api/llm/setup-directories", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            modelsPath: this.modelsPath,
            cachePath: this.cachePath,
          }),
        });
      }
    } catch (error) {
      console.error("Failed to create directories:", error);
    }
  }
  async checkModelAvailability(): Promise<{
    embedding: boolean;
    chat: boolean;
    classification: boolean;
  }> {
    const availability = {
      embedding: false,
      chat: false,
      classification: false,
    };

    try {
      if (typeof window !== "undefined" && window.__TAURI__) {
        // Check through Tauri
        const models = await tauriLLM.getAvailableModels();
        availability.embedding = models.some((m) => m.type === "embedding");
        availability.chat = models.some((m) => m.type === "chat");
        availability.classification = models.some(
          (m) => m.type === "classification",
        );
      } else {
        // Check through web API
        const response = await fetch("/api/llm/models/status");
        const data = await response.json();
        Object.assign(availability, data.availability);
      }
    } catch (error) {
      console.error("Failed to check model availability:", error);
    }
    return availability;
  }
  async downloadModel(
    modelType: "embedding" | "chat" | "classification",
  ): Promise<boolean> {
    try {
      const model = localLLMConfig.models[modelType];

      if (typeof window !== "undefined" && window.__TAURI__) {
        // Download through Tauri
        return await this.downloadModelTauri(model);
      } else {
        // Download through web API
        return await this.downloadModelWeb(model);
      }
    } catch (error) {
      console.error(`Failed to download ${modelType} model:`, error);
      return false;
    }
  }
  private async downloadModelTauri(model: any): Promise<boolean> {
    // This would integrate with Tauri's file download capabilities
    // For now, we'll provide instructions for manual download
    console.log(`Please download ${model.name} from ${model.url}`);
    console.log(`Place the model files in: ${this.modelsPath}/${model.name}/`);
    return false;
  }
  private async downloadModelWeb(model: any): Promise<boolean> {
    const response = await fetch("/api/llm/models/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        modelName: model.name,
        modelUrl: model.url,
        destination: this.modelsPath,
      }),
    });

    return response.ok;
  }
  async getOptimalProvider(
    queryType: "embedding" | "chat" | "classification",
  ): Promise<"local" | "cloud"> {
    if (!localLLMConfig.development.preferLocal) {
      return "cloud";
    }
    const availability = await this.checkModelAvailability();

    if (availability[queryType]) {
      return "local";
    }
    return localLLMConfig.development.fallbackToCloud ? "cloud" : "local";
  }
  async getSystemInfo(): Promise<{
    isDesktop: boolean;
    hasGPU: boolean;
    availableMemory: number;
    supportedModels: string[];
  }> {
    const systemInfo = {
      isDesktop: typeof window !== "undefined" && !!window.__TAURI__,
      hasGPU: false,
      availableMemory: 0,
      supportedModels: [] as string[],
    };

    try {
      if (systemInfo.isDesktop) {
        // Get system info through Tauri
        const info = await fetch("http://localhost:3000/api/system/info").then(
          (r) => r.json(),
        );
        Object.assign(systemInfo, info);
      } else {
        // Web environment - limited system info
        systemInfo.availableMemory =
          (navigator as any).deviceMemory * 1024 || 4096; // MB
        systemInfo.hasGPU = !!(navigator as any).gpu;
      }
    } catch (error) {
      console.error("Failed to get system info:", error);
    }
    return systemInfo;
  }
}
// Singleton instance
export const localLLMManager = new LocalLLMManager();

// Auto-initialize
if (typeof window !== "undefined") {
  localLLMManager.initialize().catch(console.error);
}
export default localLLMManager;
