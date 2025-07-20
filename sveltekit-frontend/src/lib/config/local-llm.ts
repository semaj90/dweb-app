// Local LLM Configuration for Ollama and llama.cpp integration
// Manages local model paths and configurations for the legal AI assistant

import { existsSync } from "fs";
import { join } from "path";

// Base paths for local LLM installations (relative to project root)
const projectRoot = process.cwd().includes("sveltekit-frontend")
  ? join(process.cwd(), "..", "..") // If running from sveltekit-frontend
  : process.cwd().includes("web-app")
    ? join(process.cwd(), "..") // If running from web-app
    : process.cwd(); // If running from project root

export const LOCAL_LLM_PATHS = {
  // Ollama installation
  ollama: {
    executable: join(projectRoot, "Ollama", "ollama.exe"),
    app: join(projectRoot, "Ollama", "ollama app.exe"),
    baseUrl: "http://localhost:11434",
    modelsPath:
      process.env.OLLAMA_MODELS ||
      join(process.env.USERPROFILE || "", ".ollama", "models"),
  },

  // llama.cpp installation
  llamaCpp: {
    basePath: join(projectRoot, "llama.cpp"),
    executable: join(projectRoot, "llama.cpp", "llama-server.exe"),
    baseUrl: "http://localhost:8080",
  },

  // Local Gemma3 Q4_K_M model
  gemmaModel: {
    path: join(projectRoot, "gemma3Q4_K_M", "mo16.gguf"),
    name: "gemma3-legal", // Use the custom model we'll create
    ollamaModel: "gemma3-legal", // Custom Ollama model name
    fallbackModel: "gemma3:12b", // Fallback to existing model
    format: "gguf",
    size: "16B", // Estimated from filename mo16.gguf
  },
};

// Model configurations for different use cases
export const MODEL_CONFIGS = {
  // Legal AI assistant configuration
  legal: {
    temperature: 0.7,
    maxTokens: 512,
    topP: 0.9,
    topK: 40,
    repeatPenalty: 1.1,
    systemPrompt: `You are a specialized legal AI assistant. You provide accurate, professional legal analysis and guidance based on the provided context. Always cite your sources and clearly state when information is insufficient. Use appropriate legal terminology and maintain professional standards.`,
    contextWindow: 4096,
  },

  // Fast response configuration
  fast: {
    temperature: 0.3,
    maxTokens: 256,
    topP: 0.8,
    topK: 20,
    repeatPenalty: 1.05,
    systemPrompt: `You are a helpful legal AI assistant. Provide concise, accurate responses to legal questions.`,
    contextWindow: 2048,
  },

  // Detailed analysis configuration
  detailed: {
    temperature: 0.8,
    maxTokens: 1024,
    topP: 0.95,
    topK: 50,
    repeatPenalty: 1.15,
    systemPrompt: `You are an expert legal AI assistant specializing in comprehensive legal analysis. Provide detailed, thorough responses with multiple perspectives and considerations. Always cite relevant legal principles and precedents when available.`,
    contextWindow: 8192,
  },
};

// Check if local installations are available
export function checkLocalInstallations() {
  const status = {
    ollama: {
      available: existsSync(LOCAL_LLM_PATHS.ollama.executable),
      path: LOCAL_LLM_PATHS.ollama.executable,
    },
    llamaCpp: {
      available: existsSync(LOCAL_LLM_PATHS.llamaCpp.basePath),
      path: LOCAL_LLM_PATHS.llamaCpp.basePath,
    },
    gemmaModel: {
      available: existsSync(LOCAL_LLM_PATHS.gemmaModel.path),
      path: LOCAL_LLM_PATHS.gemmaModel.path,
    },
  };

  return status;
}
// Get preferred local LLM provider
export function getPreferredProvider() {
  const installations = checkLocalInstallations();

  // Priority: Ollama > llama.cpp (Ollama is easier to manage)
  if (installations.ollama.available) {
    return "ollama";
  } else if (installations.llamaCpp.available) {
    return "llamacpp";
  }
  return null;
}
// Environment variables for development
export const ENV_CONFIG = {
  // Ollama configuration
  OLLAMA_BASE_URL:
    process.env.OLLAMA_BASE_URL || LOCAL_LLM_PATHS.ollama.baseUrl,
  OLLAMA_MODELS: process.env.OLLAMA_MODELS || LOCAL_LLM_PATHS.ollama.modelsPath,

  // llama.cpp configuration
  LLAMA_CPP_BASE_URL:
    process.env.LLAMA_CPP_BASE_URL || LOCAL_LLM_PATHS.llamaCpp.baseUrl,

  // Model preferences
  DEFAULT_MODEL: process.env.DEFAULT_LLM_MODEL || "gemma2:2b",
  PREFERRED_PROVIDER:
    process.env.PREFERRED_LLM_PROVIDER || getPreferredProvider(),

  // Performance settings
  MAX_CONCURRENT_REQUESTS: parseInt(
    process.env.MAX_CONCURRENT_LLM_REQUESTS || "3",
  ),
  REQUEST_TIMEOUT: parseInt(process.env.LLM_REQUEST_TIMEOUT || "30000"),

  // Development settings
  DEBUG_LLM: process.env.DEBUG_LLM === "true",
  LOG_LLM_REQUESTS: process.env.LOG_LLM_REQUESTS === "true",
};

// Helper function to start local services
export async function startLocalServices() {
  const { spawn } = await import("child_process");
  const status = checkLocalInstallations();

  const services: Array<{
    name: string;
    pid: number | undefined;
    url: string;
  }> = [];

  // Start Ollama if available
  if (status.ollama.available) {
    try {
      console.log("🚀 Starting Ollama service...");
      const ollamaProcess = spawn(
        LOCAL_LLM_PATHS.ollama.executable,
        ["serve"],
        {
          detached: true,
          stdio: "pipe",
        },
      );

      services.push({
        name: "Ollama",
        pid: ollamaProcess.pid,
        url: LOCAL_LLM_PATHS.ollama.baseUrl,
      });

      // Wait a moment for startup
      await new Promise((resolve) => setTimeout(resolve, 3000));
    } catch (error) {
      console.error("Failed to start Ollama:", error);
    }
  }
  return services;
}
// Helper function to load Gemma model into Ollama
export async function loadGemmaModel() {
  const status = checkLocalInstallations();

  if (!status.ollama.available || !status.gemmaModel.available) {
    throw new Error("Ollama or Gemma model not available");
  }
  try {
    // First, create a Modelfile for the GGUF model
    const modelfilePath = join(process.cwd(), "Modelfile.gemma3");
    const { writeFileSync } = await import("fs");

    const modelfileContent = `FROM ${LOCAL_LLM_PATHS.gemmaModel.path}

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Set the template
TEMPLATE """{{ if .System "<|start_header_id|>system<|end_header_id|>

{{ .System "<|eot_id|>{{ end "{{ if .Prompt "<|start_header_id|>user<|end_header_id|>

{{ .Prompt "<|eot_id|>{{ end "<|start_header_id|>assistant<|end_header_id|>

{{ .Response "<|eot_id|>"""

# Set system message
SYSTEM """You are a specialized legal AI assistant. You provide accurate, professional legal analysis and guidance based on the provided context. Always cite your sources and clearly state when information is insufficient. Use appropriate legal terminology and maintain professional standards."""
`;

    writeFileSync(modelfilePath, modelfileContent);

    // Import the model into Ollama
    const { spawn } = await import("child_process");
    const importProcess = spawn(LOCAL_LLM_PATHS.ollama.executable, [
      "create",
      LOCAL_LLM_PATHS.gemmaModel.name,
      "-f",
      modelfilePath,
    ]);

    return new Promise((resolve, reject) => {
      importProcess.on("close", (code) => {
        if (code === 0) {
          console.log(
            `✅ Successfully loaded Gemma model as '${LOCAL_LLM_PATHS.gemmaModel.name}'`,
          );
          resolve(LOCAL_LLM_PATHS.gemmaModel.name);
        } else {
          reject(new Error(`Failed to load Gemma model, exit code: ${code}`));
        }
      });

      importProcess.on("error", reject);
    });
  } catch (error) {
    console.error("Failed to load Gemma model:", error);
    throw error;
  }
}
export default {
  LOCAL_LLM_PATHS,
  MODEL_CONFIGS,
  ENV_CONFIG,
  checkLocalInstallations,
  getPreferredProvider,
  startLocalServices,
  loadGemmaModel,
};
