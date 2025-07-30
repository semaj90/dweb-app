// LLM Endpoint Auto-Detection Service
// Detects vLLM (WSL/Linux) or Ollama (Windows) and returns the healthy endpoint

const VLLM_URL = "http://localhost:8000/v1";
const OLLAMA_URL = "http://localhost:11434/v1";

export async function getHealthyLlmEndpoint(): Promise<string> {
  // Try vLLM first (WSL/Linux)
  try {
    const vllmHealth = await fetch(`${VLLM_URL}/models`, {
      method: "GET",
      signal: AbortSignal.timeout(2000),
    });
    if (vllmHealth.ok) return VLLM_URL;
  } catch {}
  // Fallback to Ollama (Windows)
  try {
    const ollamaHealth = await fetch(`${OLLAMA_URL}/models`, {
      method: "GET",
      signal: AbortSignal.timeout(2000),
    });
    if (ollamaHealth.ok) return OLLAMA_URL;
  } catch {}
  throw new Error("No healthy LLM endpoint detected (vLLM or Ollama)");
}

// Usage example:
// const endpoint = await getHealthyLlmEndpoint();
// fetch(`${endpoint}/chat/completions`, ...)
