// LLM Endpoint Service
// Returns the Ollama endpoint

const OLLAMA_URL = "http://localhost:11434/v1";

export async function getHealthyLlmEndpoint(): Promise<string> {
  // Try Ollama
  try {
    const ollamaHealth = await fetch(`${OLLAMA_URL}/models`, {
      method: "GET",
      signal: AbortSignal.timeout(2000),
    });
    if (ollamaHealth.ok) return OLLAMA_URL;
  } catch {}
  throw new Error("No healthy LLM endpoint detected (Ollama)");
}

// Usage example:
// const endpoint = await getHealthyLlmEndpoint();
// fetch(`${endpoint}/chat/completions`, ...)
