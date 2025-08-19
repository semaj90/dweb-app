// @ts-nocheck
// Use process.env instead of SvelteKit env for server-side code
// import { env } from "$env/dynamic/private";

export interface EmbeddingProvider {
  name: string;
  endpoint: string;
  model: string;
  dimensions: number;
}
// Configure embedding providers
const providers: Record<string, EmbeddingProvider> = {
  ollama: {
    name: "Ollama",
    endpoint: "http://localhost:11434/api/embeddings",
    model: "nomic-embed-text",
    dimensions: 768,
  },
  openai: {
    name: "OpenAI",
    endpoint: "https://api.openai.com/v1/embeddings",
    model: "text-embedding-ada-002",
    dimensions: 1536,
  },
};

export async function getEmbedding(
  text: string,
  provider: string = "ollama",
): Promise<number[]> {
  const config = providers[provider];
  if (!config) {
    throw new Error(`Unknown embedding provider: ${provider}`);
  }
  try {
    // Clean and truncate text
    const cleanText = text.replace(/[^\w\s.,;:!?-]/g, " ").trim();
    const truncatedText = cleanText.slice(0, 8000); // Safe limit for most models

    if (provider === "ollama") {
      return await getOllamaEmbedding(truncatedText, config);
    } else if (provider === "openai") {
      return await getOpenAIEmbedding(truncatedText, config);
    }
    throw new Error(
      `Embedding method not implemented for provider: ${provider}`,
    );
  } catch (error) {
    console.error(`Error getting embedding from ${provider}:`, error);
    throw new Error(
      `Failed to generate embedding: ${(error as Error).message || String(error)}`,
    );
  }
}
async function getOllamaEmbedding(
  text: string,
  config: EmbeddingProvider,
): Promise<number[]> {
  const response = await fetch(config.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: config.model,
      prompt: text,
    }),
  });

  if (!response.ok) {
    throw new Error(
      `Ollama API error: ${response.status} ${response.statusText}`,
    );
  }
  const data = await response.json();
  return data.embedding;
}
async function getOpenAIEmbedding(
  text: string,
  config: EmbeddingProvider,
): Promise<number[]> {
  const apiKey = import.meta.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY environment variable not set");
  }
  const response = await fetch(config.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: config.model,
      input: text,
    }),
  });

  if (!response.ok) {
    throw new Error(
      `OpenAI API error: ${response.status} ${response.statusText}`,
    );
  }
  const data = await response.json();
  return data.data[0].embedding;
}
export async function similaritySearch(
  embedding: number[],
  limit: number = 10,
  threshold: number = 0.7,
): Promise<Array<{ id: string; similarity: number; content: string }>> {
  // This would integrate with your existing vector search
  // For now, return empty array - implement with your Qdrant setup
  return [];
}
export { providers };
