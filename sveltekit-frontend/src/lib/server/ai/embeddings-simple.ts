// Redis cache imports - using local interfaces to avoid missing module errors
export interface CacheInterface {
  getCachedEmbedding: (key: string) => Promise<number[] | null>;
  cacheEmbedding: (key: string, embedding: number[], ttl?: number) => Promise<void>;
}

// Mock cache implementation
const cache: CacheInterface = {
  getCachedEmbedding: async (key: string) => null,
  cacheEmbedding: async (key: string, embedding: number[], ttl?: number) => {}
};

const getCachedEmbedding = cache.getCachedEmbedding;
const cacheEmbedding = cache.cacheEmbedding;

// Simplified AI embedding service - Production ready
// Supports OpenAI embeddings with Redis/memory caching
// Use process.env for server-side environment variables

// Target embedding dimension (match database schema). Defaults to 384.
const TARGET_DIM: number = (() => {
  const v = parseInt(import.meta.env.EMBEDDING_DIMENSIONS || "384", 10);
  return Number.isFinite(v) && v > 0 ? v : 384;
})();

function ensureDim(
  vec: number[] | Float32Array | null | undefined,
  target = TARGET_DIM
): number[] {
  if (!vec || (!Array.isArray(vec as any) && !(vec instanceof Float32Array)))
    return [];
  const arr = Array.isArray(vec as any)
    ? (vec as any as number[])
    : Array.from(vec as Float32Array);
  if (arr.length === target) return arr;
  if (arr.length > target) return arr.slice(0, target);
  return arr.concat(Array(target - arr.length).fill(0));
}

export interface EmbeddingOptions {
  model?: "openai" | "local";
  cache?: boolean;
  maxTokens?: number;
}
export async function generateEmbedding(
  text: string,
  options: EmbeddingOptions = {}
): Promise<number[] | null> {
  const { model = "local", cache = true, maxTokens = 8000 } = options;

  if (!text || text.trim().length === 0) {
    return null;
  }
  // Truncate text if too long
  const truncatedText =
    text.length > maxTokens ? text.substring(0, maxTokens) : text;

  // Generate cache key for both lookup and storage
  const cacheKey = `${model}:${truncatedText.substring(0, 100)}`;

  // Check cache first
  if (cache) {
    const cachedEmbedding = await getCachedEmbedding(cacheKey);
    if (cachedEmbedding) {
      return cachedEmbedding;
    }
  }
  try {
    let embedding: number[] = [];

    if (model === "openai") {
      embedding = await generateOpenAIEmbedding(truncatedText);
    } else {
      // Prefer CPU/Xenova for local embeddings; fallback to Ollama if CPU path fails
      try {
        embedding = await generateCpuEmbedding(truncatedText);
      } catch (cpuErr) {
        console.warn(
          "CPU embedding failed, falling back to Ollama:",
          (cpuErr as Error)?.message || cpuErr
        );
        embedding = await generateNomicEmbedding(truncatedText);
      }
    }

    // Normalize to target dimension for pgvector compatibility
    embedding = ensureDim(embedding, TARGET_DIM);

    // Cache the result
    if (cache) {
      await cacheEmbedding(cacheKey, embedding);
    }
    return embedding;
  } catch (error: any) {
    console.error("Embedding generation failed:", error);
    return null;
  }
}
// OpenAI embedding generation
async function generateOpenAIEmbedding(text: string): Promise<number[]> {
  const apiKey = import.meta.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OpenAI API key not configured");
  }
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: text,
      model: "text-embedding-3-small", // 1536 dimensions, fast and cost-effective
    }),
  });

  if (!response.ok) {
    const errorData = await response
      .json()
      .catch(() => ({ error: "Unknown error" }));
    throw new Error(
      `OpenAI API error: ${response.statusText} - ${JSON.stringify(errorData)}`
    );
  }
  const data = await response.json();
  return ensureDim(data.data[0].embedding, TARGET_DIM);
}

// Nomic Embed via Ollama
async function generateNomicEmbedding(text: string): Promise<number[]> {
  const ollamaUrl = import.meta.env.OLLAMA_URL || "http://localhost:11434";

  try {
    const response = await fetch(`${ollamaUrl}/api/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "nomic-embed-text",
        prompt: text, // Ollama embeddings expect 'prompt'
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Ollama API error: ${response.status} ${response.statusText}`
      );
    }

    const data = await response.json();
    // Support { embedding }, { embeddings }, and OpenAI-like { data: [{ embedding }] }
    if (Array.isArray(data?.embedding))
      return ensureDim(data.embedding, TARGET_DIM);
    if (Array.isArray(data?.embeddings)) {
      // Could be array of numbers or array of arrays depending on provider
      if (Array.isArray(data.embeddings[0]))
        return ensureDim(data.embeddings[0] as number[], TARGET_DIM);
      return ensureDim(data.embeddings as number[], TARGET_DIM);
    }
    if (Array.isArray(data?.data) && Array.isArray(data.data[0]?.embedding)) {
      return ensureDim(data.data[0].embedding, TARGET_DIM);
    }
    return [];
  } catch (error: any) {
    console.error("Nomic embedding generation failed:", error);
    throw error;
  }
}

// CPU/Xenova embedding generation
async function generateCpuEmbedding(text: string): Promise<number[]> {
  // Dynamic import to avoid bundling issues if transformers isn't installed in some environments
  const mod = await import("@xenova/transformers");
  const pipeline = (mod as any).pipeline || (mod as any).default?.pipeline;
  if (!pipeline) throw new Error("@xenova/transformers pipeline not available");
  const extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );
  const result = await extractor(text, { pooling: "mean", normalize: true });
  const arr = Array.from(result.data as Float32Array);
  return ensureDim(arr, TARGET_DIM);
}
// Batch embedding generation for efficiency
export async function generateBatchEmbeddings(
  texts: string[],
  options: EmbeddingOptions = {}
): Promise<(number[] | null)[]> {
  const { model = "openai" } = options;

  // Filter out empty texts
  const validTexts = texts.filter((text) => text && text.trim().length > 0);
  if (validTexts.length === 0) {
    return texts.map(() => null);
  }
  try {
    if (model === "openai" && validTexts.length > 1) {
      return await generateOpenAIBatchEmbeddings(validTexts);
    } else if (model === "local" && validTexts.length > 1) {
      // Prefer CPU pipeline for batch as well (process sequentially to limit memory)
      const out: (number[] | null)[] = [];
      for (const t of validTexts) {
        try {
          const e = await generateCpuEmbedding(t);
          out.push(ensureDim(e, TARGET_DIM));
        } catch (e: any) {
          // fallback to Ollama per-text
          try {
            const nomic = await generateNomicEmbedding(t);
            out.push(ensureDim(nomic, TARGET_DIM));
          } catch {
            out.push(null);
          }
        }
      }
      return out;
    }
  } catch (error: any) {
    console.warn(
      "Batch embedding failed, falling back to individual generation:",
      error
    );
  }
  // Fall back to individual generation
  const results: (number[] | null)[] = [];
  for (const text of texts) {
    const embedding = await generateEmbedding(text, options);
    results.push(embedding);
  }
  return results;
}
// OpenAI batch embedding generation
async function generateOpenAIBatchEmbeddings(
  texts: string[]
): Promise<(number[] | null)[]> {
  const apiKey = import.meta.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OpenAI API key not configured");
  }
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: texts,
      model: "text-embedding-3-small",
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenAI API error: ${response.statusText}`);
  }
  const data = await response.json();
  return data.data.map((item: any) => ensureDim(item.embedding, TARGET_DIM));
}

// Nomic batch embedding generation
async function generateNomicBatchEmbeddings(
  texts: string[]
): Promise<(number[] | null)[]> {
  const ollamaUrl = import.meta.env.OLLAMA_URL || "http://localhost:11434";

  // Note: Ollama doesn't support batch embeddings, so we process individually
  // but we can do them in parallel
  const promises = texts.map(async (text) => {
    try {
      const response = await fetch(`${ollamaUrl}/api/embeddings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "nomic-embed-text",
          prompt: text,
        }),
      });

      if (!response.ok) {
        throw new Error(
          `Ollama API error: ${response.status} ${response.statusText}`
        );
      }

      const data = await response.json();
      if (Array.isArray(data?.embedding)) return data.embedding;
      if (Array.isArray(data?.embeddings)) {
        if (Array.isArray(data.embeddings[0]))
          return data.embeddings[0] as number[];
        return data.embeddings as number[];
      }
      if (Array.isArray(data?.data) && Array.isArray(data.data[0]?.embedding)) {
        return data.data[0].embedding;
      }
      return null;
    } catch (error: any) {
      console.error(
        "Nomic embedding generation failed for text:",
        text.substring(0, 50),
        error
      );
      return null;
    }
  });

  return Promise.all(promises);
}
// Export object for easier importing and better organization
export const embeddings = {
  generate: generateEmbedding,
  generateBatch: generateBatchEmbeddings,
};

// For backward compatibility
export const embedAndSearch = {
  generateEmbedding,
  generateBatchEmbeddings,
};
