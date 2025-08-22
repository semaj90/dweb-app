import type { ChatMessage, RAGContext } from "$lib/types/ai-chat";

export interface RankedChunk {
  id: string;
  text: string;
  score: number;
  documentId?: string;
  sectionId?: string;
}

export interface IntentPoint {
  x: number;
  y: number;
  intent: string;
}

export interface RAGInputs {
  context: RAGContext | undefined | null;
  history: ChatMessage[];
  embeddings: (text: string) => Promise<number[]>; // embedding fn (e.g., Ollama endpoint)
  search: (queryVec: number[], limit: number) => Promise<RankedChunk[]>; // pgvector/Qdrant
}

export async function buildIntentAwareRetrieval(input: RAGInputs) {
  const { context, history, embeddings, search } = input;
  const recent = history.slice(-20); // light window
  const lastUser = [...recent].reverse().find((m) => m.role === "user");
  const queryText = lastUser?.content || "";

  if (!queryText.trim()) {
    return { chunks: [], confidence: 0, intent: "unknown" };
  }

  try {
    // Get embeddings for the query
    const queryVector = await embeddings(queryText);
    
    // Search for relevant chunks
    const chunks = await search(queryVector, 10);
    
    // Calculate confidence based on scores
    const avgScore = chunks.length > 0 
      ? chunks.reduce((sum, chunk) => sum + chunk.score, 0) / chunks.length
      : 0;
    
    const confidence = Math.min(avgScore * 100, 100); // Convert to percentage
    
    // Simple intent detection based on query keywords
    const intent = detectIntent(queryText);
    
    return {
      chunks,
      confidence,
      intent,
      queryText
    };
  } catch (error) {
    console.error("Error in buildIntentAwareRetrieval:", error);
    return { chunks: [], confidence: 0, intent: "error" };
  }
}

function detectIntent(queryText: string): string {
  const query = queryText.toLowerCase();
  
  if (query.includes("legal") || query.includes("case") || query.includes("law")) {
    return "legal-research";
  }
  if (query.includes("evidence") || query.includes("document")) {
    return "document-analysis";
  }
  if (query.includes("search") || query.includes("find")) {
    return "search";
  }
  if (query.includes("analyze") || query.includes("summary")) {
    return "analysis";
  }
  
  return "general-inquiry";
}

export function normalize(vector: number[]): number[] {
  const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  return magnitude > 0 ? vector.map(val => val / magnitude) : vector;
}

export class SOMGrid {
  private grid: IntentPoint[][];
  private width: number;
  private height: number;

  constructor(width: number = 10, height: number = 10) {
    this.width = width;
    this.height = height;
    this.grid = [];
    
    // Initialize grid
    for (let i = 0; i < height; i++) {
      this.grid[i] = [];
      for (let j = 0; j < width; j++) {
        this.grid[i][j] = {
          x: j,
          y: i,
          intent: "unknown"
        };
      }
    }
  }

  findBestMatch(vector: number[]): IntentPoint {
    // Simple implementation - return center point
    return this.grid[Math.floor(this.height / 2)][Math.floor(this.width / 2)];
  }

  train(vectors: number[][], intents: string[]): void {
    // Placeholder for SOM training algorithm
    // In a real implementation, this would update the grid weights
    console.log("SOM training not implemented yet");
  }
}