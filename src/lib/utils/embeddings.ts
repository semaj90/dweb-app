// Shared embedding utilities
// Centralizes serialization and parsing logic for numeric vector embeddings stored as JSON strings

export type EmbeddingInput = number[] | Float32Array | null | undefined;

/**
 * Serialize an embedding array to a compact JSON string suitable for storage in a text column.
 * Falls back to '[]' for nullish inputs. Accepts number[] or Float32Array.
 */
export function serializeEmbedding(vec: EmbeddingInput): string {
  if (!vec) return "[]";
  if (Array.isArray(vec)) return JSON.stringify(vec);
  if (vec instanceof Float32Array) return JSON.stringify(Array.from(vec));
  return "[]";
}

/**
 * Parse a stored JSON string back into number[] safely.
 */
export function parseEmbedding(raw: string | null | undefined): number[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map(Number) : [];
  } catch {
    return [];
  }
}

/**
 * Quick dimension helper without full parse (best effort).
 */
export function embeddingDimensions(raw: string | null | undefined): number {
  if (!raw) return 0;
  try {
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr.length : 0;
  } catch {
    return 0;
  }
}
