// Simple embedding adapter abstraction to enable unit testing & future swap
export interface EmbeddingAdapterOptions {
  dimensions?: number;
  deterministic?: boolean; // for tests
}

export interface EmbeddingResult {
  vector: Float32Array;
  model: string;
  input: string;
}

export class EmbeddingAdapter {
  private dimensions: number;
  private deterministic: boolean;
  private model: string;

  constructor(model = 'mock-embedding-model', opts: EmbeddingAdapterOptions = {}) {
    this.dimensions = opts.dimensions ?? 64;
    this.deterministic = !!opts.deterministic;
    this.model = model;
  }

  async embed(text: string): Promise<EmbeddingResult> {
    if (!text || !text.trim()) throw new Error('Text required');
    const vector = new Float32Array(this.dimensions);
    if (this.deterministic) {
      let hash = 0;
      for (let i = 0; i < text.length; i++) hash = (hash * 31 + text.charCodeAt(i)) >>> 0;
      for (let i = 0; i < this.dimensions; i++) {
        const v = Math.sin(hash + i) * 10000;
        vector[i] = (v - Math.floor(v));
      }
    } else {
      for (let i = 0; i < this.dimensions; i++) vector[i] = Math.random();
    }
    return { vector, model: this.model, input: text };
  }
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error('Vector length mismatch');
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
