/**
 * Lightweight SvelteKit GPU cache integration
 * Provides a simple in-memory cache for GPU-backed resources (metadata only)
 * This file is self-contained and has no external dependencies to avoid type errors.
 */

export interface GPUCacheEntry {
  id: string;
  width: number;
  height: number;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

export class SveltekitGPUCacheIntegration {
  private cache: Map<string, GPUCacheEntry>;

  constructor() {
	this.cache = new Map();
  }

  has(id: string): boolean {
	return this.cache.has(id);
  }

  get(id: string): GPUCacheEntry | undefined {
	return this.cache.get(id);
  }

  set(entry: GPUCacheEntry): void {
	entry.timestamp = Date.now();
	this.cache.set(entry.id, entry);
  }

  delete(id: string): boolean {
	return this.cache.delete(id);
  }

  clear(): void {
	this.cache.clear();
  }

  // Return entries older than msThreshold (useful for trimming)
  findOlderThan(msThreshold: number): GPUCacheEntry[] {
	const cutoff = Date.now() - msThreshold;
	const results: GPUCacheEntry[] = [];
	for (const entry of this.cache.values()) {
	  if (entry.timestamp < cutoff) results.push(entry);
	}
	return results;
  }

  // Simple size getter
  size(): number {
	return this.cache.size;
  }
}

export const gpuCache = new SveltekitGPUCacheIntegration();
export default gpuCache;
