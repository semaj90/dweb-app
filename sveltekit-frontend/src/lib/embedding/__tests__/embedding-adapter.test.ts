import { describe, it, expect } from 'vitest';
import { EmbeddingAdapter, cosineSimilarity } from '../../embedding/embedding-adapter';

describe('EmbeddingAdapter', () => {
  it('produces deterministic embeddings when enabled', async () => {
    const adapter = new EmbeddingAdapter('test-model', { dimensions: 16, deterministic: true });
    const a = await adapter.embed('contract breach damages');
    const b = await adapter.embed('contract breach damages');
    expect(Array.from(a.vector)).toEqual(Array.from(b.vector));
  });

  it('cosine similarity is high for identical text', async () => {
    const adapter = new EmbeddingAdapter('test-model', { dimensions: 32, deterministic: true });
    const a = await adapter.embed('liability clause');
    const b = await adapter.embed('liability clause');
    const sim = cosineSimilarity(a.vector, b.vector);
    expect(sim).toBeGreaterThan(0.99);
  });

  it('throws on empty text', async () => {
    const adapter = new EmbeddingAdapter();
    await expect(adapter.embed('')).rejects.toThrow();
  });
});
