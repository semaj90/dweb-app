
// PgVector-backed implementation of EmbeddingRepository.
import { db } from '$lib/db/drizzle';
import { documentChunks } from '$lib/db/schema-postgres';
import { sql } from 'drizzle-orm';
import { splitText } from './text-splitter';
import { getEmbedding } from '$lib/ai/embedding-service';
import type { EmbeddingRepository, IngestionJobRequest, SimilarityQueryOptions, SimilarityResult, IngestionJobStatus } from './embedding-repository';
import { enqueue, processNext as queueProcessNext, getStatus } from './ingestion-queue';

const DEFAULT_MODEL = 'nomic-embed-text';

async function embedContent(text: string, model: string) {
  const emb = await getEmbedding(text, { model });
  return emb?.embedding || emb || [];
}

async function enqueueIngestion(job: IngestionJobRequest): Promise<IngestionJobStatus> {
  // Normalize optional overrides into payload (queue just stores object)
  return enqueue(job);
}

async function processNextJob(): Promise<IngestionJobStatus | null> {
  return queueProcessNext(async (payload, update) => {
    const { evidenceId, textContent, model = DEFAULT_MODEL, chunkSize, chunkOverlap } = payload;
    if (!textContent) throw new Error('Missing textContent');
    const chunks = splitText(textContent, { maxWords: chunkSize || 220, overlap: chunkOverlap ?? 30 });
    update({ totalChunks: chunks.length });
    let processed = 0;
    for (const { content, index } of chunks) {
      const embedding = await embedContent(content, model);
      await db.insert(documentChunks).values({
        documentId: evidenceId, // reuse evidenceId as document linkage for now
        documentType: 'evidence',
        chunkIndex: index,
        content,
        embedding
      });
      processed++;
      update({ processedChunks: processed });
    }
  });
}

function getJobStatus(jobId: string) {
  return getStatus(jobId);
}

async function querySimilar(query: string, options: SimilarityQueryOptions = {}): Promise<SimilarityResult[]> {
  const model = options.model || DEFAULT_MODEL;
  const queryEmbedding = await embedContent(query, model);
  const limit = options.limit || 8;
  const rows = await db.execute(sql`SELECT id, document_id, document_type, chunk_index, content, embedding <=> ${queryEmbedding} AS distance
                                     FROM document_chunks
                                     ORDER BY embedding <=> ${queryEmbedding}
                                     LIMIT ${limit}`);
  return rows.rows.map(r => ({
    id: r.id,
    documentId: r.document_id,
    documentType: r.document_type,
    chunkIndex: r.chunk_index,
    content: r.content,
    score: 1 - Number(r.distance)
  }));
}

export const pgvectorEmbeddingRepository: EmbeddingRepository = {
  enqueueIngestion,
  processNextJob,
  getJobStatus: async (jobId: string) => getJobStatus(jobId) || null,
  querySimilar
};

// Named exports (optional direct usage)
export { enqueueIngestion, processNextJob, getJobStatus, querySimilar };
