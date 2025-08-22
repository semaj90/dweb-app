import { json } from "@sveltejs/kit";
import { db, documents, embeddings } from "drizzle-orm";
import { sql } from 'drizzle-orm';

export const GET = async () => {
  try {
    const TARGET_DIM = parseInt(import.meta.env.EMBEDDING_DIM || import.meta.env.VECTOR_DIM || '768', 10);

    // Counts
  const docsCountRes = await db.execute(sql`SELECT COUNT(*) AS count FROM documents`) as any;
  const embCountRes = await db.execute(sql`SELECT COUNT(*) AS count FROM legal_embeddings`) as any;

    const documentsCount = Number(docsCountRes.rows?.[0]?.count || 0);
    const embeddingsCount = Number(embCountRes.rows?.[0]?.count || 0);

    // Sample a few embeddings to check length
  const sample = await db.execute(sql`SELECT id, document_id, embedding FROM legal_embeddings LIMIT 25`) as any;
    const anomalies = [] as unknown[];
    let ok = 0;
    for (const row of sample.rows) {
      const vec: number[] = (row as any).embedding;
      if (Array.isArray(vec) && vec.length === TARGET_DIM) ok++; else anomalies.push({ id: (row as any).id, documentId: (row as any).document_id, len: Array.isArray(vec)?vec.length: null });
    }

    return json({
      success: true,
      targetDim: TARGET_DIM,
      counts: { documents: documentsCount, embeddings: embeddingsCount },
      sampleChecked: sample.rows.length,
      sampleOk: ok,
      sampleAnomalies: anomalies,
      anomalyRatio: sample.rows.length ? anomalies.length / sample.rows.length : 0,
      timestamp: new Date().toISOString()
    });
  } catch (err: any) {
    return json({ success: false, error: err.message }, { status: 500 });
  }
};
