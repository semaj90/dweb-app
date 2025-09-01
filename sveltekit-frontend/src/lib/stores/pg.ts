/// <reference types="vite/client" />
import { Pool } from 'pg';

const pool = new Pool({ connectionString: import.meta.env.PG_CONN });

function tempToPgRadius(temp = 0.3) {
  const minRadius = 0.1; // tight
  const maxRadius = 0.5; // wide
  const t = Math.max(0, Math.min(1, temp));
  return minRadius + (maxRadius - minRadius) * t;
}

export async function queryPgvector(vec: number[], limit = 200, temperature = 0.3): Promise<any> {
  const radius = tempToPgRadius(temperature);
  const rows = (
    await pool.query(
      `SELECT id, text, metadata, embedding
       FROM chunks
       WHERE embedding <-> $1 < $2
       ORDER BY embedding <-> $1
       LIMIT $3`,
      [vec, radius, limit]
    )
  ).rows;
  return {
    ann: rows.map((r: any) => ({
      id: String(r.id),
      text: r.text,
      metadata: r.metadata ?? {},
      embedding: r.embedding,
    })),
  };
}
