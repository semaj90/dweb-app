import { Pool } from 'pg';
const pool = new Pool({ connectionString: process.env.PG_CONN });

/** Map temperature [0..1] -> distance threshold for pgvector (<-> returns smaller = closer)
 *  Lower temperature -> tighter radius (more precise)
 *  Higher temperature -> wider radius (more recall)
 */
function tempToPgRadius(temp = 0.3) {
  // tune these constants to your embedding scale; these are conservative defaults
  const minRadius = 0.10; // very tight neighborhood
  const maxRadius = 0.50; // very wide
  const t = Math.max(0, Math.min(1, temp));
  return minRadius + (maxRadius - minRadius) * t;
}

/** Query pgvector with temperature-aware radius */
export async function queryPgvector(vec: number[], limit = 200, temperature = 0.3) {
  const radius = tempToPgRadius(temperature);

  // Note: `embedding <-> $1` is distance (smaller=closer). We filter by radius and sort.
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

  // return items shaped for orchestrator
  return {
    ann: rows.map((r: unknown) => ({
      id: String(r.id),
      text: r.text,
      metadata: r.metadata ?? {},
      embedding: r.embedding
    }))
  };
}

// Legacy export for backward compatibility
export { queryPgvector as default };
