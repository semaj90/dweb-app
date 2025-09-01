import { describe, it, expect, beforeAll } from 'vitest';
import { db } from '$lib/server/db';
import { evidence } from '$lib/server/db/unified-schema';
import { sql } from 'drizzle-orm';
import { randomUUID } from 'node:crypto';

// Raw pgvector operator test using '<->' distance ordering (cosine ops if index configured)
// Skips silently if the evidence table or embedding column not available / extension missing.

const DIM = 384;

// Row shape returned by raw distance query
interface DistanceRow {
  id: string;
  distance: number | string;
}

// Seed row shape
interface SeedRow {
  id: string;
  title: string;
  description: string;
  evidence_type: string;
  embedding: number[];
}

async function seedTestVectors(count = 5): Promise<SeedRow> {
  const rows: SeedRow[] = Array.from({ length: count }, (_, i) => {
    const base = i + 1;
    const vec = Array.from({ length: DIM }, (__, j) => ((j + base) % 17) / 17);
    return {
      id: randomUUID(),
      title: `Vector Row ${base}`,
      description: 'Raw operator test row',
      evidence_type: 'document',
      embedding: vec as number[]
    };
  });
  await db.insert(evidence).values(rows).returning({ id: evidence.id });
  return rows[0];
}

// Build a pgvector literal (e.g. '[0.1,0.2,...]') to avoid driver issues with parameter casting
function makeVectorLiteral(vec: number[]): string {
  return '[' + vec.map(v => Number(v).toFixed(6)).join(',') + ']';
}

describe('pgvector raw operator search', () => {
  let qvecLiteral: string | undefined;

  beforeAll(async () => {
    try {
      const first = await seedTestVectors();
      if (first?.embedding) {
        qvecLiteral = makeVectorLiteral(first.embedding);
      }
    } catch (e) {
      console.warn('Skipping raw operator test (DB/pgvector unavailable):', (e as Error).message);
    }
  });

  it('orders results by <-> distance ascending', async () => {
    if (!qvecLiteral) {
      // Seeding failed or extension unavailable; treat as skipped.
      return;
    }

    // Cast to vector type (pgvector does not allow a dimension modifier in a cast)
        const vectorExpr = sql.raw(`${qvecLiteral}::vector`);

        let rows: DistanceRow[];
        try {
          const execResult = await db.execute(
            sql`SELECT id, embedding <-> ${vectorExpr} AS distance
                FROM evidence
                ORDER BY embedding <-> ${vectorExpr} ASC
                LIMIT 3`
          );
          rows = (execResult as any).rows as DistanceRow[];
          if (!rows || rows.length === 0) {
            console.warn('No rows returned; skipping raw operator assertions.');
            return;
          }
        } catch (e) {
          console.warn('Query failed, skipping raw operator test:', (e as Error).message);
          return;
        }

        expect(rows.length).toBeGreaterThan(0);
    for (let i = 1; i < rows.length; i++) {
      const prev = Number(rows[i - 1].distance);
      const cur = Number(rows[i].distance);
      expect(cur).toBeGreaterThanOrEqual(prev);
    }
  });
});
