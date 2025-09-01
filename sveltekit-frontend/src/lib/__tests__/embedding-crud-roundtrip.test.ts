import { describe, it, expect, beforeAll } from 'vitest';
import { db, legal_documents } from '$lib/server/db';
import { sql } from 'drizzle-orm';
import { randomUUID } from 'crypto';

let available = true;
beforeAll(async () => { try { await db.execute(sql`SELECT 1`); } catch { available = false; } });

function maybe(name: string, fn: () => any) { // @ts-ignore
  return available ? it(name, fn, 15_000) : it.skip(name, () => {}); }

describe('Embedding CRUD Round Trip (simplified)', () => {
  maybe('insert + retrieve legal document', async () => {
    const id = randomUUID();
    await db.insert(legal_documents).values({
      id,
      title: 'Test Contract',
      document_type: 'contract',
      content: 'Sample contract terms and conditions for embedding test.'
    });
    const rows = await db.select().from(legal_documents).where(sql`id = ${id}`).limit(1);
    expect(rows[0]?.title).toBe('Test Contract');
  });
});
