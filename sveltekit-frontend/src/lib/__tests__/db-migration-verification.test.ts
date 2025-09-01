import { describe, it, expect, beforeAll } from 'vitest';
import { db } from '$lib/server/db';
import { sql } from 'drizzle-orm';

let dbAvailable = true;

beforeAll(async () => {
  try {
    await db.execute(sql`SELECT 1 AS ok`);
  } catch (e) {
    dbAvailable = false;
  }
});

function maybe(name: string, fn: Parameters<typeof it>[1], timeout = 15_000) {
  it(name, async (ctx) => {
    if (!dbAvailable) {
      ctx.skip();
    }
    await fn?.(ctx as any);
  }, timeout);
}

// Normalize drizzle execute result (can be an array or an object with rows)
function asRows(res: any): any[] {
  if (Array.isArray(res)) return res;
  if (res && Array.isArray(res.rows)) return res.rows;
  return [];
}

describe('Database / Migration / pgvector verification', () => {
  maybe('pgvector extension installed', async () => {
    const res = await db.execute(sql`SELECT extname FROM pg_extension WHERE extname = 'vector'`);
    const rows = asRows(res);
    expect(rows.some((r: any) => r.extname === 'vector')).toBe(true);
  });

  maybe('core auth tables exist', async () => {
    const res = await db.execute(sql`
      SELECT tablename FROM pg_catalog.pg_tables
      WHERE schemaname='public' AND tablename IN ('users','sessions','keys')
    `);
    const rows = asRows(res);
    const names = rows.map((r: any) => r.tablename);
    expect(names).toContain('users');
    expect(names).toContain('sessions');
  });

  maybe('evidence table present (legacy snake_case)', async () => {
    const res = await db.execute(sql`
      SELECT tablename FROM pg_catalog.pg_tables
      WHERE schemaname='public' AND tablename = 'evidence'
    `);
    const rows = asRows(res);
    const names = rows.map((r: any) => r.tablename);
    expect(names).toContain('evidence');
  });
});
