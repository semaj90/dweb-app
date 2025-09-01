import { describe, it, expect, beforeAll } from 'vitest';
import { db, users } from '$lib/server/db';
import { createUserSession, validateSession, invalidateSession } from '$lib/server/lucia';
import { sql } from 'drizzle-orm';
import { randomUUID } from 'crypto';

let dbUp = true;
beforeAll(async () => { try { await db.execute(sql`SELECT 1`); } catch { dbUp = false; } });

function maybe(name: string, fn: () => any) { // @ts-ignore
  return dbUp ? it(name, fn, 20_000) : it.skip(name, () => {}); }

describe('Auth Session Flow', () => {
  const email = `test_${Date.now()}@example.com`;
  let userId: string; let sessionId: string;
  maybe('create user + session then validate and invalidate', async () => {
    userId = randomUUID();
    await db.insert(users).values({ id: userId, email, role: 'user' });
    const { sessionId: sid } = await createUserSession(userId, 1);
    sessionId = sid;
    const validated = await validateSession(sessionId);
    expect(validated.user?.email).toBe(email);
    await invalidateSession(sessionId);
    const after = await validateSession(sessionId);
    expect(after.user).toBeNull();
  });
});
