// Frontend layer re-exports the canonical server db defined at root to avoid duplicate connections.
import { db as rootDb } from '../../../../src/lib/server/db/index';
export const db = rootDb;
export * from './unified-schema';
export async function dbHealth() { try { await db.execute('SELECT 1'); return { ok: true }; } catch (e: any) { return { ok: false, error: e.message }; } }