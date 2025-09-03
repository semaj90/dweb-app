# Database Migrations Review (SvelteKit Frontend)

## Summary
This document consolidates and normalizes the state of SQL/DDL migrations relevant to the SvelteKit application layer. A new OCR enhancement migration was introduced (`003_add_ocr_columns_to_evidence.sql`) but was previously not inside an organized `migrations/` folder at this level. This review ensures a canonical ordering and provides audit references.

## Canonical Migration Chain (Frontend Context)
| Order | File | Purpose |
|-------|------|---------|
| 001 | `src/lib/server/db/migrations/001_init_pgvector.sql` | Enable pgvector extension (initial) |
| 002 | `src/lib/server/db/migrations/002_enhanced_schema_with_qdrant.sql` | Extended schema + vector search preparation |
| 003 | `migrations/003_add_ocr_columns_to_evidence.sql` | Adds OCR metrics columns to evidence table |

> Note: Earlier duplicate/alternate pgvector setup exists at `src/lib/db/migrations/001_setup_pgvector.sql`; prefer the `server/db/migrations` path as canonical; mark the other as legacy.

## OCR Migration (003)
Adds nullable columns (safe forward-add):
- `ocr_confidence` (varchar 32)
- `ocr_word_count` (varchar 32)
- `ocr_processing_time_ms` (varchar 32)
- `ocr_metadata` (jsonb default '{}')

All columns are optional; idempotency is advised via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` pattern.

## Recommended Normalizations
1. Introduce a lightweight migration runner script (PowerShell + Node) to apply only unapplied migrations (tracking table: `app_migrations`).
2. Deprecate `src/lib/db/migrations/001_setup_pgvector.sql` or rename with `_legacy` suffix.
3. Add checksum recording (SHA256) and execution timestamp.
4. Provide rollback notes for reversible operations (OCR columns are additive; rollback would `ALTER TABLE evidence DROP COLUMN ...`).

## Tracking Table (to create if absent)
```sql
CREATE TABLE IF NOT EXISTS app_migrations (
  id serial primary key,
  filename text not null unique,
  checksum text not null,
  applied_at timestamptz not null default now()
);
```

## Example Runner (pseudo)
```bash
for each *.sql in canonical order:
  if not in app_migrations:
    compute checksum
    psql -f file
    insert record
```

## Verification Checklist
- [x] OCR columns present in `schema-postgres.ts`
- [ ] Migration executed on target database (run: `SELECT column_name FROM information_schema.columns WHERE table_name='evidence';`)
- [ ] app_migrations tracking table created
- [ ] Duplicate pgvector migration cleaned or marked legacy
- [ ] Add future: indexes for OCR metrics if query patterns emerge

## Next Actions
- Implement migration runner script (optional request: say "runner" to have it created).
- Confirm production DB alignment before deploying features depending on OCR metrics.
- Extend evidence ingestion pipeline to populate OCR metric fields.
