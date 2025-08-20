# PowerShell Environment Variable Examples for Autosolve & Suggestions

Set an environment variable for the current process only:

```powershell
$env:SUGGEST_ENRICH = '1'
node scripts/generate-suggestions-concurrent.mjs
```

Run extended autosolve pipeline with enrichment and dry-run:

```powershell
$env:SUGGEST_ENRICH = '1'
$env:DRY_RUN = '1'
node scripts/check-auto-solve-extended.mjs
```

One-liner (scoped to command session):

```powershell
powershell -NoProfile -Command "$env:SUGGEST_ENRICH='1'; node scripts/generate-suggestions-concurrent.mjs"
```

Clear after use:

```powershell
Remove-Item Env:SUGGEST_ENRICH
Remove-Item Env:DRY_RUN
```

Common flags:
- `SUGGEST_ENRICH=1` Enables file-content enrichment heuristics.
- `DRY_RUN=1` Prevents write operations in apply phase (metrics still recorded).

Metrics file: `.vscode/autosolve-metrics.jsonl` (append-only). Each line: `{ timestamp, baseline:{errors,warnings}, post:{errors,warnings}, delta:{}, dryRun }`.
