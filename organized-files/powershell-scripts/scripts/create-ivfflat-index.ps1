# Creates an IVFFLAT index for a pgvector column based on env/params
param(
  [string]$ConnectionString = $env:PG_CONN_STRING,
  [string]$Table = $env:VECTOR_TABLE,
  [string]$Column = $env:VECTOR_COLUMN,
  [ValidateSet('cosine','l2','ip')][string]$Metric = $env:DISTANCE_METRIC,
  [int]$Lists = 100
)

function Write-Info($msg){ Write-Host "[ivfflat] $msg" -ForegroundColor Cyan }
function Write-Warn($msg){ Write-Host "[ivfflat] $msg" -ForegroundColor Yellow }
function Write-Err($msg){ Write-Host "[ivfflat] $msg" -ForegroundColor Red }

if (-not $ConnectionString) { Write-Err "PG connection string not provided. Set PG_CONN_STRING or pass -ConnectionString"; exit 1 }
if (-not $Table) { $Table = 'documents' }
if (-not $Column) { $Column = 'embedding' }
if (-not $Metric) { $Metric = 'cosine' }

$opclass = switch ($Metric) {
  'l2' { 'vector_l2_ops' }
  'ip' { 'vector_ip_ops' }
  default { 'vector_cosine_ops' }
}

$safeTable = $Table -replace '[^a-zA-Z0-9_]', '_'
$safeColumn = $Column -replace '[^a-zA-Z0-9_]', '_'
$indexName = "idx_${safeTable}_${safeColumn}_ivfflat"

$sql = @"
CREATE EXTENSION IF NOT EXISTS vector;
-- Create IVFFLAT index (adjust lists for your data size: ~ #rows / 1000 as a starting point)
CREATE INDEX CONCURRENTLY IF NOT EXISTS $indexName
  ON $Table USING ivfflat ($Column $opclass)
  WITH (lists = $Lists);

-- Optional: analyze to update stats
ANALYZE $Table;
"@

Write-Info "Target: $Table.$Column (metric=$Metric opclass=$opclass lists=$Lists)"
Write-Info "Index:  $indexName"

$psql = Get-Command psql -ErrorAction SilentlyContinue
if ($null -eq $psql) {
  $outFile = Join-Path (Split-Path $PSCommandPath -Parent) "..\sql\ivfflat-index.sql"
  New-Item -Force -ItemType Directory -Path (Split-Path $outFile) | Out-Null
  $sql | Set-Content -NoNewline -Path $outFile
  Write-Warn "psql not found in PATH. Wrote SQL to: $outFile"
  Write-Info "Run it manually with your Postgres client."
  exit 0
}

Write-Info "Executing via psql..."
"$sql" | psql "$ConnectionString"
if ($LASTEXITCODE -ne 0) {
  Write-Err "psql exited with code $LASTEXITCODE"
  exit $LASTEXITCODE
}
Write-Info "IVFFLAT index ensured successfully."
