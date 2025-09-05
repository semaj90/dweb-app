$files = @(
  'src/routes/api/analyze/+server.ts',
  'src/lib/stores/ai.ts',
  'src/lib/services/indexeddb.ts',
  'src/lib/services/context7Service.ts',
  'src/lib/services/inlineSuggestionService.ts',
  'src/lib/services/ocr-processor.ts',
  'src/lib/services/enhancedRAG.ts',
  'src/lib/services/nomic-embedding-service.ts',
  'src/lib/stores/avatarStore.ts',
  'src/lib/stores/chat-store.ts'
)
$base = Get-Location
$destRoot = Join-Path $base 'svelte-check-temp'
foreach ($f in $files) {
  $src = Join-Path $base $f
  if (Test-Path $src) {
    $dest = Join-Path $destRoot $f
    New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
    Copy-Item $src $dest -Force
    Write-Output "COPIED: $f"
  } else {
    Write-Output "MISSING: $f"
  }
}
Write-Output 'COPY_STEP_DONE'
