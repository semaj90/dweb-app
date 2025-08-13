param(
  [ValidateSet('chat','embeddings')][string]$Mode = 'chat',
  [string]$Model = 'gemma3-legal',
  [string]$Prompt = 'test',
  [switch]$Stream,
  [string]$OutFile = '.\logs\ollama-response.json'
)

function Ensure-Dir($path) {
  $dir = Split-Path -Parent $path
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
}

$effectiveModel = if ($Mode -eq 'embeddings' -and $Model -eq 'gemma3-legal') { 'nomic-embed-text' } else { $Model }

$body = switch ($Mode) {
  'chat' {
    @{ model = $effectiveModel; stream = [bool]$Stream; messages = @(@{ role='user'; content=$Prompt }) }
  }
  'embeddings' {
  @{ model = $effectiveModel; input = $Prompt }
  }
}

$url = if ($Mode -eq 'chat') { 'http://localhost:11434/api/chat' } else { 'http://localhost:11434/api/embeddings' }

try {
  $resp = Invoke-RestMethod -Uri $url -Method Post -Body ($body | ConvertTo-Json -Depth 5) -ContentType 'application/json' -TimeoutSec 60
  Ensure-Dir $OutFile
  $resp | ConvertTo-Json -Depth 10 | Set-Content -Path $OutFile -Encoding UTF8
  Write-Host "Saved Ollama $Mode response to $OutFile"
} catch {
  Write-Host "Ollama request failed: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}
