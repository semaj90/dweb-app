# Start core services in background and run health checks
Set-StrictMode -Version Latest
$repo = 'C:\Users\james\Desktop\deeds-web\deeds-web-app'
Set-Location $repo
if (-not (Test-Path .\logs)) { New-Item -ItemType Directory -Path .\logs | Out-Null }

# Start SvelteKit (npm run dev) in the sveltekit-frontend folder
$svelteDir = Join-Path $repo 'sveltekit-frontend'
if (Test-Path $svelteDir) {
  try {
    Write-Host "Starting SvelteKit (npm run dev) in $svelteDir"
    $s = Start-Process -FilePath 'npm' -ArgumentList 'run','dev' -WorkingDirectory $svelteDir -RedirectStandardOutput (Join-Path $repo 'logs\sveltekit.log') -RedirectStandardError (Join-Path $repo 'logs\sveltekit.err.log') -PassThru -ErrorAction Stop
    Write-Host "  SvelteKit PID: $($s.Id)"
  } catch {
    Write-Host "  Failed to start SvelteKit: $($_.Exception.Message)"
  }
} else { Write-Host "SvelteKit folder not found: $svelteDir" }

# Start Enhanced RAG executable
$enhanced = Join-Path $repo 'go-microservice\bin\enhanced-rag.exe'
if (Test-Path $enhanced) {
  try {
    Write-Host "Starting Enhanced RAG: $enhanced"
    $r = Start-Process -FilePath $enhanced -RedirectStandardOutput (Join-Path $repo 'logs\enhanced-rag.log') -RedirectStandardError (Join-Path $repo 'logs\enhanced-rag.err.log') -PassThru -ErrorAction Stop
    Write-Host "  enhanced-rag PID: $($r.Id)"
  } catch {
    Write-Host "  Failed to start enhanced-rag: $($_.Exception.Message)"
  }
} else { Write-Host "enhanced-rag.exe not found: $enhanced" }

# Start Context7 MCP server (node)
$ctx = Join-Path $repo 'mcp-servers\context7-server.js'
if (Test-Path $ctx) {
  try {
    Write-Host "Starting Context7 MCP server: $ctx"
    $c = Start-Process -FilePath 'node' -ArgumentList $ctx -RedirectStandardOutput (Join-Path $repo 'logs\context7.log') -RedirectStandardError (Join-Path $repo 'logs\context7.err.log') -PassThru -ErrorAction Stop
    Write-Host "  context7 PID: $($c.Id)"
  } catch {
    Write-Host "  Failed to start context7 server: $($_.Exception.Message)"
  }
} else { Write-Host "context7-server.js not found: $ctx" }

Start-Sleep -Seconds 2

# Health checks (HTTP)
function DoCheck($url, $label) {
  try {
    $out = & curl.exe -sS $url 2>&1
  if ($LASTEXITCODE -eq 0) { Write-Host ("{0}: {1}" -f $label, $out) } else { Write-Host ("{0}: curl failed (exit {1}) - {2}" -f $label, $LASTEXITCODE, $out) }
  } catch {
  Write-Host ("{0}: curl exception - {1}" -f $label, $_.Exception.Message)
  }
}

Write-Host "\nHealth checks:"
DoCheck 'http://localhost:5173/' 'sveltekit'
DoCheck 'http://localhost:8094/health' 'enhanced-rag'
DoCheck 'http://localhost:4000/health' 'context7'

Write-Host 'Done.'
