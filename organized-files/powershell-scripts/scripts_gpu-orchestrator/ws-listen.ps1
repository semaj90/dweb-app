param(
  [int]$Port = $(if($env:GPU_ORCHESTRATOR_PORT){$env:GPU_ORCHESTRATOR_PORT}else{8095})
)

$uri = "ws://localhost:$Port/ws"
Write-Host "🔌 Connecting to $uri..." -ForegroundColor Cyan
Add-Type -AssemblyName System.Net.WebSockets
$ws = [System.Net.WebSockets.ClientWebSocket]::new()
try {
  $ws.ConnectAsync([Uri]$uri, [Threading.CancellationToken]::None).Wait()
  Write-Host "✅ Connected. Listening for events (Ctrl+C to stop)" -ForegroundColor Green
  $buffer = New-Object byte[] 8192
  while($ws.State -eq [System.Net.WebSockets.WebSocketState]::Open) {
    $seg = [ArraySegment[byte]]::new($buffer,0,$buffer.Length)
    $result = $ws.ReceiveAsync($seg, [Threading.CancellationToken]::None).Result
    if($result.Count -gt 0){
      $msg = [Text.Encoding]::UTF8.GetString($buffer,0,$result.Count)
      Write-Host ("📥 " + $msg)
    }
    if($result.MessageType -eq [System.Net.WebSockets.WebSocketMessageType]::Close){ break }
  }
} catch {
  Write-Host "❌ WebSocket error: $($_.Exception.Message)" -ForegroundColor Red
} finally {
  if($ws){ $ws.Dispose() }
  Write-Host "🔌 Disconnected" -ForegroundColor Yellow
}
