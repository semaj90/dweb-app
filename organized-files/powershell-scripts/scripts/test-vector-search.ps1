Param(
  [Parameter(Mandatory=$false)][string]$Query = "agreement",
  [Parameter(Mandatory=$false)][int]$Limit = 5,
  [Parameter(Mandatory=$false)][string]$Url = "http://localhost:5173/api/ai/vector-search"
)

$body = @{ query = $Query; limit = $Limit } | ConvertTo-Json -Compress
try {
  $resp = Invoke-RestMethod -Method Post -Uri $Url -ContentType 'application/json' -Body $body -TimeoutSec 15
  $resp | ConvertTo-Json -Depth 8
} catch {
  Write-Error $_.Exception.Message
  if ($_.Exception.Response -and $_.Exception.Response.GetResponseStream) {
    $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
    $reader.ReadToEnd() | Write-Output
  }
  exit 1
}
