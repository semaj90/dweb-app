param(
  [string]$Query = 'contract liability terms',
  [int]$Limit = 5,
  [double]$MinSim = 0.2,
  [string]$Url = 'http://localhost:5179/api/ai/vector-search'
)

$bodyObj = @{ query = $Query; options = @{ limit = $Limit; minSimilarity = $MinSim } }
$bodyJson = $bodyObj | ConvertTo-Json -Depth 5 -Compress

$resp = Invoke-RestMethod -Uri $Url -Method POST -ContentType 'application/json' -Body $bodyJson
$resp | ConvertTo-Json -Depth 8
