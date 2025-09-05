$logContent = Get-Content "error-logs/test-clean-errors.log" -Raw
$body = @{
    logFile = "test-clean-errors.log"
    content = $logContent
    timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8096/api/process-error-log" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 30

Write-Host "📊 Processing Result:" -ForegroundColor Green
Write-Host "Message: $($response.message)"
Write-Host "Recommendations: $($response.recommendationCount)"
Write-Host "Errors found: $($response.errors.Count)"
Write-Host "Processing time: $($response.processingTime)"

if ($response.recommendations -and $response.recommendations.Count -gt 0) {
    Write-Host "`n🔧 Sample Recommendation:" -ForegroundColor Yellow
    $rec = $response.recommendations[0]
    Write-Host "File: $($rec.file)"
    Write-Host "Issue: $($rec.issue)"
    Write-Host "Solution: $($rec.solution)"
    Write-Host "Confidence: $($rec.confidence)"
}