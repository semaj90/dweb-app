# Compute sizes for untracked files listed in untracked.txt
param(
    [string]$RepoRoot = "c:\Users\james\Desktop\deeds-web\deeds-web-app",
    [string]$UntrackedFile = "untracked.txt",
    [int]$Top = 20
)
Set-Location $RepoRoot
if (-not (Test-Path $UntrackedFile)) {
    Write-Host "untracked.txt not found in $RepoRoot"
    exit 1
}
$paths = Get-Content $UntrackedFile | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
if ($paths.Count -eq 0) { Write-Host 'No untracked files'; exit 0 }
$results = @()
foreach ($p in $paths) {
    $full = Join-Path $RepoRoot $p
    if (Test-Path $full) {
        $sum = Get-ChildItem -LiteralPath $full -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum
n        $bytes = $sum.Sum
        if ($null -eq $bytes) { $bytes = 0 }
    } else {
        $bytes = 0
    }
    $results += [PSCustomObject]@{ Path = $p; Bytes = [int64]$bytes }
}
$sorted = $results | Sort-Object -Property Bytes -Descending
$sorted | Select-Object -First $Top | Format-Table @{Label='Bytes';Expression={ $_.Bytes };Align='Right'},@{Label='Path';Expression={ $_.Path }} -AutoSize
$total = ($results | Measure-Object -Property Bytes -Sum).Sum
Write-Host "Total untracked bytes: $total"
