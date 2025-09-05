$root = 'c:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice\proto'
$dirs = Get-ChildItem -Path $root -Recurse -Directory | Where-Object { $_.Name -eq 'github.com' }
foreach ($d in $dirs) {
    Write-Host "Removing $($d.FullName)"
    Remove-Item -LiteralPath $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Host 'Done'
