# Check CUDA in PATH
Write-Host "Checking PATH for CUDA installations..."
$env:PATH -split ';' | Where-Object { $_ -match 'cuda' -or $_ -match 'CUDA' } | ForEach-Object {
    Write-Host "CUDA PATH entry: $_"
}

Write-Host ""
Write-Host "Checking common CUDA installation paths..."
$cudaPaths = @(
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin',
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin',
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin',
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin'
)

foreach ($path in $cudaPaths) {
    if (Test-Path $path) {
        Write-Host "Found CUDA at: $path"
        if ($env:PATH -split ';' -contains $path) {
            Write-Host "  Already in PATH"
        } else {
            Write-Host "  Not in PATH"
        }
    }
}

Write-Host ""
Write-Host "CUDA Environment Variables:"
Get-ChildItem env: | Where-Object { $_.Name -match 'CUDA' } | ForEach-Object {
    Write-Host "$($_.Name) = $($_.Value)"
}