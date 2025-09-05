# Check for VS Code installation
$paths = @(
    'C:\Program Files\Microsoft VS Code',
    'C:\Users\' + $env:USERNAME + '\AppData\Local\Programs\Microsoft VS Code'
)

foreach ($path in $paths) {
    if (Test-Path $path) {
        Write-Host "Found VS Code at: $path"
        $codePath = Join-Path $path 'bin\code.cmd'
        if (Test-Path $codePath) {
            Write-Host "Code executable found at: $codePath"
            & $codePath --list-extensions
            break
        } else {
            Write-Host "Code executable not found in bin folder"
        }
    } else {
        Write-Host "VS Code not found at: $path"
    }
}

# Also check if it's in PATH
Write-Host "`nChecking PATH for VS Code..."
$env:PATH -split ';' | Where-Object { $_ -like '*VS Code*' } | ForEach-Object { Write-Host "VS Code in PATH: $_" }