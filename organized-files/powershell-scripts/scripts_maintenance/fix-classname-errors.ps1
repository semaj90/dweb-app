# PowerShell script to fix className="${1}" errors across the SvelteKit frontend

Write-Host "🔧 Starting systematic fix for className errors..." -ForegroundColor Green

# Navigate to the sveltekit-frontend directory
$frontendPath = "c:\Users\james\Desktop\web-app\sveltekit-frontend"
Set-Location $frontendPath

# Get all .svelte files recursively
$svelteFiles = Get-ChildItem -Path . -Recurse -Filter "*.svelte" | Where-Object { !$_.PSIsContainer }

Write-Host "📁 Found $($svelteFiles.Count) Svelte files to process" -ForegroundColor Yellow

$totalReplacements = 0

foreach ($file in $svelteFiles) {
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content
    
    # Fix className="${1}" to class=""
    $content = $content -replace 'className="\$\{1\}"', 'class=""'
    
    # Fix className="${1} to class=""  (handle cases where closing quote might be missing)
    $content = $content -replace 'className="\$\{1\}', 'class=""'
    
    # Fix any remaining className= patterns to class=
    $content = $content -replace 'className=', 'class='
    
    # Count replacements for this file
    if ($content -ne $originalContent) {
        $replacements = ([regex]::Matches($originalContent, 'className')).Count
        $totalReplacements += $replacements
        
        # Write the fixed content back to the file
        Set-Content -Path $file.FullName -Value $content -NoNewline
        
        Write-Host "✅ Fixed $replacements className errors in: $($file.Name)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "🎉 Completed! Fixed $totalReplacements className errors across $($svelteFiles.Count) files" -ForegroundColor Green
Write-Host ""

# Also fix any ${1} placeholder errors in textarea rows and other attributes
Write-Host "🔧 Fixing placeholder errors like rows=`${1}`..." -ForegroundColor Yellow

foreach ($file in $svelteFiles) {
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content
    
    # Fix rows="${1}" to rows="3" (common default)
    $content = $content -replace 'rows="\$\{1\}"', 'rows="3"'
    
    # Fix any other ${1} placeholders to empty string or appropriate defaults
    $content = $content -replace '\$\{1\}', '""'
    
    if ($content -ne $originalContent) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        Write-Host "✅ Fixed placeholder errors in: $($file.Name)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "🚀 Running svelte-check to verify fixes..." -ForegroundColor Blue

# Run svelte-check to see if errors are reduced
& npm run check

Write-Host ""
Write-Host "✨ Script completed! The className errors should now be fixed." -ForegroundColor Green
Write-Host "💡 If there are still issues, they may be different types of errors." -ForegroundColor Cyan
