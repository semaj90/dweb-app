# Simplified PowerShell script to generate TODO.md from npm run check

Write-Host "🔄 Running project checks..." -ForegroundColor Yellow

# Run npm run check and capture output
$checkOutput = npm run check 2>&1

Write-Host "✅ Checks complete. Generating TODO.md..." -ForegroundColor Green

# Prepare the output file
$outputFile = "TODO.md"
$date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Start the markdown content
$content = @"
# ✅ Project Issues Todo List

Generated on $date

## Summary
This file contains all TypeScript, Svelte, and other issues found by running ``npm run check``.

## Issues Found

"@

# Process each line of output
$currentFile = ""
$issueCount = 0

foreach ($line in $checkOutput) {
    $lineStr = $line.ToString()
    
    # Look for file paths
    if ($lineStr -match "src[/\\].*\.(svelte|ts|js)") {
        $matches = [regex]::Matches($lineStr, "src[/\\].*\.(svelte|ts|js)")
        if ($matches.Count -gt 0) {
            $newFile = $matches[0].Value
            if ($newFile -ne $currentFile) {
                $currentFile = $newFile
                $content += "`n### File: ``$currentFile```n"
            }
        }
    }
    
    # Look for error/warning indicators
    if ($lineStr -match "(Error:|Warning:|×|✖)" -and $currentFile -ne "") {
        $issueCount++
        $cleanLine = $lineStr -replace "^[✖×]\s*", "" -replace "^\s*Error:\s*", "" -replace "^\s*Warning:\s*", ""
        $content += "- **Issue #$issueCount:** $cleanLine`n"
    }
}

# Add summary
if ($issueCount -eq 0) {
    $content += "`n✅ **No issues found!** Your project is clean.`n"
} else {
    $content += "`n📊 **Total Issues Found:** $issueCount`n"
}

# Write to file
$content | Out-File -FilePath $outputFile -Encoding UTF8

Write-Host "🎉 Success! Generated $outputFile with $issueCount issues found." -ForegroundColor Green

# Display the file content
Write-Host "`n📄 Content of TODO.md:" -ForegroundColor Cyan
Get-Content $outputFile | Write-Host
