# PowerShell script to automatically generate a markdown todo list from npm run check

Write-Host "🔄 Running project checks and generating error log..." -ForegroundColor Yellow

# Step 1: Run the check and capture all output to a log file
npm run check > errors.log 2>&1

Write-Host "✅ Checks complete. Parsing log file..." -ForegroundColor Green

# Step 2: Prepare the output markdown file
$OUTPUT_FILE = "TODO.md"
$date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

@"
# ✅ Project Issues Todo List

Generated on $date

"@ | Out-File -FilePath $OUTPUT_FILE -Encoding UTF8

# Step 3: Parse the log file and format the output
$content = Get-Content errors.log
$currentFile = ""
$lastFile = ""
$todoContent = @()

foreach ($line in $content) {
    # Match a line containing a file path (e.g., .../src/...)
    if ($line -match "src[/\\].*") {
        $matches = [regex]::Matches($line, "src[/\\].*")
        if ($matches.Count -gt 0) {
            $currentFile = $matches[0].Value
            $currentFile = $currentFile -replace ":[0-9]+:[0-9]+$", ""
        }
    }
    
    # Match a line that starts with "Warn:" or "Error:"
    if ($line -match "^(Warn:|Error:)") {
        if ($currentFile -and $currentFile -ne "") {
            # Print the file header if it is new
            if ($currentFile -ne $lastFile) {
                $todoContent += "`n- **File:** ``$currentFile```"
            }
            
            # Clean up the message
            $issue = $line -replace "^(Warn:|Error:)\s*", ""
            $todoContent += "  - **Issue:** $issue"
            
            # Track the last file to avoid duplicate headers
            $lastFile = $currentFile
            $currentFile = ""
        }
    }
}

# Append the todo content to the file
$todoContent | Out-File -FilePath $OUTPUT_FILE -Append -Encoding UTF8

Write-Host "🎉 Success! Your todo list has been generated at ./TODO.md" -ForegroundColor Green
