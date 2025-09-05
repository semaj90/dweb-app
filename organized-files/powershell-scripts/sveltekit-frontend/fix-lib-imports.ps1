# PowerShell script to fix all $lib imports in the SvelteKit frontend
# This script replaces $lib imports with correct relative paths

$frontendDir = "c:\Users\james\Downloads\Deeds-App-doesn-t-work--main (2)\web-app\sveltekit-frontend\src"

Write-Host "Starting to fix all $lib imports..."

# Get all TypeScript, JavaScript, and Svelte files
$files = Get-ChildItem -Path $frontendDir -Recurse -Include "*.ts", "*.js", "*.svelte" | Where-Object { $_.Name -ne "app.d.ts" }

$totalFiles = $files.Count
$processed = 0

foreach ($file in $files) {
    $processed++
    $relativePath = $file.FullName.Substring($frontendDir.Length + 1)
    Write-Host "[$processed/$totalFiles] Processing: $relativePath"
    
    # Calculate the relative path depth to get back to src
    $depth = ($relativePath -split '\\').Length - 1
    $backPath = ""
    for ($i = 0; $i -lt $depth; $i++) {
        $backPath += "../"
    }
    
    # Read file content
    $content = Get-Content -Path $file.FullName -Raw
    $originalContent = $content
    
    # Replace $lib imports with relative paths
    # For imports like: from '$lib/server/db'
    $content = $content -replace "from '`$lib/", "from '$($backPath)lib/"
    $content = $content -replace 'from "`$lib/', "from `"$($backPath)lib/"
    
    # For dynamic imports like: await import('$lib/server/db')
    $content = $content -replace "import\('`$lib/", "import('$($backPath)lib/"
    $content = $content -replace 'import\("`$lib/', "import(`"$($backPath)lib/"
    
    # Add .js extensions for proper ESM compatibility
    $content = $content -replace "from '([^']+)';", "from '`$1.js';"
    $content = $content -replace 'from "([^"]+)";', 'from "$1.js";'
    
    # But fix cases where .js was added incorrectly (like to .svelte files)
    $content = $content -replace "\.svelte\.js", ".svelte"
    $content = $content -replace "\.css\.js", ".css"
    $content = $content -replace "\.scss\.js", ".scss"
    
    # Don't add .js to SvelteKit built-ins or external packages
    $content = $content -replace "\$app/([^'\"]+)\.js", "`$app/`$1"
    $content = $content -replace "svelte/([^'\"]+)\.js", "svelte/`$1"
    $content = $content -replace "vite/([^'\"]+)\.js", "vite/`$1"
    $content = $content -replace "@([^/\s'\"]+)/([^'\"]+)\.js", "@`$1/`$2"
    $content = $content -replace "([^/\.])/([^'\"./]+)\.js", "`$1/`$2"
    
    # Write back only if content changed
    if ($content -ne $originalContent) {
        Set-Content -Path $file.FullName -Value $content
        Write-Host "  ✓ Updated imports in $relativePath"
    } else {
        Write-Host "  - No changes needed in $relativePath"
    }
}

Write-Host "`nCompleted fixing $lib imports in $totalFiles files."
Write-Host "Next steps:"
Write-Host "1. Run 'npm run dev' to test the dev server"
Write-Host "2. Run 'npm run test' to test Playwright"
Write-Host "3. Fix any remaining import errors manually"
