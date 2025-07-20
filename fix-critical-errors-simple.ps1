# SvelteKit Critical Fixes - Simplified Version
# Based on official SvelteKit documentation and Context7 best practices

param(
    [string]$ProjectPath = (Get-Location).Path
)

$frontendPath = Join-Path $ProjectPath "sveltekit-frontend"

Write-Host "Starting SvelteKit critical fixes..." -ForegroundColor Cyan
Write-Host "Project path: $ProjectPath" -ForegroundColor Gray
Write-Host ""

# Function to create missing UI components
function Create-UIComponents {
    Write-Host "Creating missing UI components..." -ForegroundColor Yellow

    # Create Button component directory
    $buttonDir = Join-Path $frontendPath "src\lib\components\ui\button"
    if (-not (Test-Path $buttonDir)) {
        New-Item -ItemType Directory -Force -Path $buttonDir | Out-Null
    }

    # Create Button.svelte
    $buttonPath = Join-Path $buttonDir "Button.svelte"
    $buttonContent = @'
<script>
	let { variant = 'default', class: className, children, ...restProps } = $props();
</script>

<button class="btn btn-{variant} {className}" {...restProps}>
	{#if children}
		{@render children()}
	{:else}
		<slot />
	{/if}
</button>

<style>
	.btn {
		padding: 0.5rem 1rem;
		border: none;
		border-radius: 0.25rem;
		cursor: pointer;
	}
	.btn-default {
		background: #f3f4f6;
		color: #1f2937;
	}
	.btn-primary {
		background: #3b82f6;
		color: white;
	}
</style>
'@

    Set-Content $buttonPath $buttonContent
    Write-Host "Created Button component" -ForegroundColor Green

    # Create Modal component
    $modalDir = Join-Path $frontendPath "src\lib\components\ui\modal"
    if (-not (Test-Path $modalDir)) {
        New-Item -ItemType Directory -Force -Path $modalDir | Out-Null
    }

    $modalPath = Join-Path $modalDir "Modal.svelte"
    $modalContent = @'
<script>
	let { open = false, title, children } = $props();
</script>

{#if open}
	<div class="modal-overlay">
		<div class="modal-content">
			<div class="modal-header">
				<h2>{title}</h2>
				<button onclick={() => open = false}>×</button>
			</div>
			<div class="modal-body">
				{#if children}
					{@render children()}
				{:else}
					<slot />
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	.modal-overlay {
		position: fixed;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		background: rgba(0, 0, 0, 0.5);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 1000;
	}
	.modal-content {
		background: white;
		border-radius: 0.5rem;
		padding: 1rem;
		max-width: 500px;
		width: 90%;
	}
	.modal-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 1rem;
	}
</style>
'@

    Set-Content $modalPath $modalContent
    Write-Host "Created Modal component" -ForegroundColor Green
}

# Function to create proper TypeScript definitions
function Fix-TypeDefinitions {
    Write-Host "Creating TypeScript definitions..." -ForegroundColor Yellow

    # Create app.d.ts
    $appTypesPath = Join-Path $frontendPath "src\app.d.ts"
    $appTypesContent = @'
declare global {
	namespace App {
		interface Error {
			message: string;
			code?: string;
		}
		interface Locals {
			user?: {
				id: string;
				email: string;
			};
		}
		interface PageData {}
		interface Platform {}
	}
}

export {};
'@

    Set-Content $appTypesPath $appTypesContent
    Write-Host "Created app.d.ts" -ForegroundColor Green
}

# Function to fix common Svelte syntax errors
function Fix-SvelteSyntaxErrors {
    Write-Host "Fixing common Svelte syntax errors..." -ForegroundColor Yellow

    $svelteFiles = Get-ChildItem -Path (Join-Path $frontendPath "src") -Filter "*.svelte" -Recurse

    foreach ($file in $svelteFiles) {
        try {
            $content = Get-Content $file.FullName -Raw
            $originalContent = $content

            # Fix disabled attribute format
            $content = $content -replace 'disabled="true"', 'disabled'
            $content = $content -replace 'disabled="false"', ''

            # Fix readonly attribute format
            $content = $content -replace 'readonly="false"', ''

            # Only write if content changed
            if ($content -ne $originalContent) {
                Set-Content $file.FullName $content -NoNewline
                Write-Host "Fixed syntax in $($file.Name)" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "Warning: Could not process $($file.Name): $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

# Function to install missing dependencies
function Install-Dependencies {
    Write-Host "Installing missing dependencies..." -ForegroundColor Yellow

    Push-Location $frontendPath
    try {
        npm install
        Write-Host "Dependencies installed" -ForegroundColor Green
    }
    catch {
        Write-Host "Warning: Could not install dependencies" -ForegroundColor Yellow
    }
    finally {
        Pop-Location
    }
}

# Main execution
try {
    if (-not (Test-Path $frontendPath)) {
        Write-Host "Error: SvelteKit frontend directory not found at $frontendPath" -ForegroundColor Red
        exit 1
    }

    # Apply fixes
    Fix-TypeDefinitions
    Create-UIComponents
    Fix-SvelteSyntaxErrors
    Install-Dependencies

    # Generate SvelteKit types
    Write-Host "Generating SvelteKit types..." -ForegroundColor Yellow
    Push-Location $frontendPath
    try {
        npx svelte-kit sync
        Write-Host "Generated SvelteKit types" -ForegroundColor Green
    }
    catch {
        Write-Host "Warning: Could not generate types" -ForegroundColor Yellow
    }
    finally {
        Pop-Location
    }

    Write-Host ""
    Write-Host "SvelteKit critical fixes completed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Summary of fixes applied:" -ForegroundColor Cyan
    Write-Host "  - Created proper app.d.ts with SvelteKit types" -ForegroundColor White
    Write-Host "  - Created missing UI components" -ForegroundColor White
    Write-Host "  - Fixed common Svelte syntax errors" -ForegroundColor White
    Write-Host "  - Installed missing dependencies" -ForegroundColor White
    Write-Host "  - Generated SvelteKit types" -ForegroundColor White
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Run npm run check to verify fixes" -ForegroundColor White
    Write-Host "2. Run npm run dev to test the application" -ForegroundColor White
    Write-Host "3. Address any remaining component-specific issues" -ForegroundColor White
    Write-Host ""
}
catch {
    Write-Host "Error during fixes: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Check the error message above and try running individual fixes" -ForegroundColor Yellow
    exit 1
}
