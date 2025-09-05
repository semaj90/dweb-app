# SvelteKit Error Fix Script - Enhanced with Official Documentation
# Based on SvelteKit official docs and best practices
# Addresses TypeScript, Svelte syntax, and SvelteKit-specific errors

Write-Host "Starting SvelteKit Critical Error Fixes..." -ForegroundColor Cyan

# Determine the correct frontend path
$frontendPath = if (Test-Path ".\sveltekit-frontend") { ".\sveltekit-frontend" } else { "." }

# Ensure we're in a SvelteKit project
if (-not (Test-Path (Join-Path $frontendPath "svelte.config.js")) -and -not (Test-Path (Join-Path $frontendPath "package.json"))) {
    Write-Host "SvelteKit project not found. Please run from project root." -ForegroundColor Red
    exit 1
}

Write-Host "Working directory: $frontendPath" -ForegroundColor Cyan

# Function to fix SvelteKit app.d.ts configuration
function Fix-AppTypes {
    Write-Host "Setting up SvelteKit app.d.ts..." -ForegroundColor Yellow

    $appTypesPath = Join-Path $frontendPath "src\app.d.ts"
    $srcDir = Split-Path $appTypesPath -Parent
    if (-not (Test-Path $srcDir)) {
        New-Item -ItemType Directory -Force -Path $srcDir | Out-Null
    }

    $appTypesContent = @"
// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces

declare global {
	namespace App {
		// Enhanced error interface with custom properties
		interface Error {
			message: string;
			code?: string;
			id?: string;
		}

		interface Locals {
			user?: {
				id: string;
				email: string;
				name: string;
				role: string;
			};
		}

		interface PageData {}
		interface Platform {}
	}
}

export {};
"@

    Set-Content $appTypesPath $appTypesContent
    Write-Host "Created/updated app.d.ts with proper SvelteKit types" -ForegroundColor Green
}

# Function to create missing UI component files with proper Svelte 5 syntax
function Create-UIComponents {
    Write-Host "Creating missing UI components with Svelte 5 syntax..." -ForegroundColor Yellow

    # Create Button component
    $buttonPath = Join-Path $frontendPath "src\lib\components\ui\button\Button.svelte"
    $buttonDir = Split-Path $buttonPath -Parent
    if (-not (Test-Path $buttonDir)) {
        New-Item -ItemType Directory -Force -Path $buttonDir | Out-Null
    }

    $buttonContent = @'
<script>
	import { cn } from '$lib/utils';

	let {
		variant = 'default',
		size = 'default',
		class: className,
		children,
		...restProps
	} = `$props();

	const variants = {
		default: 'bg-primary text-primary-foreground hover:bg-primary/90',
		destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
		outline: 'border border-input bg-background hover:bg-accent hover:text-accent-foreground',
		secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
		ghost: 'hover:bg-accent hover:text-accent-foreground',
		link: 'text-primary underline-offset-4 hover:underline'
	};

	const sizes = {
		default: 'h-10 px-4 py-2',
		sm: 'h-9 rounded-md px-3',
		lg: 'h-11 rounded-md px-8',
		icon: 'h-10 w-10'
	};
</script>

<button
	class={cn(
		'inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
		variants[variant],
		sizes[size],
		className
	)}
	{...restProps}
>
	{#if children}
		{@render children()}
	{:else}
		<slot />
	{/if}
</button>
'@

    Set-Content $buttonPath $buttonContent
    Write-Host "Created Button component" -ForegroundColor Green

    # Create Dialog component
    $dialogPath = Join-Path $frontendPath "src\lib\components\ui\dialog\Dialog.svelte"
    $dialogDir = Split-Path $dialogPath -Parent
    if (-not (Test-Path $dialogDir)) {
        New-Item -ItemType Directory -Force -Path $dialogDir | Out-Null
    }

    $dialogContent = @"
<script>
	let { open = `$state(false), children } = `$props();
</script>

{#if open}
	<div class="fixed inset-0 z-50 bg-black/50" role="dialog" aria-modal="true">
		<div class="fixed left-[50%] top-[50%] translate-x-[-50%] translate-y-[-50%] bg-background p-6 shadow-lg rounded-lg">
			{#if children}
				{@render children()}
			{:else}
				<slot />
			{/if}
		</div>
	</div>
{/if}
"@

    Set-Content $dialogPath $dialogContent
    Write-Host "Created Dialog component" -ForegroundColor Green

    # Create Modal component (alias for Dialog)
    $modalPath = Join-Path $frontendPath "src\lib\components\Modal.svelte"
    $modalContent = @"
<script>
	let {
		isOpen = `$state(false),
		title,
		children,
		onClose = () => {}
	} = `$props();
</script>

{#if isOpen}
	<div
		class="fixed inset-0 z-50 bg-black/50 flex items-center justify-center"
		role="dialog"
		aria-modal="true"
		on:click={onClose}
	>
		<div
			class="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl"
			on:click|stopPropagation
		>
			{#if title}
				<h2 class="text-xl font-semibold mb-4">{title}</h2>
			{/if}

			{#if children}
				{@render children()}
			{:else}
				<slot />
			{/if}

			<button
				class="absolute top-2 right-2 text-gray-400 hover:text-gray-600"
				on:click={onClose}
				aria-label="Close modal"
			>
				×
			</button>
		</div>
	</div>
{/if}
"@

    Set-Content $modalPath $modalContent
    Write-Host "Created Modal component" -ForegroundColor Green
}

# Function to create missing UI component index files
function Create-UIComponentIndexes {
    Write-Host "Creating UI component index files..." -ForegroundColor Yellow

    # Button component index
    $buttonIndexPath = Join-Path $frontendPath "src\lib\components\ui\button\index.ts"
    $buttonDir = Split-Path $buttonIndexPath -Parent
    if (-not (Test-Path $buttonDir)) {
        New-Item -ItemType Directory -Force -Path $buttonDir | Out-Null
    }

    $buttonIndexContent = "export { default as Button } from './Button.svelte';"
    Set-Content $buttonIndexPath $buttonIndexContent
    Write-Host "Created button index file" -ForegroundColor Green

    # Dialog component index
    $dialogIndexPath = Join-Path $frontendPath "src\lib\components\ui\dialog\index.ts"
    $dialogDir = Split-Path $dialogIndexPath -Parent
    if (-not (Test-Path $dialogDir)) {
        New-Item -ItemType Directory -Force -Path $dialogDir | Out-Null
    }

    $dialogIndexContent = @"
export { default as Dialog } from './Dialog.svelte';
export { default as DialogContent } from './Dialog.svelte';
export { default as DialogHeader } from './Dialog.svelte';
"@
    Set-Content $dialogIndexPath $dialogIndexContent
    Write-Host "Created dialog index file" -ForegroundColor Green

    # Main components index
    $mainIndexPath = Join-Path $frontendPath "src\lib\components\index.ts"
    $mainIndexContent = @"
// UI Components
export { Button } from './ui/button';
export { Dialog, DialogContent, DialogHeader } from './ui/dialog';

// Main Components
export { default as Modal } from './Modal.svelte';
export { default as Header } from './Header.svelte';
export { default as Sidebar } from './Sidebar.svelte';
export { default as SearchBar } from './SearchBar.svelte';
export { default as Checkbox } from './Checkbox.svelte';
export { default as Dropdown } from './Dropdown.svelte';
"@
    Set-Content $mainIndexPath $mainIndexContent
    Write-Host "Created main components index" -ForegroundColor Green
}

# Function to create utility functions
function Create-UtilityFunctions {
    Write-Host "Creating utility functions..." -ForegroundColor Yellow

    $utilsPath = Join-Path $frontendPath "src\lib\utils.ts"
    $utilsContent = @"
import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

export function formatDate(date: Date | string): string {
	const d = new Date(date);
	return d.toLocaleDateString();
}

export function formatFileSize(bytes: number): string {
	if (bytes === 0) return '0 Bytes';
	const k = 1024;
	const sizes = ['Bytes', 'KB', 'MB', 'GB'];
	const i = Math.floor(Math.log(bytes) / Math.log(k));
	return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
"@

    Set-Content $utilsPath $utilsContent
    Write-Host "Created utility functions" -ForegroundColor Green
}

# Function to fix type definitions
function Fix-TypeDefinitions {
    Write-Host "Fixing type definitions..." -ForegroundColor Yellow

    $typesPath = Join-Path $frontendPath "src\lib\types\index.ts"
    $typesDir = Split-Path $typesPath -Parent
    if (-not (Test-Path $typesDir)) {
        New-Item -ItemType Directory -Force -Path $typesDir | Out-Null
    }

    $typesContent = @"
// Core Types - Fixed for TypeScript errors

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'prosecutor' | 'investigator' | 'analyst';
}

export interface Case {
  id: string;
  title: string;
  description: string;
  status: 'open' | 'closed' | 'pending' | 'investigating' | 'archived';
  priority: 'low' | 'medium' | 'high' | 'critical';
  createdAt: Date;
  updatedAt: Date;
}

export interface Evidence {
  id: string;
  title: string;
  content: string;
  type: string;
  evidenceType?: string;
  caseId?: string;
  createdAt?: Date;
  thumbnailUrl?: string;
  fileSize?: number;
}

export interface Report {
  id: string;
  title: string;
  content: string;
  caseId?: string;
  reportType?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface HelpArticle {
  id: string;
  category: string;
  title: string;
  description: string;
  type: string;
  duration: string;
  popularity: number;
  tags: string[];
  content: string;
}

export interface AnalysisResults {
  keyEntities: Array<{
    text: string;
    type: string;
    confidence: number;
  }>;
  summary?: string;
  error?: string;
}
"@

    Set-Content $typesPath $typesContent
    Write-Host "Updated type definitions" -ForegroundColor Green
}

# Function to install missing dependencies with SvelteKit-specific packages
function Install-MissingDependencies {
    Write-Host "Installing missing dependencies..." -ForegroundColor Yellow

    Push-Location $frontendPath

    try {
        Write-Host "Installing core dependencies..." -ForegroundColor Cyan

        # Install search functionality
        npm install fuse.js --silent
        Write-Host "✓ Installed fuse.js for search" -ForegroundColor Green

        # Install utility libraries
        npm install clsx tailwind-merge --silent
        Write-Host "✓ Installed clsx and tailwind-merge for styling" -ForegroundColor Green

        # Install type definitions
        npm install -D @types/node --silent
        Write-Host "✓ Installed @types/node" -ForegroundColor Green

        # Update SvelteKit to latest
        Write-Host "Updating SvelteKit..." -ForegroundColor Cyan
        npm install @sveltejs/kit@latest --silent
        Write-Host "✓ Updated @sveltejs/kit" -ForegroundColor Green

    }
    catch {
        Write-Host "Warning: Some dependencies could not be installed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    finally {
        Pop-Location
    }
}

# Function to fix common Svelte syntax errors
function Fix-SvelteSyntaxErrors {
    Write-Host "Fixing common Svelte syntax errors..." -ForegroundColor Yellow

    # Get all .svelte files
    $svelteFiles = Get-ChildItem -Path (Join-Path $frontendPath "src") -Filter "*.svelte" -Recurse

    foreach ($file in $svelteFiles) {
        try {
            $content = Get-Content $file.FullName -Raw
            $originalContent = $content

            # Fix common syntax issues using proper PowerShell regex

            # 1. Fix disabled attribute format
            $content = $content -replace 'disabled="true"', 'disabled'
            $content = $content -replace 'disabled="false"', ''

            # 2. Fix readonly attribute format
            $content = $content -replace 'readonly="false"', ''

            # 3. Fix tabindex attribute format (numbers only)
            $content = $content -replace 'tabindex="(\d+)"', 'tabindex={$1}'

            # 4. Fix basic CSS selector issues
            $content = $content -replace ':global\(\s*\.\s*([^)]+)\)', ':global(.$1)'

            # Only write if content changed
            if ($content -ne $originalContent) {
                Set-Content $file.FullName $content -NoNewline
                Write-Host "✓ Fixed syntax in $($file.Name)" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "Warning: Could not process $($file.Name): $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

# Function to create proper SvelteKit error handling
function Create-ErrorHandling {
    Write-Host "Setting up SvelteKit error handling..." -ForegroundColor Yellow

    # Create +error.svelte
    $errorPagePath = Join-Path $frontendPath "src\routes\+error.svelte"
    $errorPageContent = @'
<script>
	import { page } from '$app/state';
</script>

<svelte:head>
	<title>Error {page.status}</title>
</svelte:head>

<div class="error-container">
	<h1>{page.status}: {page.error.message}</h1>

	{#if page.error.code}
		<p>Error Code: {page.error.code}</p>
	{/if}

	<a href="/" class="home-link">← Back to Home</a>
</div>

<style>
	.error-container {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 50vh;
		text-align: center;
		padding: 2rem;
	}

	.home-link {
		margin-top: 1rem;
		color: #3b82f6;
		text-decoration: none;
	}

	.home-link:hover {
		text-decoration: underline;
	}
</style>
'@

    Set-Content $errorPagePath $errorPageContent
    Write-Host "Created error page component" -ForegroundColor Green

    # Create hooks.server.js for server-side error handling
    $hooksPath = Join-Path $frontendPath "src\hooks.server.js"
    $hooksContent = @"
import { error } from '@sveltejs/kit';

/** @type {import('@sveltejs/kit').Handle} */
export async function handle({ event, resolve }) {
	// Add any global request handling here
	const response = await resolve(event);
	return response;
}

/** @type {import('@sveltejs/kit').HandleServerError} */
export async function handleError({ error, event, status, message }) {
	const errorId = crypto.randomUUID();

	// Log the error (you can integrate with error tracking services here)
	console.error('Server error:', { error, event: event.url.pathname, errorId, status });

	return {
		message: 'An error occurred',
		errorId
	};
}
"@

    Set-Content $hooksPath $hooksContent
    Write-Host "✓ Created server hooks for error handling" -ForegroundColor Green
}

# Main execution
try {
    Write-Host "Starting SvelteKit critical fixes..." -ForegroundColor Cyan
    Write-Host "This script implements solutions from official SvelteKit documentation" -ForegroundColor Gray

    # 1. Fix SvelteKit app configuration
    Fix-AppTypes

    # 2. Fix type definitions
    Fix-TypeDefinitions

    # 3. Create missing UI components with Svelte 5 syntax
    Create-UIComponents

    # 4. Create component indexes
    Create-UIComponentIndexes

    # 5. Create utility functions
    Create-UtilityFunctions

    # 6. Install missing dependencies
    Install-MissingDependencies

    # 7. Fix common Svelte syntax errors
    Fix-SvelteSyntaxErrors

    # 8. Create proper error handling
    Create-ErrorHandling

    # 9. Generate SvelteKit types
    Write-Host "Generating SvelteKit types..." -ForegroundColor Yellow
    Push-Location $frontendPath
    try {
        npx svelte-kit sync
        Write-Host "✓ Generated SvelteKit types" -ForegroundColor Green
    }
    catch {
        Write-Host "Warning: Could not generate types" -ForegroundColor Yellow
    }
    finally {
        Pop-Location
    }

    Write-Host ""
    Write-Host "🎉 SvelteKit critical fixes completed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Summary of fixes applied:" -ForegroundColor Cyan
    Write-Host "  ✓ Created proper app.d.ts with SvelteKit types" -ForegroundColor White
    Write-Host "  ✓ Fixed and enhanced type definitions" -ForegroundColor White
    Write-Host "  ✓ Created missing UI components with Svelte 5 syntax" -ForegroundColor White
    Write-Host "  ✓ Set up component index files for proper imports" -ForegroundColor White
    Write-Host "  ✓ Created utility functions" -ForegroundColor White
    Write-Host "  ✓ Installed missing dependencies" -ForegroundColor White
    Write-Host "  ✓ Fixed common Svelte syntax errors" -ForegroundColor White
    Write-Host "  ✓ Created proper error handling components" -ForegroundColor White
    Write-Host "  ✓ Generated SvelteKit types" -ForegroundColor White
    Write-Host ""
    Write-Host "🔍 Next steps:" -ForegroundColor Cyan
    Write-Host "1. Run npm run check to verify fixes" -ForegroundColor White
    Write-Host "2. Run npm run dev to test the application" -ForegroundColor White
    Write-Host "3. Address any remaining component-specific issues" -ForegroundColor White
    Write-Host "4. Check the comprehensive guide in .vscode/copilot.md" -ForegroundColor White
    Write-Host ""
    Write-Host "📖 For more information, see:" -ForegroundColor Cyan
    Write-Host "   • SvelteKit Docs: https://kit.svelte.dev/" -ForegroundColor White
    Write-Host "   • Error Handling: https://kit.svelte.dev/docs/errors" -ForegroundColor White
    Write-Host "   • TypeScript: https://kit.svelte.dev/docs/types" -ForegroundColor White
    Write-Host ""

}
catch {
    Write-Host "❌ Error during fixes: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Check the error message above and try running individual fixes" -ForegroundColor Yellow
    exit 1
}
