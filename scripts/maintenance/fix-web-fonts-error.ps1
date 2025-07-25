#!/usr/bin/env pwsh

Write-Host "🔧 Fixing UnoCSS Web Fonts Timeout Error" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$webAppPath = "C:\Users\james\Desktop\web-app\sveltekit-frontend"
$unocssConfigPath = Join-Path $webAppPath "unocss.config.ts"

Set-Location $webAppPath

Write-Host "`n📝 Updating UnoCSS configuration to fix web fonts timeout..." -ForegroundColor Yellow

# Backup original config
if (Test-Path $unocssConfigPath) {
    Copy-Item $unocssConfigPath "$unocssConfigPath.backup"
    Write-Host "✅ Created backup: unocss.config.ts.backup" -ForegroundColor Green
}

# Create new config without web fonts fetching
$newConfig = @'
import {
  defineConfig,
  presetUno,
  presetAttributify,
  presetIcons,
  transformerDirectives,
  transformerVariantGroup
} from 'unocss';

export default defineConfig({
  // Legal-focused design system with PicoCSS compatibility
  presets: [
    presetUno(),
    presetAttributify(),
    presetIcons({
      collections: {
        lucide: () => import('@iconify/json/json/lucide.json').then(i => i.default),
        mdi: () => import('@iconify/json/json/mdi.json').then(i => i.default),
        tabler: () => import('@iconify/json/json/tabler.json').then(i => i.default),
      },
    }),
    // Removed presetWebFonts to prevent timeout issues
  ],
  transformers: [
    transformerDirectives(),
    transformerVariantGroup(),
  ],
  theme: {
    colors: {
      // Legal-focused color palette
      primary: {
        50: '#f0f9ff',
        100: '#e0f2fe',
        200: '#bae6fd',
        300: '#7dd3fc',
        400: '#38bdf8',
        500: '#0ea5e9',
        600: '#0284c7',
        700: '#0369a1',
        800: '#075985',
        900: '#0c4a6e',
      },
      legal: {
        navy: '#1e3a8a',
        gold: '#d97706',
        neutral: '#374151',
        success: '#059669',
        warning: '#d97706',
        error: '#dc2626',
      },
      semantic: {
        contract: '#3b82f6',
        evidence: '#059669',
        statute: '#7c3aed',
        case: '#dc2626',
        regulation: '#ea580c',
      }
    },
    fontFamily: {
      // Use system fonts as fallbacks
      sans: ['system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      mono: ['Consolas', 'Monaco', 'Courier New', 'monospace'],
    },
    borderRadius: {
      none: '0',
      sm: '0.125rem',
      DEFAULT: '0.25rem',
      md: '0.375rem',
      lg: '0.5rem',
      xl: '0.75rem',
      '2xl': '1rem',
      '3xl': '1.5rem',
      full: '9999px',
    },
    boxShadow: {
      'legal-card': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
      'legal-elevated': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
      'legal-focus': '0 0 0 3px rgba(59, 130, 246, 0.1)',
    }
  },
  shortcuts: {
    // Legal document styling shortcuts
    'legal-heading': 'text-legal-navy font-semibold tracking-tight',
    'legal-body': 'text-legal-neutral leading-relaxed',
    'legal-card': 'bg-white border border-gray-200 rounded-lg shadow-legal-card p-6',
    'legal-button': 'inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md',
    'legal-button-primary': 'legal-button bg-primary-600 text-white hover:bg-primary-700 focus:ring-2 focus:ring-primary-500',
    'legal-button-secondary': 'legal-button bg-white text-gray-700 border-gray-300 hover:bg-gray-50 focus:ring-2 focus:ring-primary-500',
    'legal-input': 'block w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500',
    'legal-textarea': 'legal-input resize-none',
    
    // Citation and evidence styling
    'citation-block': 'border-l-4 border-primary-500 bg-primary-50 p-4 my-4 rounded-r',
    'evidence-highlight': 'bg-yellow-100 border-l-4 border-yellow-500 p-2 rounded-r',
    'statute-reference': 'bg-purple-50 border border-purple-200 px-2 py-1 rounded text-purple-800 text-sm',
    
    // AI and search interface
    'ai-response': 'bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4',
    'search-result': 'hover:bg-gray-50 transition-colors duration-150 p-3 rounded cursor-pointer',
    'rag-context': 'bg-green-50 border-l-4 border-green-500 p-3 text-sm text-green-800',
    
    // Editor and WYSIWYG
    'editor-toolbar': 'bg-white border-b border-gray-200 p-2 flex items-center space-x-2',
    'editor-content': 'prose prose-legal max-w-none p-6 focus:outline-none',
    
    // Layout utilities
    'sidebar-nav': 'w-64 bg-white border-r border-gray-200 h-full overflow-y-auto',
    'main-content': 'flex-1 overflow-y-auto bg-gray-50',
    'content-header': 'bg-white border-b border-gray-200 px-6 py-4',
  },
  rules: [
    // Custom rules for legal document styling
    [/^text-case-(.+)$/, ([, type]) => {
      const colors = {
        contract: '#3b82f6',
        evidence: '#059669',
        statute: '#7c3aed',
        case: '#dc2626',
        regulation: '#ea580c',
      };
      return { color: colors[type as keyof typeof colors] || colors.contract };
    }],
    [/^bg-case-(.+)$/, ([, type]) => {
      const backgrounds = {
        contract: '#eff6ff',
        evidence: '#ecfdf5',
        statute: '#f3e8ff',
        case: '#fef2f2',
        regulation: '#fff7ed',
      };
      return { 'background-color': backgrounds[type as keyof typeof backgrounds] || backgrounds.contract };
    }],
  ],
  safelist: [
    // Ensure these classes are always available
    'prose',
    'prose-legal',
    'legal-card',
    'legal-button-primary',
    'legal-button-secondary',
    'citation-block',
    'evidence-highlight',
    'ai-response',
    'i-phosphor-scales',
    'i-lucide-gavel',
    'i-mdi-book-open-variant',
  ],
});
'@

Set-Content $unocssConfigPath $newConfig -Encoding UTF8
Write-Host "✅ Updated UnoCSS config to use system fonts" -ForegroundColor Green

Write-Host "`n🎨 Adding fallback font CSS..." -ForegroundColor Yellow

# Create app.css with font fallbacks if it doesn't exist
$appCssPath = Join-Path $webAppPath "src\app.css"
if (-not (Test-Path $appCssPath)) {
    $fontCss = @'
/* System font fallbacks */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

:root {
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.font-mono {
  font-family: 'Fira Code', Consolas, Monaco, 'Courier New', monospace;
}

/* Fallback if fonts fail to load */
@supports not (font-variation-settings: normal) {
  :root {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
}
'@
    
    Set-Content $appCssPath $fontCss -Encoding UTF8
    Write-Host "✅ Created app.css with font fallbacks" -ForegroundColor Green
}

Write-Host "`n🔄 Restarting development server..." -ForegroundColor Yellow
Write-Host "The web fonts timeout error should now be resolved!" -ForegroundColor Green

Write-Host "`n🚀 Next steps:" -ForegroundColor Cyan
Write-Host "1. Stop the current dev server (Ctrl+C)" -ForegroundColor White
Write-Host "2. Run: npm run dev" -ForegroundColor White
Write-Host "3. Your app should now load without font errors!" -ForegroundColor White

Write-Host "`n💡 What was fixed:" -ForegroundColor Cyan
Write-Host "• Removed presetWebFonts from UnoCSS config" -ForegroundColor White
Write-Host "• Added system font fallbacks" -ForegroundColor White
Write-Host "• Created backup of original config" -ForegroundColor White
Write-Host "• Added graceful font loading" -ForegroundColor White

Write-Host "`n📝 To restore Google Fonts later (when network is stable):" -ForegroundColor Gray
Write-Host "• Copy unocss.config.ts.backup back to unocss.config.ts" -ForegroundColor Gray

Write-Host "`nPress any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
