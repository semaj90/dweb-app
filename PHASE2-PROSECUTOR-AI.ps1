# PROSECUTOR AI - Advanced Integration Setup Script
# Phase 2: Enhanced UI/UX with AI Foundations + 7-Phase Roadmap
param(
    [switch]$SkipClean = $false,
    [switch]$Force = $false,
    [switch]$InstallAdvanced = $true
)

# Set console colors for better visibility
$Host.UI.RawUI.BackgroundColor = "Black"
$Host.UI.RawUI.ForegroundColor = "Green"
Clear-Host

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "🎯 PROSECUTOR AI - ADVANCED INTEGRATION SETUP" -ForegroundColor Yellow
Write-Host "🚀 Phase 2: Enhanced UI/UX with AI Foundations" -ForegroundColor Magenta
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "🔥 COMPREHENSIVE INTEGRATION FEATURES:" -ForegroundColor Yellow
Write-Host "✅ Melt UI + Bits UI v2 Integration" -ForegroundColor Green
Write-Host "✅ AI Command Parsing & Real-time Updates" -ForegroundColor Green
Write-Host "✅ XState Machine for AI Command Processing" -ForegroundColor Green
Write-Host "✅ Enhanced Component System with Prop Merging" -ForegroundColor Green
Write-Host "✅ shadcn-svelte + UnoCSS Integration" -ForegroundColor Green
Write-Host ""

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "📋 EVIDENCE SYSTEM FEATURES" -ForegroundColor Yellow
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The evidence system supports:" -ForegroundColor White
Write-Host "1. 📂 " -NoNewline -ForegroundColor Yellow
Write-Host "Drag files" -NoNewline -ForegroundColor White
Write-Host " from your computer onto the evidence board" -ForegroundColor Gray
Write-Host "2. ➕ " -NoNewline -ForegroundColor Yellow
Write-Host "Click `"ADD EVIDENCE`"" -NoNewline -ForegroundColor White
Write-Host " to open the upload dialog" -ForegroundColor Gray
Write-Host "3. 📄 " -NoNewline -ForegroundColor Yellow
Write-Host "Multiple file types:" -NoNewline -ForegroundColor White
Write-Host " PDF, images (JPG, PNG, GIF), videos (MP4, MOV, AVI), documents" -ForegroundColor Gray
Write-Host "4. 🏷️  " -NoNewline -ForegroundColor Yellow
Write-Host "File metadata:" -NoNewline -ForegroundColor White
Write-Host " Automatic file size, type, and thumbnail generation" -ForegroundColor Gray
Write-Host "5. 📊 " -NoNewline -ForegroundColor Yellow
Write-Host "Evidence organization:" -NoNewline -ForegroundColor White
Write-Host " Categorize and prioritize uploaded evidence" -ForegroundColor Gray
Write-Host ""

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "🗺️ 7-PHASE ROADMAP" -ForegroundColor Yellow
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Phase 1: ✅ " -NoNewline -ForegroundColor Green
Write-Host "Foundation Setup (Complete)" -ForegroundColor White
Write-Host "Phase 2: 🔥 " -NoNewline -ForegroundColor Yellow
Write-Host "Enhanced UI/UX with AI foundations (Current)" -ForegroundColor White
Write-Host "Phase 3: 🚀 " -NoNewline -ForegroundColor Cyan
Write-Host "AI Core (LLM + Vector search + RAG)" -ForegroundColor Gray
Write-Host "Phase 4: 📊 " -NoNewline -ForegroundColor Cyan
Write-Host "Data Management (Loki.js + Redis + RabbitMQ + Neo4j)" -ForegroundColor Gray
Write-Host "Phase 5: 🤖 " -NoNewline -ForegroundColor Cyan
Write-Host "AI-driven UI updates in real-time" -ForegroundColor Gray
Write-Host "Phase 6: 🧠 " -NoNewline -ForegroundColor Cyan
Write-Host "Advanced AI (self-prompting + recommendations)" -ForegroundColor Gray
Write-Host "Phase 7: 🏭 " -NoNewline -ForegroundColor Cyan
Write-Host "Production optimization" -ForegroundColor Gray
Write-Host ""

# Set execution policy temporarily
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# Set the correct directory
$ProjectPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
Set-Location $ProjectPath

Write-Host "📍 Working in: $(Get-Location)" -ForegroundColor Green
Write-Host ""

# Function to check if command exists
function Test-Command($command) {
    $null = Get-Command $command -ErrorAction SilentlyContinue
    return $?
}

# Check prerequisites
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow

if (!(Test-Command "node")) {
    Write-Host "❌ Node.js not found!" -ForegroundColor Red
    Write-Host "📥 Please install Node.js from: https://nodejs.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

if (!(Test-Command "npm")) {
    Write-Host "❌ npm not found! Please install npm." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$nodeVersion = node --version
$npmVersion = npm --version
Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
Write-Host "✅ npm: $npmVersion" -ForegroundColor Green
Write-Host ""

# Clean previous builds if requested
if (!$SkipClean) {
    Write-Host "🧹 Cleaning previous builds..." -ForegroundColor Yellow
    
    $foldersToClean = @(".svelte-kit", "build", "dist", ".vite-temp")
    foreach ($folder in $foldersToClean) {
        if (Test-Path $folder) {
            Remove-Item $folder -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "   🗑️ Removed $folder" -ForegroundColor Gray
        }
    }
    
    Write-Host "✅ Cleanup complete" -ForegroundColor Green
    Write-Host ""
}

# Install advanced dependencies
if ($InstallAdvanced) {
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host "📦 INSTALLING ADVANCED DEPENDENCIES" -ForegroundColor Yellow
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "🔧 Installing UnoCSS integration packages..." -ForegroundColor Yellow
    npm install -D @unocss/preset-wind @unocss/preset-typography @unocss/preset-icons
    npm install -D @unocss/transformer-variant-group
    npm install -D @unocss/svelte-scoped
    
    Write-Host "🎨 Installing UI component dependencies..." -ForegroundColor Yellow
    npm install tailwind-merge
    npm install bits-ui@latest clsx@latest
    npm install class-variance-authority
    
    Write-Host "🤖 Installing AI and state management..." -ForegroundColor Yellow
    npm install @xstate/svelte xstate
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️ Some packages may have had issues, but continuing..." -ForegroundColor Yellow
    } else {
        Write-Host "✅ Advanced dependencies installed" -ForegroundColor Green
    }
    Write-Host ""
}

# Install or update core dependencies
Write-Host "📦 Checking core dependencies..." -ForegroundColor Yellow

if (!(Test-Path "node_modules") -or $Force) {
    Write-Host "   📥 Installing fresh dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ npm install failed!" -ForegroundColor Red
        Write-Host "🔧 Try running: npm cache clean --force" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✅ Dependencies already installed" -ForegroundColor Green
}

Write-Host ""

# Run SvelteKit sync
Write-Host "🔄 Running SvelteKit sync..." -ForegroundColor Yellow
npm run prepare
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ SvelteKit sync had issues, but continuing..." -ForegroundColor Yellow
}

Write-Host ""

# Check for and fix configuration issues
Write-Host "🔍 Checking and fixing configuration..." -ForegroundColor Yellow

# Check PostCSS config
if (Test-Path "postcss.config.js") {
    $postcssContent = Get-Content "postcss.config.js" -Raw
    if ($postcssContent -like "*tailwindcss*" -and $postcssContent -notlike "*@tailwindcss/postcss*") {
        Write-Host "   🔧 Fixing PostCSS configuration..." -ForegroundColor Yellow
        $newPostCSSConfig = @"
import tailwindcss from "@tailwindcss/postcss";
import autoprefixer from "autoprefixer";

export default {
  plugins: [tailwindcss(), autoprefixer()],
};
"@
        $newPostCSSConfig | Out-File "postcss.config.js" -Encoding UTF8
        Write-Host "   ✅ PostCSS configuration updated" -ForegroundColor Green
    }
}

# Check Vite config for conflicts
if (Test-Path "vite.config.ts") {
    $viteContent = Get-Content "vite.config.ts" -Raw
    if ($viteContent -like "*UnoCSS()*" -and $viteContent -like "*tailwindcss()*") {
        Write-Host "   🔧 Fixing Vite config conflicts..." -ForegroundColor Yellow
        # Remove UnoCSS to prevent conflicts with TailwindCSS
        $viteContent = $viteContent -replace "import UnoCSS from `"unocss/vite`";", ""
        $viteContent = $viteContent -replace ", UnoCSS\(\)", ""
        $viteContent | Out-File "vite.config.ts" -Encoding UTF8
        Write-Host "   ✅ Vite configuration cleaned" -ForegroundColor Green
    }
}

Write-Host ""

# Create advanced AI integration files
Write-Host "🤖 Setting up AI integration files..." -ForegroundColor Yellow

# Create AI command parser
$aiCommandParser = @"
import { writable } from 'svelte/store';
import { interpret } from 'xstate';
import { aiCommandMachine } from './ai-command-machine.js';

/**
 * AI Command Parser for real-time UI updates
 * Part of Phase 2: Enhanced UI/UX with AI foundations
 */

// AI Command Store
export const aiCommandStore = writable({
  currentCommand: '',
  isProcessing: false,
  lastResult: null,
  history: [],
  error: null
});

// XState service for AI command processing
export const aiCommandService = interpret(aiCommandMachine).start();

/**
 * Parse AI command and trigger real-time UI updates
 * @param {string} command - The AI command to parse
 * @returns {Promise<object>} - Command result
 */
export function parseAICommand(command) {
  return new Promise((resolve, reject) => {
    try {
      aiCommandStore.update(store => ({
        ...store,
        currentCommand: command,
        isProcessing: true,
        error: null
      }));

      // Send command to XState machine
      aiCommandService.send({ type: 'PROCESS_COMMAND', command });

      // Simulate AI processing (replace with actual AI integration in Phase 3)
      setTimeout(() => {
        const result = {
          command,
          timestamp: new Date().toISOString(),
          success: true,
          aiResponse: `Processed: ${command}`,
          uiUpdates: generateUIUpdates(command)
        };

        aiCommandStore.update(store => ({
          ...store,
          isProcessing: false,
          lastResult: result,
          history: [...store.history, result]
        }));

        resolve(result);
      }, 1000);

    } catch (error) {
      aiCommandStore.update(store => ({
        ...store,
        isProcessing: false,
        error: error.message
      }));
      reject(error);
    }
  });
}

/**
 * Generate UI updates based on AI command
 * @param {string} command - The processed command
 * @returns {object} - UI update instructions
 */
function generateUIUpdates(command) {
  const updates = {
    classes: [],
    attributes: {},
    content: null
  };

  // AI-controlled class application
  if (command.includes('highlight')) {
    updates.classes.push('ai-highlight', 'animate-pulse');
  }
  
  if (command.includes('evidence')) {
    updates.classes.push('evidence-focus', 'border-accent');
  }

  if (command.includes('priority')) {
    updates.classes.push('priority-high', 'bg-destructive/10');
  }

  // Legacy .yorha-* class support
  if (command.includes('yorha')) {
    updates.classes.push('yorha-enhanced', 'yorha-ai-active');
  }

  return updates;
}

/**
 * Apply AI-controlled classes to elements
 * @param {HTMLElement} element - Target element
 * @param {Array} classes - Classes to apply
 */
export function applyAIClasses(element, classes) {
  if (!element || !classes) return;
  
  classes.forEach(className => {
    element.classList.add(className);
  });
}

// Export for real-time updates
export { aiCommandStore as default };
"@

if (!(Test-Path "src/lib/stores")) {
    New-Item -ItemType Directory -Path "src/lib/stores" -Force | Out-Null
}

$aiCommandParser | Out-File "src/lib/stores/ai-command-parser.js" -Encoding UTF8

# Create XState machine
$aiCommandMachine = @"
import { createMachine, assign } from 'xstate';

/**
 * XState Machine for AI Command Processing
 * Part of Phase 2: Enhanced UI/UX with AI foundations
 */

export const aiCommandMachine = createMachine({
  id: 'aiCommand',
  initial: 'idle',
  context: {
    command: null,
    result: null,
    error: null,
    retryCount: 0
  },
  states: {
    idle: {
      on: {
        PROCESS_COMMAND: {
          target: 'processing',
          actions: assign({
            command: (context, event) => event.command,
            error: null,
            retryCount: 0
          })
        }
      }
    },
    processing: {
      invoke: {
        id: 'processCommand',
        src: 'processCommand',
        onDone: {
          target: 'success',
          actions: assign({
            result: (context, event) => event.data
          })
        },
        onError: {
          target: 'error',
          actions: assign({
            error: (context, event) => event.data
          })
        }
      }
    },
    success: {
      after: {
        2000: 'idle'
      },
      on: {
        PROCESS_COMMAND: {
          target: 'processing',
          actions: assign({
            command: (context, event) => event.command,
            error: null
          })
        }
      }
    },
    error: {
      on: {
        RETRY: {
          target: 'processing',
          cond: (context) => context.retryCount < 3,
          actions: assign({
            retryCount: (context) => context.retryCount + 1
          })
        },
        PROCESS_COMMAND: {
          target: 'processing',
          actions: assign({
            command: (context, event) => event.command,
            error: null,
            retryCount: 0
          })
        }
      },
      after: {
        5000: 'idle'
      }
    }
  }
}, {
  services: {
    processCommand: async (context) => {
      // This will be replaced with actual AI service in Phase 3
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            command: context.command,
            processed: true,
            timestamp: new Date().toISOString()
          });
        }, 1000);
      });
    }
  }
});
"@

$aiCommandMachine | Out-File "src/lib/stores/ai-command-machine.js" -Encoding UTF8

Write-Host "   ✅ AI integration files created" -ForegroundColor Green
Write-Host ""

# Start the development server
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "🚀 LAUNCHING PROSECUTOR AI - PHASE 2" -ForegroundColor Yellow
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📱 Server will be available at: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:5173" -ForegroundColor Cyan
Write-Host "🛑 Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Try multiple startup approaches
$attempts = @(
    @{ Name = "Standard dev"; Command = "npm run dev" },
    @{ Name = "Clean dev"; Command = "npm run dev:clean" },
    @{ Name = "Safe dev"; Command = "npm run dev:safe" },
    @{ Name = "Local dev"; Command = "npm run dev:local" }
)

$success = $false
foreach ($attempt in $attempts) {
    Write-Host "🔥 Trying: $($attempt.Name)..." -ForegroundColor Yellow
    
    try {
        Invoke-Expression $attempt.Command
        $success = $true
        break
    } catch {
        Write-Host "   ❌ $($attempt.Name) failed: $($_.Exception.Message)" -ForegroundColor Red
        continue
    }
}

if (!$success) {
    Write-Host ""
    Write-Host "❌ All startup attempts failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔧 Manual troubleshooting steps:" -ForegroundColor Yellow
    Write-Host "   1. npm run fix:all" -ForegroundColor White
    Write-Host "   2. npm run clean" -ForegroundColor White
    Write-Host "   3. npm install" -ForegroundColor White
    Write-Host "   4. npm run dev" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host "🎉 PROSECUTOR AI PHASE 2 LAUNCHED!" -ForegroundColor Yellow
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Your application is running at: " -NoNewline -ForegroundColor White
    Write-Host "http://localhost:5173" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🔥 PHASE 2 FEATURES ACTIVE:" -ForegroundColor Yellow
    Write-Host "• Melt UI + Bits UI v2 Integration" -ForegroundColor White
    Write-Host "• AI Command Parsing with parseAICommand()" -ForegroundColor White
    Write-Host "• XState Machine for AI Command Processing" -ForegroundColor White
    Write-Host "• Real-time UI updates via ai-controlled classes" -ForegroundColor White
    Write-Host "• Enhanced Component System with Prop Merging" -ForegroundColor White
    Write-Host "• Legacy .yorha-* class support" -ForegroundColor White
    Write-Host ""
    Write-Host "📋 EVIDENCE SYSTEM:" -ForegroundColor Yellow
    Write-Host "• Drag & Drop Evidence Upload" -ForegroundColor White
    Write-Host "• Multiple File Type Support" -ForegroundColor White
    Write-Host "• Automatic Metadata Generation" -ForegroundColor White
    Write-Host "• Evidence Organization & Categorization" -ForegroundColor White
    Write-Host ""
    Write-Host "🗺️ NEXT PHASES:" -ForegroundColor Yellow
    Write-Host "Phase 3: AI Core (LLM + Vector search + RAG)" -ForegroundColor Gray
    Write-Host "Phase 4: Data Management (Loki.js + Redis + RabbitMQ + Neo4j)" -ForegroundColor Gray
    Write-Host "Phase 5: AI-driven UI updates in real-time" -ForegroundColor Gray
    Write-Host ""
}

Read-Host "Press Enter to exit"