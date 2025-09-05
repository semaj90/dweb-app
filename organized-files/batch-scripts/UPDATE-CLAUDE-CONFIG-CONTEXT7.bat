@echo off
setlocal enabledelayedexpansion
title Update Claude Config with Context 7
color 0A

echo ========================================
echo CLAUDE DESKTOP CONFIG - CONTEXT 7 UPDATE
echo Legal AI Project Integration
echo ========================================
echo.

echo Stopping Claude Desktop if running...
taskkill /f /im Claude.exe > nul 2>&1
timeout /t 2 > nul

set "CLAUDE_CONFIG=%APPDATA%\Claude\claude_desktop_config.json"

echo Backing up existing configuration...
if exist "%CLAUDE_CONFIG%" (
    copy "%CLAUDE_CONFIG%" "%CLAUDE_CONFIG%.backup.%date:~-4,4%%date:~-10,2%%date:~-7,2%" > nul
    echo ✅ Backup created
) else (
    echo Creating Claude config directory...
    mkdir "%APPDATA%\Claude" > nul 2>&1
)

echo Creating enhanced Claude configuration with Context 7...
> "%CLAUDE_CONFIG%" (
echo {
echo   "mcpServers": {
echo     "filesystem": {
echo       "name": "Legal AI Filesystem Access",
echo       "command": "npx",
echo       "args": [
echo         "--yes",
echo         "@modelcontextprotocol/server-filesystem",
echo         "C:/Users/james/Desktop/deeds-web/deeds-web-app",
echo         "--write-access"
echo       ],
echo       "enabled": true,
echo       "description": "Full read/write access to Legal AI Assistant project directory"
echo     },
echo     "context7": {
echo       "name": "Legal AI Project Context",
echo       "command": "npx",
echo       "args": [
echo         "--yes",
echo         "@modelcontextprotocol/server-context7",
echo         "--project-root",
echo         "C:/Users/james/Desktop/deeds-web/deeds-web-app",
echo         "--project-type",
echo         "legal-ai-assistant",
echo         "--framework",
echo         "sveltekit-typescript"
echo       ],
echo       "enabled": true,
echo       "description": "Context-aware assistance for Legal AI project development"
echo     }
echo   },
echo   "projectContext": {
echo     "name": "Legal AI Assistant",
echo     "type": "web-application",
echo     "description": "AI-powered legal case management system for prosecutors",
echo     "framework": "SvelteKit 2.0",
echo     "language": "TypeScript",
echo     "database": "PostgreSQL with pgvector",
echo     "aiIntegration": "Ollama + Gemma 3 Legal",
echo     "currentPhase": "Phase 2 - Enhanced UI/UX",
echo     "rootDirectory": "C:/Users/james/Desktop/deeds-web/deeds-web-app",
echo     "frontendDirectory": "C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend",
echo     "keyDirectories": {
echo       "frontend": "sveltekit-frontend/",
echo       "database": "database/",
echo       "docker": "docker/",
echo       "scripts": "scripts/",
echo       "docs": "docs/"
echo     },
echo     "techStack": {
echo       "frontend": [
echo         "SvelteKit 2.0",
echo         "TypeScript",
echo         "UnoCSS",
echo         "Bits UI v2",
echo         "XState v5",
echo         "Superforms",
echo         "Zod validation"
echo       ],
echo       "backend": [
echo         "PostgreSQL",
echo         "Drizzle ORM",
echo         "pgvector extension",
echo         "Redis",
echo         "Qdrant vector DB"
echo       ],
echo       "ai": [
echo         "Ollama",
echo         "Gemma 3 Legal model",
echo         "RAG pipeline",
echo         "Vector embeddings",
echo         "Streaming chat"
echo       ],
echo       "infrastructure": [
echo         "Docker",
echo         "Docker Compose",
echo         "Playwright testing",
echo         "Vite build tool"
echo       ]
echo     },
echo     "architecture": {
echo       "phase1": {
echo         "status": "complete",
echo         "description": "Foundation setup with SvelteKit, PostgreSQL, authentication"
echo       },
echo       "phase2": {
echo         "status": "in-progress",
echo         "completion": "75%%",
echo         "description": "Enhanced UI/UX with NieR aesthetic, component system"
echo       },
echo       "phase3": {
echo         "status": "ready",
echo         "description": "AI core integration with Ollama and vector search"
echo       },
echo       "phases4to7": {
echo         "status": "planned",
echo         "description": "Advanced data management, real-time AI, agents, production"
echo       }
echo     },
echo     "coreFeatures": {
echo       "caseManagement": "Multi-case handling for prosecutors",
echo       "evidenceSystem": "Upload, categorize, and analyze evidence",
echo       "aiAssistant": "Legal document analysis and prosecution recommendations",
echo       "vectorSearch": "Semantic search across legal documents and cases",
echo       "interactiveCanvas": "Visual evidence organization and relationship mapping",
echo       "realTimeCollaboration": "Multi-user case collaboration features"
echo     },
echo     "currentIssues": [
echo       "TypeScript errors in component prop merging",
echo       "UnoCSS/TailwindCSS dependency conflicts",
echo       "XState v5 integration completion",
echo       "Interactive canvas stabilization"
echo     ],
echo     "upcomingFeatures": [
echo       "Ollama AI integration",
echo       "Vector search implementation",
echo       "RAG pipeline for legal documents",
echo       "Real-time collaboration",
echo       "Advanced evidence analysis"
echo     ]
echo   },
echo   "assistantInstructions": {
echo     "projectAwareness": "This is a legal AI assistant for prosecutors with case management, evidence handling, and AI-powered analysis capabilities. Always consider the legal and ethical implications of suggestions.",
echo     "codeStyle": "Use TypeScript, modern Svelte 5 syntax with runes ^($state, $props^), UnoCSS utilities, and maintain the cyberpunk/terminal aesthetic.",
echo     "architecturePatterns": "Follow the established patterns: component composition with Bits UI, XState for complex workflows, Drizzle ORM for database, and proper error handling.",
echo     "securityConsiderations": "This handles sensitive legal data. Always implement proper authentication, data validation, and secure coding practices.",
echo     "performanceTargets": "Aim for LCP ^< 2s, API responses ^< 500ms, AI responses ^< 3s, and vector search ^< 100ms."
echo   }
echo }
)

echo ✅ Claude Desktop configuration updated successfully!
echo.
echo 📁 Config location: %CLAUDE_CONFIG%
echo.
echo 🔧 MCP Servers enabled:
echo   • filesystem - Full read/write access to project
echo   • context7 - Project-aware assistance
echo.
echo ⚠️  IMPORTANT: Restart Claude Desktop for changes to take effect
echo.
echo 🧪 Test commands to try in Claude:
echo   • "What is the current phase of the Legal AI project?"
echo   • "List the main directories in this project"
echo   • "Help fix TypeScript errors in components"
echo   • "Show me the project architecture"
echo.
pause