@echo off
setlocal enabledelayedexpansion
title AI Feature Implementation
color 0A

echo ========================================
echo AI FEATURE IMPLEMENTATION
echo ========================================
echo.

set "FRONTEND_PATH=%~dp0sveltekit-frontend"

echo [1/5] Installing AI packages...
cd "%FRONTEND_PATH%"
npm install @langchain/ollama @langchain/postgres ioredis

echo [2/5] Creating AI chat component...
mkdir src\lib\components\ai 2>nul
> src\lib\components\ai\LegalAIChat.svelte (
echo ^<script lang="ts"^>
echo   import { Button } from '$lib/components/ui/button';
echo   import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
echo   import { Input } from '$lib/components/ui/input';
echo.  
echo   let messages = $state^([]^);
echo   let input = $state^(''^);
echo   let loading = $state^(false^);
echo.
echo   async function sendMessage^(^) {
echo     if ^(!input.trim^(^)^) return;
echo.    
echo     messages.push^({ role: 'user', content: input }^);
echo     const query = input;
echo     input = '';
echo     loading = true;
echo.
echo     try {
echo       const response = await fetch^('/api/ai/chat', {
echo         method: 'POST',
echo         headers: { 'Content-Type': 'application/json' },
echo         body: JSON.stringify^({ message: query }^)
echo       }^);
echo.      
echo       const data = await response.json^(^);
echo       messages.push^({ role: 'assistant', content: data.response }^);
echo     } catch ^(error^) {
echo       messages.push^({ role: 'error', content: 'AI service error' }^);
echo     } finally {
echo       loading = false;
echo     }
echo   }
echo ^</script^>
echo.
echo ^<Card class="h-96 flex flex-col"^>
echo   ^<CardHeader^>
echo     ^<CardTitle^>Legal AI Assistant^</CardTitle^>
echo   ^</CardHeader^>
echo   ^<CardContent class="flex-1 flex flex-col space-y-4"^>
echo     ^<div class="flex-1 overflow-y-auto space-y-2 p-2 border rounded"^>
echo       {#each messages as message}
echo         ^<div class="p-2 rounded {message.role === 'user' ? 'bg-blue-100' : 'bg-gray-100'}"^>
echo           ^<strong^>{message.role}:^</strong^> {message.content}
echo         ^</div^>
echo       {/each}
echo       {#if loading}
echo         ^<div class="text-center p-2"^>AI thinking...^</div^>
echo       {/if}
echo     ^</div^>
echo     ^<div class="flex space-x-2"^>
echo       ^<Input bind:value={input} placeholder="Ask legal question..." /^>
echo       ^<Button onclick={sendMessage} disabled={loading}^>Send^</Button^>
echo     ^</div^>
echo   ^</CardContent^>
echo ^</Card^>
)

echo [3/5] Creating AI API endpoint...
mkdir src\routes\api\ai\chat 2>nul
> src\routes\api\ai\chat\+server.ts (
echo import { json } from '@sveltejs/kit';
echo import type { RequestHandler } from './$types';
echo.
echo export const POST: RequestHandler = async ^({ request }^) =^> {
echo   try {
echo     const { message } = await request.json^(^);
echo.    
echo     const response = await fetch^('http://localhost:11434/api/generate', {
echo       method: 'POST',
echo       headers: { 'Content-Type': 'application/json' },
echo       body: JSON.stringify^({
echo         model: 'gemma3-legal',
echo         prompt: message,
echo         stream: false
echo       }^)
echo     }^);
echo.    
echo     const data = await response.json^(^);
echo     return json^({ response: data.response }^);
echo   } catch ^(error^) {
echo     return json^({ error: 'AI service unavailable' }, { status: 500 }^);
echo   }
echo };
)

echo [4/5] Creating evidence analysis component...
> src\lib\components\ai\EvidenceAnalyzer.svelte (
echo ^<script lang="ts"^>
echo   import { Button } from '$lib/components/ui/button';
echo   import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
echo   import { Textarea } from '$lib/components/ui/textarea';
echo.  
echo   let files = $state^([]^);
echo   let analysis = $state^(''^);
echo   let analyzing = $state^(false^);
echo.
echo   async function analyzeEvidence^(^) {
echo     if ^(!files.length^) return;
echo.    
echo     analyzing = true;
echo     try {
echo       const content = await files[0].text^(^);
echo       const response = await fetch^('/api/ai/analyze', {
echo         method: 'POST',
echo         headers: { 'Content-Type': 'application/json' },
echo         body: JSON.stringify^({ content }^)
echo       }^);
echo.      
echo       const data = await response.json^(^);
echo       analysis = data.analysis;
echo     } catch ^(error^) {
echo       analysis = 'Analysis failed';
echo     } finally {
echo       analyzing = false;
echo     }
echo   }
echo.
echo   function handleFileUpload^(event^) {
echo     const input = event.target;
echo     if ^(input.files^) {
echo       files = Array.from^(input.files^);
echo     }
echo   }
echo ^</script^>
echo.
echo ^<Card^>
echo   ^<CardHeader^>
echo     ^<CardTitle^>Evidence Analysis^</CardTitle^>
echo   ^</CardHeader^>
echo   ^<CardContent class="space-y-4"^>
echo     ^<input type="file" onchange={handleFileUpload} accept=".txt,.pdf,.doc" /^>
echo     ^<Button onclick={analyzeEvidence} disabled={analyzing}^>
echo       {analyzing ? 'Analyzing...' : 'Analyze Evidence'}
echo     ^</Button^>
echo     {#if analysis}
echo       ^<Textarea readonly value={analysis} rows={10} /^>
echo     {/if}
echo   ^</CardContent^>
echo ^</Card^>
)

echo [5/5] Creating analyze API endpoint...
mkdir src\routes\api\ai\analyze 2>nul
> src\routes\api\ai\analyze\+server.ts (
echo import { json } from '@sveltejs/kit';
echo import type { RequestHandler } from './$types';
echo.
echo export const POST: RequestHandler = async ^({ request }^) =^> {
echo   try {
echo     const { content } = await request.json^(^);
echo.    
echo     const prompt = `Analyze this legal evidence for key findings, relevance, and recommendations:\n\n${content}`;
echo.    
echo     const response = await fetch^('http://localhost:11434/api/generate', {
echo       method: 'POST',
echo       headers: { 'Content-Type': 'application/json' },
echo       body: JSON.stringify^({
echo         model: 'gemma3-legal',
echo         prompt,
echo         stream: false
echo       }^)
echo     }^);
echo.    
echo     const data = await response.json^(^);
echo     return json^({ analysis: data.response }^);
echo   } catch ^(error^) {
echo     return json^({ error: 'Analysis service unavailable' }, { status: 500 }^);
echo   }
echo };
)

cd ..
echo.
echo ✅ AI features implemented!
echo 🎯 Components: LegalAIChat, EvidenceAnalyzer
echo 🔌 API endpoints: /api/ai/chat, /api/ai/analyze
pause
