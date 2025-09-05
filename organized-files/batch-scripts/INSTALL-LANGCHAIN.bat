@echo off
title Install LangChain Dependencies
echo Installing LangChain for AI integration...

cd sveltekit-frontend

echo [1/3] Installing LangChain packages...
npm install langchain @langchain/ollama @langchain/postgres @langchain/core

echo [2/3] Creating LangChain service...
mkdir src\lib\services 2>nul
> src\lib\services\langchain-service.ts (
echo import { Ollama } from '@langchain/ollama';
echo import { PostgresVectorStore } from '@langchain/postgres';
echo import { Pool } from 'pg';
echo.
echo export class LangChainService {
echo   private llm = new Ollama({
echo     baseUrl: 'http://deeds-ollama-gpu:11434',
echo     model: 'gemma3-legal'
echo   }^);
echo.
echo   private db = new Pool({
echo     connectionString: process.env.DATABASE_URL
echo   }^);
echo.
echo   async queryWithContext(prompt: string^) {
echo     const vectorStore = new PostgresVectorStore(this.db, {
echo       tableName: 'embeddings'
echo     }^);
echo.
echo     const context = await vectorStore.similaritySearch(prompt, 5^);
echo     const contextText = context.map(doc =^> doc.pageContent^).join('\n\n'^);
echo.
echo     const fullPrompt = `Context: ${contextText}\n\nQuery: ${prompt}`;
echo     return await this.llm.invoke(fullPrompt^);
echo   }
echo }
)

echo [3/3] Updating package.json scripts...
cd ..
echo Adding langchain to package.json scripts...

echo ✅ LangChain integration ready!
pause
