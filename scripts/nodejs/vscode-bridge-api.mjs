// VS Code Extension Bridge - Express API
// File: vscode-bridge-api.mjs

import express from 'express';
import cors from 'cors';
import fetch from 'node-fetch';

const app = express();
app.use(cors());
app.use(express.json());

class VSCodeBridge {
  constructor() {
    this.setupRoutes();
  }

  setupRoutes() {
    // Environment awareness
    app.get('/api/environment', (req, res) => {
      res.json({
        web: { url: 'http://localhost:5173', status: 'active' },
        desktop: { platform: process.platform, arch: process.arch },
        ollama: { url: 'http://localhost:11434', model: 'gemma3-legal' },
        mcp: { server: 'legal-ai-server', status: 'active' }
      });
    });

    // Synthesis through VS Code
    app.post('/api/vscode/synthesize', async (req, res) => {
      try {
        const response = await fetch('http://localhost:5173/api/evidence/synthesize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(req.body)
        });
        
        const result = await response.json();
        res.json({ success: true, ...result });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });

    // RAG Studio VS Code interface
    app.post('/api/vscode/rag', async (req, res) => {
      try {
        const response = await fetch('http://localhost:5173/api/enhanced-rag/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(req.body)
        });
        
        const result = await response.json();
        res.json(result);
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });

    // File context analysis
    app.post('/api/vscode/analyze-file', async (req, res) => {
      const { content, filePath, language } = req.body;
      
      try {
        const response = await fetch('http://localhost:11434/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'gemma3-legal',
            prompt: `Analyze this ${language} file: ${filePath}\n\nContent:\n${content}\n\nProvide legal context analysis:`,
            stream: false
          })
        });
        
        const result = await response.json();
        res.json({ analysis: result.response, filePath, language });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });
  }

  start(port = 3001) {
    app.listen(port, () => {
      console.log(`VS Code Bridge API running on port ${port}`);
    });
  }
}

new VSCodeBridge().start();
