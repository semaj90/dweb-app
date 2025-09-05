// vite.config.enhanced-rag.example.ts
// Example integration of Enhanced RAG plugin with SvelteKit 2

import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import { enhancedRagPlugin } from './vite-enhanced-rag.plugin';

export default defineConfig({
  plugins: [
    sveltekit(),
    
    // Enhanced RAG Plugin - Add this to your existing vite.config.ts
    enhancedRagPlugin({
      logFile: 'enhanced-rag/logs/vite-errors.log',
      enableErrorEmbedding: true,
      enableDevFeedback: true,
      ollamaHost: 'http://localhost:11434',
      embeddingModel: 'nomic-embed-text'
    })
  ],
  
  server: {
    // Enhanced development server with RAG logging
    host: '0.0.0.0',
    port: 5173,
    
    // Custom middleware for RAG integration
    middlewareMode: false
  },
  
  build: {
    // Optimize for legal AI application
    target: 'esnext',
    sourcemap: true,
    
    rollupOptions: {
      output: {
        // Code splitting for enhanced RAG components
        manualChunks: {
          'enhanced-rag': [
            'lokijs',
            'fuse.js',
            'src/lib/components/enhanced-rag/EnhancedRAGInterface.svelte'
          ]
        }
      }
    }
  },
  
  define: {
    // Environment variables for RAG system
    __OLLAMA_HOST__: JSON.stringify(process.env.OLLAMA_HOST || 'http://localhost:11434'),
    __EMBED_MODEL__: JSON.stringify('nomic-embed-text'),
    __LEGAL_MODEL__: JSON.stringify('gemma3-legal')
  }
});

// To integrate this into your existing vite.config.ts:
// 1. Copy the enhancedRagPlugin() to your plugins array
// 2. Add the build.rollupOptions.output.manualChunks for code splitting
// 3. Include the define block for environment variables
// 4. Optionally add the server configuration for development