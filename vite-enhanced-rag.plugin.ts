
// Enhanced RAG Vite Plugin
import type { Plugin } from 'vite';

export const enhancedRagPlugin = (): Plugin => ({
  name: 'enhanced-rag-logging',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      console.log(`[VITE RAG] ${req.method} ${req.url}`);
      next();
    });

    server.watcher.on('error', (err) => {
      const errorData = {
        timestamp: new Date().toISOString(),
        type: 'vite-watch-error',
        message: err.message,
        stack: err.stack,
        source: 'vite-watcher'
      };
      
      console.error('[VITE RAG ERROR]', errorData);
      
      // Write to enhanced RAG log
      require('fs').appendFileSync('enhanced-rag-2025-08-04T05-38-20\logs\vite-errors.log', 
        JSON.stringify(errorData) + '\n'
      );
    });
  },
  
  handleHotUpdate(ctx) {
    console.log(`[VITE RAG HMR] ${ctx.file}`);
  }
});