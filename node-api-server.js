// Simple Node.js API server for Enhanced RAG V2
// This replaces the Go service temporarily until Go is installed

const http = require('http');
const url = require('url');

const PORT = 8084;

// Simple in-memory storage
const documents = [];
const cache = new Map();

// Create server
const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const path = parsedUrl.pathname;
    const method = req.method;
    
    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    if (method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }
    
    // Health check endpoint
    if (path === '/api/health' || path === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            status: 'healthy',
            service: 'Enhanced RAG V2 (Node.js)',
            timestamp: new Date().toISOString(),
            endpoints: [
                '/api/health',
                '/api/chat',
                '/api/ai/summarize',
                '/api/metrics'
            ]
        }));
        return;
    }
    
    // Chat endpoint
    if (path === '/api/chat' && method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const response = {
                    message: `Response to: ${data.message || 'Hello'}`,
                    timestamp: new Date().toISOString(),
                    model: 'mock-enhanced-rag-v2'
                };
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(response));
            } catch (e) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Invalid JSON' }));
            }
        });
        return;
    }
    
    // Summarize endpoint
    if (path === '/api/ai/summarize' && method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const response = {
                    summary: `Summary of document: ${(data.text || '').substring(0, 100)}...`,
                    wordCount: (data.text || '').split(' ').length,
                    timestamp: new Date().toISOString()
                };
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(response));
            } catch (e) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Invalid JSON' }));
            }
        });
        return;
    }
    
    // Metrics endpoint
    if (path === '/api/metrics') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            uptime: process.uptime(),
            memory: process.memoryUsage(),
            requests_total: cache.size,
            documents_processed: documents.length,
            timestamp: new Date().toISOString()
        }));
        return;
    }
    
    // Default 404
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
});

// Start server
server.listen(PORT, () => {
    console.log(`
╔════════════════════════════════════════╗
║   Enhanced RAG V2 - Node.js Service   ║
╚════════════════════════════════════════╝

[✓] Server running on port ${PORT}
[✓] Health check: http://localhost:${PORT}/api/health
[✓] Chat API: http://localhost:${PORT}/api/chat
[✓] Summarize API: http://localhost:${PORT}/api/ai/summarize
[✓] Metrics: http://localhost:${PORT}/api/metrics

Press Ctrl+C to stop the server
    `);
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down gracefully...');
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});