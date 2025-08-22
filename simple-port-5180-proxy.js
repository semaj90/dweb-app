#!/usr/bin/env node

/**
 * Simple Port 5180 Proxy Server
 * Forwards requests to frontend running on port 5173
 * Uses only built-in Node.js modules
 */

const http = require('http');
const url = require('url');

console.log('🚀 Starting Simple Port 5180 Proxy Server...');
console.log('📡 Forwarding requests from port 5180 to port 5173\n');

// Create a simple proxy server
const server = http.createServer((req, res) => {
  const targetUrl = `http://localhost:5173${req.url}`;
  
  console.log(`📥 ${req.method} ${req.url} -> Port 5173`);
  
  // Create request options
  const options = {
    hostname: 'localhost',
    port: 5173,
    path: req.url,
    method: req.method,
    headers: req.headers
  };
  
  // Forward the request
  const proxyReq = http.request(options, (proxyRes) => {
    // Copy response headers
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    
    // Pipe the response body
    proxyRes.pipe(res);
  });
  
  // Handle proxy request errors
  proxyReq.on('error', (err) => {
    console.error('❌ Proxy Request Error:', err.message);
    if (!res.headersSent) {
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      res.end('Proxy Error: ' + err.message);
    }
  });
  
  // Handle client disconnect
  req.on('close', () => {
    proxyReq.destroy();
  });
  
  // Pipe the request body
  req.pipe(proxyReq);
});

// Start the server on port 5180
const PORT = 5180;
server.listen(PORT, () => {
  console.log(`✅ Proxy server running on port ${PORT}`);
  console.log(`🌐 Access your app at: http://localhost:${PORT}`);
  console.log(`🔗 Original frontend: http://localhost:5173`);
  console.log('\n📋 Proxy Status: ACTIVE');
  console.log('   • Port 5180 → Port 5173');
  console.log('   • All requests forwarded');
  console.log('   • Simple HTTP proxy');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down proxy server...');
  server.close(() => {
    console.log('✅ Proxy server stopped');
    process.exit(0);
  });
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Shutting down proxy server...');
  server.close(() => {
    console.log('✅ Proxy server stopped');
    process.exit(0);
  });
});
