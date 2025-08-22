#!/usr/bin/env node

/**
 * Simple Port 5180 Proxy
 * Forwards all requests to your frontend on port 5173
 */

const http = require('http');

console.log('🚀 Starting Port 5180 Proxy...');
console.log('📡 Forwarding to: http://localhost:5173\n');

const server = http.createServer((req, res) => {
  console.log(`📥 ${req.method} ${req.url}`);
  
  // Create proxy request
  const proxyReq = http.request({
    hostname: 'localhost',
    port: 5173,
    path: req.url,
    method: req.method,
    headers: req.headers
  }, (proxyRes) => {
    // Copy response headers
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    
    // Pipe response
    proxyRes.pipe(res);
  });
  
  // Handle errors
  proxyReq.on('error', (err) => {
    console.error('❌ Proxy Error:', err.message);
    if (!res.headersSent) {
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      res.end('Proxy Error: ' + err.message);
    }
  });
  
  // Handle client disconnect
  req.on('close', () => proxyReq.destroy());
  
  // Pipe request body
  req.pipe(proxyReq);
});

// Start server
server.listen(5180, '0.0.0.0', () => {
  console.log('✅ Proxy server running on port 5180');
  console.log('🌐 Access your app at: http://localhost:5180');
  console.log('🔗 Original frontend: http://localhost:5173');
  console.log('\n📋 Status: ACTIVE');
  console.log('   • Port 5180 → Port 5173');
  console.log('   • All requests forwarded');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down...');
  server.close(() => process.exit(0));
});
