#!/usr/bin/env node

/**
 * Port 5180 Proxy Server
 * Forwards requests to frontend running on port 5173
 */

const http = require('http');
const httpProxy = require('http-proxy');

console.log('🚀 Starting Port 5180 Proxy Server...');
console.log('📡 Forwarding requests from port 5180 to port 5173\n');

// Create a proxy server
const proxy = httpProxy.createProxyServer({
  target: 'http://localhost:5173',
  changeOrigin: true,
  ws: true // Support WebSocket connections
});

// Handle proxy errors
proxy.on('error', (err, req, res) => {
  console.error('❌ Proxy Error:', err.message);
  if (!res.headersSent) {
    res.writeHead(500, { 'Content-Type': 'text/plain' });
    res.end('Proxy Error: ' + err.message);
  }
});

// Create the proxy server
const server = http.createServer((req, res) => {
  console.log(`📥 ${req.method} ${req.url} -> Port 5173`);
  
  // Forward the request to port 5173
  proxy.web(req, res, { target: 'http://localhost:5173' });
});

// Handle WebSocket upgrades
server.on('upgrade', (req, socket, head) => {
  console.log(`🔌 WebSocket upgrade: ${req.url}`);
  proxy.ws(req, socket, head);
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
  console.log('   • WebSocket support enabled');
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
