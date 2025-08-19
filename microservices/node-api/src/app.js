import { createServer } from 'node:http';
import { parse } from 'node:url';

const PORT = process.env.PORT || process.env.NODE_API_PORT || 3000;

// Simple health check handler
const healthHandler = async () => {
  return new Response(JSON.stringify({ 
    status: 'ok', 
    service: 'node-api',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  }), { 
    status: 200, 
    headers: { 'content-type': 'application/json' } 
  });
};

const clusterHandler = async () => {
  return new Response(JSON.stringify({ 
    cluster: 'active',
    nodes: ['node-1'],
    services: ['nats', 'api']
  }), { 
    status: 200, 
    headers: { 'content-type': 'application/json' } 
  });
};

// Simple router
const routes = new Map([
  ['GET:/healthz', healthHandler],
  ['GET:/health', healthHandler],
  ['GET:/api/v1/cluster', clusterHandler],
  ['POST:/api/v1/cluster', clusterHandler]
]);

const server = createServer(async (req, res) => {
  const { pathname } = parse(req.url);
  const routeKey = `${req.method}:${pathname}`;
  
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }
  
  const handler = routes.get(routeKey);
  
  if (handler) {
    try {
      const response = await handler();
      
      // Send the response
      const responseBody = await response.text();
      const headers = Object.fromEntries(response.headers.entries());
      res.writeHead(response.status, headers);
      res.end(responseBody);
    } catch (error) {
      console.error('Route handler error:', error);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Internal server error' }));
    }
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Route not found' }));
  }
});


server.listen(PORT, () => {
  console.log(`🚀 Node API microservice running on port ${PORT}`);
  console.log(`📍 Health check: http://localhost:${PORT}/healthz`);
  console.log(`📍 Cluster API: http://localhost:${PORT}/api/v1/cluster`);
  console.log(`📍 NATS API: http://localhost:${PORT}/api/v1/nats/*`);
});

export default server;