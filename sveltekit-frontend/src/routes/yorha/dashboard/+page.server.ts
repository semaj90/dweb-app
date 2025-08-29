import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ url, fetch }) => {
  try {
    // Fetch initial system status for SSR
    const response = await fetch('/api/yorha/system/status');
    let systemStatus;
    
    if (response.ok) {
      systemStatus = await response.json();
    } else {
      // Fallback to mock data
      systemStatus = generateMockSystemStatus();
    }

    // Provide mock multicore status for SSR compatibility
    const multicoreStatus = {
      totalWorkers: 4,
      healthyWorkers: 4,
      busyWorkers: 0,
      queueSize: 0,
      activeTasks: 0,
      totalTasks: 1234,
      completedTasks: 1230,
      failedTasks: 4
    };

    // Generate initial graph data
    const graphData = generateSystemGraphData();

    return {
      systemStatus,
      multicoreStatus,
      graphData,
      initialLoad: true,
      timestamp: new Date().toISOString()
    };
  } catch (error: any) {
    console.error('Error loading dashboard data:', error);
    
    // Return fallback data
    return {
      systemStatus: generateMockSystemStatus(),
      multicoreStatus: null,
      graphData: generateSystemGraphData(),
      initialLoad: false,
      timestamp: new Date().toISOString(),
      error: 'Failed to load system data'
    };
  }
};

function generateMockSystemStatus() {
  return {
    database: {
      connected: true,
      latency: Math.floor(Math.random() * 50) + 15,
      activeConnections: Math.floor(Math.random() * 15) + 8,
      queryCount: Math.floor(Math.random() * 5000) + 10000
    },
    backend: {
      healthy: true,
      uptime: Math.floor(Date.now() / 1000) - (Math.random() * 86400),
      activeServices: 8,
      cpuUsage: Math.floor(Math.random() * 30) + 35,
      memoryUsage: Math.floor(Math.random() * 25) + 55
    },
    frontend: {
      renderFPS: Math.floor(Math.random() * 10) + 55,
      componentCount: 778,
      activeComponents: Math.floor(Math.random() * 50) + 150,
      webGPUEnabled: true
    },
    timestamp: new Date().toISOString(),
    systemLoad: Math.floor(Math.random() * 30) + 45,
    gpuUtilization: Math.floor(Math.random() * 20) + 78,
    networkLatency: Math.floor(Math.random() * 30) + 20
  };
}

function generateSystemGraphData() {
  return {
    nodes: [
      {
        id: 'postgres',
        type: 'database',
        label: 'PostgreSQL + pgvector',
        position: { x: 0, y: 0, z: 0 },
        metrics: { connections: 12, queries: 15847 },
        status: 'healthy'
      },
      {
        id: 'redis',
        type: 'database',
        label: 'Redis Cache',
        position: { x: 2, y: 0, z: 0 },
        metrics: { memory: '2.1GB', keys: 45823 },
        status: 'healthy'
      },
      {
        id: 'neo4j',
        type: 'database',
        label: 'Neo4j Graph',
        position: { x: 1, y: 0, z: 1 },
        metrics: { nodes: 15847, relationships: 89423 },
        status: 'healthy'
      },
      {
        id: 'ollama',
        type: 'service',
        label: 'Ollama Cluster',
        position: { x: 0, y: 1, z: 0 },
        metrics: { models: 5, requests: 1847 },
        status: 'healthy'
      },
      {
        id: 'enhanced-rag',
        type: 'service',
        label: 'Enhanced RAG',
        position: { x: 1, y: 1, z: 0 },
        metrics: { port: 8094, uptime: '99.9%' },
        status: 'healthy'
      },
      {
        id: 'upload-service',
        type: 'service',
        label: 'Upload Service',
        position: { x: 2, y: 1, z: 0 },
        metrics: { port: 8093, files: 2847 },
        status: 'healthy'
      },
      {
        id: 'sveltekit',
        type: 'component',
        label: 'SvelteKit Frontend',
        position: { x: 1, y: 2, z: 0 },
        metrics: { components: 778, fps: 60 },
        status: 'healthy'
      },
      {
        id: 'context7',
        type: 'service',
        label: 'Context7 Multicore',
        position: { x: 0, y: 1, z: 1 },
        metrics: { workers: 4, tasks: 1234 },
        status: 'healthy'
      }
    ],
    edges: [
      {
        from: 'sveltekit',
        to: 'postgres',
        type: 'api',
        traffic: 85,
        latency: 23
      },
      {
        from: 'sveltekit',
        to: 'redis',
        type: 'api',
        traffic: 65,
        latency: 12
      },
      {
        from: 'enhanced-rag',
        to: 'postgres',
        type: 'data',
        traffic: 90,
        latency: 15
      },
      {
        from: 'enhanced-rag',
        to: 'ollama',
        type: 'grpc',
        traffic: 75,
        latency: 8
      },
      {
        from: 'upload-service',
        to: 'postgres',
        type: 'data',
        traffic: 45,
        latency: 18
      },
      {
        from: 'context7',
        to: 'ollama',
        type: 'api',
        traffic: 55,
        latency: 25
      },
      {
        from: 'postgres',
        to: 'neo4j',
        type: 'data',
        traffic: 30,
        latency: 35
      }
    ]
  };
}