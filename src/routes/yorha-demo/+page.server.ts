import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ url, locals }) => {
  // Get system status for the demo navigation
  const systemStatus = {
    services: {
      ollama: 'http://localhost:11434',
      orchestrator: 'http://localhost:40000',
      sveltekit: 'http://localhost:5173',
      redis: 'http://localhost:6379',
      postgres: 'postgresql://localhost:5432'
    },
    features: {
      gpu_acceleration: true,
      context7_mcp: true,
      real_time_websocket: true,
      legal_ai_models: ['gemma3-legal:latest', 'nomic-embed-text'],
      ui_framework: ['bits-ui', 'unocss', 'yorha-design']
    },
    performance: {
      worker_processes: 8,
      thread_pools: 4,
      gpu_layers: 35,
      model_size: '11.8B'
    }
  };

  // Available demo routes for validation
  const availableRoutes = [
    '/production-ai',
    '/gpu-orchestrator',
    '/semantic-3d',
    '/test-gemma3',
    '/auth/register',
    '/auth/login',
    '/auth/profile',
    '/auth/roles',
    '/cases/create',
    '/cases',
    '/evidence/upload',
    '/cases/analysis',
    '/ai/find',
    '/ai/analyze',
    '/ai/search',
    '/ai/chat',
    '/ui/bits-demo',
    '/ui/gaming',
    '/ui/canvas',
    '/ui/dataviz',
    '/dev/mcp-tools',
    '/dev/context7',
    '/dev/health',
    '/dev/performance'
  ];

  return {
    systemStatus,
    availableRoutes,
    timestamp: new Date().toISOString()
  };
};