
/**
 * Complete Routes Configuration for YoRHa Navigation
 * Comprehensive mapping of all available demo routes and features
 */

export interface RouteDefinition {
  id: string;
  label: string;
  route: string;
  icon: string;
  description: string;
  category: 'main' | 'demo' | 'admin' | 'dev' | 'ai' | 'legal';
  status: 'active' | 'beta' | 'experimental' | 'deprecated';
  tags: string[];
}

export const allRoutes: RouteDefinition[] = [
  // === MAIN OPERATIONS ===
  {
    id: 'command-center',
    label: 'Command Center',
    route: '/',
    icon: '⚡',
    description: 'Enhanced RAG System with AI Model Orchestration',
    category: 'main',
    status: 'active',
    tags: ['rag', 'ai', 'orchestration']
  },
  {
    id: 'cases',
    label: 'Case Management',
    route: '/cases',
    icon: '📁',
    description: 'Legal case management with AI analysis',
    category: 'main',
    status: 'active',
    tags: ['legal', 'cases', 'management']
  },
  {
    id: 'evidence',
    label: 'Evidence Analysis',
    route: '/evidence',
    icon: '🔍',
    description: 'Digital evidence processing with OCR and AI',
    category: 'main',
    status: 'active',
    tags: ['evidence', 'ocr', 'analysis']
  },
  {
    id: 'detective',
    label: 'YoRHa Detective Center',
    route: '/yorha/detective',
    icon: '🕵️',
    description: 'YoRHa-themed detective command center with case management',
    category: 'main',
    status: 'active',
    tags: ['detective', 'yorha', 'cases', 'command-center']
  },
  {
    id: 'persons',
    label: 'Persons of Interest',
    route: '/detective',
    icon: '👤',
    description: 'Person tracking and relationship mapping',
    category: 'main',
    status: 'active',
    tags: ['detective', 'persons', 'tracking']
  },
  {
    id: 'ai-assistant',
    label: 'AI Assistant',
    route: '/ai-assistant',
    icon: '🤖',
    description: 'Multi-agent AI assistant with specialized legal knowledge',
    category: 'ai',
    status: 'active',
    tags: ['ai', 'assistant', 'legal']
  },
  {
    id: 'search',
    label: 'Legal Search',
    route: '/search',
    icon: '🔎',
    description: 'Semantic search across legal documents and case law',
    category: 'main',
    status: 'active',
    tags: ['search', 'semantic', 'legal']
  },
  {
    id: 'documents',
    label: 'Document Processing',
    route: '/legal/documents',
    icon: '📄',
    description: 'Legal document analysis and processing',
    category: 'legal',
    status: 'active',
    tags: ['documents', 'legal', 'processing']
  },
  {
    id: 'reports',
    label: 'Report Generation',
    route: '/reports',
    icon: '📊',
    description: 'Automated report generation with AI insights',
    category: 'main',
    status: 'active',
    tags: ['reports', 'generation', 'ai']
  },
  {
    id: 'memory',
    label: 'Memory Dashboard',
    route: '/memory-dashboard',
    icon: '🧠',
    description: 'AI memory and context management',
    category: 'ai',
    status: 'active',
    tags: ['memory', 'ai', 'context']
  },
  {
    id: 'chat',
    label: 'AI Chat Interface',
    route: '/chat',
    icon: '💬',
    description: 'Interactive chat with legal AI models',
    category: 'ai',
    status: 'active',
    tags: ['chat', 'ai', 'conversation']
  },

  // === AI DEMONSTRATIONS ===
  {
    id: 'demo-overview',
    label: 'Demo Overview',
    route: '/demo',
    icon: '🎯',
    description: 'Overview of all AI demonstrations and capabilities',
    category: 'demo',
    status: 'active',
    tags: ['demo', 'overview', 'ai']
  },
  {
    id: 'demo-ai-assistant',
    label: 'AI Assistant Demo',
    route: '/demo/ai-assistant',
    icon: '🤖',
    description: 'Advanced AI assistant with legal specialization',
    category: 'demo',
    status: 'active',
    tags: ['ai', 'assistant', 'demo']
  },
  {
    id: 'demo-ai-complete-test',
    label: 'AI Complete Test',
    route: '/demo/ai-complete-test',
    icon: '🧪',
    description: 'Comprehensive AI system testing interface',
    category: 'demo',
    status: 'active',
    tags: ['ai', 'testing', 'complete']
  },
  {
    id: 'demo-ai-dashboard',
    label: 'AI Dashboard',
    route: '/demo/ai-dashboard',
    icon: '📊',
    description: 'Real-time AI performance and analytics dashboard',
    category: 'demo',
    status: 'active',
    tags: ['ai', 'dashboard', 'analytics']
  },
  {
    id: 'demo-ai-integration',
    label: 'AI Integration Demo',
    route: '/demo/ai-integration',
    icon: '🔗',
    description: 'Multi-service AI integration showcase',
    category: 'demo',
    status: 'active',
    tags: ['ai', 'integration', 'showcase']
  },
  {
    id: 'demo-ai-pipeline',
    label: 'AI Pipeline',
    route: '/demo/ai-pipeline',
    icon: '⚙️',
    description: 'AI processing pipeline visualization',
    category: 'demo',
    status: 'active',
    tags: ['ai', 'pipeline', 'processing']
  },
  {
    id: 'demo-ai-summary',
    label: 'AI Document Summary',
    route: '/demo/ai-summary',
    icon: '📝',
    description: 'Intelligent document summarization with AI',
    category: 'demo',
    status: 'active',
    tags: ['ai', 'summary', 'documents']
  },
  {
    id: 'demo-ai-test',
    label: 'AI Testing Suite',
    route: '/demo/ai-test',
    icon: '🔬',
    description: 'AI model testing and evaluation tools',
    category: 'demo',
    status: 'active',
    tags: ['ai', 'testing', 'evaluation']
  },
  {
    id: 'demo-component-gallery',
    label: 'Component Gallery',
    route: '/demo/component-gallery',
    icon: '🎨',
    description: 'Showcase of UI components and design patterns',
    category: 'demo',
    status: 'active',
    tags: ['components', 'ui', 'gallery']
  },
  {
    id: 'demo-document-ai',
    label: 'Document AI',
    route: '/demo/document-ai',
    icon: '📑',
    description: 'AI-powered document analysis and extraction',
    category: 'demo',
    status: 'active',
    tags: ['document', 'ai', 'analysis']
  },
  {
    id: 'demo-enhanced-rag-semantic',
    label: 'Enhanced RAG Semantic',
    route: '/demo/enhanced-rag-semantic',
    icon: '🧠',
    description: 'Advanced semantic search with RAG architecture',
    category: 'demo',
    status: 'active',
    tags: ['rag', 'semantic', 'enhanced']
  },
  {
    id: 'demo-enhanced-semantic-architecture',
    label: 'Semantic Architecture',
    route: '/demo/enhanced-semantic-architecture',
    icon: '🏗️',
    description: 'Advanced semantic processing architecture',
    category: 'demo',
    status: 'active',
    tags: ['semantic', 'architecture', 'advanced']
  },
  {
    id: 'demo-gpu-legal-ai',
    label: 'GPU Legal AI',
    route: '/demo/gpu-legal-ai',
    icon: '🚀',
    description: 'GPU-accelerated legal AI processing',
    category: 'demo',
    status: 'active',
    tags: ['gpu', 'legal', 'ai']
  },
  {
    id: 'demo-inline-suggestions',
    label: 'Inline AI Suggestions',
    route: '/demo/inline-suggestions',
    icon: '💡',
    description: 'Real-time AI suggestions and completions',
    category: 'demo',
    status: 'active',
    tags: ['ai', 'suggestions', 'inline']
  },
  {
    id: 'demo-langextract-ollama',
    label: 'LangExtract Ollama',
    route: '/demo/langextract-ollama',
    icon: '🔤',
    description: 'Language extraction with Ollama integration',
    category: 'demo',
    status: 'active',
    tags: ['language', 'extraction', 'ollama']
  },
  {
    id: 'demo-legal-ai-complete',
    label: 'Complete Legal AI',
    route: '/demo/legal-ai-complete',
    icon: '⚖️',
    description: 'Complete legal AI system demonstration',
    category: 'demo',
    status: 'active',
    tags: ['legal', 'ai', 'complete']
  },
  {
    id: 'demo-live-agents',
    label: 'Live AI Agents',
    route: '/demo/live-agents',
    icon: '🤖',
    description: 'Real-time multi-agent AI coordination',
    category: 'demo',
    status: 'active',
    tags: ['agents', 'live', 'coordination']
  },
  {
    id: 'demo-neural-sprite-engine',
    label: 'Neural Sprite Engine',
    route: '/demo/neural-sprite-engine',
    icon: '🎮',
    description: 'Neural network-powered sprite rendering',
    category: 'demo',
    status: 'experimental',
    tags: ['neural', 'sprite', 'engine']
  },
  {
    id: 'demo-notes',
    label: 'AI Note Taking',
    route: '/demo/notes',
    icon: '📝',
    description: 'Intelligent note-taking with AI enhancement',
    category: 'demo',
    status: 'active',
    tags: ['notes', 'ai', 'taking']
  },
  {
    id: 'demo-phase5',
    label: 'Phase 5 Demo',
    route: '/demo/phase5',
    icon: '🚀',
    description: 'Phase 5 development milestone demonstration',
    category: 'demo',
    status: 'active',
    tags: ['phase5', 'milestone', 'demo']
  },
  {
    id: 'demo-professional-editor',
    label: 'Professional Editor',
    route: '/demo/professional-editor',
    icon: '✏️',
    description: 'Professional document editing with AI assistance',
    category: 'demo',
    status: 'active',
    tags: ['editor', 'professional', 'ai']
  },
  {
    id: 'demo-simple-test',
    label: 'Simple Test Interface',
    route: '/demo/simple-test',
    icon: '🔍',
    description: 'Simple testing interface for quick validation',
    category: 'demo',
    status: 'active',
    tags: ['simple', 'test', 'interface']
  },
  {
    id: 'demo-system-summary',
    label: 'System Summary',
    route: '/demo/system-summary',
    icon: '📋',
    description: 'Comprehensive system status and metrics',
    category: 'demo',
    status: 'active',
    tags: ['system', 'summary', 'metrics']
  },
  {
    id: 'demo-unified-architecture',
    label: 'Unified Architecture',
    route: '/demo/unified-architecture',
    icon: '🏛️',
    description: 'Unified system architecture demonstration',
    category: 'demo',
    status: 'active',
    tags: ['unified', 'architecture', 'system']
  },
  {
    id: 'demo-unocss-svelte5',
    label: 'UnoCSS + Svelte 5',
    route: '/demo/unocss-svelte5',
    icon: '🎨',
    description: 'UnoCSS styling with Svelte 5 features',
    category: 'demo',
    status: 'active',
    tags: ['unocss', 'svelte5', 'styling']
  },
  {
    id: 'demo-vector-intelligence',
    label: 'Vector Intelligence',
    route: '/demo/vector-intelligence',
    icon: '🧮',
    description: 'Vector-based intelligence and similarity search',
    category: 'demo',
    status: 'active',
    tags: ['vector', 'intelligence', 'search']
  },
  {
    id: 'demo-vector-search',
    label: 'Vector Search',
    route: '/demo/vector-search',
    icon: '🔍',
    description: 'Advanced vector similarity search interface',
    category: 'demo',
    status: 'active',
    tags: ['vector', 'search', 'similarity']
  },
  {
    id: 'demo-webgpu-acceleration',
    label: 'WebGPU Acceleration',
    route: '/demo/webgpu-acceleration',
    icon: '⚡',
    description: 'WebGPU-accelerated AI processing',
    category: 'demo',
    status: 'experimental',
    tags: ['webgpu', 'acceleration', 'ai']
  },
  {
    id: 'demo-yorha-tables',
    label: 'YoRHa Tables',
    route: '/demo/yorha-tables',
    icon: '📊',
    description: 'YoRHa-themed data tables and visualization',
    category: 'demo',
    status: 'active',
    tags: ['yorha', 'tables', 'visualization']
  },

  // === DEVELOPER TOOLS ===
  {
    id: 'dev-mcp-tools',
    label: 'MCP Tools',
    route: '/dev/mcp-tools',
    icon: '🔧',
    description: 'Model Context Protocol development tools',
    category: 'dev',
    status: 'active',
    tags: ['mcp', 'tools', 'development']
  },
  {
    id: 'dev-context7-test',
    label: 'Context7 Test',
    route: '/dev/context7-test',
    icon: '🧪',
    description: 'Context7 integration testing interface',
    category: 'dev',
    status: 'active',
    tags: ['context7', 'test', 'integration']
  },
  {
    id: 'dev-copilot-optimizer',
    label: 'Copilot Optimizer',
    route: '/dev/copilot-optimizer',
    icon: '🚁',
    description: 'AI copilot optimization and tuning',
    category: 'dev',
    status: 'active',
    tags: ['copilot', 'optimizer', 'ai']
  },
  {
    id: 'dev-self-prompting',
    label: 'Self-Prompting Demo',
    route: '/dev/self-prompting-demo',
    icon: '🔄',
    description: 'Self-prompting AI system demonstration',
    category: 'dev',
    status: 'experimental',
    tags: ['self-prompting', 'ai', 'demo']
  },
  {
    id: 'dev-vector-search',
    label: 'Vector Search Dev',
    route: '/dev/vector-search-demo',
    icon: '🔍',
    description: 'Vector search development and testing tools',
    category: 'dev',
    status: 'active',
    tags: ['vector', 'search', 'development']
  },
  {
    id: 'dev-dynamic-routing-test',
    label: 'Dynamic Routing Test',
    route: '/dev/dynamic-routing-test',
    icon: '🛣️',
    description: 'Dynamic routing system testing and demonstration',
    category: 'dev',
    status: 'active',
    tags: ['routing', 'dynamic', 'testing']
  },
  {
    id: 'routes-index',
    label: 'Routes Index',
    route: '/routes',
    icon: '🗺️',
    description: 'Complete navigation index of all available routes and APIs',
    category: 'admin',
    status: 'active',
    tags: ['navigation', 'index', 'routes', 'admin', 'api']
  },

  // === ADMINISTRATIVE ===
  {
    id: 'settings',
    label: 'System Settings',
    route: '/settings',
    icon: '⚙️',
    description: 'System configuration and preferences',
    category: 'admin',
    status: 'active',
    tags: ['settings', 'configuration', 'system']
  },
  {
    id: 'security',
    label: 'Security Center',
    route: '/security',
    icon: '🛡️',
    description: 'Security monitoring and access control',
    category: 'admin',
    status: 'active',
    tags: ['security', 'monitoring', 'access']
  },
  {
    id: 'profile',
    label: 'User Profile',
    route: '/profile',
    icon: '👤',
    description: 'User profile management and preferences',
    category: 'admin',
    status: 'active',
    tags: ['profile', 'user', 'management']
  },
  {
    id: 'help',
    label: 'Help & Documentation',
    route: '/help',
    icon: '❓',
    description: 'System help and documentation',
    category: 'admin',
    status: 'active',
    tags: ['help', 'documentation', 'support']
  },

  // === SPECIALIZED INTERFACES ===
  {
    id: 'enhanced-rag',
    label: 'Enhanced RAG Demo',
    route: '/enhanced-ai-demo',
    icon: '🚀',
    description: 'Enhanced RAG system with advanced AI capabilities',
    category: 'demo',
    status: 'active',
    tags: ['rag', 'enhanced', 'ai']
  },
  {
    id: 'context7',
    label: 'Context7 Integration',
    route: '/context7-demo',
    icon: '🔗',
    description: 'Context7 MCP integration demonstration',
    category: 'demo',
    status: 'active',
    tags: ['context7', 'mcp', 'integration']
  },
  {
    id: 'gaming-demo',
    label: 'Gaming Interface',
    route: '/gaming-demo',
    icon: '🎮',
    description: 'Gaming-style interface with NieR Automata theme',
    category: 'demo',
    status: 'active',
    tags: ['gaming', 'interface', 'nier']
  },
  {
    id: 'yorha-dashboard',
    label: 'YoRHa Dashboard',
    route: '/yorha-dashboard',
    icon: '🤖',
    description: 'Complete YoRHa-themed administrative dashboard',
    category: 'demo',
    status: 'active',
    tags: ['yorha', 'dashboard', 'admin']
  }
];

// Helper functions for route management
export function getRoutesByCategory(category: RouteDefinition['category']): RouteDefinition[] {
  return allRoutes.filter(route => route.category === category);
}

export function getActiveRoutes(): RouteDefinition[] {
  return allRoutes.filter(route => route.status === 'active');
}

export function searchRoutes(query: string): RouteDefinition[] {
  const lowerQuery = query.toLowerCase();
  return allRoutes.filter(route =>
    route.label.toLowerCase().includes(lowerQuery) ||
    route.description.toLowerCase().includes(lowerQuery) ||
    route.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
  );
}

export function getRouteById(id: string): RouteDefinition | undefined {
  return allRoutes.find(route => route.id === id);
}

export function getRoutesByTag(tag: string): RouteDefinition[] {
  return allRoutes.filter(route => route.tags.includes(tag));
}

// Route categories with metadata
export const routeCategories = {
  main: {
    label: 'CORE OPERATIONS',
    icon: '⚡',
    description: 'Primary system operations and tools',
    color: '#ffbf00'
  },
  demo: {
    label: 'AI DEMONSTRATIONS',
    icon: '🎯',
    description: 'AI capabilities and technology showcases',
    color: '#00ff41'
  },
  ai: {
    label: 'AI SYSTEMS',
    icon: '🤖',
    description: 'Artificial intelligence tools and interfaces',
    color: '#ff6b6b'
  },
  legal: {
    label: 'LEGAL OPERATIONS',
    icon: '⚖️',
    description: 'Legal-specific tools and workflows',
    color: '#4ecdc4'
  },
  dev: {
    label: 'DEVELOPMENT TOOLS',
    icon: '🔧',
    description: 'Development and debugging utilities',
    color: '#a78bfa'
  },
  admin: {
    label: 'ADMINISTRATION',
    icon: '⚙️',
    description: 'System administration and configuration',
    color: '#fb7185'
  }
};

// Statistics for dashboard
export const routeStats = {
  total: allRoutes.length,
  active: getActiveRoutes().length,
  experimental: allRoutes.filter(r => r.status === 'experimental').length,
  beta: allRoutes.filter(r => r.status === 'beta').length,
  deprecated: allRoutes.filter(r => r.status === 'deprecated').length,
  byCategory: Object.keys(routeCategories).reduce((acc, category) => {
    acc[category] = getRoutesByCategory(category as RouteDefinition['category']).length;
    return acc;
  }, {} as Record<string, number>)
};

// Additional interfaces and types for compatibility
export interface DynamicRouteConfig {
  path: string;
  component?: any;
  metadata?: Record<string, any>;
}

export interface GeneratedRoute {
  path: string;
  handler: any;
  config: DynamicRouteConfig;
}

export interface NavigationGuard {
  name: string;
  condition: (route: RouteDefinition) => boolean;
}

// Route registry for dynamic route management
export const routeRegistry = new Map<string, RouteDefinition>();
// Initialize route registry
allRoutes.forEach(route => {
  routeRegistry.set(route.id, route);
});

// Dynamic route management functions
export function getRoute(id: string): RouteDefinition | undefined {
  return routeRegistry.get(id) || getRouteById(id);
}

export function getAllDynamicRoutes(): RouteDefinition[] {
  return Array.from(routeRegistry.values());
}

export function registerDynamicRoute(config: DynamicRouteConfig): GeneratedRoute {
  return {
    path: config.path,
    handler: config.component,
    config
  };
}