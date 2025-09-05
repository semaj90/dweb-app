import { readdir, stat } from 'fs/promises';
import { join } from 'path';
import type { PageServerLoad } from './$types';

interface RouteInfo {
  path: string;
  hasPageFile: boolean;
  hasServerFile: boolean;
  hasLayoutFile: boolean;
  isParametric: boolean;
  category: string;
  priority: 'high' | 'medium' | 'low';
  description: string;
  status: 'functional' | 'api' | 'empty' | 'layout';
}

const ROUTE_CATEGORIES = {
  core: {
    name: 'Core Application',
    patterns: [/^\/$/, /^\/cases/, /^\/evidence/, /^\/documents/, /^\/legal/, /^\/search/],
    priority: 'high' as const,
    icon: '⚖️',
    description: 'Main legal AI application features'
  },
  auth: {
    name: 'Authentication',
    patterns: [/^\/login/, /^\/register/, /^\/logout/, /^\/profile/],
    priority: 'high' as const,
    icon: '🔐',
    description: 'User authentication and profile management'
  },
  admin: {
    name: 'Administrative',
    patterns: [/^\/admin/],
    priority: 'medium' as const,
    icon: '👨‍💼',
    description: 'System administration and management'
  },
  api: {
    name: 'API Endpoints',
    patterns: [/^\/api/],
    priority: 'high' as const,
    icon: '🔌',
    description: 'Backend API endpoints and services'
  },
  ai: {
    name: 'AI Features',
    patterns: [/^\/ai/, /^\/rag/, /^\/context7/, /^\/semantic/],
    priority: 'high' as const,
    icon: '🤖',
    description: 'Artificial intelligence and machine learning features'
  },
  demos: {
    name: 'Demos & Testing',
    patterns: [/demo/, /test/, /showcase/, /^\/dev/, /^\/perf/],
    priority: 'low' as const,
    icon: '🧪',
    description: 'Development demos, testing, and showcases'
  },
  utilities: {
    name: 'Utility Pages',
    patterns: [/^\/help/, /^\/settings/, /^\/dashboard/, /^\/upload/, /^\/export/, /^\/import/],
    priority: 'medium' as const,
    icon: '⚙️',
    description: 'Utility functions and configuration pages'
  }
};

async function scanRoutesDirectory(dir: string, basePath = ''): Promise<RouteInfo[]> {
  const routes: RouteInfo[] = [];
  
  try {
    const entries = await readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      if (entry.isDirectory() && !entry.name.startsWith('.')) {
        const fullPath = join(dir, entry.name);
        const routePath = basePath + '/' + entry.name;
        
        // Check for page files
        const hasPageFile = await checkForFile(fullPath, '+page.svelte');
        const hasServerFile = await checkForServerFiles(fullPath);
        const hasLayoutFile = await checkForLayoutFiles(fullPath);
        
        const category = categorizeRoute(routePath);
        
        routes.push({
          path: routePath,
          hasPageFile,
          hasServerFile,
          hasLayoutFile,
          isParametric: entry.name.includes('['),
          category: category.name,
          priority: category.priority,
          description: getRouteDescription(routePath, category),
          status: hasPageFile ? 'functional' : hasServerFile ? 'api' : hasLayoutFile ? 'layout' : 'empty'
        });
        
        // Recursively scan subdirectories
        const subRoutes = await scanRoutesDirectory(fullPath, routePath);
        routes.push(...subRoutes);
      }
    }
  } catch (error) {
    console.warn(`Could not scan ${dir}:`, error);
  }
  
  return routes;
}

async function checkForFile(dir: string, filename: string): Promise<boolean> {
  try {
    await stat(join(dir, filename));
    return true;
  } catch {
    return false;
  }
}

async function checkForServerFiles(dir: string): Promise<boolean> {
  const serverFiles = ['+page.server.ts', '+page.server.js', '+layout.server.ts', '+layout.server.js'];
  for (const file of serverFiles) {
    if (await checkForFile(dir, file)) return true;
  }
  return false;
}

async function checkForLayoutFiles(dir: string): Promise<boolean> {
  const layoutFiles = ['+layout.svelte', '+layout.ts', '+layout.js'];
  for (const file of layoutFiles) {
    if (await checkForFile(dir, file)) return true;
  }
  return false;
}

function categorizeRoute(routePath: string) {
  for (const [key, category] of Object.entries(ROUTE_CATEGORIES)) {
    if (category.patterns.some(pattern => pattern.test(routePath))) {
      return category;
    }
  }
  return {
    name: 'Uncategorized',
    priority: 'low' as const,
    icon: '📄',
    description: 'Miscellaneous routes and features'
  };
}

function getRouteDescription(routePath: string, category: any): string {
  // Custom descriptions for specific routes
  const customDescriptions: Record<string, string> = {
    '/': 'YoRHa Legal AI Dashboard - Main system interface',
    '/cases': 'Legal case management and tracking',
    '/evidence': 'Evidence collection and analysis',
    '/documents': 'Document processing and management',
    '/ai': 'AI-powered legal assistance tools',
    '/rag-demo': 'Retrieval Augmented Generation demonstration',
    '/context7-demo': 'Context7 MCP tools testing interface',
    '/demo': 'Component gallery and feature showcase',
    '/admin': 'System administration panel',
    '/login': 'User authentication portal',
    '/all-routes': 'Complete route directory (this page)'
  };

  if (customDescriptions[routePath]) {
    return customDescriptions[routePath];
  }

  // Generate description based on route path
  const routeName = routePath.split('/').pop() || 'root';
  const cleanName = routeName.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  
  return `${cleanName} - ${category.description}`;
}

export const load: PageServerLoad = async () => {
  try {
    const routesDir = join(process.cwd(), 'src', 'routes');
    const allRoutes = await scanRoutesDirectory(routesDir);
    
    // Filter and organize routes
    const functionalRoutes = allRoutes.filter(r => r.hasPageFile);
    const apiRoutes = allRoutes.filter(r => r.hasServerFile && !r.hasPageFile);
    const emptyRoutes = allRoutes.filter(r => !r.hasPageFile && !r.hasServerFile && !r.hasLayoutFile);
    
    // Group by category
    const routesByCategory = allRoutes.reduce((acc, route) => {
      if (!acc[route.category]) {
        acc[route.category] = [];
      }
      acc[route.category].push(route);
      return acc;
    }, {} as Record<string, RouteInfo[]>);
    
    // Generate statistics
    const stats = {
      total: allRoutes.length,
      functional: functionalRoutes.length,
      api: apiRoutes.length,
      empty: emptyRoutes.length,
      categories: Object.keys(routesByCategory).length
    };

    return {
      routes: allRoutes,
      routesByCategory,
      functionalRoutes,
      apiRoutes,
      stats,
      categories: ROUTE_CATEGORIES
    };
  } catch (error) {
    console.error('Failed to load routes:', error);
    return {
      routes: [],
      routesByCategory: {},
      functionalRoutes: [],
      apiRoutes: [],
      stats: { total: 0, functional: 0, api: 0, empty: 0, categories: 0 },
      categories: ROUTE_CATEGORIES
    };
  }
};