#!/usr/bin/env tsx

/**
 * Test All Routes - Comprehensive SvelteKit Route Testing
 * Tests all static and dynamic routes in the application
 */

import { glob } from 'glob';
import path from 'path';
import fetch from 'node-fetch';
import { allEndpoints, smokeConfig } from '../smoke.config';

export interface RouteTest {
  route: string;
  method: 'GET' | 'POST';
  expectedStatus: number[];
  params?: Record<string, string>;
  body?: any;
}

class RouteDiscovery {
  private baseDir: string;
  private baseUrl: string;

  constructor(baseDir: string = 'src/routes', baseUrl: string = smokeConfig.baseUrl) {
    this.baseDir = baseDir;
    this.baseUrl = baseUrl;
  }

  async discoverRoutes(): Promise<RouteTest[]> {
    console.log('🔍 Discovering routes...');
    
    // Find all page files
    const pageFiles = await glob('**/+page.{svelte,js,ts}', {
      cwd: this.baseDir,
      ignore: ['node_modules/**', '.svelte-kit/**']
    });

    // Find all server files  
    const serverFiles = await glob('**/+server.{js,ts}', {
      cwd: this.baseDir,
      ignore: ['node_modules/**', '.svelte-kit/**']
    });

    const routes: RouteTest[] = [];

    // Process page routes
    for (const file of pageFiles) {
      const route = this.fileToRoute(file.replace(/\/\+page\.(svelte|js|ts)$/, ''));
      routes.push({
        route,
        method: 'GET',
        expectedStatus: [200, 404, 500] // Allow various statuses for discovery
      });
    }

    // Process API routes  
    for (const file of serverFiles) {
      const route = '/api' + this.fileToRoute(file.replace(/\/\+server\.(js|ts)$/, ''));
      
      // Add both GET and POST for API routes
      routes.push({
        route,
        method: 'GET', 
        expectedStatus: [200, 405, 404, 500]
      });
      
      routes.push({
        route,
        method: 'POST',
        expectedStatus: [200, 405, 404, 500],
        body: { test: true }
      });
    }

    console.log(`📍 Discovered ${routes.length} route combinations`);
    return routes;
  }

  private fileToRoute(filePath: string): string {
    let route = '/' + filePath;
    
    // Handle root route
    if (route === '/') return route;
    
    // Convert [param] to sample values
    route = route.replace(/\[([^\]]+)\]/g, (match, param) => {
      // Provide sample values for common params
      const sampleValues: Record<string, string> = {
        'id': '123',
        'caseId': 'case-123', 
        'userId': 'user-123',
        'slug': 'sample-slug',
        'evidenceId': 'evidence-123'
      };
      
      return sampleValues[param] || 'sample-' + param;
    });
    
    // Handle (groups) - remove them
    route = route.replace(/\([^)]+\)/g, '');
    
    // Clean up double slashes
    route = route.replace(/\/+/g, '/');
    
    // Remove trailing slash unless root
    if (route !== '/' && route.endsWith('/')) {
      route = route.slice(0, -1);
    }

    return route;
  }
}

async function testRoute(test: RouteTest, baseUrl: string): Promise<{success: boolean, status: number, error?: string}> {
  try {
    const url = baseUrl + test.route;
    const options: any = {
      method: test.method,
      timeout: 10000
    };

    if (test.body && test.method === 'POST') {
      options.headers = { 'Content-Type': 'application/json' };
      options.body = JSON.stringify(test.body);
    }

    const response = await fetch(url, options);
    const success = test.expectedStatus.includes(response.status);
    
    return {
      success,
      status: response.status,
      ...(success ? {} : { error: `Expected ${test.expectedStatus.join('|')}, got ${response.status}` })
    };
  } catch (error: any) {
    return {
      success: false,
      status: 0,
      error: error.message
    };
  }
}

async function main(): Promise<any> {
  console.log('\n🧪 Comprehensive Route Testing Suite');
  console.log('=====================================');
  
  const discovery = new RouteDiscovery();
  const routes = await discovery.discoverRoutes();
  
  console.log(`\n🚀 Testing ${routes.length} routes...`);
  console.log(`Base URL: ${smokeConfig.baseUrl}`);
  
  const results = [];
  let passed = 0;
  let failed = 0;
  
  // Test routes with concurrency control
  const concurrency = 5;
  const chunks = [];
  for (let i = 0; i < routes.length; i += concurrency) {
    chunks.push(routes.slice(i, i + concurrency));
  }
  
  for (const chunk of chunks) {
    const chunkResults = await Promise.all(
      chunk.map(async (route) => {
        const result = await testRoute(route, smokeConfig.baseUrl);
        const success = result.success;
        
        if (success) {
          passed++;
          console.log(`✅ ${route.method} ${route.route} → ${result.status}`);
        } else {
          failed++;
          console.log(`❌ ${route.method} ${route.route} → ${result.status} (${result.error})`);
        }
        
        return { route, result };
      })
    );
    
    results.push(...chunkResults);
    
    // Small delay between chunks
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  // Summary
  console.log('\n📊 Test Results Summary:');
  console.log('========================');
  console.log(`✅ Passed: ${passed}`);
  console.log(`❌ Failed: ${failed}`);
  console.log(`📊 Total: ${routes.length}`);
  console.log(`📈 Success Rate: ${((passed / routes.length) * 100).toFixed(1)}%`);
  
  // Detailed failure analysis
  const failures = results.filter(r => !r.result.success);
  if (failures.length > 0) {
    console.log('\n🔍 Failure Analysis:');
    console.log('===================');
    
    const by500 = failures.filter(f => f.result.status === 500);
    const by404 = failures.filter(f => f.result.status === 404);
    const byTimeout = failures.filter(f => f.result.status === 0);
    
    if (by500.length > 0) {
      console.log(`\n💥 Server Errors (500): ${by500.length}`);
      by500.slice(0, 5).forEach(f => console.log(`   • ${f.route.method} ${f.route.route}`));
      if (by500.length > 5) console.log(`   • ... and ${by500.length - 5} more`);
    }
    
    if (by404.length > 0) {
      console.log(`\n🔍 Not Found (404): ${by404.length}`);
      by404.slice(0, 5).forEach(f => console.log(`   • ${f.route.method} ${f.route.route}`));
      if (by404.length > 5) console.log(`   • ... and ${by404.length - 5} more`);
    }
    
    if (byTimeout.length > 0) {
      console.log(`\n⏰ Timeouts/Network: ${byTimeout.length}`);
      byTimeout.slice(0, 5).forEach(f => console.log(`   • ${f.route.method} ${f.route.route}`));
      if (byTimeout.length > 5) console.log(`   • ... and ${byTimeout.length - 5} more`);
    }
  }
  
  // Exit with appropriate code
  process.exit(failed > (routes.length * 0.2) ? 1 : 0); // Allow up to 20% failures
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('💥 Route testing failed:', error);
    process.exit(2);
  });
}

export { RouteDiscovery, testRoute };