import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';
import { endpoints, allEndpoints, smokeConfig } from '../smoke.config';

type RouteEntry = { url: string; method?: string } | string;

function parseArgs() {
  const args = process.argv.slice(2);
  const out: any = { 
    crawlRoutes: false, 
    host: smokeConfig.baseUrl || 'http://localhost:5173', 
    routesFile: undefined, 
    clientCheck: false,
    verbose: smokeConfig.verbose || false,
    criticalOnly: smokeConfig.criticalOnly || false
  };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--crawl-routes' || a === '-c') out.crawlRoutes = true;
    else if (a === '--host' && args[i + 1]) { out.host = args[++i]; }
    else if (a === '--routes-file' && args[i + 1]) { out.routesFile = args[++i]; }
    else if (a === '--client-check') out.clientCheck = true;
    else if (a === '--verbose' || a === '-v') out.verbose = true;
    else if (a === '--critical-only') out.criticalOnly = true;
  }
  return out;
}

async function fetchWithTimeout(url: string, opts: any = {}, timeout = smokeConfig.timeout || 7000): Promise<any> {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeout);
  try {
    const res = await fetch(url, { ...opts, signal: ac.signal } as any);
    return res;
  } finally {
    clearTimeout(timer);
  }
}

async function runHealthChecks(verbose = false, criticalOnly = false): Promise<any> {
  console.log('\n🔥 Running Service Health Checks...');
  
  // Use comprehensive endpoint list or legacy endpoints
  const endpointsToCheck = allEndpoints && allEndpoints.length > 0 ? allEndpoints : endpoints;
  
  // Filter critical endpoints if requested
  const filteredEndpoints = criticalOnly 
    ? endpointsToCheck.filter((ep: any) => ep.critical !== false)
    : endpointsToCheck;

  if (verbose) {
    console.log(`Checking ${filteredEndpoints.length} endpoints${criticalOnly ? ' (critical only)' : ''}...`);
  }

  const results = await Promise.all(
    filteredEndpoints.map(async (ep: any) => {
      const endpoint = {
        name: ep.name || ep.url,
        url: ep.url,
        method: ep.method || 'GET',
        expectedStatus: ep.expectedStatus || [200],
        critical: ep.critical !== false,
        category: ep.category || 'unknown'
      };

      try {
        const res = await fetchWithTimeout(endpoint.url, { method: endpoint.method }, smokeConfig.timeout);
        const statusMatch = Array.isArray(endpoint.expectedStatus) 
          ? endpoint.expectedStatus.includes(res.status)
          : endpoint.expectedStatus === res.status;
        
        const result = { 
          ...endpoint,
          status: res.status, 
          ok: statusMatch || res.ok,
          responseTime: Date.now() // Simplified - would need proper timing
        };

        const icon = result.ok ? '✅' : (endpoint.critical ? '❌' : '⚠️');
        const criticalLabel = endpoint.critical ? '[CRITICAL]' : '';
        console.log(`${icon} ${criticalLabel} ${result.name} → ${result.status} (${result.url})`);
        
        if (verbose && !result.ok) {
          console.log(`    Expected: ${JSON.stringify(endpoint.expectedStatus)}, Got: ${result.status}`);
        }

        return result;
      } catch (err: any) {
        const result = { 
          ...endpoint,
          status: 'ERROR', 
          ok: false, 
          error: err?.message || 'Network error',
          responseTime: 0
        };

        const icon = endpoint.critical ? '❌' : '⚠️';
        const criticalLabel = endpoint.critical ? '[CRITICAL]' : '';
        console.log(`${icon} ${criticalLabel} ${result.name} → ERROR (${result.url}) - ${result.error}`);
        
        return result;
      }
    })
  );

  return results.filter((r) => !r.ok);
}

function resolveRoutesFromFile(filePath: string, host: string): RouteEntry[] {
  const abs = path.resolve(process.cwd(), filePath);
  if (!fs.existsSync(abs)) return [];
  const content = fs.readFileSync(abs, 'utf8').trim();
  try {
    const parsed = JSON.parse(content);
    if (Array.isArray(parsed)) return parsed.map((p) => {
      if (typeof p === 'string') return p.replace(/<HOST>/g, host);
      if (p.url) return { url: p.url.replace(/<HOST>/g, host), method: p.method || 'GET' };
      return p;
    });
  } catch (e: any) {
    return content.split(/\r?\n/).map((l) => l.trim()).filter(Boolean).map((r) => r.replace(/<HOST>/g, host));
  }
  return [];
}

function generateRoutesIfNeeded(outFile = 'routes.txt') {
  const gen = path.resolve(process.cwd(), 'scripts', 'generate-routes.mjs');
  const out = path.resolve(process.cwd(), outFile);
  if (fs.existsSync(gen)) {
    console.log('Generating routes file using scripts/generate-routes.mjs ...');
    const r = spawnSync(process.execPath, [gen], { stdio: 'inherit', cwd: process.cwd() });
    if (r.status !== 0) console.warn('Route generation script exited with code', r.status);
    if (fs.existsSync(out)) return out;
  }
  return null;
}

async function runRouteCrawl(opts: { host: string; routesFile?: string; clientCheck?: boolean; verbose?: boolean }): Promise<any> {
  console.log('\n🔍 Running Route Crawl...');
  let routes: RouteEntry[] = [];
  
  if (opts.routesFile) {
    routes = resolveRoutesFromFile(opts.routesFile, opts.host);
    if (opts.verbose) console.log(`Loaded ${routes.length} routes from ${opts.routesFile}`);
  }
  
  if (routes.length === 0) {
    const generated = generateRoutesIfNeeded('routes.txt');
    if (generated) {
      routes = resolveRoutesFromFile(generated, opts.host);
      if (opts.verbose) console.log(`Generated and loaded ${routes.length} routes`);
    }
  }
  
  if (routes.length === 0) {
    const fallback = path.resolve(process.cwd(), '..', 'scripts', 'routes-to-test.json');
    if (fs.existsSync(fallback)) {
      routes = resolveRoutesFromFile(fallback, opts.host);
      if (opts.verbose) console.log(`Loaded ${routes.length} fallback routes`);
    }
  }
  
  if (routes.length === 0) {
    console.log('⚠️  No routes found to crawl. Create a routes.txt or pass --routes-file');
    return [];
  }

  const ssrResults: any[] = [];
  console.log(`Testing ${routes.length} routes...`);

  for (const r of routes) {
    let url = typeof r === 'string' ? `${opts.host}${r}` : (r.url.startsWith('http') ? r.url : `${opts.host}${r.url}`);
    const method = typeof r === 'string' ? 'GET' : (r.method || 'GET');
    
    try {
      const res = await fetchWithTimeout(url, { method }, smokeConfig.timeout);
      ssrResults.push({ url, ok: res.ok, status: res.status });
      console.log(`${res.ok ? '✅' : '❌'} [SSR] ${method} ${url} -> ${res.status}`);
    } catch (err: any) {
      ssrResults.push({ url, ok: false, status: 'ERROR', error: err?.message });
      console.log(`❌ [SSR] ${method} ${url} -> ERROR ${err?.message}`);
    }
  }

  const clientResults: any[] = [];
  if (opts.clientCheck) {
    let playwright: any = null;
    try {
      playwright = await import('playwright');
    } catch (e: any) {
      try { playwright = await import('@playwright/test'); } catch (_) { playwright = null; }
    }
    
    if (!playwright) {
      console.log('⚠️  Playwright not available; skipping client-side checks. Install playwright to enable this.');
    } else {
      console.log('\n⚙️  Running client-side Playwright checks...');
      const b = playwright.chromium || (playwright as any).chromium;
      const browser = await b.launch({ headless: true });
      const context = await browser.newContext();
      const page = await context.newPage();
      
      for (const r of routes.slice(0, 5)) { // Limit client checks to avoid long test times
        const url = typeof r === 'string' ? `${opts.host}${r}` : (r.url.startsWith('http') ? r.url : `${opts.host}${r.url}`);
        try {
          const resp = await page.goto(url, { waitUntil: 'networkidle' as any, timeout: 10000 });
          const ok = resp && resp.status && resp.status() < 400;
          clientResults.push({ url, ok, status: resp?.status() });
          console.log(`${ok ? '✅' : '❌'} [CLIENT] GET ${url} -> ${resp?.status()}`);
        } catch (err: any) {
          clientResults.push({ url, ok: false, error: err?.message });
          console.log(`❌ [CLIENT] GET ${url} -> ERROR ${err?.message}`);
        }
      }
      await browser.close();
    }
  }

  const failures = ssrResults.filter((r) => !r.ok).concat(clientResults.filter((r) => !r.ok));
  return failures;
}

async function main(): Promise<any> {
  const opts = parseArgs();
  console.log('\n🔥 Smoke Test Runner');
  console.log(`Host: ${opts.host}`);
  console.log(`Verbose: ${opts.verbose ? 'ON' : 'OFF'}`);
  console.log(`Critical Only: ${opts.criticalOnly ? 'ON' : 'OFF'}`);
  
  const failedHealth = await runHealthChecks(opts.verbose, opts.criticalOnly);
  let failedRoutes: any[] = [];
  
  if (opts.crawlRoutes) {
    failedRoutes = await runRouteCrawl({ 
      host: opts.host, 
      routesFile: opts.routesFile, 
      clientCheck: opts.clientCheck,
      verbose: opts.verbose
    });
  }

  const totalFailures = failedHealth.length + failedRoutes.length;
  const criticalFailures = failedHealth.filter(f => f.critical).length;

  console.log('\n📊 Smoke Test Results:');
  console.log(`├─ Health checks: ${failedHealth.length} failures`);
  if (criticalFailures > 0) {
    console.log(`├─ Critical failures: ${criticalFailures} ⚠️`);
  }
  if (opts.crawlRoutes) {
    console.log(`├─ Route checks: ${failedRoutes.length} failures`);
  }
  console.log(`└─ Total: ${totalFailures} failures`);

  if (criticalFailures > 0) {
    console.log('\n❌ Smoke test FAILED — Critical services are down');
    process.exit(2);
  } else if (totalFailures > 0) {
    console.log(`\n⚠️  Smoke test completed with ${totalFailures} non-critical issues`);
    process.exit(1);
  } else {
    console.log('\n✅ Smoke test passed — All checks healthy');
    process.exit(0);
  }
}

main().catch((e: any) => {
  console.error('💥 Smoke runner error:', e);
  process.exit(2);
});