// Context7 Autosolve API - Complete TypeScript Error Fixing Integration
// REST API for managing the autosolve system with Context7 best practices

import { json, type RequestHandler } from "@sveltejs/kit";
// Orphaned content: import {

import { databaseOrchestrator } from "$lib/services/comprehensive-database-orchestrator";
// Orphaned content: import {

// GET /api/context7-autosolve - Get autosolve system status
export const GET: RequestHandler = async ({ url }) => {
  try {
    const action = url.searchParams.get('action');
    const limit = parseInt(url.searchParams.get('limit') || '10');

    switch (action) {
      case 'status':
        return json({
          success: true,
          autosolve_status: context7AutosolveIntegration.getStatus(),
          orchestrator_status: databaseOrchestrator.getStatus(),
          timestamp: new Date().toISOString(),
        });

      case 'history':
        const history = await context7AutosolveIntegration.getAutosolvHistory(limit);
        return json({
          success: true,
          history,
          count: history.length,
          limit,
          timestamp: new Date().toISOString(),
        });

      case 'health':
        return await performComprehensiveHealthCheck();

      default:
        // Default: return comprehensive status
        const status = context7AutosolveIntegration.getStatus();
        const recentHistory = await context7AutosolveIntegration.getAutosolvHistory(5);

        return json({
          success: true,
          message: 'Context7 Autosolve System Active',
          status,
          recent_cycles: recentHistory,
          capabilities: [
            'Automatic TypeScript error detection',
            'AI-powered fix recommendations',
            'Ollama summary generation',
            'PostgreSQL persistence',
            'Real-time health monitoring',
            'Context7 best practices integration',
          ],
          endpoints: {
            trigger: 'POST /api/context7-autosolve (action: trigger)',
            health: 'GET /api/context7-autosolve?action=health',
            history: 'GET /api/context7-autosolve?action=history&limit=10',
            configure: 'POST /api/context7-autosolve (action: configure)',
          },
          timestamp: new Date().toISOString(),
        });
    }
  } catch (error: any) {
    return json(
      {
        success: false,
        error: error?.message,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};

// POST /api/context7-autosolve - Control autosolve operations
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action, data } = await request.json();

    switch (action) {
      case 'trigger':
        return await triggerAutosolve();

      case 'health_check':
        return await performComprehensiveHealthCheck();

      case 'configure':
        return await configureAutosolve(data);

      case 'force_cycle':
        return await forceAutosolveCycle(data);

      case 'analyze_errors':
        return await analyzeCurrentErrors();

      case 'generate_summary':
        return await generateSystemSummary();

      default:
        return json(
          {
            success: false,
            error: `Unknown action: ${action}`,
            available_actions: [
              'trigger',
              'health_check',
              'configure',
              'force_cycle',
              'analyze_errors',
              'generate_summary',
            ],
          },
          { status: 400 }
        );
    }
  } catch (error: any) {
    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};

// Trigger autosolve cycle
async function triggerAutosolve() {
  try {
    console.log('🚀 Triggering Context7 autosolve cycle via API...');

    const result = await context7AutosolveIntegration.triggerManualAutosolve();

    return json({
      success: true,
      message: 'Autosolve cycle triggered successfully',
      result,
      recommendations: generateRecommendations(result),
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `Autosolve trigger failed: ${error instanceof Error ? error.message : String(error)}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// Perform comprehensive health check
async function performComprehensiveHealthCheck() {
  try {
    const healthResults = {
      autosolve_system: context7AutosolveIntegration.getStatus(),
      database_orchestrator: databaseOrchestrator.getStatus(),
      services: {},
      typescript_status: null,
      postgresql_status: null,
      overall_health: 'unknown',
    };

    // Check key services
    const services = [
      { name: 'ollama', url: 'http://localhost:11434/api/tags' },
      { name: 'enhanced_rag', url: 'http://localhost:8097/health' },
      { name: 'aggregate_server', url: 'http://localhost:8123/health' },
      { name: 'error_processor', url: 'http://localhost:9099/health' },
      { name: 'recommendation_service', url: 'http://localhost:8096/health' },
    ];

    const serviceChecks = await Promise.allSettled(
      services.map(async (service) => {
        try {
          const controller = new AbortController();
          const t = setTimeout(() => controller.abort(), 3000);
          const response = await fetch(service.url, { method: 'GET', signal: controller.signal });
          clearTimeout(t);
          return {
            service: service.name,
            status: response.ok ? 'healthy' : 'unhealthy',
            response_code: response.status,
            url: service.url,
          };
        } catch (error: any) {
          return {
            service: service.name,
            status: 'error',
            error: error instanceof Error ? error.message : String(error),
            url: service.url,
          };
        }
      })
    );

    serviceChecks.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        healthResults.services[services[index].name] = result.value;
      } else {
        healthResults.services[services[index].name] = {
          service: services[index].name,
          status: 'error',
          error: result.reason.message,
        };
      }
    });

    // Check TypeScript status
    try {
      const tsController = new AbortController();
      const tsTimeout = setTimeout(() => tsController.abort(), 10000);
      const tsResponse = await fetch('http://localhost:5173/api/system/typescript-check', {
        method: 'POST',
        signal: tsController.signal,
      });
      clearTimeout(tsTimeout);
      healthResults.typescript_status = {
        status: tsResponse.ok ? 'available' : 'unavailable',
        response_code: tsResponse.status,
      };
    } catch (error: any) {
      healthResults.typescript_status = {
        status: 'error',
        error: error instanceof Error ? error.message : String(error),
      };
    }

    // Check PostgreSQL
    try {
      await databaseOrchestrator.queryDatabase({}, 'cases');
      healthResults.postgresql_status = {
        status: 'healthy',
        connection: 'active',
      };
    } catch (error: any) {
      healthResults.postgresql_status = {
        status: 'error',
        error: error instanceof Error ? error.message : String(error),
      };
    }

    // Determine overall health
    const healthyServices = Object.values(healthResults.services as Record<string, any>).filter(
      (s: any) => s && s.status === 'healthy'
    ).length;
    const totalServices = Object.keys(healthResults.services).length;

    if (
      healthyServices === totalServices &&
      healthResults.postgresql_status.status === 'healthy' &&
      healthResults.autosolve_system.integration_active
    ) {
      healthResults.overall_health = 'excellent';
    } else if (healthyServices >= totalServices * 0.7) {
      healthResults.overall_health = 'good';
    } else if (healthyServices >= totalServices * 0.5) {
      healthResults.overall_health = 'degraded';
    } else {
      healthResults.overall_health = 'critical';
    }

    return json({
      success: true,
      health_check: healthResults,
      healthy_services: healthyServices,
      total_services: totalServices,
      health_score: Math.round((healthyServices / totalServices) * 100),
      recommendations: generateHealthRecommendations(healthResults),
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `Health check failed: ${error instanceof Error ? error.message : String(error)}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// Configure autosolve system
async function configureAutosolve(configuration: any) {
  try {
    const configSettings = {
      autosolve_interval: configuration.interval || 180000,
      max_errors_per_cycle: configuration.max_errors || 10,
      enable_ai_recommendations: configuration.enable_ai !== true,
      // fixed logic: now true unless explicitly disabled
      enable_ollama_summary: configuration.enable_ollama !== true,
      backup_before_fix: configuration.backup !== false,
      ...configuration,
    };

    // Save configuration to database
    await databaseOrchestrator.saveToDatabase(
      {
        configuration: configSettings,
        configured_at: new Date(),
        configured_via: 'api',
      },
      'autosolve_configurations'
    );

    return json({
      success: true,
      message: 'Autosolve system configured successfully',
      configuration: configSettings,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `Configuration failed: ${error instanceof Error ? error.message : String(error)}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// Force immediate autosolve cycle
async function forceAutosolveCycle(options: any = {}) {
  try {
    const forceOptions = {
      skip_health_check: options.skip_health_check || false,
      max_fixes: options.max_fixes || 50,
      target_errors: options.target_errors || [],
    };

    console.log('🔧 Forcing autosolve cycle with options:', forceOptions);

    const result = await context7AutosolveIntegration.triggerManualAutosolve();

    // Log forced cycle
    await databaseOrchestrator.saveToDatabase(
      {
        type: 'forced_cycle',
        options: forceOptions,
        result,
        triggered_at: new Date(),
        triggered_via: 'api',
      },
      'autosolve_logs'
    );

    return json({
      success: true,
      message: 'Forced autosolve cycle completed',
      options: forceOptions,
      result,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `Forced cycle failed: ${error instanceof Error ? error.message : String(error)}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// Analyze current TypeScript errors
async function analyzeCurrentErrors() {
  try {
    // Trigger TypeScript check
    const tsCheckResponse = await fetch('http://localhost:5173/api/system/typescript-check', {
      method: 'POST',
    });

    if (!tsCheckResponse.ok) {
      throw new Error('TypeScript check service unavailable');
    }

    const tsResult = await tsCheckResponse.json();

    // Analyze error patterns
    interface FileErrorStats {
      count: number;
      errors: Record<string, number>;
      warnings: number;
      errorsCount: number;
    }
    const errorAnalysis: {
      total_errors: number;
      error_patterns: Record<string, number>;
      severity_breakdown: { error: number; warning: number };
      file_analysis: Record<string, FileErrorStats>;
      recommendations: string[];
      [key: string]: any;
    } = {
      total_errors: 0,
      error_patterns: {},
      severity_breakdown: { error: 0, warning: 0 },
      file_analysis: {},
      recommendations: [],
    };
    // Enhanced TypeScript error parsing + optional Context7 multicore analyzer
    try {
      const raw = tsResult.output || '';
      const lines = raw.split(/\r?\n/);

      interface ParsedError {
        file: string;
        line: number | null;
        column: number | null;
        code: string | null;
        severity: 'error' | 'warning';
        message: string;
        raw: string;
      }

      const parsedErrors: ParsedError[] = [];
      const tsErrorRegex =
        /^(?<file>[^(\s][^:(]+)\((?<line>\d+),(?<column>\d+)\): (?<severity>error|warning) (?<code>TS\d+): (?<message>.*)$/;

      for (const line of lines) {
        const m = tsErrorRegex.exec(line.trim());
        if (m && m.groups) {
          parsedErrors.push({
            file: m.groups.file,
            line: parseInt(m.groups.line, 10),
            column: parseInt(m.groups.column, 10),
            code: m.groups.code,
            severity: m.groups.severity as 'error' | 'warning',
            message: m.groups.message,
            raw: line,
          });
        }
      }

      // Aggregate patterns
      for (const pe of parsedErrors) {
        if (pe.code) {
          errorAnalysis.error_patterns[pe.code] = (errorAnalysis.error_patterns[pe.code] || 0) + 1;
        }
        errorAnalysis.severity_breakdown[pe.severity] =
          (errorAnalysis.severity_breakdown[pe.severity] || 0) + 1;
        if (!errorAnalysis.file_analysis[pe.file]) {
          errorAnalysis.file_analysis[pe.file] = {
            count: 0,
            errors: {},
            warnings: 0,
            errorsCount: 0,
          };
        }
        const fa = errorAnalysis.file_analysis[pe.file];
        fa.count++;
        if (pe.severity === 'warning') {
          fa.warnings++;
        } else {
          fa.errorsCount++;
        }
        if (pe.code) {
          fa.errors[pe.code] = (fa.errors[pe.code] || 0) + 1;
        }
      }
      // --- GPU / PM2 / zx runtime augmentation (optional, best-effort, non-fatal) ---
      try {
        const runtime: any = {
          captured_at: new Date().toISOString(),
          node_version: typeof process !== 'undefined' ? process.version : undefined,
        };

        // Helper with timeout
        const runCmd = async (cmd: string, args: string[] = [], timeoutMs = 1500) => {
          try {
            const { spawn } = await import('node:child_process');
            return await new Promise<string>((resolve, reject) => {
              const child = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
              let out = '';
              let err = '';
              const to = setTimeout(() => {
                child.kill('SIGKILL');
                reject(new Error('timeout'));
              }, timeoutMs);
              child.stdout.on('data', (d) => (out += d.toString()));
              child.stderr.on('data', (d) => (err += d.toString()));
              child.on('error', reject);
              child.on('close', (code) => {
                clearTimeout(to);
                if (code === 0) resolve(out.trim());
                else reject(new Error(err || `exit ${code}`));
              });
            });
          } catch {
            return null;
          }
        };

        // Try zx (if installed) for nicer shell (optional)
        let zx$: any = null;
        try {
          zx$ = await import('zx').catch(() => null);
        } catch {
          /* ignore */
        }

        // GPU detection via nvidia-smi
        let gpuInfo: any = null;
        const nvidiaOutput =
          (await runCmd('nvidia-smi', [
            '--query-gpu=name,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits',
          ])) || null;

        if (nvidiaOutput) {
          gpuInfo = nvidiaOutput
            .split('\n')
            .map((l) => l.split(',').map((s) => s.trim()))
            .filter((r) => r.length >= 4)
            .map(([name, memTotal, memUsed, util]) => ({
              name,
              memory_total_mb: Number(memTotal),
              memory_used_mb: Number(memUsed),
              utilization_percent: Number(util),
            }));
        } else if (process?.env?.NVIDIA_VISIBLE_DEVICES) {
          gpuInfo = { visible_devices: import.meta.env.NVIDIA_VISIBLE_DEVICES };
        }

        if (gpuInfo) runtime.gpu = gpuInfo;

        // PM2 process list (best effort)
        let pm2List: any = null;
        // Preferred: pm2 jlist (prints JSON)
        const pm2Raw = (await runCmd('pm2', ['jlist'], 1200)) || null;
        if (pm2Raw) {
          try {
            pm2List = JSON.parse(pm2Raw)
              .slice(0, 8)
              .map((p: any) => ({
                name: p.name,
                pid: p.pid,
                status: p.pm2_env?.status,
                restarts: p.pm2_env?.restart_time,
                cpu: p.monit?.cpu,
                memory_mb: p.monit?.memory ? Math.round(p.monit.memory / 1024 / 1024) : undefined,
              }));
          } catch {
            /* ignore */
          }
        }

        if (pm2List) runtime.pm2 = pm2List;

        // If zx present, optionally capture quick disk usage (non-blocking, small)
        if (zx$ && typeof (zx$ as any).$ === 'function') {
          try {
            // Use tagged template only if available (zx exports `$` function). Avoid referencing undeclared identifier.
            const zxExec: any = (zx$ as any).$;
            const du = await zxExec`bash -c "df -h . | tail -1"`.catch(() => null);
            if (du) {
              runtime.disk = String(du).trim().split(/\s+/).slice(1, 5); // size, used, avail, use%
            }
          } catch {
            /* ignore */
          }
        }

        if (runtime.gpu || runtime.pm2 || runtime.disk) {
          errorAnalysis.runtime_environment = runtime;
        }
      } catch {
        // ignore runtime augmentation failures
      }
      errorAnalysis.total_errors =
        typeof tsResult.error_count === 'number' ? tsResult.error_count : parsedErrors.length;
      // Attempt optional multicore deep analysis (context7-multicore.js)
      // Non-fatal if unavailable. Adds remote fallback using context7 worker + MCP docs.
      let multicoreModule: any = null;
      try {
        // Try in-project service locations first (won't throw build if absent due to dynamic import)
        multicoreModule =
          (await import('$lib/services/context7-multicore').catch(() => null)) ||
          (await import('$lib/services/context7-multicore.js').catch(() => null));
      } catch {
        // ignore
      }

      // Helper: parse possible go-simd / tensor textual payloads into JSON-friendly arrays
      const parseGoSimdTensor = (rawStr: string) => {
        if (!rawStr || typeof rawStr !== 'string') return null;
        // Accept formats like "[[0.12 0.5 -0.9],[1.0 2.0 3.0]]" or "0.1 0.2 0.3"
        const clean = rawStr
          .trim()
          .replace(/,+/g, ',')
          .replace(/\s+/g, ' ')
          .replace(/\s*\]\s*\[/g, ']|['); // normalize boundary
        try {
          // Try JSON first
          if (/[\[\{].*[\]\}]/.test(clean)) {
            const jsonLike = clean
              .replace(/(\d)-\s+(\d)/g, '$1- $2') // minor cleanup
              .replace(/(\d)\s+(\d)/g, '$1,$2') // space-delimited to commas (simple heuristic)
              .replace(/\s+/g, ' ');
            try {
              return JSON.parse(jsonLike);
            } catch {
              /* fallthrough */
            }
          }
          // Space-separated floats
          const nums = clean
            .replace(/^\[+|\]+$/g, '')
            .split(/[,\s]+/)
            .map((n) => Number(n))
            .filter((n) => Number.isFinite(n));
          return nums.length ? nums : null;
        } catch {
          return null;
        }
      };

      // Remote fallback to context7-worker if local module not importable
      if (!multicoreModule) {
        const remoteAnalyze = async (payload: { raw: string; parsed: any[] }) => {
          try {
            const endpoint = 'http://localhost:4100/semantic-analysis';
            const resp = await fetch(endpoint, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ text: payload.raw, context: 'typescript error log' }),
            }).catch(() => null);

            if (!resp || !resp.ok) return null;
            const data = await resp.json();

            // Derive recommendations from legal concepts / sentiment
            const concepts = (data.semantic_analysis?.legal_concepts || []).flatMap(
              (c: any) => c.terms || []
            );
            const sentiment = data.semantic_analysis?.sentiment_score;
            const extraRec: string[] = [];

            if (concepts.length) {
              extraRec.push(
                `Detected legal-domain terms: ${[...new Set(concepts)].slice(0, 8).join(', ')}`
              );
            }
            if (typeof sentiment === 'number') {
              extraRec.push(`Semantic sentiment score: ${sentiment.toFixed(2)}`);
            }

            // Attempt tensor parsing if embeddings present
            const embeddings = Array.isArray(data.embeddings)
              ? data.embeddings
              : parseGoSimdTensor(String(data.embeddings || ''));
            if (embeddings && Array.isArray(embeddings)) {
              extraRec.push(`Embedding vector length: ${embeddings.length}`);
            }

            return {
              recommendations: extraRec,
              hotspots: data.semantic_analysis?.legal_concepts,
              clusters: null,
            };
          } catch {
            return null;
          }
        };

        multicoreModule = {
          analyzeErrors: remoteAnalyze,
        };
      }

      // Optional: fetch MCP library docs to enrich recommendations (non-blocking)
      (async () => {
        try {
          const { mcpContext72GetLibraryDocs } = await import(
            '$lib/mcp-context72-get-library-docs'
          ).catch(() => ({ mcpContext72GetLibraryDocs: null }));
          if (!mcpContext72GetLibraryDocs) return;
          // Example: gather SvelteKit / drizzle docs if TS errors relate to them
          const suspectedLibs = Object.keys(errorAnalysis.error_patterns || {}).some((k) =>
            /drizzle|orm/i.test(k)
          )
            ? ['drizzle-orm', 'sveltekit']
            : ['sveltekit'];
          const docsResults: any[] = [];
          for (const lib of suspectedLibs) {
            const docs = await mcpContext72GetLibraryDocs(lib).catch(() => null);
            if (docs) docsResults.push({ lib, snippet: JSON.stringify(docs).slice(0, 400) });
          }
          if (docsResults.length) {
            errorAnalysis.mcp_docs = docsResults;
          }
        } catch {
          /* ignore */
        }
      })();

      if (multicoreModule?.analyzeErrors) {
        try {
          const advanced = await multicoreModule.analyzeErrors({
            raw,
            parsed: parsedErrors,
            concurrency: 4,
          });
          if (advanced?.recommendations?.length) {
            errorAnalysis.recommendations = advanced.recommendations;
          }
          if (advanced?.hotspots) {
            errorAnalysis.hotspots = advanced.hotspots;
          }
          if (advanced?.clusters) {
            errorAnalysis.clusters = advanced.clusters;
          }
        } catch {
          // ignore advanced analysis failures
        }
      }

      // Base recommendations if none supplied
      if (!errorAnalysis.recommendations || errorAnalysis.recommendations.length === 0) {
        const rec: string[] = [];
        if (errorAnalysis.total_errors === 0) {
          rec.push('No TypeScript errors detected');
        } else {
          rec.push('Prioritize most frequent TS error codes first');
          const topCodes = Object.entries(errorAnalysis.error_patterns)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([c]) => c);
          if (topCodes.length) rec.push(`Top recurring codes: ${topCodes.join(', ')}`);
          const heavyFiles = Object.entries(errorAnalysis.file_analysis)
            .sort((a: any, b: any) => b[1].count - a[1].count)
            .slice(0, 2)
            .map(([f]) => f);
          if (heavyFiles.length) rec.push(`Focus files: ${heavyFiles.join(', ')}`);
        }
        rec.push('Run autosolve cycle to address clustered issues');
        rec.push('Validate ambient type declarations and module paths');
        errorAnalysis.recommendations = rec;
      }

      // Attach parsed errors (trim to avoid huge payloads)
      errorAnalysis.sample_errors = parsedErrors.slice(0, 25);
    } catch (parseErr) {
      errorAnalysis.parse_error = parseErr instanceof Error ? parseErr.message : String(parseErr);
      errorAnalysis.total_errors = tsResult.error_count || 0;
      errorAnalysis.recommendations = [
        'Parsing failed; review raw output formatting',
        'Ensure TypeScript compiler output is standard (no custom wrappers)',
      ];
    }

    return json({
      success: true,
      message: 'TypeScript error analysis completed',
      analysis: errorAnalysis,
      raw_output: tsResult.output?.substring(0, 1000) + '...',
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `Error analysis failed: ${error instanceof Error ? error.message : String(error)}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// Generate comprehensive system summary
async function generateSystemSummary() {
  try {
    const autosolvHistory = await context7AutosolveIntegration.getAutosolvHistory(10);
    const systemStatus = context7AutosolveIntegration.getStatus();

    const summary = {
      system_overview: {
        autosolve_active: systemStatus.integration_active,
        total_cycles: systemStatus.cycle_count,
        currently_running: systemStatus.is_running,
      },
      recent_performance: {
        cycles_analyzed: autosolvHistory.length,
        average_fixes:
          autosolvHistory.length > 0
            ? autosolvHistory.reduce((sum, cycle) => sum + cycle.fixes_applied, 0) /
              autosolvHistory.length
            : 0,
        success_rate:
          autosolvHistory.length > 0
            ? (autosolvHistory.filter((cycle) => cycle.status === 'success').length /
                autosolvHistory.length) *
              100
            : 0,
      },
      recommendations: [
        'Monitor autosolve cycles for efficiency',
        'Review error patterns for preventive measures',
        'Ensure all services are healthy for optimal performance',
      ],
    };

    return json({
      success: true,
      message: 'System summary generated successfully',
      summary,
      recent_history: autosolvHistory.slice(0, 5),
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `Summary generation failed: ${error instanceof Error ? error.message : String(error)}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// Helper functions
function generateRecommendations(result: any): string[] {
  const recommendations = [];

  if (result.status === 'success') {
    recommendations.push('✅ All errors fixed successfully');
    recommendations.push('Consider running preventive type checks');
  } else if (result.status === 'partial') {
    recommendations.push('🔄 Some errors remain - consider manual review');
    recommendations.push('Run another autosolve cycle for remaining issues');
  } else {
    recommendations.push('❌ Autosolve cycle failed - check service health');
    recommendations.push('Review error logs for troubleshooting');
  }

  if (result.error_count > 20) {
    recommendations.push('High error count detected - consider code review');
  }

  return recommendations;
}
function generateHealthRecommendations(healthResults: any): string[] {
  const recommendations: string[] = [];

  if (healthResults.overall_health === 'critical') {
    recommendations.push('🚨 Critical: Multiple services down - check service startup');
    recommendations.push('Verify PostgreSQL and key services are running');
  } else if (healthResults.overall_health === 'degraded') {
    recommendations.push('⚠️ Some services unavailable - check specific endpoints');
    recommendations.push('Autosolve functionality may be limited');
  } else if (healthResults.overall_health === 'good') {
    recommendations.push('✅ System mostly healthy');
    recommendations.push('Monitor for any developing issues');
  }

  // Service-specific recommendations
  if (healthResults.services?.ollama?.status !== 'healthy') {
    recommendations.push('Start Ollama service for AI recommendations');
  }

  if (healthResults.postgresql_status?.status !== 'healthy') {
    recommendations.push('Check PostgreSQL connection for data persistence');
  }

  return recommendations;
}
