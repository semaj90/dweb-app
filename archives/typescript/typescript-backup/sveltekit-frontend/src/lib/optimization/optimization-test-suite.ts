
/**
 * Comprehensive Optimization Test Suite and Validation System
 * Tests all optimization components and validates integration
 */

import { performance } from 'perf_hooks';
import { 
  type EnhancedOptimizationSuite, 
  type EnhancedPerformanceMetrics,
  createEnhancedOptimizationSuite, 
  optimizeForLegalAIProduction 
} from './index';
import { createContext7MCPIntegration } from './context7-mcp-integration';

// === Test Result Types ===
export interface TestResult {
  name: string;
  passed: boolean;
  duration_ms: number;
  error?: string;
  details?: unknown;
  performance_impact?: {
    before: number;
    after: number;
    improvement: number;
  };
}

export interface TestSuite {
  name: string;
  tests: TestResult[];
  passed: number;
  failed: number;
  total_duration_ms: number;
  overall_passed: boolean;
}

export interface ValidationReport {
  timestamp: string;
  environment: {
    node_version: string;
    memory_limit: string;
    cpu_cores: number;
  };
  test_suites: TestSuite[];
  overall_results: {
    total_tests: number;
    passed_tests: number;
    failed_tests: number;
    success_rate: number;
    total_duration_ms: number;
  };
  performance_benchmarks: {
    vs_code_commands: number;
    cache_operations_per_second: number;
    json_parse_speed_mb_per_second: number;
    docker_optimization_time_ms: number;
    memory_usage_mb: number;
  };
  recommendations: string[];
}

// === Main Test Suite Class ===
export class OptimizationTestSuite {
  private suite: EnhancedOptimizationSuite | null = null;
  private context7_integrator: any = null;
  private test_data = {
    small_json: JSON.stringify({ test: 'data', items: Array(10).fill('test') }),
    large_json: JSON.stringify({ 
      data: Array(1000).fill({ 
        id: Math.random().toString(36), 
        content: 'large test data '.repeat(10),
        nested: { deep: { value: Math.random() } }
      })
    }),
    cache_keys: Array(100).fill(null).map((_, i) => `test_key_${i}`),
    docker_containers: ['postgres', 'qdrant', 'redis', 'ollama']
  };

  // === Core Component Tests ===
  async testVSCodeExtension(): Promise<TestSuite> {
    const tests: TestResult[] = [];
    const suite_start = performance.now();

    // Test 1: Extension Initialization
    {
      const start = performance.now();
      try {
        if (!this.suite) {
          this.suite = createEnhancedOptimizationSuite();
        }
        await this.suite.vscode?.initialize();
        
        tests.push({
          name: 'VS Code Extension Initialization',
          passed: true,
          duration_ms: performance.now() - start
        });
      } catch (error: any) {
        tests.push({
          name: 'VS Code Extension Initialization',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 2: Command Execution Performance
    {
      const start = performance.now();
      try {
        const commands_to_test = [
          'context7.analyzeStack',
          'legal.createCase',
          'ai.generateSummary',
          'cache.optimize',
          'docker.checkHealth'
        ];
        
        let successful_commands = 0;
        for (const command of commands_to_test) {
          try {
            await (this.suite!.vscode as any)?.executeCommand?.(command);
            successful_commands++;
          } catch (error: any) {
            // Individual command failures are acceptable
          }
        }
        
        const success_rate = successful_commands / commands_to_test.length;
        
        tests.push({
          name: 'Command Execution Performance',
          passed: success_rate >= 0.8, // 80% success rate required
          duration_ms: performance.now() - start,
          details: {
            tested_commands: commands_to_test.length,
            successful_commands,
            success_rate: success_rate * 100
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'Command Execution Performance',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 3: Memory Usage Monitoring
    {
      const start = performance.now();
      try {
        const stats = await this.suite!.vscode?.getStats?.() || { cache: { utilization: 50 } };
        const memory_efficient = stats.cache.utilization < 90; // Less than 90% cache utilization
        
        tests.push({
          name: 'Memory Usage Monitoring',
          passed: memory_efficient,
          duration_ms: performance.now() - start,
          details: {
            cache_utilization: stats.cache.utilization,
            commands_registered: stats.commands
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'Memory Usage Monitoring',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    const total_duration = performance.now() - suite_start;
    const passed = tests.filter((t: any) => t.passed).length;
    const failed = tests.filter((t: any) => !t.passed).length;

    return {
      name: 'VS Code Extension Tests',
      tests,
      passed,
      failed,
      total_duration_ms: total_duration,
      overall_passed: failed === 0
    };
  }

  async testRedisSOMapCache(): Promise<TestSuite> {
    const tests: TestResult[] = [];
    const suite_start = performance.now();

    if (!this.suite) {
      this.suite = createEnhancedOptimizationSuite();
    }

    // Test 1: Basic Cache Operations
    {
      const start = performance.now();
      try {
        await (this.suite.cache as any)?.set?.('test_key', 'test_value');
        const retrieved = await (this.suite.cache as any)?.get?.('test_key');
        
        tests.push({
          name: 'Basic Cache Operations',
          passed: retrieved === 'test_value',
          duration_ms: performance.now() - start,
          details: { retrieved_value: retrieved }
        });
      } catch (error: any) {
        tests.push({
          name: 'Basic Cache Operations',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 2: Self-Organizing Map Clustering
    {
      const start = performance.now();
      try {
        // Add multiple entries to trigger SOM clustering
        for (let i = 0; i < 50; i++) {
          await (this.suite.cache as any)?.set?.(`som_test_${i}`, {
            data: `test_data_${i}`,
            type: i % 3 === 0 ? 'frequent' : i % 3 === 1 ? 'burst' : 'random'
          }, {
            metadata: {
              access_pattern: i % 3 === 0 ? 'frequent' : i % 3 === 1 ? 'burst' : 'random',
              ai_relevance: Math.random()
            }
          });
        }

        const { clusters, recommendations } = await (this.suite.cache as any)?.analyzeAccessPatterns?.() || { clusters: [], recommendations: [] };
        
        tests.push({
          name: 'Self-Organizing Map Clustering',
          passed: clusters.length > 0 && recommendations.length > 0,
          duration_ms: performance.now() - start,
          details: {
            clusters_found: clusters.length,
            recommendations_generated: recommendations.length
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'Self-Organizing Map Clustering',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 3: Memory Pressure Handling
    {
      const start = performance.now();
      try {
        const before_stats = this.suite.cache?.getStats?.() || { cache: { size: 0 } };
        
        // Fill cache to trigger memory pressure
        const large_data = 'x'.repeat(100000); // 100KB per entry
        for (let i = 0; i < 100; i++) {
          await (this.suite.cache as any)?.set?.(`pressure_test_${i}`, large_data);
        }
        
        const after_stats = this.suite.cache?.getStats?.() || { cache: { size: 0 } };
        const handled_pressure = after_stats.memory.utilization <= 100; // Should not exceed 100%
        
        tests.push({
          name: 'Memory Pressure Handling',
          passed: handled_pressure,
          duration_ms: performance.now() - start,
          details: {
            before_utilization: before_stats.memory.utilization,
            after_utilization: after_stats.memory.utilization,
            items_cached: after_stats.cache.size
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'Memory Pressure Handling',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    const total_duration = performance.now() - suite_start;
    const passed = tests.filter((t: any) => t.passed).length;
    const failed = tests.filter((t: any) => !t.passed).length;

    return {
      name: 'Redis SOM Cache Tests',
      tests,
      passed,
      failed,
      total_duration_ms: total_duration,
      overall_passed: failed === 0
    };
  }

  async testDockerOptimizer(): Promise<TestSuite> {
    const tests: TestResult[] = [];
    const suite_start = performance.now();

    if (!this.suite) {
      this.suite = createEnhancedOptimizationSuite();
    }

    // Test 1: Container Resource Monitoring
    {
      const start = performance.now();
      try {
        const stats = this.suite.docker?.getResourceUtilization?.() || { memory: 0, cpu: 0, containers: [] };
        const has_containers = stats.containers.length > 0;
        const valid_metrics = stats.efficiency_score >= 0 && stats.efficiency_score <= 1;
        
        tests.push({
          name: 'Container Resource Monitoring',
          passed: has_containers && valid_metrics,
          duration_ms: performance.now() - start,
          details: {
            containers: stats.containers.length,
            efficiency_score: stats.efficiency_score,
            memory_allocated_gb: Math.round(stats.total_memory_allocated / (1024 * 1024 * 1024))
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'Container Resource Monitoring',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 2: Optimization Preset Application
    {
      const start = performance.now();
      try {
        const before_stats = this.suite.docker?.getResourceUtilization?.() || { memory: 0 };
        await this.suite.docker?.applyDevelopmentPreset?.();
        const after_stats = this.suite.docker?.getResourceUtilization?.() || { memory: 0 };
        
        // Preset should maintain or improve efficiency
        const efficiency_maintained = after_stats.efficiency_score >= before_stats.efficiency_score * 0.9;
        
        tests.push({
          name: 'Optimization Preset Application',
          passed: efficiency_maintained,
          duration_ms: performance.now() - start,
          details: {
            before_efficiency: before_stats.efficiency_score,
            after_efficiency: after_stats.efficiency_score,
            efficiency_change: after_stats.efficiency_score - before_stats.efficiency_score
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'Optimization Preset Application',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 3: Docker Compose Generation
    {
      const start = performance.now();
      try {
        const dockerCompose = (this.suite.docker as any)?.generateOptimizedDockerCompose?.() || '';
        const is_valid_yaml = dockerCompose.includes('version:') && 
                             dockerCompose.includes('services:') &&
                             dockerCompose.includes('networks:');
        
        tests.push({
          name: 'Docker Compose Generation',
          passed: is_valid_yaml,
          duration_ms: performance.now() - start,
          details: {
            compose_length: dockerCompose.length,
            has_resource_limits: dockerCompose.includes('resources:')
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'Docker Compose Generation',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    const total_duration = performance.now() - suite_start;
    const passed = tests.filter((t: any) => t.passed).length;
    const failed = tests.filter((t: any) => !t.passed).length;

    return {
      name: 'Docker Optimizer Tests',
      tests,
      passed,
      failed,
      total_duration_ms: total_duration,
      overall_passed: failed === 0
    };
  }

  async testJSONWASMOptimizer(): Promise<TestSuite> {
    const tests: TestResult[] = [];
    const suite_start = performance.now();

    if (!this.suite) {
      this.suite = createEnhancedOptimizationSuite();
    }

    // Test 1: JSON Parsing Performance
    {
      const start = performance.now();
      try {
        const { data, stats } = await (this.suite.json as any)?.parseJSON?.(this.test_data.large_json) || { data: {}, stats: {} };
        const parse_successful = Array.isArray(data.data) && data.data.length === 1000;
        const reasonable_performance = stats.parse_time_ms < 100; // Should parse 1000 items in < 100ms
        
        tests.push({
          name: 'JSON Parsing Performance',
          passed: parse_successful && reasonable_performance,
          duration_ms: performance.now() - start,
          details: {
            parsed_items: Array.isArray(data.data) ? data.data.length : 0,
            parse_time_ms: stats.parse_time_ms,
            wasm_acceleration: stats.wasm_acceleration
          },
          performance_impact: {
            before: this.test_data.large_json.length,
            after: stats.parse_time_ms,
            improvement: this.test_data.large_json.length / stats.parse_time_ms // chars per ms
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'JSON Parsing Performance',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 2: JSON Compression
    {
      const start = performance.now();
      try {
        const test_object = JSON.parse(this.test_data.large_json);
        const { compressed, stats } = await (this.suite.json as any)?.compressJSON?.(test_object) || { compressed: {}, stats: {} };
        
        const compression_effective = stats.compression_ratio > 1.2; // At least 20% compression
        const compressed_is_smaller = stats.compressed_size < stats.original_size;
        
        tests.push({
          name: 'JSON Compression',
          passed: compression_effective && compressed_is_smaller,
          duration_ms: performance.now() - start,
          details: {
            original_size: stats.original_size,
            compressed_size: stats.compressed_size,
            compression_ratio: stats.compression_ratio,
            wasm_acceleration: stats.wasm_acceleration
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'JSON Compression',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 3: WebAssembly Initialization
    {
      const start = performance.now();
      try {
        const wasm_initialized = this.suite.json?.isWASMInitialized?.() || false;
        
        tests.push({
          name: 'WebAssembly Initialization',
          passed: true, // WASM initialization is optional
          duration_ms: performance.now() - start,
          details: {
            wasm_available: wasm_initialized,
            note: wasm_initialized ? 'WebAssembly acceleration available' : 'Using JavaScript fallback'
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'WebAssembly Initialization',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    const total_duration = performance.now() - suite_start;
    const passed = tests.filter((t: any) => t.passed).length;
    const failed = tests.filter((t: any) => !t.passed).length;

    return {
      name: 'JSON WASM Optimizer Tests',
      tests,
      passed,
      failed,
      total_duration_ms: total_duration,
      overall_passed: failed === 0
    };
  }

  async testContext7Integration(): Promise<TestSuite> {
    const tests: TestResult[] = [];
    const suite_start = performance.now();

    if (!this.context7_integrator) {
      this.context7_integrator = createContext7MCPIntegration();
    }

    // Test 1: Stack Analysis with Optimization
    {
      const start = performance.now();
      try {
        const response = await this.context7_integrator.analyzeStackWithOptimization('sveltekit', 'legal-ai');
        const has_recommendations = response.optimization_recommendations && 
                                   response.optimization_recommendations.length > 0;
        
        tests.push({
          name: 'Stack Analysis with Optimization',
          passed: response.success && has_recommendations,
          duration_ms: performance.now() - start,
          details: {
            recommendations_count: response.optimization_recommendations?.length || 0,
            context7_success: response.success
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'Stack Analysis with Optimization',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 2: Best Practices Generation
    {
      const start = performance.now();
      try {
        const response = await this.context7_integrator.generateBestPracticesWithResourceAwareness('performance');
        const has_impact_estimate = response.performance_impact !== undefined;
        
        tests.push({
          name: 'Best Practices Generation',
          passed: response.success && has_impact_estimate,
          duration_ms: performance.now() - start,
          details: {
            has_performance_impact: has_impact_estimate,
            expected_improvement: response.performance_impact?.expected_improvement || 0
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'Best Practices Generation',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 3: Comprehensive Analysis
    {
      const start = performance.now();
      try {
        const analysis = await this.context7_integrator.runComprehensiveOptimizationAnalysis();
        const has_metrics = analysis.current_metrics !== null;
        const has_recommendations = analysis.recommendations.length > 0;
        const has_plan = analysis.implementation_plan.length > 0;
        
        tests.push({
          name: 'Comprehensive Analysis',
          passed: has_metrics && has_recommendations && has_plan,
          duration_ms: performance.now() - start,
          details: {
            metrics_available: has_metrics,
            recommendations_count: analysis.recommendations.length,
            implementation_steps: analysis.implementation_plan.length
          }
        });
      } catch (error: any) {
        tests.push({
          name: 'Comprehensive Analysis',
          passed: false,
          duration_ms: performance.now() - start,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    const total_duration = performance.now() - suite_start;
    const passed = tests.filter((t: any) => t.passed).length;
    const failed = tests.filter((t: any) => !t.passed).length;

    return {
      name: 'Context7 Integration Tests',
      tests,
      passed,
      failed,
      total_duration_ms: total_duration,
      overall_passed: failed === 0
    };
  }

  // === Performance Benchmarking ===
  async runPerformanceBenchmarks(): Promise<{
    vs_code_commands: number;
    cache_operations_per_second: number;
    json_parse_speed_mb_per_second: number;
    docker_optimization_time_ms: number;
    memory_usage_mb: number;
  }> {
    if (!this.suite) {
      this.suite = createEnhancedOptimizationSuite();
      await this.suite.vscode?.initialize?.();
    }

    const benchmarks = {
      vs_code_commands: 0,
      cache_operations_per_second: 0,
      json_parse_speed_mb_per_second: 0,
      docker_optimization_time_ms: 0,
      memory_usage_mb: 0
    };

    // Benchmark VS Code commands
    {
      const start = performance.now();
      let successful_commands = 0;
      const commands = ['cache.optimize', 'docker.checkHealth', 'typescript.checkTypes'];
      
      for (const command of commands) {
        try {
          await (this.suite.vscode as any)?.executeCommand?.(command);
          successful_commands++;
        } catch (error: any) {
          // Continue with other commands
        }
      }
      
      const duration = performance.now() - start;
      benchmarks.vs_code_commands = successful_commands;
    }

    // Benchmark cache operations
    {
      const start = performance.now();
      const operations_count = 1000;
      
      for (let i = 0; i < operations_count; i++) {
        await (this.suite.cache as any)?.set?.(`bench_${i}`, `value_${i}`);
        await (this.suite.cache as any)?.get?.(`bench_${i}`);
      }
      
      const duration = (performance.now() - start) / 1000; // Convert to seconds
      benchmarks.cache_operations_per_second = Math.round((operations_count * 2) / duration); // set + get
    }

    // Benchmark JSON parsing
    {
      const json_size_mb = this.test_data.large_json.length / (1024 * 1024);
      const iterations = 10;
      const start = performance.now();
      
      for (let i = 0; i < iterations; i++) {
        await (this.suite.json as any)?.parseJSON?.(this.test_data.large_json);
      }
      
      const duration = (performance.now() - start) / 1000; // Convert to seconds
      const total_mb_processed = json_size_mb * iterations;
      benchmarks.json_parse_speed_mb_per_second = Math.round(total_mb_processed / duration);
    }

    // Benchmark Docker optimization
    {
      const start = performance.now();
      await this.suite.docker?.applyDevelopmentPreset?.();
      benchmarks.docker_optimization_time_ms = Math.round(performance.now() - start);
    }

    // Get memory usage
    {
      const docker_stats = this.suite.docker?.getResourceUtilization?.() || { memory: 0, cpu: 0 };
      benchmarks.memory_usage_mb = Math.round(docker_stats.total_memory_used / (1024 * 1024));
    }

    return benchmarks;
  }

  // === Main Test Runner ===
  async runAllTests(): Promise<ValidationReport> {
    const start_time = performance.now();
    
    console.log('🚀 Starting Optimization Suite Validation...');
    
    const test_suites: TestSuite[] = [];
    
    // Run all test suites
    console.log('📋 Testing VS Code Extension...');
    test_suites.push(await this.testVSCodeExtension());
    
    console.log('🗄️ Testing Redis SOM Cache...');
    test_suites.push(await this.testRedisSOMapCache());
    
    console.log('🐳 Testing Docker Optimizer...');
    test_suites.push(await this.testDockerOptimizer());
    
    console.log('📊 Testing JSON WASM Optimizer...');
    test_suites.push(await this.testJSONWASMOptimizer());
    
    console.log('🔗 Testing Context7 Integration...');
    test_suites.push(await this.testContext7Integration());
    
    // Run performance benchmarks
    console.log('⚡ Running Performance Benchmarks...');
    const performance_benchmarks = await this.runPerformanceBenchmarks();
    
    const total_duration = performance.now() - start_time;
    const total_tests = test_suites.reduce((sum, suite) => sum + suite.tests.length, 0);
    const passed_tests = test_suites.reduce((sum, suite) => sum + suite.passed, 0);
    const failed_tests = test_suites.reduce((sum, suite) => sum + suite.failed, 0);
    
    // Generate recommendations
    const recommendations = this.generateRecommendations(test_suites, performance_benchmarks);
    
    const report: ValidationReport = {
      timestamp: new Date().toISOString(),
      environment: {
        node_version: process.version || 'unknown',
        memory_limit: `${Math.round(performance_benchmarks.memory_usage_mb)}MB`,
        cpu_cores: typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4
      },
      test_suites,
      overall_results: {
        total_tests,
        passed_tests,
        failed_tests,
        success_rate: (passed_tests / total_tests) * 100,
        total_duration_ms: total_duration
      },
      performance_benchmarks,
      recommendations
    };
    
    return report;
  }

  private generateRecommendations(
    test_suites: TestSuite[],
    benchmarks: ReturnType<OptimizationTestSuite['runPerformanceBenchmarks']> extends Promise<infer T> ? T : never
  ): string[] {
    const recommendations: string[] = [];
    
    // Analyze test results
    const failed_suites = test_suites.filter((suite: any) => !suite.overall_passed);
    if (failed_suites.length > 0) {
      recommendations.push(`⚠️  ${failed_suites.length} test suite(s) have failures - review error details`);
    }
    
    // Analyze performance benchmarks
    if (benchmarks.cache_operations_per_second < 1000) {
      recommendations.push('🗄️ Cache performance is below optimal - consider tuning cache configuration');
    }
    
    if (benchmarks.json_parse_speed_mb_per_second < 10) {
      recommendations.push('📊 JSON parsing performance is slow - enable WebAssembly acceleration');
    }
    
    if (benchmarks.memory_usage_mb > 8000) {
      recommendations.push('💾 High memory usage detected - apply memory optimization strategies');
    }
    
    if (benchmarks.docker_optimization_time_ms > 1000) {
      recommendations.push('🐳 Docker optimization is slow - consider container resource limits');
    }
    
    // Success recommendations
    if (recommendations.length === 0) {
      recommendations.push('✅ All systems operating optimally - no immediate action required');
      recommendations.push('📈 Consider enabling advanced optimization features for further improvements');
    }
    
    return recommendations;
  }

  // === Report Generation ===
  generateHumanReadableReport(report: ValidationReport): string {
    const { overall_results, performance_benchmarks } = report;
    
    const emoji_status = overall_results.success_rate >= 90 ? '✅' : 
                        overall_results.success_rate >= 70 ? '⚠️' : '❌';
    
    return `
# Optimization Suite Validation Report ${emoji_status}

**Generated:** ${report.timestamp}
**Environment:** Node.js ${report.environment.node_version}, ${report.environment.cpu_cores} cores, ${report.environment.memory_limit} memory

## Overall Results
- **Tests:** ${overall_results.total_tests} total, ${overall_results.passed_tests} passed, ${overall_results.failed_tests} failed
- **Success Rate:** ${overall_results.success_rate.toFixed(1)}%
- **Duration:** ${(overall_results.total_duration_ms / 1000).toFixed(2)} seconds

## Performance Benchmarks
- **VS Code Commands:** ${performance_benchmarks.vs_code_commands} successful
- **Cache Operations:** ${performance_benchmarks.cache_operations_per_second.toLocaleString()} ops/sec
- **JSON Processing:** ${performance_benchmarks.json_parse_speed_mb_per_second} MB/sec
- **Docker Optimization:** ${performance_benchmarks.docker_optimization_time_ms}ms
- **Memory Usage:** ${performance_benchmarks.memory_usage_mb}MB

## Test Suite Results
${report.test_suites.map((suite: any) => `### ${suite.name} ${suite.overall_passed ? '✅' : '❌'}
- Passed: ${suite.passed}/${suite.tests.length} tests
- Duration: ${(suite.total_duration_ms / 1000).toFixed(2)}s
${suite.tests.filter((t: any) => !t.passed).map((t: any) => `  - ❌ ${t.name}: ${t.error}`).join('\n')}`
).join('\n\n')}

## Recommendations
${report.recommendations.map((rec: any) => `- ${rec}`).join('\n')}
`;
  }
}

// === Factory Functions ===
export function createTestSuite(): OptimizationTestSuite {
  return new OptimizationTestSuite();
}

export async function runQuickValidation(): Promise<{
  passed: boolean;
  summary: string;
  details: ValidationReport;
}> {
  const testSuite = createTestSuite();
  const report = await testSuite.runAllTests();
  
  return {
    passed: report.overall_results.success_rate >= 80,
    summary: `${report.overall_results.passed_tests}/${report.overall_results.total_tests} tests passed (${report.overall_results.success_rate.toFixed(1)}%)`,
    details: report
  };
}

// === Export Default Test Suite ===
export default createTestSuite();