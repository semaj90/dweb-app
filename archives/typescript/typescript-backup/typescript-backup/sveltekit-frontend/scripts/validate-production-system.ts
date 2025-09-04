#!/usr/bin/env tsx
/**
 * Production System Validation Script
 * 
 * Usage:
 * npm run validate:system
 * npm run validate:system -- --verbose
 * npm run validate:system -- --report-only
 * npm run validate:system -- --benchmark
 * 
 * Environment:
 * VALIDATION_ENDPOINT=http://localhost:5173 (default)
 * VALIDATION_TIMEOUT=60000 (default: 60 seconds)
 * VALIDATION_RETRIES=3 (default: 3 retries)
 */

import { performance } from 'perf_hooks';

export interface ValidationConfig {
  endpoint: string;
  timeout: number;
  retries: number;
  verbose: boolean;
  reportOnly: boolean;
  benchmark: boolean;
}

export interface ApiResponse {
  success: boolean;
  data?: any;
  error?: string;
  meta?: any;
}

class ProductionValidator {
  private config: ValidationConfig;
  private startTime: number = 0;

  constructor() {
    this.config = {
      endpoint: process.env.VALIDATION_ENDPOINT || 'http://localhost:5173',
      timeout: parseInt(process.env.VALIDATION_TIMEOUT || '60000'),
      retries: parseInt(process.env.VALIDATION_RETRIES || '3'),
      verbose: process.argv.includes('--verbose'),
      reportOnly: process.argv.includes('--report-only'),
      benchmark: process.argv.includes('--benchmark')
    };
  }

  async run(): Promise<void> {
    this.startTime = performance.now();
    
    console.log('🚀 Production System Validation');
    console.log('=' .repeat(50));
    console.log(`Platform: ${process.platform}`);
    console.log(`Node.js: ${process.version}`);
    console.log(`Endpoint: ${this.config.endpoint}`);
    console.log(`Timeout: ${this.config.timeout}ms`);
    console.log('=' .repeat(50));
    console.log('');

    try {
      if (this.config.reportOnly) {
        await this.getValidationReport();
      } else if (this.config.benchmark) {
        await this.runBenchmarks();
      } else {
        await this.runFullValidation();
      }
      
      console.log('\n✅ Validation completed successfully');
      console.log(`⏱️  Total time: ${Math.round(performance.now() - this.startTime)}ms`);
      
    } catch (error: any) {
      console.error('\n❌ Validation failed:', error instanceof Error ? error.message : error);
      process.exit(1);
    }
  }

  private async runFullValidation(): Promise<void> {
    console.log('🔍 Running comprehensive system validation...\n');

    // Step 1: Quick health check
    await this.quickHealthCheck();
    
    // Step 2: Full validation
    const validationResponse = await this.makeRequest('/api/validation/system?action=validate');
    
    if (!validationResponse.success) {
      throw new Error(`Validation failed: ${validationResponse.error}`);
    }

    const report = validationResponse.data;
    
    // Display results
    this.displayValidationReport(report);
    
    // Save report to file if requested
    if (this.config.verbose) {
      await this.saveReportToFile(report);
    }
  }

  private async quickHealthCheck(): Promise<void> {
    console.log('🏥 Quick health check...');
    
    const response = await this.makeRequest('/api/validation/system?action=health');
    
    if (!response.success) {
      console.log(`❌ Health check failed: ${response.error}\n`);
      return;
    }

    const health = response.data;
    console.log(`✅ Health: ${health.status} (${health.healthy}/${health.total} services)`);
    console.log(`⏱️  Response time: ${health.processingTime}ms\n`);
  }

  private async getValidationReport(): Promise<void> {
    console.log('📊 Retrieving last validation report...\n');
    
    const response = await this.makeRequest('/api/validation/system?action=report');
    
    if (!response.success) {
      throw new Error(`Report retrieval failed: ${response.error}`);
    }

    const report = response.data;
    const reportAge = response.meta?.reportAge || 0;
    
    console.log(`📅 Report age: ${Math.floor(reportAge / 60)} minutes ${reportAge % 60} seconds`);
    if (response.meta?.isStale) {
      console.log('⚠️  Report is stale - consider running fresh validation\n');
    }
    
    this.displayValidationReport(report);
  }

  private async runBenchmarks(): Promise<void> {
    console.log('⚡ Running system benchmarks...\n');
    
    const response = await this.makeRequest('/api/validation/system', {
      method: 'POST',
      body: JSON.stringify({ action: 'benchmark' }),
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!response.success) {
      throw new Error(`Benchmark failed: ${response.error}`);
    }

    const results = response.data;
    
    console.log('📈 Benchmark Results:');
    console.log(`Platform: ${results.platform} (${results.environment})`);
    console.log(`Timestamp: ${new Date(results.timestamp).toLocaleString()}\n`);
    
    Object.entries(results.tests).forEach(([testName, testResult]: [string, any]) => {
      console.log(`🔸 ${testName}:`);
      console.log(`   Score: ${testResult.score}/100`);
      Object.entries(testResult.details).forEach(([key, value]) => {
        console.log(`   ${key}: ${value}`);
      });
      console.log('');
    });
  }

  private displayValidationReport(report: any): void {
    console.log('📊 Validation Report Summary:');
    console.log(`Overall Status: ${this.getStatusEmoji(report.overall.status)} ${report.overall.status.toUpperCase()}`);
    console.log(`Overall Score: ${report.overall.score}/100`);
    console.log(`Platform: ${report.overall.platform}`);
    console.log(`Environment: ${report.overall.environment}`);
    console.log(`Timestamp: ${new Date(report.overall.timestamp).toLocaleString()}\n`);

    // Performance metrics
    console.log('⚡ Performance Metrics:');
    console.log(`Average Latency: ${report.performance.averageLatency}ms`);
    console.log(`Memory Usage: ${report.performance.totalMemoryUsage}MB`);
    if (report.performance.gpuUtilization !== undefined) {
      console.log(`GPU Utilization: ${report.performance.gpuUtilization}%`);
    }
    console.log(`Integration Score: ${report.performance.integrationScore}/100\n`);

    // Service status
    console.log('🔧 Service Status:');
    report.services.forEach((service: any) => {
      const emoji = this.getStatusEmoji(service.status);
      const score = `(${service.performanceScore}/100)`;
      const latency = `${service.latency}ms`;
      const memory = `${Math.round(service.resourceUsage.memory)}MB`;
      
      console.log(`${emoji} ${service.service.padEnd(25)} ${score} ${latency.padStart(8)} ${memory.padStart(10)}`);
      
      if (this.config.verbose && service.errors.length > 0) {
        service.errors.forEach((error: string) => console.log(`    ❌ ${error}`));
      }
      if (this.config.verbose && service.warnings.length > 0) {
        service.warnings.forEach((warning: string) => console.log(`    ⚠️  ${warning}`));
      }
    });
    console.log('');

    // Recommendations
    if (report.recommendations.length > 0) {
      console.log('💡 Recommendations:');
      report.recommendations.forEach((rec: string, index: number) => {
        console.log(`${index + 1}. ${rec}`);
      });
      console.log('');
    }

    // Critical issues
    if (report.criticalIssues.length > 0) {
      console.log('🚨 Critical Issues:');
      report.criticalIssues.forEach((issue: string, index: number) => {
        console.log(`${index + 1}. ${issue}`);
      });
      console.log('');
    }

    // Service breakdown
    const healthyServices = report.services.filter((s: any) => s.status === 'healthy').length;
    const degradedServices = report.services.filter((s: any) => s.status === 'degraded').length;
    const failedServices = report.services.filter((s: any) => s.status === 'failed').length;
    
    console.log('📈 Service Breakdown:');
    console.log(`✅ Healthy: ${healthyServices}/${report.services.length}`);
    console.log(`⚠️  Degraded: ${degradedServices}/${report.services.length}`);
    console.log(`❌ Failed: ${failedServices}/${report.services.length}`);
  }

  private getStatusEmoji(status: string): string {
    switch (status) {
      case 'healthy': return '✅';
      case 'degraded': return '⚠️';
      case 'failed': return '❌';
      default: return '❓';
    }
  }

  private async makeRequest(path: string, options?: RequestInit): Promise<ApiResponse> {
    const url = `${this.config.endpoint}${path}`;
    
    for (let attempt = 1; attempt <= this.config.retries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);
        
        const response = await fetch(url, {
          ...options,
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok && response.status !== 409) { // 409 is acceptable for validation in progress
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
        
      } catch (error: any) {
        if (attempt === this.config.retries) {
          throw error;
        }
        
        console.log(`⚠️  Request failed (attempt ${attempt}/${this.config.retries}), retrying...`);
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt)); // Exponential backoff
      }
    }
    
    throw new Error('Max retries exceeded');
  }

  private async saveReportToFile(report: any): Promise<void> {
    const fs = await import('fs/promises');
    const path = await import('path');
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `validation-report-${timestamp}.json`;
    const filepath = path.join(process.cwd(), '.vscode', filename);
    
    try {
      await fs.writeFile(filepath, JSON.stringify(report, null, 2));
      console.log(`💾 Report saved to: ${filepath}`);
    } catch (error: any) {
      console.log(`⚠️  Could not save report: ${error instanceof Error ? error.message : error}`);
    }
  }
}

// Script execution
async function main(): Promise<any> {
  const validator = new ProductionValidator();
  await validator.run();
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { ProductionValidator };