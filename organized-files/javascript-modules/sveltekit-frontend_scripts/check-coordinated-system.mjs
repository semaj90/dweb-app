#!/usr/bin/env node
/**
 * 🔍 Coordinated System Health Check
 * Validates Master Service Coordinator and all integrated services
 */

import chalk from 'chalk';
import ora from 'ora';
import boxen from 'boxen';

const style = {
  primary: (text) => chalk.hex('#f4f4f4')(text),
  secondary: (text) => chalk.hex('#8b9dc3')(text),
  accent: (text) => chalk.hex('#dca561')(text),
  success: (text) => chalk.hex('#51cf66')(text),
  warning: (text) => chalk.hex('#ff6b6b')(text),
  error: (text) => chalk.hex('#ff4757')(text),
  bold: (text) => chalk.bold(text),
  dim: (text) => chalk.dim(text)
};

class CoordinatedSystemChecker {
  constructor() {
    this.coordinatorURL = 'http://localhost:5173/api/v1/coordinator';
    this.healthDashboardURL = 'http://localhost:5173/system/health';
    this.results = {
      coordinator: { status: 'unknown', details: {} },
      services: [],
      errors: [],
      performance: {},
      recommendations: []
    };
  }

  async check() {
    console.log(boxen(
      style.bold('🔍 COORDINATED SYSTEM HEALTH CHECK\n') +
      style.secondary('Master Service Coordinator Integration Status\n') +
      style.dim('Validating 38+ Services + Error Resolution'),
      {
        padding: 1,
        margin: 1,
        borderStyle: 'double',
        borderColor: '#dca561'
      }
    ));

    try {
      // Phase 1: Check TypeScript errors
      await this.checkTypeScriptErrors();
      
      // Phase 2: Check Master Service Coordinator
      await this.checkCoordinator();
      
      // Phase 3: Check service health via coordinator
      await this.checkServices();
      
      // Phase 4: Check error resolution system
      await this.checkErrorResolution();
      
      // Phase 5: Display comprehensive report
      this.displayReport();
      
    } catch (error) {
      console.error(style.error('❌ Health check failed:'), error.message);
      process.exit(1);
    }
  }

  async checkTypeScriptErrors() {
    const spinner = ora('🔍 Checking TypeScript errors...').start();
    
    try {
      // This matches what the coordinated startup does
      const { exec } = await import('child_process');
      const { promisify } = await import('util');
      const execAsync = promisify(exec);
      
      const { stdout, stderr } = await execAsync(
        'NODE_OPTIONS="--max-old-space-size=2048" timeout 15s npm run check:ultra-fast',
        { timeout: 20000 }
      );
      
      if (stderr && stderr.includes('error TS')) {
        const errorCount = (stderr.match(/error TS/g) || []).length;
        this.results.errors.push({
          type: 'typescript',
          count: errorCount,
          status: 'detected'
        });
        
        spinner.warn(style.warning(`TypeScript errors detected: ${errorCount}`));
      } else {
        spinner.succeed(style.success('TypeScript check passed'));
      }
      
    } catch (error) {
      this.results.errors.push({
        type: 'typescript_check',
        message: error.message,
        status: 'failed'
      });
      
      spinner.fail(style.error('TypeScript check failed'));
    }
  }

  async checkCoordinator() {
    const spinner = ora('🎯 Checking Master Service Coordinator...').start();
    
    try {
      const response = await fetch(`${this.coordinatorURL}?action=status`, {
        timeout: 5000
      });
      
      if (response.ok) {
        const status = await response.json();
        this.results.coordinator = {
          status: 'available',
          details: status
        };
        
        spinner.succeed(style.success('Master Service Coordinator available'));
        
        // Check health endpoint too
        await this.checkCoordinatorHealth();
        
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
      
    } catch (error) {
      this.results.coordinator = {
        status: 'unavailable',
        error: error.message
      };
      
      spinner.fail(style.error('Master Service Coordinator unavailable'));
      this.results.recommendations.push('Start SvelteKit frontend: npm run dev');
      this.results.recommendations.push('Check coordinator files are present');
    }
  }

  async checkCoordinatorHealth() {
    try {
      const response = await fetch(`${this.coordinatorURL}?action=health`, {
        timeout: 5000
      });
      
      if (response.ok) {
        const health = await response.json();
        this.results.coordinator.health = health;
        
        console.log(style.accent(
          `📊 System Health: ${health.services?.length || 0} services monitored`
        ));
        
        if (health.errorResolution?.enabled) {
          console.log(style.success(
            `🛠️ Error Resolution: ${health.errorResolution.resolvedCount || 0} errors resolved`
          ));
        }
      }
    } catch (error) {
      // Non-critical, health endpoint might not be available
    }
  }

  async checkServices() {
    const spinner = ora('📡 Checking Go microservices...').start();
    
    if (this.results.coordinator.status === 'available') {
      try {
        const response = await fetch(`${this.coordinatorURL}?action=services`, {
          timeout: 10000
        });
        
        if (response.ok) {
          const services = await response.json();
          this.results.services = services.services || [];
          
          const healthyCount = this.results.services.filter(s => s.status === 'healthy').length;
          const totalCount = this.results.services.length;
          
          if (healthyCount === totalCount && totalCount > 0) {
            spinner.succeed(style.success(`All ${totalCount} services healthy`));
          } else if (healthyCount > 0) {
            spinner.warn(style.warning(`${healthyCount}/${totalCount} services healthy`));
          } else {
            spinner.fail(style.error('No healthy services detected'));
          }
        } else {
          throw new Error('Services endpoint not available');
        }
        
      } catch (error) {
        spinner.fail(style.error('Failed to check services via coordinator'));
        await this.checkServicesDirectly();
      }
    } else {
      spinner.info(style.dim('Coordinator unavailable - checking services directly'));
      await this.checkServicesDirectly();
    }
  }

  async checkServicesDirectly() {
    const criticalServices = [
      { name: 'Enhanced RAG', port: 8094, path: '/health' },
      { name: 'Upload Service', port: 8093, path: '/health' },
      { name: 'Vector Service', port: 8095, path: '/api/health' },
      { name: 'gRPC Server', port: 50051, path: '/health' }
    ];
    
    console.log(style.dim('🔍 Direct service health checks:'));
    
    for (const service of criticalServices) {
      try {
        const response = await fetch(`http://localhost:${service.port}${service.path}`, {
          timeout: 3000
        });
        
        if (response.ok) {
          console.log(style.success(`  ✅ ${service.name} (${service.port})`));
          this.results.services.push({
            name: service.name,
            port: service.port,
            status: 'healthy'
          });
        } else {
          throw new Error(`HTTP ${response.status}`);
        }
      } catch (error) {
        console.log(style.warning(`  ⚠️ ${service.name} (${service.port}) - ${error.message}`));
        this.results.services.push({
          name: service.name,
          port: service.port,
          status: 'unavailable',
          error: error.message
        });
      }
    }
  }

  async checkErrorResolution() {
    const spinner = ora('🛠️ Checking error resolution system...').start();
    
    if (this.results.coordinator.health?.errorResolution) {
      const errorSystem = this.results.coordinator.health.errorResolution;
      
      if (errorSystem.enabled) {
        spinner.succeed(style.success(
          `Error resolution active: ${errorSystem.resolvedCount || 0} resolved`
        ));
        
        this.results.performance.errorResolution = {
          enabled: true,
          resolved: errorSystem.resolvedCount || 0,
          successRate: errorSystem.successRate || 0
        };
      } else {
        spinner.info(style.dim('Error resolution available but not active'));
      }
    } else {
      spinner.warn(style.warning('Error resolution system status unknown'));
      
      if (this.results.errors.some(e => e.type === 'typescript')) {
        this.results.recommendations.push('Error resolution would help with TypeScript errors');
      }
    }
  }

  displayReport() {
    const healthyServices = this.results.services.filter(s => s.status === 'healthy').length;
    const totalServices = this.results.services.length;
    const hasErrors = this.results.errors.length > 0;
    const coordinatorAvailable = this.results.coordinator.status === 'available';
    
    // Overall status
    let overallStatus = 'excellent';
    if (!coordinatorAvailable) overallStatus = 'degraded';
    else if (healthyServices < totalServices * 0.8) overallStatus = 'poor';
    else if (hasErrors) overallStatus = 'good';
    
    const statusColor = {
      excellent: '#51cf66',
      good: '#dca561', 
      degraded: '#ff6b6b',
      poor: '#ff4757'
    };
    
    // System summary
    const summary = [
      style.bold('📋 SYSTEM STATUS REPORT'),
      '',
      `🎯 Master Service Coordinator: ${coordinatorAvailable ? style.success('✅ Available') : style.error('❌ Unavailable')}`,
      `📡 Services Health: ${style.success(`${healthyServices}`)}/${totalServices} healthy`,
      hasErrors ? `🐛 Errors Detected: ${style.warning(this.results.errors.length.toString())}` : style.success('✅ No errors detected'),
      this.results.performance.errorResolution?.enabled 
        ? `🛠️ Error Resolution: ${style.success('Active')} (${this.results.performance.errorResolution.resolved} resolved)`
        : style.dim('🛠️ Error Resolution: Available'),
      '',
      style.secondary('📡 Access Points:'),
      coordinatorAvailable ? style.success('• Coordinator API: http://localhost:5173/api/v1/coordinator') : style.dim('• Coordinator API: Unavailable'),
      style.dim('• Health Dashboard: http://localhost:5173/system/health'),
      style.dim('• Frontend: http://localhost:5173'),
      '',
      `📊 Overall Status: ${style.bold(overallStatus.toUpperCase())}`
    ].filter(Boolean);
    
    console.log(boxen(summary.join('\n'), {
      padding: 1,
      margin: 1,
      borderStyle: 'round',
      borderColor: statusColor[overallStatus]
    }));
    
    // Service details if available
    if (this.results.services.length > 0) {
      console.log(style.secondary('\n📋 Service Details:'));
      
      this.results.services.forEach(service => {
        const status = service.status === 'healthy' 
          ? style.success('✅ Healthy')
          : style.warning('⚠️ Unavailable');
        
        console.log(style.dim(`  ${service.name} (${service.port}): ${status}`));
      });
    }
    
    // Recommendations
    if (this.results.recommendations.length > 0) {
      console.log(style.accent('\n💡 Recommendations:'));
      this.results.recommendations.forEach(rec => {
        console.log(style.dim(`  • ${rec}`));
      });
    }
    
    // Quick start commands
    console.log(style.secondary('\n🚀 Quick Commands:'));
    console.log(style.dim('  • Start coordinated system: npm run dev:full'));
    console.log(style.dim('  • Check coordinator status: npm run coordinator:status'));
    console.log(style.dim('  • View health dashboard: http://localhost:5173/system/health'));
    
    // Exit with appropriate code
    process.exit(overallStatus === 'excellent' || overallStatus === 'good' ? 0 : 1);
  }
}

// Run the health check
const checker = new CoordinatedSystemChecker();
checker.check().catch(console.error);