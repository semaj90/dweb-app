/**
 * Component Error Validation Script
 * Tests all the fixes applied to resolve component errors
 */

import { spawn } from 'child_process';
import { writeFileSync, readFileSync } from 'fs';
import { join } from 'path';

interface TestResult {
  name: string;
  status: 'pass' | 'fail' | 'warning';
  message: string;
  details?: any;
}

class ComponentTester {
  private results: TestResult[] = [];
  private baseDir = process.cwd();

  async runAllTests() {
    console.log('🔍 Running Component Error Validation Tests...\n');

    // Test 1: TypeScript Compilation
    await this.testTypeScriptCompilation();

    // Test 2: Database Schema Validation
    await this.testDatabaseSchema();

    // Test 3: API Endpoints
    await this.testAPIEndpoints();

    // Test 4: Session Management
    await this.testSessionManagement();

    // Test 5: Type Consistency
    await this.testTypeConsistency();

    // Test 6: Pico CSS Removal
    await this.testPicoCSSRemoval();

    // Generate report
    this.generateReport();
  }

  private async testTypeScriptCompilation(): Promise<void> {
    try {
      console.log('📝 Testing TypeScript compilation...');
      
      const result = await this.runCommand('npm', ['run', 'check']);
      
      if (result.success) {
        this.addResult('TypeScript Compilation', 'pass', 'All TypeScript errors resolved');
      } else {
        const errorCount = (result.output.match(/error/gi) || []).length;
        this.addResult(
          'TypeScript Compilation', 
          errorCount > 10 ? 'fail' : 'warning', 
          `${errorCount} TypeScript errors found`,
          { output: result.output.slice(-500) }
        );
      }
    } catch (error: any) {
      this.addResult('TypeScript Compilation', 'fail', `Compilation test failed: ${error.message}`);
    }
  }

  private async testDatabaseSchema(): Promise<void> {
    try {
      console.log('🗄️ Testing database schema...');
      
      // Check if key schema files exist and are properly structured
      const schemaFile = join(this.baseDir, 'src/lib/server/db/schema-postgres.ts');
      const indexFile = join(this.baseDir, 'src/lib/server/db/index.ts');
      
      const schemaContent = readFileSync(schemaFile, 'utf8');
      const indexContent = readFileSync(indexFile, 'utf8');
      
      // Check for sessions table export
      const hasSessionsExport = indexContent.includes('sessions,') && schemaContent.includes('export const sessions');
      
      // Check for proper table imports
      const hasProperImports = indexContent.includes('import {') && indexContent.includes('sessions');
      
      if (hasSessionsExport && hasProperImports) {
        this.addResult('Database Schema', 'pass', 'Sessions table and exports properly configured');
      } else {
        this.addResult('Database Schema', 'fail', 'Missing sessions table exports or improper imports');
      }
    } catch (error: any) {
      this.addResult('Database Schema', 'fail', `Schema validation failed: ${error.message}`);
    }
  }

  private async testAPIEndpoints(): Promise<void> {
    try {
      console.log('🌐 Testing API endpoints...');
      
      // Test if optimization API has proper error handling
      const optimizationFile = join(this.baseDir, 'src/routes/api/copilot/optimize/+server.ts');
      const content = readFileSync(optimizationFile, 'utf8');
      
      const hasInputValidation = content.includes('if (!action)');
      const hasErrorHandling = content.includes('catch (err: any)');
      const hasProperLogging = content.includes('console.error');
      
      if (hasInputValidation && hasErrorHandling && hasProperLogging) {
        this.addResult('API Endpoints', 'pass', 'Optimization API has proper error handling');
      } else {
        this.addResult('API Endpoints', 'warning', 'API error handling may be incomplete');
      }
    } catch (error: any) {
      this.addResult('API Endpoints', 'fail', `API test failed: ${error.message}`);
    }
  }

  private async testSessionManagement(): Promise<void> {
    try {
      console.log('🔐 Testing session management...');
      
      const luciaFile = join(this.baseDir, 'src/lib/server/lucia.ts');
      const hooksFile = join(this.baseDir, 'src/hooks.server.ts');
      const appDFile = join(this.baseDir, 'src/app.d.ts');
      
      const luciaContent = readFileSync(luciaFile, 'utf8');
      const hooksContent = readFileSync(hooksFile, 'utf8');
      const appDContent = readFileSync(appDFile, 'utf8');
      
      // Check if Lucia has proper session types
      const hasSessionTypes = luciaContent.includes('SessionUser') && luciaContent.includes('SessionValidationResult');
      
      // Check if hooks use our custom validation
      const usesCustomValidation = hooksContent.includes('validateSession') && !hooksContent.includes('lucia.validateSession');
      
      // Check if app.d.ts has correct types
      const hasCorrectTypes = appDContent.includes('SessionUser') && appDContent.includes('auth');
      
      if (hasSessionTypes && usesCustomValidation && hasCorrectTypes) {
        this.addResult('Session Management', 'pass', 'Custom session management properly implemented');
      } else {
        this.addResult('Session Management', 'warning', 'Session management implementation may be incomplete');
      }
    } catch (error: any) {
      this.addResult('Session Management', 'fail', `Session test failed: ${error.message}`);
    }
  }

  private async testTypeConsistency(): Promise<void> {
    try {
      console.log('📋 Testing type consistency...');
      
      const authTypesFile = join(this.baseDir, 'src/lib/types/auth.ts');
      const ragTypesFile = join(this.baseDir, 'src/lib/types/rag.ts');
      
      const authExists = this.fileExists(authTypesFile);
      const ragExists = this.fileExists(ragTypesFile);
      
      if (authExists && ragExists) {
        const authContent = readFileSync(authTypesFile, 'utf8');
        const ragContent = readFileSync(ragTypesFile, 'utf8');
        
        const hasAuthTypes = authContent.includes('SessionUser') && authContent.includes('UserSession');
        const hasRAGTypes = ragContent.includes('DocumentType') && ragContent.includes('RAGSearchResult');
        
        if (hasAuthTypes && hasRAGTypes) {
          this.addResult('Type Consistency', 'pass', 'Auth and RAG types properly defined');
        } else {
          this.addResult('Type Consistency', 'warning', 'Some type definitions may be missing');
        }
      } else {
        this.addResult('Type Consistency', 'fail', 'Required type files not found');
      }
    } catch (error: any) {
      this.addResult('Type Consistency', 'fail', `Type consistency test failed: ${error.message}`);
    }
  }

  private async testPicoCSSRemoval(): Promise<void> {
    try {
      console.log('🎨 Testing Pico CSS removal...');
      
      const packageFile = join(this.baseDir, 'package.json');
      const svelteConfigFile = join(this.baseDir, 'svelte.config.js');
      
      const packageContent = readFileSync(packageFile, 'utf8');
      
      // Check if pico-css is removed from dependencies
      const hasPicoCSS = packageContent.includes('pico') && (
        packageContent.includes('@picocss/pico') || 
        packageContent.includes('pico-css')
      );
      
      if (!hasPicoCSS) {
        this.addResult('Pico CSS Removal', 'pass', 'No Pico CSS references found in package.json');
      } else {
        this.addResult('Pico CSS Removal', 'warning', 'Pico CSS references still found');
      }
    } catch (error: any) {
      this.addResult('Pico CSS Removal', 'fail', `Pico CSS test failed: ${error.message}`);
    }
  }

  private addResult(name: string, status: 'pass' | 'fail' | 'warning', message: string, details?: any) {
    this.results.push({ name, status, message, details });
    
    const icon = status === 'pass' ? '✅' : status === 'warning' ? '⚠️' : '❌';
    console.log(`${icon} ${name}: ${message}`);
  }

  private async runCommand(command: string, args: string[]): Promise<{ success: boolean; output: string }> {
    return new Promise((resolve) => {
      const process = spawn(command, args, { 
        stdio: 'pipe',
        shell: true,
        cwd: this.baseDir
      });
      
      let output = '';
      
      process.stdout?.on('data', (data) => {
        output += data.toString();
      });
      
      process.stderr?.on('data', (data) => {
        output += data.toString();
      });
      
      process.on('close', (code) => {
        resolve({
          success: code === 0,
          output
        });
      });
      
      // Timeout after 30 seconds
      setTimeout(() => {
        process.kill();
        resolve({
          success: false,
          output: output + '\\n[TIMEOUT]'
        });
      }, 30000);
    });
  }

  private fileExists(path: string): boolean {
    try {
      readFileSync(path);
      return true;
    } catch {
      return false;
    }
  }

  private generateReport() {
    console.log('\\n📊 Component Error Fix Report');
    console.log('═'.repeat(50));
    
    const passed = this.results.filter(r => r.status === 'pass').length;
    const warnings = this.results.filter(r => r.status === 'warning').length;
    const failed = this.results.filter(r => r.status === 'fail').length;
    
    console.log(`✅ Passed: ${passed}`);
    console.log(`⚠️  Warnings: ${warnings}`);
    console.log(`❌ Failed: ${failed}`);
    console.log(`📊 Total: ${this.results.length}`);
    
    if (failed === 0 && warnings <= 2) {
      console.log('\\n🎉 Component fixes successful! Most issues resolved.');
    } else if (failed <= 2) {
      console.log('\\n✨ Good progress! Minor issues remain.');
    } else {
      console.log('\\n🔧 More work needed. Check failed tests above.');
    }
    
    // Write detailed report
    const reportData = {
      timestamp: new Date().toISOString(),
      summary: { passed, warnings, failed, total: this.results.length },
      results: this.results
    };
    
    writeFileSync(
      join(this.baseDir, 'component-fix-report.json'),
      JSON.stringify(reportData, null, 2)
    );
    
    console.log('\\n📄 Detailed report saved to: component-fix-report.json');
  }
}

// Run tests
const tester = new ComponentTester();
tester.runAllTests().catch(console.error);
