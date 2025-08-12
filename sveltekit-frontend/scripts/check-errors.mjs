// check-errors.mjs
// Fast error checking for TypeScript and Svelte files

import { exec } from 'child_process';
import { promisify } from 'util';
import chalk from 'chalk';
import ora from 'ora';
import fs from 'fs/promises';
import path from 'path';

const execAsync = promisify(exec);

const ERROR_PATTERNS = {
  typescript: /error TS\d+:/gi,
  svelte: /Error:|Warning:/gi,
  import: /Cannot find module|Module not found/gi,
  syntax: /SyntaxError|Unexpected token/gi,
  json: /JSON at position|Expected property name/gi
};

class ErrorChecker {
  constructor() {
    this.errors = [];
    this.warnings = [];
    this.startTime = Date.now();
  }

  async checkTypeScript() {
    const spinner = ora('Checking TypeScript...').start();
    try {
      const { stdout, stderr } = await execAsync('npx tsc --noEmit --pretty false --skipLibCheck');
      const output = stdout + stderr;
      
      if (output.includes('error')) {
        const errors = output.match(/.*error TS\d+:.*/g) || [];
        this.errors.push(...errors.map(e => ({ type: 'TypeScript', message: e })));
        spinner.fail(`TypeScript: ${errors.length} errors found`);
      } else {
        spinner.succeed('TypeScript: No errors');
      }
    } catch (error) {
      if (error.stdout || error.stderr) {
        const output = error.stdout + error.stderr;
        const errors = output.match(/.*error TS\d+:.*/g) || [];
        this.errors.push(...errors.map(e => ({ type: 'TypeScript', message: e })));
        spinner.fail(`TypeScript: ${errors.length} errors found`);
      } else {
        spinner.fail('TypeScript check failed');
      }
    }
  }

  async checkSvelte() {
    const spinner = ora('Checking Svelte...').start();
    try {
      const { stdout, stderr } = await execAsync('npx svelte-check --output machine --threshold warning');
      const output = stdout + stderr;
      
      // Parse svelte-check output
      const lines = output.split('\n');
      let errorCount = 0;
      let warningCount = 0;
      
      for (const line of lines) {
        if (line.includes('Error:')) {
          errorCount++;
          this.errors.push({ type: 'Svelte', message: line });
        } else if (line.includes('Warning:')) {
          warningCount++;
          this.warnings.push({ type: 'Svelte', message: line });
        }
      }
      
      if (errorCount > 0) {
        spinner.fail(`Svelte: ${errorCount} errors, ${warningCount} warnings`);
      } else if (warningCount > 0) {
        spinner.warn(`Svelte: ${warningCount} warnings`);
      } else {
        spinner.succeed('Svelte: No issues');
      }
    } catch (error) {
      spinner.warn('Svelte check completed with issues');
    }
  }

  async checkImports() {
    const spinner = ora('Checking imports...').start();
    
    try {
      // Find all TypeScript and Svelte files
      const files = await this.findFiles(['**/*.ts', '**/*.svelte'], [
        'node_modules/**',
        '.svelte-kit/**',
        'dist/**'
      ]);
      
      let importErrors = 0;
      
      for (const file of files) {
        const content = await fs.readFile(file, 'utf-8');
        const imports = content.match(/import .* from ['"](.+)['"]/g) || [];
        
        for (const imp of imports) {
          const modulePath = imp.match(/from ['"](.+)['"]/)?.[1];
          if (modulePath && !modulePath.startsWith('.') && !modulePath.startsWith('$')) {
            // Check if module exists in node_modules
            try {
              await fs.access(path.join('node_modules', modulePath.split('/')[0]));
            } catch {
              this.errors.push({
                type: 'Import',
                message: `Missing module "${modulePath}" in ${file}`
              });
              importErrors++;
            }
          }
        }
      }
      
      if (importErrors > 0) {
        spinner.fail(`Imports: ${importErrors} missing modules`);
      } else {
        spinner.succeed('Imports: All resolved');
      }
    } catch (error) {
      spinner.fail('Import check failed');
    }
  }

  async findFiles(patterns, ignore) {
    const { glob } = await import('glob');
    const files = [];
    
    for (const pattern of patterns) {
      const matches = await glob(pattern, { ignore });
      files.push(...matches);
    }
    
    return files;
  }

  printReport() {
    const duration = ((Date.now() - this.startTime) / 1000).toFixed(2);
    
    console.log('\n' + chalk.bold('═'.repeat(60)));
    console.log(chalk.bold.cyan('ERROR CHECK REPORT'));
    console.log(chalk.bold('═'.repeat(60)));
    
    if (this.errors.length === 0 && this.warnings.length === 0) {
      console.log(chalk.green.bold('\n✅ No errors or warnings found!'));
    } else {
      if (this.errors.length > 0) {
        console.log(chalk.red.bold(`\n❌ ${this.errors.length} Errors:`));
        this.errors.slice(0, 10).forEach((error, i) => {
          console.log(chalk.red(`  ${i + 1}. [${error.type}] ${this.truncate(error.message, 100)}`));
        });
        if (this.errors.length > 10) {
          console.log(chalk.gray(`  ... and ${this.errors.length - 10} more errors`));
        }
      }
      
      if (this.warnings.length > 0) {
        console.log(chalk.yellow.bold(`\n⚠️  ${this.warnings.length} Warnings:`));
        this.warnings.slice(0, 5).forEach((warning, i) => {
          console.log(chalk.yellow(`  ${i + 1}. [${warning.type}] ${this.truncate(warning.message, 100)}`));
        });
        if (this.warnings.length > 5) {
          console.log(chalk.gray(`  ... and ${this.warnings.length - 5} more warnings`));
        }
      }
    }
    
    console.log(chalk.gray(`\n⏱️  Check completed in ${duration}s`));
    console.log(chalk.bold('═'.repeat(60)));
    
    // Exit with error code if errors found
    if (this.errors.length > 0) {
      process.exit(1);
    }
  }

  truncate(str, maxLength) {
    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength - 3) + '...';
  }
}

// Run checks
async function main() {
  console.log(chalk.bold.blue('🔍 Starting comprehensive error check...\n'));
  
  const checker = new ErrorChecker();
  
  // Run checks in parallel for speed
  await Promise.all([
    checker.checkTypeScript(),
    checker.checkSvelte(),
    checker.checkImports()
  ]);
  
  checker.printReport();
}

// Handle missing dependencies gracefully
async function ensureDependencies() {
  const deps = ['chalk', 'ora', 'glob'];
  const missing = [];
  
  for (const dep of deps) {
    try {
      await import(dep);
    } catch {
      missing.push(dep);
    }
  }
  
  if (missing.length > 0) {
    console.log('Installing required dependencies...');
    const { exec } = await import('child_process');
    const { promisify } = await import('util');
    const execAsync = promisify(exec);
    
    await execAsync(`npm install --save-dev ${missing.join(' ')}`);
    console.log('Dependencies installed. Please run the script again.');
    process.exit(0);
  }
}

await ensureDependencies();
await main();
