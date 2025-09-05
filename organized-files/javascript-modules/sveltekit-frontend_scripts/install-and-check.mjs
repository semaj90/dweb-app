#!/usr/bin/env node
/**
 * Comprehensive Installation and Error Checking Script
 * Installs all dependencies and checks for errors throughout the system
 */

import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m'
};

const log = {
  info: (msg) => console.log(`${colors.blue}ℹ${colors.reset}  ${msg}`),
  success: (msg) => console.log(`${colors.green}✅${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}❌${colors.reset} ${msg}`),
  warning: (msg) => console.log(`${colors.yellow}⚠️${colors.reset}  ${msg}`),
  section: (msg) => console.log(`\n${colors.cyan}${colors.bright}=== ${msg} ===${colors.reset}`),
  subsection: (msg) => console.log(`\n${colors.magenta}📋 ${msg}${colors.reset}`)
};

// Execute command and return promise
function execCommand(command, args = [], options = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, {
      cwd: projectRoot,
      shell: true,
      stdio: 'pipe',
      ...options
    });

    let stdout = '';
    let stderr = '';

    proc.stdout?.on('data', (data) => {
      stdout += data.toString();
      if (!options.silent) {
        process.stdout.write(data);
      }
    });

    proc.stderr?.on('data', (data) => {
      stderr += data.toString();
      if (!options.silent && !options.ignoreStderr) {
        process.stderr.write(data);
      }
    });

    proc.on('close', (code) => {
      resolve({ code, stdout, stderr });
    });

    proc.on('error', (err) => {
      reject(err);
    });
  });
}

// Check if a command exists
async function commandExists(command) {
  try {
    const result = await execCommand('where', [command], { silent: true });
    return result.code === 0;
  } catch {
    return false;
  }
}

// Main installation and checking process
async function main() {
  log.section('COMPREHENSIVE INSTALLATION & ERROR CHECK');
  const startTime = Date.now();
  const errors = [];
  const warnings = [];
  
  // 1. Check Node.js version
  log.subsection('Checking Node.js version');
  try {
    const { stdout } = await execCommand('node', ['--version'], { silent: true });
    const version = stdout.trim();
    log.success(`Node.js version: ${version}`);
    
    const major = parseInt(version.slice(1).split('.')[0]);
    if (major < 18) {
      errors.push(`Node.js version ${version} is too old. Required: v18+`);
      log.error('Node.js version is too old!');
    }
  } catch (err) {
    errors.push('Failed to check Node.js version');
    log.error('Failed to check Node.js version');
  }

  // 2. Check npm version
  log.subsection('Checking npm version');
  try {
    const { stdout } = await execCommand('npm', ['--version'], { silent: true });
    log.success(`npm version: ${stdout.trim()}`);
  } catch (err) {
    errors.push('npm not found');
    log.error('npm not found!');
  }

  // 3. Clean install dependencies
  log.subsection('Installing dependencies');
  log.info('Running npm ci for faster, cleaner installation...');
  
  try {
    // First, ensure package-lock.json exists
    const lockFileExists = await fs.access(path.join(projectRoot, 'package-lock.json'))
      .then(() => true)
      .catch(() => false);
    
    if (lockFileExists) {
      const { code, stderr } = await execCommand('npm', ['ci']);
      if (code === 0) {
        log.success('Dependencies installed successfully with npm ci');
      } else {
        log.warning('npm ci failed, falling back to npm install');
        const installResult = await execCommand('npm', ['install']);
        if (installResult.code === 0) {
          log.success('Dependencies installed successfully with npm install');
        } else {
          errors.push('Failed to install dependencies');
          log.error('Failed to install dependencies!');
        }
      }
    } else {
      log.info('No package-lock.json found, running npm install...');
      const { code } = await execCommand('npm', ['install']);
      if (code === 0) {
        log.success('Dependencies installed successfully');
      } else {
        errors.push('Failed to install dependencies');
        log.error('Failed to install dependencies!');
      }
    }
  } catch (err) {
    errors.push(`Installation error: ${err.message}`);
    log.error(`Installation failed: ${err.message}`);
  }

  // 4. Check for missing dependencies
  log.subsection('Checking for missing dependencies');
  try {
    const { stdout, stderr, code } = await execCommand('npm', ['ls', '--depth=0'], { silent: true });
    if (code !== 0 && stderr.includes('missing')) {
      const missingDeps = stderr.match(/missing: .+/g);
      if (missingDeps) {
        missingDeps.forEach(dep => {
          warnings.push(dep);
          log.warning(dep);
        });
      }
    } else {
      log.success('All dependencies are installed');
    }
  } catch (err) {
    warnings.push('Could not verify dependencies');
  }

  // 5. TypeScript compilation check
  log.subsection('Checking TypeScript compilation');
  try {
    log.info('Running TypeScript compiler...');
    const { code, stdout, stderr } = await execCommand('npx', ['tsc', '--noEmit', '--skipLibCheck'], { silent: true });
    
    if (code === 0) {
      log.success('TypeScript compilation successful');
    } else {
      const errorLines = stderr.split('\n').filter(line => line.includes('error'));
      errorLines.forEach(line => {
        errors.push(`TS Error: ${line}`);
      });
      log.error(`TypeScript compilation failed with ${errorLines.length} errors`);
      
      // Show first few errors
      errorLines.slice(0, 5).forEach(line => {
        console.log(`  ${colors.red}│${colors.reset} ${line}`);
      });
      if (errorLines.length > 5) {
        console.log(`  ${colors.yellow}... and ${errorLines.length - 5} more errors${colors.reset}`);
      }
    }
  } catch (err) {
    warnings.push('TypeScript check failed to run');
    log.warning('Could not run TypeScript check');
  }

  // 6. Svelte check
  log.subsection('Checking Svelte components');
  try {
    log.info('Running Svelte check...');
    const { code, stdout } = await execCommand('npx', ['svelte-check', '--threshold', 'error', '--fail-on-warnings', 'false'], { silent: true });
    
    if (code === 0) {
      log.success('Svelte components check passed');
    } else {
      const errorCount = (stdout.match(/Error:/g) || []).length;
      const warningCount = (stdout.match(/Warning:/g) || []).length;
      
      if (errorCount > 0) {
        errors.push(`Svelte check: ${errorCount} errors found`);
        log.error(`Svelte check found ${errorCount} errors`);
      }
      if (warningCount > 0) {
        warnings.push(`Svelte check: ${warningCount} warnings found`);
        log.warning(`Svelte check found ${warningCount} warnings`);
      }
    }
  } catch (err) {
    warnings.push('Svelte check failed to run');
    log.warning('Could not run Svelte check');
  }

  // 7. Check required services
  log.subsection('Checking required services');
  
  // Check PostgreSQL
  try {
    const { code } = await execCommand('pg_isready', [], { silent: true });
    if (code === 0) {
      log.success('PostgreSQL is running');
    } else {
      warnings.push('PostgreSQL is not running');
      log.warning('PostgreSQL is not running');
    }
  } catch {
    warnings.push('PostgreSQL not found or not running');
    log.warning('PostgreSQL not found');
  }

  // Check Redis
  try {
    const { code } = await execCommand('redis-cli', ['ping'], { silent: true });
    if (code === 0) {
      log.success('Redis is running');
    } else {
      warnings.push('Redis is not running');
      log.warning('Redis is not running');
    }
  } catch {
    warnings.push('Redis not found or not running');
    log.warning('Redis not found');
  }

  // Check Ollama
  try {
    const { code } = await execCommand('curl', ['-s', 'http://localhost:11434/api/tags'], { silent: true });
    if (code === 0) {
      log.success('Ollama is running');
    } else {
      warnings.push('Ollama is not running');
      log.warning('Ollama is not running');
    }
  } catch {
    warnings.push('Ollama not found or not running');
    log.warning('Ollama not found');
  }

  // 8. Check for AI model files
  log.subsection('Checking AI configuration');
  const aiFiles = [
    'src/lib/server/ai/ollama-config.ts',
    'src/lib/server/ai/ollama-service.ts',
    'src/lib/server/ai/types.ts',
    'src/routes/api/ai/generate/+server.ts',
    'src/routes/api/ai/embeddings/+server.ts',
    'src/routes/api/ai/analyze/+server.ts'
  ];

  for (const file of aiFiles) {
    const filePath = path.join(projectRoot, file);
    try {
      await fs.access(filePath);
      log.success(`Found: ${file}`);
    } catch {
      errors.push(`Missing AI file: ${file}`);
      log.error(`Missing: ${file}`);
    }
  }

  // 9. Check environment files
  log.subsection('Checking environment configuration');
  const envFiles = ['.env', '.env.local', '.env.ai'];
  let envFound = false;
  
  for (const envFile of envFiles) {
    const envPath = path.join(projectRoot, envFile);
    try {
      await fs.access(envPath);
      log.success(`Found: ${envFile}`);
      envFound = true;
    } catch {
      // Not an error, just info
    }
  }
  
  if (!envFound) {
    warnings.push('No .env file found. Create .env from .env.example');
    log.warning('No .env file found');
  }

  // 10. Security audit
  log.subsection('Running security audit');
  try {
    const { stdout, code } = await execCommand('npm', ['audit', '--json'], { silent: true });
    const audit = JSON.parse(stdout);
    
    if (audit.metadata) {
      const { vulnerabilities } = audit.metadata;
      const total = vulnerabilities.total || 0;
      
      if (total === 0) {
        log.success('No security vulnerabilities found');
      } else {
        const critical = vulnerabilities.critical || 0;
        const high = vulnerabilities.high || 0;
        
        if (critical > 0) {
          errors.push(`${critical} critical security vulnerabilities`);
          log.error(`${critical} critical vulnerabilities found`);
        }
        if (high > 0) {
          warnings.push(`${high} high security vulnerabilities`);
          log.warning(`${high} high vulnerabilities found`);
        }
        
        log.info(`Run 'npm audit fix' to fix vulnerabilities`);
      }
    }
  } catch {
    log.warning('Could not run security audit');
  }

  // 11. Test build
  log.subsection('Testing production build');
  try {
    log.info('Running build test (this may take a moment)...');
    const { code, stderr } = await execCommand('npm', ['run', 'build'], { silent: true });
    
    if (code === 0) {
      log.success('Production build successful');
    } else {
      errors.push('Production build failed');
      log.error('Production build failed');
      
      // Show build errors if any
      if (stderr) {
        const errorLines = stderr.split('\n').slice(0, 5);
        errorLines.forEach(line => {
          if (line.trim()) {
            console.log(`  ${colors.red}│${colors.reset} ${line}`);
          }
        });
      }
    }
  } catch (err) {
    warnings.push('Build test failed');
    log.warning('Could not test build');
  }

  // Final Summary
  const duration = ((Date.now() - startTime) / 1000).toFixed(2);
  
  log.section('INSTALLATION & CHECK SUMMARY');
  console.log(`\nCompleted in ${duration} seconds\n`);
  
  if (errors.length === 0 && warnings.length === 0) {
    log.success('🎉 SYSTEM IS FULLY OPERATIONAL!');
    log.success('All checks passed successfully');
    console.log('\n' + colors.green + '✨ Ready to start development:' + colors.reset);
    console.log('   npm run dev        - Start development server');
    console.log('   npm run dev:full   - Start with all services');
    console.log('   npm run test       - Run tests');
  } else {
    if (errors.length > 0) {
      console.log(`\n${colors.red}${colors.bright}ERRORS (${errors.length}):${colors.reset}`);
      errors.forEach((err, i) => {
        console.log(`  ${i + 1}. ${err}`);
      });
    }
    
    if (warnings.length > 0) {
      console.log(`\n${colors.yellow}${colors.bright}WARNINGS (${warnings.length}):${colors.reset}`);
      warnings.forEach((warn, i) => {
        console.log(`  ${i + 1}. ${warn}`);
      });
    }
    
    console.log('\n' + colors.yellow + '💡 Recommended actions:' + colors.reset);
    
    if (errors.some(e => e.includes('TypeScript'))) {
      console.log('   • Fix TypeScript errors: npm run check:typescript');
    }
    if (errors.some(e => e.includes('Svelte'))) {
      console.log('   • Fix Svelte errors: npm run check:svelte');
    }
    if (warnings.some(w => w.includes('PostgreSQL'))) {
      console.log('   • Start PostgreSQL: pg_ctl start');
    }
    if (warnings.some(w => w.includes('Redis'))) {
      console.log('   • Start Redis: redis-server');
    }
    if (warnings.some(w => w.includes('Ollama'))) {
      console.log('   • Start Ollama: ollama serve');
    }
    if (warnings.some(w => w.includes('vulnerabilities'))) {
      console.log('   • Fix vulnerabilities: npm audit fix');
    }
  }
  
  // Write results to file
  const results = {
    timestamp: new Date().toISOString(),
    duration: `${duration}s`,
    errors: errors.length,
    warnings: warnings.length,
    errorDetails: errors,
    warningDetails: warnings
  };
  
  await fs.writeFile(
    path.join(projectRoot, 'installation-check-results.json'),
    JSON.stringify(results, null, 2)
  );
  
  console.log('\n' + colors.cyan + '📄 Detailed results saved to: installation-check-results.json' + colors.reset);
  
  process.exit(errors.length > 0 ? 1 : 0);
}

// Run the script
main().catch(err => {
  log.error(`Script failed: ${err.message}`);
  process.exit(1);
});
