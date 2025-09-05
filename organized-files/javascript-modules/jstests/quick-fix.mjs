#!/usr/bin/env node
// Quick error check and fix runner

import { execSync } from 'child_process';
import { existsSync, mkdirSync } from 'fs';
import chalk from 'chalk';

console.log(chalk.bold.cyan(`
╔═══════════════════════════════════════════════╗
║       SvelteKit Error Fix Tool v1.0           ║
╚═══════════════════════════════════════════════╝
`));

// Ensure logs directory exists
if (!existsSync('logs')) {
  mkdirSync('logs');
  console.log(chalk.green('✅ Created logs directory'));
}

// Install dependencies first
console.log(chalk.yellow('\n📦 Installing dependencies...'));
try {
  execSync('npm install', { stdio: 'inherit' });
  execSync('cd sveltekit-frontend && npm install fuse.js', { stdio: 'inherit' });
} catch (error) {
  console.log(chalk.red('⚠️  Some dependencies failed to install'));
}

// Run the fix
console.log(chalk.yellow('\n🔧 Running automatic fixes...'));
try {
  // Run specific fixes
  execSync('node fix-specific-errors.mjs', { stdio: 'inherit' });
  
  // Run TypeScript fixes
  execSync('node fix-all-typescript-imports.mjs', { stdio: 'inherit' });
  
  console.log(chalk.green('\n✅ Fixes applied successfully!'));
} catch (error) {
  console.log(chalk.red('\n⚠️  Some fixes failed, but continuing...'));
}

// Final check
console.log(chalk.yellow('\n📋 Running final check...'));
try {
  execSync('cd sveltekit-frontend && npm run check', { stdio: 'inherit' });
  console.log(chalk.green('\n🎉 All checks passed!'));
} catch (error) {
  console.log(chalk.yellow('\n⚠️  Some errors remain. This is normal for a first run.'));
  console.log(chalk.cyan('\nTo see detailed logs:'));
  console.log(chalk.white('  - Check the logs/ directory'));
  console.log(chalk.white('  - Run: npm run check'));
}

console.log(chalk.bold.green('\n✨ Error fix process complete!'));
console.log(chalk.cyan('\nNext steps:'));
console.log(chalk.white('1. Run: npm run dev'));
console.log(chalk.white('2. Test your application'));
console.log(chalk.white('3. Run: npm run check (to verify)'));
