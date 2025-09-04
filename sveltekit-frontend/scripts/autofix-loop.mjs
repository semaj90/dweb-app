#!/usr/bin/env node

import { execSync } from 'child_process';
import { readFileSync, writeFileSync } from 'fs';
import { glob } from 'glob';
import path from 'path';

// Autofix loop for Svelte project
async function runAutofixLoop() {
  console.log('Starting autofix loop...');

  const maxIterations = 10;
  let iteration = 0;

  while (iteration < maxIterations) {
	iteration++;
	console.log(`\nIteration ${iteration}:`);

	try {
	  // Run type checking
	  console.log('Running type check...');
	  execSync('npm run check', { stdio: 'inherit' });

	  // Run linting
	  console.log('Running linter...');
	  execSync('npm run lint', { stdio: 'inherit' });

	  console.log('✅ All checks passed!');
	  break;

	} catch (error) {
	  console.log(`❌ Issues found in iteration ${iteration}`);

	  // Try to auto-fix common issues
	  try {
		console.log('Attempting auto-fixes...');

		// Run prettier
		execSync('npm run format', { stdio: 'inherit' });

		// Run eslint with --fix
		execSync('npx eslint --fix src/', { stdio: 'inherit' });

		console.log('Auto-fixes applied');

	  } catch (fixError) {
		console.log('Auto-fix failed:', fixError.message);
		break;
	  }
	}
  }

  if (iteration >= maxIterations) {
	console.log('\n⚠️  Max iterations reached. Manual intervention may be required.');
  }
}

runAutofixLoop().catch(console.error);
