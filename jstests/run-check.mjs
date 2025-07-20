#!/usr/bin/env node

import { execSync } from 'child_process';
import { writeFileSync } from 'fs';

console.log('🔍 Running TypeScript check to assess current state...');
console.log('=' * 60);

try {
  // Change to the sveltekit-frontend directory
  process.chdir('./sveltekit-frontend');
  
  console.log('📁 Working directory:', process.cwd());
  
  // Run npm run check
  console.log('\n🔧 Running: npm run check');
  console.log('-' * 40);
  
  const output = execSync('npm run check', { 
    encoding: 'utf8',
    timeout: 120000 // 2 minutes timeout
  });
  
  console.log('✅ Success! No TypeScript errors found.');
  console.log('\n📄 Output:');
  console.log(output);
  
  // Save the output
  writeFileSync('check-success-output.txt', output);
  console.log('\n📋 Output saved to: check-success-output.txt');
  
} catch (error) {
  console.log('⚠️ TypeScript check found issues:');
  console.log('\n📄 Error output:');
  console.log(error.stdout || error.message);
  
  // Save the error output
  const errorOutput = error.stdout || error.stderr || error.message;
  writeFileSync('check-error-output.txt', errorOutput);
  console.log('\n📋 Error output saved to: check-error-output.txt');
  
  // Parse the output to count errors and warnings
  const errorLines = (error.stdout || '').split('\n');
  const summaryLine = errorLines.find(line => 
    line.includes('found') && (line.includes('error') || line.includes('warning'))
  );
  
  if (summaryLine) {
    console.log('\n📊 SUMMARY:');
    console.log(summaryLine);
  }
  
  process.exit(1);
}
