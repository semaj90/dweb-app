#!/usr/bin/env node
// Validate canvas implementation

import { existsSync } from 'fs';
import { join } from 'path';
import chalk from 'chalk';

console.log(chalk.bold.cyan('\n🔍 Validating Canvas Implementation\n'));

const checks = [
  {
    name: 'Canvas component exists',
    path: 'sveltekit-frontend/src/routes/cases/[id]/canvas/+page.svelte',
    required: true
  },
  {
    name: 'Upload API endpoint exists',
    path: 'sveltekit-frontend/src/routes/api/evidence/upload/+server.ts',
    required: true
  },
  {
    name: 'Case service exists',
    path: 'sveltekit-frontend/src/lib/services/caseService.ts',
    required: true
  },
  {
    name: 'Context menu components exist',
    path: 'sveltekit-frontend/src/lib/components/ui/context-menu/index.ts',
    required: true
  },
  {
    name: 'Upload directory exists',
    path: 'sveltekit-frontend/static/uploads',
    required: false
  },
  {
    name: 'Button component exists',
    path: 'sveltekit-frontend/src/lib/components/ui/button/index.ts',
    required: true
  }
];

let allPassed = true;

for (const check of checks) {
  const fullPath = join(process.cwd(), check.path);
  const exists = existsSync(fullPath);
  
  if (exists) {
    console.log(chalk.green(`✅ ${check.name}`));
  } else if (check.required) {
    console.log(chalk.red(`❌ ${check.name} - MISSING`));
    allPassed = false;
  } else {
    console.log(chalk.yellow(`⚠️  ${check.name} - Optional, not found`));
  }
}

console.log('\n' + chalk.bold('📋 Implementation Checklist:'));

const features = [
  'Drag and drop file upload',
  'Context menu with right-click',
  'Progress tracking during upload',
  'TypeScript type safety',
  'Proper error handling',
  'Authentication check',
  'File type validation',
  'Responsive visual feedback'
];

features.forEach(feature => {
  console.log(chalk.gray(`  ☐ ${feature}`));
});

if (allPassed) {
  console.log(chalk.bold.green('\n✨ All required files are in place!'));
  console.log(chalk.cyan('\nNext steps:'));
  console.log('1. Run: npm run check:fix');
  console.log('2. Run: npm run dev');
  console.log('3. Test file upload at /cases/[id]/canvas');
} else {
  console.log(chalk.bold.red('\n⚠️  Some required files are missing!'));
  console.log(chalk.yellow('\nRun the setup script to create missing files:'));
  console.log('npm run check:fix');
}

// Create upload directory if it doesn't exist
const uploadDir = join(process.cwd(), 'sveltekit-frontend', 'static', 'uploads', 'evidence');
if (!existsSync(uploadDir)) {
  console.log(chalk.yellow('\n📁 Creating upload directory...'));
  require('fs').mkdirSync(uploadDir, { recursive: true });
  console.log(chalk.green('✅ Upload directory created'));
}

console.log('\n');
