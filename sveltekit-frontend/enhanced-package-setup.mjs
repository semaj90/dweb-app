// Update package.json with modern development scripts
const fs = require('fs');

try {
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));

  // Add modern development scripts
  const enhancedScripts = {
    ...packageJson.scripts,
ECHO is on.
    // Development
    'dev:modern': 'concurrently "npm run websocket:start" "vite dev --host localhost --port 5173"',
    'dev:bits-ui': 'vite dev --host localhost --port 5173',
    'dev:xstate': 'XSTATE_INSPECT=true npm run dev',
ECHO is on.
    // Building and testing
    'build:modern': 'vite build 
    'test:components': 'vitest run tests/',
    'test:e2e': 'playwright test',
    'test:e2e:ui': 'playwright test --ui',
    'test:all': 'npm run test:components 
ECHO is on.
    // Type checking and linting
    'check:modern': 'svelte-kit sync 
    'check:xstate': 'npm run check:modern 
"XState machines validated"',
    'lint:fix': 'prettier --write . 
ECHO is on.
    // Database operations
    'db:modern': 'drizzle-kit generate 
    'db:studio:modern': 'drizzle-kit studio --port 3001',
    'db:seed:modern': 'tsx src/lib/server/db/seed.ts',
ECHO is on.
    // Modern deployment
    'preview:modern': 'vite preview --host localhost --port 4173',
    'deploy:check': 'npm run check:modern 
ECHO is on.
    // Debugging and development tools
    'debug:xstate': 'XSTATE_INSPECT=true XSTATE_DEVTOOLS=true npm run dev',
    'debug:components': 'npm run test:components -- --reporter=verbose',
    'analyze:bundle': 'npm run build 
Need to install the following packages:
vite-bundle-analyzer@1.1.0
Ok to proceed? (y) 
ECHO is on.
    // Quick fixes and maintenance
    'fix:all:modern': 'npm run lint:fix 
    'clean:modern': 'rimraf .svelte-kit dist node_modules/.vite',
    'reset:modern': 'npm run clean:modern 

> prosecutor-web-frontend@0.0.1 prepare
> svelte-kit sync || echo ''


up to date, audited 1381 packages in 4s

273 packages are looking for funding
  run `npm fund` for details

13 vulnerabilities (6 low, 6 moderate, 1 critical)

To address issues that do not require attention, run:
  npm audit fix

To address all issues possible (including breaking changes), run:
  npm audit fix --force

Some issues need review, and may require choosing
a different dependency.

Run `npm audit` for details.
  };

  packageJson.scripts = enhancedScripts;

  // Add modern dev dependencies
  const modernDevDeps = {
    ...packageJson.devDependencies,
    '@testing-library/svelte': '5.0.0',
    '@testing-library/jest-dom': '6.1.0',
    'vitest': '2.0.0',
    'happy-dom': '15.0.0',
    'concurrently': '8.2.2'
  };

  packageJson.devDependencies = modernDevDeps;

  fs.writeFileSync('package.json', JSON.stringify(packageJson, null, 2));
  console.log('✅ Enhanced package.json with modern scripts');
} catch (error) {
  console.error('Error updating package.json:', error);
}
