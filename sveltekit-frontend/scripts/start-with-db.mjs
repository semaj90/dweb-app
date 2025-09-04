#!/usr/bin/env node
import { execSync } from 'child_process';

console.log('Running DB migration...');
try {
  execSync('psql -U postgres -d legal_ai_db -f migrations/0002_integrate_wiring.sql', { stdio: 'inherit' });
  console.log('Migration complete');
} catch (e) {
  console.log('Migration failed:', e.message);
}

console.log('Starting system...');
execSync('npm run dev', { stdio: 'inherit' });
