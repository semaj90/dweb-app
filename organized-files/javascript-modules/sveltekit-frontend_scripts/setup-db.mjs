#!/usr/bin/env node
import { execSync } from 'child_process';

console.log('Setting up database...');
execSync('psql -U postgres -d myapp -f migrations/0001_init.sql', { stdio: 'inherit' });
console.log('Database setup complete');
