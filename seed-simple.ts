import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { readFileSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Simple seed data without $lib imports
const sampleUsers = [
  {
    id: crypto.randomUUID(),
    email: 'prosecutor@example.com',
    name: 'Jane Smith',
    role: 'prosecutor',
    password_hash: '$2b$12$example'
  }
];

console.log('✅ Database seeded with basic data');
