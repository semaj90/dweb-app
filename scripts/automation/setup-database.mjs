#!/usr/bin/env node

import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const POSTGRES_BIN = '"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe"';
const POSTGRES_USER = 'postgres';
const DATABASE_NAME = 'legal_ai_db';
const APP_USER = 'legal_admin';
const APP_PASSWORD = 'LegalSecure2024!';

console.log('🚀 Setting up Legal AI Database...\n');

async function runPsqlCommand(command, database = 'postgres') {
  const fullCommand = `${POSTGRES_BIN} -U ${POSTGRES_USER} -d ${database} -h localhost -c "${command}"`;
  console.log(`Running: ${command}`);
  
  try {
    const { stdout, stderr } = await execAsync(fullCommand);
    if (stdout) console.log('✅', stdout.trim());
    if (stderr) console.log('⚠️', stderr.trim());
    return true;
  } catch (error) {
    console.error('❌ Error:', error.message);
    return false;
  }
}

async function setupDatabase() {
  console.log('1. Creating database and user...');
  
  // Create database
  await runPsqlCommand(`CREATE DATABASE ${DATABASE_NAME};`);
  
  // Create user with password
  await runPsqlCommand(`CREATE USER ${APP_USER} WITH PASSWORD '${APP_PASSWORD}';`);
  
  // Grant privileges
  await runPsqlCommand(`GRANT ALL PRIVILEGES ON DATABASE ${DATABASE_NAME} TO ${APP_USER};`);
  
  console.log('\n2. Enabling pgvector extension...');
  
  // Enable pgvector extension
  await runPsqlCommand('CREATE EXTENSION IF NOT EXISTS vector;', DATABASE_NAME);
  
  // Grant usage on schema
  await runPsqlCommand(`GRANT ALL ON SCHEMA public TO ${APP_USER};`, DATABASE_NAME);
  
  console.log('\n3. Testing connection...');
  
  // Test connection with app user
  const testCommand = `${POSTGRES_BIN} -U ${APP_USER} -d ${DATABASE_NAME} -h localhost -c "SELECT version();"`;
  try {
    const { stdout } = await execAsync(testCommand);
    console.log('✅ Connection test successful!');
    console.log(stdout.trim());
  } catch (error) {
    console.error('❌ Connection test failed:', error.message);
  }
  
  console.log('\n✅ Database setup complete!');
  console.log(`Database: ${DATABASE_NAME}`);
  console.log(`User: ${APP_USER}`);
  console.log(`Connection string: postgresql://${APP_USER}:${APP_PASSWORD}@localhost:5432/${DATABASE_NAME}`);
}

setupDatabase().catch(console.error);