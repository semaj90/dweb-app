#!/usr/bin/env node

/**
 * Database Setup and Migration Script
 * Ensures proper database initialization for Legal AI Case Management System
 */

import { drizzle } from 'drizzle-orm/postgres-js';
import { migrate } from 'drizzle-orm/postgres-js/migrator';
import postgres from 'postgres';
import * as schema from './src/lib/db/schema.js';

// Database configuration
const DATABASE_URL = process.env.DATABASE_URL || 'postgresql://legal_admin:LegalSecure2024!@localhost:5432/legal_ai_v3';

console.log('🗄️ Legal AI Database Setup Starting...');
console.log('📍 Database URL:', DATABASE_URL.replace(/\/\/.*@/, '//[credentials]@'));

async function setupDatabase() {
  let client;
  
  try {
    // Create connection
    client = postgres(DATABASE_URL, { max: 1 });
    const db = drizzle(client, { schema });

    console.log('🔌 Testing database connection...');
    await client`SELECT 1 as test`;
    console.log('✅ Database connection successful!');

    // Check and install pgvector extension
    console.log('🧩 Checking pgvector extension...');
    const vectorCheck = await client`
      SELECT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
      ) as has_vector
    `;

    if (!vectorCheck[0].has_vector) {
      console.log('📦 Installing pgvector extension...');
      await client`CREATE EXTENSION IF NOT EXISTS vector`;
      console.log('✅ pgvector extension installed!');
    } else {
      console.log('✅ pgvector extension already installed!');
    }

    // Run migrations
    console.log('🚀 Running database migrations...');
    await migrate(db, { migrationsFolder: './drizzle' });
    console.log('✅ Database migrations completed!');

    // Verify tables
    console.log('🔍 Verifying table creation...');
    const tables = await client`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      AND table_type = 'BASE TABLE'
      ORDER BY table_name
    `;

    console.log('📋 Created tables:');
    tables.forEach(table => {
      console.log(`  ✓ ${table.table_name}`);
    });

    console.log('');
    console.log('🎉 Database setup completed successfully!');
    console.log('🔗 You can now connect to your Legal AI database.');
    
  } catch (error) {
    console.error('❌ Database setup failed:', error);
    console.error('');
    console.error('🔧 Troubleshooting steps:');
    console.error('  1. Ensure PostgreSQL is running');
    console.error('  2. Check database credentials in .env file');
    console.error('  3. Verify database exists and is accessible');
    console.error('  4. Run: npm run db:reset to start fresh');
    process.exit(1);
  } finally {
    if (client) {
      await client.end();
    }
  }
}

// Seed basic data
async function seedDatabase() {
  const client = postgres(DATABASE_URL, { max: 1 });
  const db = drizzle(client, { schema });

  try {
    console.log('🌱 Seeding initial data...');

    // Create admin user
    const adminUser = await db.insert(schema.users).values({
      email: 'admin@legal-ai.local',
      name: 'System Administrator',
      role: 'admin',
      passwordHash: '$2a$10$YourHashedPasswordHere' // Replace with actual hash
    }).returning().catch(() => {
      console.log('  ℹ️ Admin user already exists');
      return [];
    });

    if (adminUser.length > 0) {
      console.log('  ✓ Created admin user');
    }

    // Create sample case
    const sampleCase = await db.insert(schema.cases).values({
      title: 'Sample Legal Case',
      description: 'This is a sample case to demonstrate the system capabilities.',
      status: 'active',
      priority: 'medium',
      createdBy: adminUser[0]?.id || '00000000-0000-0000-0000-000000000000'
    }).returning().catch(() => {
      console.log('  ℹ️ Sample case already exists');
      return [];
    });

    if (sampleCase.length > 0) {
      console.log('  ✓ Created sample case');
    }

    console.log('✅ Initial data seeded successfully!');

  } catch (error) {
    console.log('⚠️ Seeding completed with some warnings:', error.message);
  } finally {
    await client.end();
  }
}

// Main execution
async function main() {
  await setupDatabase();
  
  // Only seed if this is a fresh setup
  if (process.argv.includes('--seed')) {
    await seedDatabase();
  }
  
  console.log('');
  console.log('🚀 Ready to launch Legal AI Case Management System!');
  console.log('📖 Next steps:');
  console.log('  1. npm run dev (start development server)');
  console.log('  2. Open http://localhost:5173');
  console.log('  3. Login with admin@legal-ai.local');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { setupDatabase, seedDatabase };