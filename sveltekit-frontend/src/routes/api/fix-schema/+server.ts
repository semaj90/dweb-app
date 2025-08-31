import type { RequestHandler } from './$types';

// Fix database schema to match Drizzle schema
import { json } from "@sveltejs/kit";
import { db, sql } from "drizzle-orm";
export async function POST(): Promise<any> {
  try {
    console.log('🔧 Adding missing columns to users table...');
    
    // Add missing columns that Drizzle schema expects
    await db.execute(sql`ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified TIMESTAMP`);
    await db.execute(sql`ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url TEXT`);
    
    console.log('✅ Added missing columns to users table');
    
    // Verify schema now matches
    const columns = await db.execute(sql`
      SELECT column_name, data_type, is_nullable 
      FROM information_schema.columns 
      WHERE table_name = 'users' AND table_schema = 'public'
      ORDER BY ordinal_position
    `);
    
    return json({
      success: true,
      message: 'Schema fixed successfully',
      columns: columns
    });
    
  } catch (error: any) {
    console.error('❌ Failed to fix schema:', error);
    return json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}