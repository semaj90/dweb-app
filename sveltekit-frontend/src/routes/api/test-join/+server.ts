import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db } from '$lib/server/db/drizzle';
import { sessions, users } from '$lib/server/db/schema-postgres';
import { eq } from 'drizzle-orm';

export const GET: RequestHandler = async ({ request }) => {
  try {
    console.log("=== DRIZZLE JOIN TEST ===");
    
    // Test 1: Direct session query
    console.log("Test 1: Simple session query");
    const directSessions = await db.select().from(sessions).limit(1);
    console.log("Sessions found:", directSessions.length);
    
    // Test 2: Manual JOIN query (what Lucia should be generating)
    console.log("Test 2: Manual JOIN query");
    const joinQuery = db
      .select({
        userId: users.id,
        userEmail: users.email,
        sessionId: sessions.id,
        sessionUserId: sessions.user_id
      })
      .from(sessions)
      .innerJoin(users, eq(sessions.user_id, users.id))
      .limit(1);
    
    // Log the SQL that would be generated
    console.log("Generated SQL:", joinQuery.toSQL());
    
    // Execute the join query
    const joinResults = await joinQuery;
    console.log("Join results:", joinResults.length);
    
    // Test 3: Simulate Lucia's getSessionAndUser query
    console.log("Test 3: Simulated Lucia query");
    const luciaQuery = db
      .select({
        // All user fields (like Lucia wants)
        ...users,
        // All session fields  
        session_id: sessions.id,
        session_user_id: sessions.user_id,
        session_expires_at: sessions.expires_at,
        session_ip_address: sessions.ip_address,
        session_user_agent: sessions.user_agent,
        session_context: sessions.session_context,
        session_created_at: sessions.created_at
      })
      .from(sessions)
      .innerJoin(users, eq(sessions.user_id, users.id))
      .where(eq(sessions.id, 'test-session-id'))
      .limit(1);
    
    console.log("Lucia-style SQL:", luciaQuery.toSQL());
    
    return json({
      success: true,
      message: 'JOIN query tests completed successfully',
      tests: {
        sessionsFound: directSessions.length,
        joinResults: joinResults.length,
        luciaQuerySQL: luciaQuery.toSQL()
      }
    });
    
  } catch (error: any) {
    console.error("JOIN Test Error:", error);
    return json({
      success: false,
      error: error.message,
      stack: error.stack,
    }, { status: 500 });
  }
};