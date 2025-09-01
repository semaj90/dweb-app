import type { RequestHandler } from './$types';

// Test login endpoint to debug authentication
import { json } from "@sveltejs/kit";

import { users } from "$lib/server/db/schema-postgres";
import { verifyPassword, createUserSession, setSessionCookie } from "drizzle-orm";
import { eq } from 'drizzle-orm';

export async function POST({ request }): Promise<any> {
  try {
    const { email, password } = await request.json();
    
    console.log(`[TEST LOGIN] Attempting login for: ${email}`);
    
    // Find user by email
    const existingUser = await db
      .select()
      .from(users)
      .where(eq(users.email, email))
      .limit(1);

    if (!existingUser.length) {
      console.log(`[TEST LOGIN] User not found: ${email}`);
      return json({ 
        success: false, 
        error: 'User not found',
        step: 'user_lookup'
      });
    }

    const user = existingUser[0];
    console.log(`[TEST LOGIN] User found: ${user.email}, has password: ${!!user.hashedPassword}, active: ${user.isActive}`);

    if (!user.hashedPassword) {
      console.log(`[TEST LOGIN] User has no password hash: ${email}`);
      return json({ 
        success: false, 
        error: 'User has no password set',
        step: 'password_check'
      });
    }

    if (!user.isActive) {
      console.log(`[TEST LOGIN] User is inactive: ${email}`);
      return json({ 
        success: false, 
        error: 'Account is deactivated',
        step: 'active_check'
      });
    }

    // Test password verification
    console.log(`[TEST LOGIN] Testing password verification for: ${email}`);
    const validPassword = await verifyPassword(user.hashedPassword, password);
    console.log(`[TEST LOGIN] Password verification result: ${validPassword}`);

    if (!validPassword) {
      return json({ 
        success: false, 
        error: 'Invalid password',
        step: 'password_verification'
      });
    }

    // Test session creation
    console.log(`[TEST LOGIN] Creating session for: ${email}`);
    const { sessionId, expiresAt } = await createUserSession(user.id);
    console.log(`[TEST LOGIN] Session created: ${sessionId}, expires: ${expiresAt}`);

    return json({
      success: true,
      message: 'Login successful',
      user: {
        id: user.id,
        email: user.email,
        role: user.role
      },
      session: {
        id: sessionId,
        expiresAt: expiresAt.toISOString()
      }
    });

  } catch (error: any) {
    console.error('[TEST LOGIN] Error:', error);
    return json({
      success: false,
      error: error.message,
      step: 'general_error'
    }, { status: 500 });
  }
}