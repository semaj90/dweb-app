// Custom Drizzle PostgreSQL Adapter for Lucia with fixed JOIN query
import type { Adapter, DatabaseSession, DatabaseUser } from "lucia";
import { db } from '$lib/server/db/drizzle';
import { sessions, users } from '$lib/server/db/schema-postgres';
import { eq, lt } from 'drizzle-orm';

export class FixedDrizzlePostgreSQLAdapter implements Adapter {
  async deleteSession(sessionId: string): Promise<void> {
    try {
      await db.delete(sessions).where(eq(sessions.id, sessionId));
    } catch (error) {
      console.error('[AUTH] Error deleting session:', error);
      throw error;
    }
  }

  async deleteUserSessions(userId: string): Promise<void> {
    await db.delete(sessions).where(eq(sessions.user_id, userId));
  }

  async getSessionAndUser(
    sessionId: string
  ): Promise<[session: DatabaseSession | null, user: DatabaseUser | null]> {
    try {
      // Check if db and db.select exist
      if (!db || typeof db.select !== 'function') {
        console.error('[AUTH] Database connection not available:', { 
          dbExists: !!db, 
          selectExists: !!(db && typeof db.select === 'function'),
          dbType: typeof db
        });
        return [null, null];
      }

      // Build the correct JOIN query manually
      const result = await db
        .select({
          // User fields
          user: users,
          // Session fields
          session: sessions
        })
        .from(sessions)
        .innerJoin(users, eq(sessions.user_id, users.id))
        .where(eq(sessions.id, sessionId))
        .limit(1);

      if (result.length === 0) {
        return [null, null];
      }

      const { user, session } = result[0];

      // Transform to Lucia's expected format
      const databaseSession: DatabaseSession = {
        id: session.id,
        userId: session.user_id,
        expiresAt: session.expires_at,
        attributes: {
          ip_address: session.ip_address,
          user_agent: session.user_agent,
          session_context: session.session_context,
          created_at: session.created_at
        }
      };

      const databaseUser: DatabaseUser = {
        id: user.id,
        attributes: {
          email: user.email,
          hashed_password: user.hashed_password,
          username: user.username,
          first_name: user.first_name,
          last_name: user.last_name,
          role: user.role,
          department: user.department,
          jurisdiction: user.jurisdiction,
          permissions: user.permissions,
          is_active: user.is_active,
          email_verified: user.email_verified,
          avatar_url: user.avatar_url,
          last_login_at: user.last_login_at,
          practice_areas: user.practice_areas,
          bar_number: user.bar_number,
          firm_name: user.firm_name,
          profile_embedding: user.profile_embedding,
          metadata: user.metadata,
          created_at: user.created_at,
          updated_at: user.updated_at,
          deleted_at: user.deleted_at
        }
      };

      return [databaseSession, databaseUser];
    } catch (error) {
      console.error('[AUTH] Error in getSessionAndUser:', error);
      return [null, null];
    }
  }

  async getUserSessions(userId: string): Promise<DatabaseSession[]> {
    const result = await db
      .select()
      .from(sessions)
      .where(eq(sessions.user_id, userId));

    return result.map(session => ({
      id: session.id,
      userId: session.user_id,
      expiresAt: session.expires_at,
      attributes: {
        ip_address: session.ip_address,
        user_agent: session.user_agent,
        session_context: session.session_context,
        created_at: session.created_at
      }
    }));
  }

  async setSession(session: DatabaseSession): Promise<void> {
    await db.insert(sessions).values({
      id: session.id,
      user_id: session.userId,
      expires_at: session.expiresAt,
      ip_address: session.attributes.ip_address || null,
      user_agent: session.attributes.user_agent || null,
      session_context: session.attributes.session_context || {},
      created_at: session.attributes.created_at || new Date()
    });
  }

  async updateSessionExpiration(sessionId: string, expiresAt: Date): Promise<void> {
    await db
      .update(sessions)
      .set({ expires_at: expiresAt })
      .where(eq(sessions.id, sessionId));
  }

  async deleteExpiredSessions(): Promise<void> {
    await db.delete(sessions).where(lt(sessions.expires_at, new Date()));
  }
}