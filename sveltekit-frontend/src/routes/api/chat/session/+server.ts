import { db } from "$lib/db/connection";
import { eq, desc } from "drizzle-orm";
import type { RequestHandler } from './$types.js';
import { URL } from "url";


export const POST: RequestHandler = async ({ request }) => {
	try {
		const sessionData = await request.json();

		if (!sessionData.id) {
			return json({ error: 'Session ID is required' }, { status: 400 });
		}

		// Create new chat session
		const [newSession] = await db
			.insert(chatSessions)
			.values({
				id: sessionData.id,
				model: sessionData.model || 'gemma3-legal',
				metadata: sessionData.metadata || {},
				createdAt: sessionData.createdAt || new Date(),
				updatedAt: new Date(),
				messageCount: 0,
				isActive: true
			})
			.returning();

		return json({
			success: true,
			session: newSession
		});

	} catch (error: any) {
		console.error('Error creating chat session:', error);
		return json({ 
			error: 'Failed to create chat session',
			details: error instanceof Error ? error.message : 'Unknown error'
		}, { status: 500 });
	}
};

export const GET: RequestHandler = async ({ url }) => {
	try {
		const sessionId = url.searchParams.get('sessionId');
		
		if (sessionId) {
			// Get specific session
			const session = await db
				.select()
				.from(chatSessions)
				.where(eq(chatSessions.id, sessionId))
				.limit(1);

			if (session.length === 0) {
				return json({ error: 'Session not found' }, { status: 404 });
			}

			return json({ session: session[0] });
		} else {
			// Get all active sessions (recent first)
			const sessions = await db
				.select()
				.from(chatSessions)
				.where(eq(chatSessions.isActive, true))
				.orderBy(desc(chatSessions.updatedAt))
				.limit(50);

			return json({ sessions });
		}

	} catch (error: any) {
		console.error('Error fetching chat sessions:', error);
		return json({ 
			error: 'Failed to fetch chat sessions',
			details: error instanceof Error ? error.message : 'Unknown error'
		}, { status: 500 });
	}
};