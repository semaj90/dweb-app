import { db } from "$lib/db/connection";
import { eq } from "drizzle-orm";
import type { RequestHandler } from './$types';


export const POST: RequestHandler = async ({ request }) => {
	try {
		const { messages, sessionId, model } = await request.json();

		if (!messages || !Array.isArray(messages)) {
			return json({ error: 'Messages array is required' }, { status: 400 });
		}

		if (!sessionId) {
			return json({ error: 'Session ID is required' }, { status: 400 });
		}

		// Ensure session exists
		const existingSession = await db
			.select()
			.from(chatSessions)
			.where(eq(chatSessions.id, sessionId))
			.limit(1);

		if (existingSession.length === 0) {
			return json({ error: 'Session not found' }, { status: 404 });
		}

		// Save messages with embeddings for vector search
		const savedMessages = [];
		
		for (const message of messages) {
			// Generate embedding for message content using nomic-embed-text
			let embedding: number[] | null = null;
			
			try {
				const embeddingResponse = await fetch('http://localhost:11434/api/embeddings', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						model: 'nomic-embed-text',
						prompt: message.content
					})
				});

				if (embeddingResponse.ok) {
					const embeddingData = await embeddingResponse.json();
					embedding = embeddingData.embedding;
				}
			} catch (error: any) {
				console.warn('Failed to generate embedding:', error);
				// Continue without embedding - not critical for basic functionality
			}

			// Insert message into database
			const [savedMessage] = await db
				.insert(chatMessages)
				.values({
					id: message.id,
					sessionId: sessionId,
					content: message.content,
					role: message.role,
					timestamp: message.timestamp || new Date(),
					embedding: embedding ? JSON.stringify(embedding) : null,
					metadata: message.metadata || {},
					model: model || 'gemma3-legal',
					confidence: message.metadata?.confidence || null
				})
				.returning();

			savedMessages.push(savedMessage);
		}

		// Update session with last activity
		await db
			.update(chatSessions)
			.set({ 
				updatedAt: new Date(),
				messageCount: existingSession[0].messageCount + messages.length
			})
			.where(eq(chatSessions.id, sessionId));

		return json({
			success: true,
			savedMessages: savedMessages.length,
			sessionId
		});

	} catch (error: any) {
		console.error('Error saving chat messages:', error);
		return json({ 
			error: 'Failed to save chat messages',
			details: error instanceof Error ? error.message : 'Unknown error'
		}, { status: 500 });
	}
};