import type { RequestHandler } from './$types';

// Legal AI Session Creation API
// Creates and manages legal AI sessions with YoRHa interface integration

import type { LegalAISession, LegalContext, SessionStatus } from '$lib/types/yorha-interface';

// Session storage (in production, use database)
const activeSessions = new Map<string, LegalAISession>();

/** POST /api/v1/legal/session/create - Create new legal AI session */
export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json();
		const { user_id, case_id, context } = body;

		// Validate required fields
		if (!user_id) {
			return json({ error: 'user_id is required' }, { status: 400 });
		}

		// Generate session ID
		const session_id = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
		const timestamp = new Date().toISOString();

		// Create legal AI session
		const session: LegalAISession = {
			session_id,
			user_id,
			case_id: case_id || `case-${Date.now()}`,
			started_at: timestamp,
			last_activity: timestamp,
			status: 'ACTIVE',
			query_count: 0,
			processing_time_total: 0,
			context: validateAndEnhanceContext(context)
		};

		// Store session
		activeSessions.set(session_id, session);

		console.log(`[Legal AI] Session created: ${session_id} for user: ${user_id}`);

		return json({
			success: true,
			session_id,
			...session,
			message: 'Legal AI session created successfully'
		});

	} catch (error: any) {
		console.error('[Legal AI] Session creation error:', error);
		return json({
			success: false,
			error: 'Failed to create legal AI session',
			details: error instanceof Error ? error.message : 'Unknown error'
		}, { status: 500 });
	}
};

/** GET /api/v1/legal/session/create - Get session creation info and active sessions */
export const GET: RequestHandler = async () => {
	return json({
		service: 'Legal AI Session Manager',
		version: '1.0.0',
		active_sessions: activeSessions.size,
		supported_context_fields: {
			jurisdiction: 'string',
			practice_area: 'string[]',
			case_type: 'string',
			priority_level: 'number (1-10)',
			security_classification: 'SecurityLevel',
			related_cases: 'string[]',
			key_entities: 'string[]'
		},
		session_statuses: ['ACTIVE', 'IDLE', 'PAUSED', 'TERMINATED', 'ERROR'],
		security_levels: ['MINIMUM', 'STANDARD', 'HIGH', 'MAXIMUM', 'CLASSIFIED'],
		endpoints: {
			create_session: 'POST /api/v1/legal/session/create',
			get_session: 'GET /api/v1/legal/session/{session_id}',
			update_session: 'PUT /api/v1/legal/session/{session_id}',
			terminate_session: 'DELETE /api/v1/legal/session/{session_id}'
		},
		timestamp: new Date().toISOString()
	});
};

// Helper functions

function validateAndEnhanceContext(context: any): LegalContext {
	const defaultContext: LegalContext = {
		jurisdiction: 'Global',
		practice_area: ['General Legal'],
		case_type: 'Investigation',
		priority_level: 5,
		security_classification: 'STANDARD',
		related_cases: [],
		key_entities: []
	};

	if (!context) return defaultContext;

	return {
		jurisdiction: context.jurisdiction || defaultContext.jurisdiction,
		practice_area: Array.isArray(context.practice_area) 
			? context.practice_area 
			: defaultContext.practice_area,
		case_type: context.case_type || defaultContext.case_type,
		priority_level: typeof context.priority_level === 'number' 
			? Math.max(1, Math.min(10, context.priority_level))
			: defaultContext.priority_level,
		security_classification: isValidSecurityLevel(context.security_classification)
			? context.security_classification
			: defaultContext.security_classification,
		related_cases: Array.isArray(context.related_cases)
			? context.related_cases
			: defaultContext.related_cases,
		key_entities: Array.isArray(context.key_entities)
			? context.key_entities
			: defaultContext.key_entities
	};
}

function isValidSecurityLevel(level: any): boolean {
	const validLevels = ['MINIMUM', 'STANDARD', 'HIGH', 'MAXIMUM', 'CLASSIFIED'];
	return typeof level === 'string' && validLevels.includes(level);
}

// Export session storage for other endpoints
export { activeSessions };