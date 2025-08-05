import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// Types for Go server integration
interface DocumentProcessRequest {
	document_id: string;
	content: string;
	document_type: string;
	case_id?: string;
	options: ProcessingOptions;
}

interface ProcessingOptions {
	extract_entities: boolean;
	generate_summary: boolean;
	assess_risk: boolean;
	generate_embedding: boolean;
	store_in_database: boolean;
	use_gemma3_legal: boolean;
}

interface DocumentProcessResponse {
	success: boolean;
	document_id: string;
	summary?: string;
	entities?: LegalEntity[];
	risk_assessment?: RiskAssessment;
	embedding?: number[];
	processing_time: string;
	metadata: Record<string, unknown>;
	error?: string;
}

interface LegalEntity {
	type: string;
	value: string;
	confidence: number;
	start_pos: number;
	end_pos: number;
}

interface RiskAssessment {
	overall_risk: string;
	risk_score: number;
	risk_factors: string[];
	recommendations: string[];
	confidence: number;
}

// Configuration
const GO_SERVER_URL = process.env.GO_SERVER_URL || 'http://localhost:8080';
const REQUEST_TIMEOUT = 60000; // 60 seconds

/**
 * Process document through Go Legal AI Server
 * Integrates with Ollama and PostgreSQL via Go microservice
 */
export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json();
		
		// Validate required fields
		if (!body.content) {
			return json({ error: 'Content is required' }, { status: 400 });
		}

		// Prepare request for Go server
		const goRequest: DocumentProcessRequest = {
			document_id: body.document_id || `doc_${Date.now()}`,
			content: body.content,
			document_type: body.document_type || 'evidence',
			case_id: body.case_id,
			options: {
				extract_entities: body.extract_entities ?? true,
				generate_summary: body.generate_summary ?? true,
				assess_risk: body.assess_risk ?? true,
				generate_embedding: body.generate_embedding ?? true,
				store_in_database: body.store_in_database ?? true,
				use_gemma3_legal: body.use_gemma3_legal ?? true
			}
		};

		console.log(`🔄 Processing document via Go server: ${goRequest.document_id}`);

		// Call Go server
		const controller = new AbortController();
		const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

		try {
			const response = await fetch(`${GO_SERVER_URL}/process-document`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify(goRequest),
				signal: controller.signal
			});

			clearTimeout(timeoutId);

			if (!response.ok) {
				const errorText = await response.text();
				console.error(`❌ Go server error (${response.status}):`, errorText);
				
				return json({ 
					error: `Go server error: ${response.status}`,
					details: errorText 
				}, { status: response.status });
			}

			const result: DocumentProcessResponse = await response.json();
			
			console.log(`✅ Document processed successfully: ${result.document_id}`);
			console.log(`📊 Processing time: ${result.processing_time}`);
			
			if (result.summary) {
				console.log(`📝 Summary generated: ${result.summary.substring(0, 100)}...`);
			}
			
			if (result.entities && result.entities.length > 0) {
				console.log(`🏷️  Entities extracted: ${result.entities.length}`);
			}
			
			if (result.risk_assessment) {
				console.log(`⚠️  Risk assessment: ${result.risk_assessment.overall_risk} (${result.risk_assessment.risk_score})`);
			}

			// Return processed result
			return json({
				success: true,
				data: result,
				processed_by: 'go-legal-ai-server',
				timestamp: new Date().toISOString()
			});

		} catch (fetchError) {
			clearTimeout(timeoutId);
			
			if (fetchError instanceof Error && fetchError.name === 'AbortError') {
				console.error('❌ Go server request timeout');
				return json({ 
					error: 'Request timeout',
					details: 'Go server did not respond within 60 seconds'
				}, { status: 408 });
			}

			console.error('❌ Go server connection error:', fetchError);
			return json({ 
				error: 'Go server connection failed',
				details: fetchError instanceof Error ? fetchError.message : 'Unknown error'
			}, { status: 503 });
		}

	} catch (error) {
		console.error('❌ API endpoint error:', error);
		return json({ 
			error: 'Internal server error',
			details: error instanceof Error ? error.message : 'Unknown error'
		}, { status: 500 });
	}
};

/**
 * Get Go server health status
 */
export const GET: RequestHandler = async () => {
	try {
		const response = await fetch(`${GO_SERVER_URL}/health`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
			}
		});

		if (!response.ok) {
			return json({ 
				error: 'Go server health check failed',
				status: response.status 
			}, { status: 503 });
		}

		const healthData = await response.json();
		
		return json({
			success: true,
			go_server_status: healthData,
			sveltekit_status: 'healthy',
			integration_status: 'connected',
			timestamp: new Date().toISOString()
		});

	} catch (error) {
		console.error('❌ Go server health check failed:', error);
		return json({ 
			error: 'Go server unreachable',
			details: error instanceof Error ? error.message : 'Unknown error',
			go_server_url: GO_SERVER_URL
		}, { status: 503 });
	}
};