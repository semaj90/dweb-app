/// <reference types="vite/client" />

/**
 * BullMQ Worker for Legal AI Document Processing
 * Integrates SvelteKit with Go Legal AI Server
 */

import { Worker, type Job } from "bullmq";
// TODO: Fix import - // Orphaned content: import {  import { evidence } from "$lib/server/db/schema-postgres";
// TODO: Fix import - // Orphaned content: import {  // Configuration
const GO_SERVER_URL = import.meta.env.GO_SERVER_URL || 'http://localhost:8080';
const REDIS_URL = import.meta.env.REDIS_URL || 'redis://localhost:6379';

// Job data interfaces
export interface LegalAIJobData {
	documentId: string;
	caseId?: string;
	content: string;
	documentType: 'evidence' | 'case' | 'legal_document';
	userId: string;
	options?: {
		extractEntities?: boolean;
		generateSummary?: boolean;
		assessRisk?: boolean;
		generateEmbedding?: boolean;
		storeInDatabase?: boolean;
		useGemma3Legal?: boolean;
	};
}

export interface GoServerResponse {
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

export interface LegalEntity {
	type: string;
	value: string;
	confidence: number;
	start_pos: number;
	end_pos: number;
}

export interface RiskAssessment {
	overall_risk: string;
	risk_score: number;
	risk_factors: string[];
	recommendations: string[];
	confidence: number;
}

/**
 * Process document through Go Legal AI Server
 */
async function processDocumentWithGoServer(jobData: LegalAIJobData): Promise<GoServerResponse> {
	const requestPayload = {
		document_id: jobData.documentId,
		content: jobData.content,
		document_type: jobData.documentType,
		case_id: jobData.caseId,
		options: {
			extract_entities: jobData.options?.extractEntities ?? true,
			generate_summary: jobData.options?.generateSummary ?? true,
			assess_risk: jobData.options?.assessRisk ?? true,
			generate_embedding: jobData.options?.generateEmbedding ?? true,
			store_in_database: jobData.options?.storeInDatabase ?? true,
			use_gemma3_legal: jobData.options?.useGemma3Legal ?? true
		}
	};

	console.log(`🔄 Sending document ${jobData.documentId} to Go server for processing...`);

	const response = await fetch(`${GO_SERVER_URL}/process-document`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify(requestPayload),
		// 5 minute timeout for complex processing
		signal: AbortSignal.timeout(300000)
	});

	if (!response.ok) {
		const errorText = await response.text();
		throw new Error(`Go server error (${response.status}): ${errorText}`);
	}

	return await response.json();
}

/**
 * Update evidence record with AI processing results
 */
async function updateEvidenceWithResults(
	documentId: string, 
	results: GoServerResponse
): Promise<void> {
	try {
		const updateData: Partial<typeof evidence.$inferInsert> = {
			updatedAt: new Date(),
		};

		// Add AI-generated summary
		if (results.summary) {
			updateData.aiSummary = results.summary;
		}

		// Add extracted entities as JSON
		if (results.entities && results.entities.length > 0) {
			updateData.aiExtractedEntities = JSON.stringify(results.entities);
		}

		// Add risk assessment
		if (results.risk_assessment) {
			updateData.aiRiskScore = results.risk_assessment.risk_score;
			updateData.aiRiskFactors = JSON.stringify(results.risk_assessment.risk_factors);
		}

		// Add processing metadata
		updateData.aiProcessingMetadata = JSON.stringify({
			processing_time: results.processing_time,
			processed_at: new Date().toISOString(),
			go_server_metadata: results.metadata,
			success: results.success
		});

		await db
			.update(evidence)
			.set(updateData)
			.where(eq(evidence.id, documentId));

		console.log(`✅ Evidence record ${documentId} updated with AI results`);

	} catch (error: any) {
		console.error(`❌ Failed to update evidence record ${documentId}:`, error);
		throw error;
	}
}

/**
 * Create and start the Legal AI worker
 */
export function createLegalAIWorker(): Worker {
	const worker = new Worker(
		'legal-ai-processing',
		async (job: Job<LegalAIJobData>) => {
			const { data } = job;
			const startTime = Date.now();

			console.log(`🚀 Processing legal AI job: ${job.id} for document: ${data.documentId}`);

			try {
				// Update job progress
				await job.updateProgress(10);

				// Process document with Go server
				const results = await processDocumentWithGoServer(data);
				await job.updateProgress(70);

				if (!results.success) {
					throw new Error(`Go server processing failed: ${results.error}`);
				}

				// Update database with results
				await updateEvidenceWithResults(data.documentId, results);
				await job.updateProgress(90);

				const processingTime = Date.now() - startTime;
				console.log(`✅ Legal AI job completed: ${job.id} in ${processingTime}ms`);

				// Return comprehensive results
				const jobResult = {
					success: true,
					documentId: data.documentId,
					processingTime: `${processingTime}ms`,
					goServerResults: results,
					summary: {
						entitiesExtracted: results.entities?.length || 0,
						summaryGenerated: !!results.summary,
						riskAssessed: !!results.risk_assessment,
						embeddingGenerated: !!results.embedding,
						overallRisk: results.risk_assessment?.overall_risk,
						riskScore: results.risk_assessment?.risk_score
					}
				};

				await job.updateProgress(100);
				return jobResult;

			} catch (error: any) {
				const processingTime = Date.now() - startTime;
				console.error(`❌ Legal AI job failed: ${job.id} after ${processingTime}ms:`, error);

				// Update evidence record with error status
				try {
					await db
						.update(evidence)
						.set({
							aiAnalysis: {
								error: error instanceof Error ? error.message : 'Unknown error',
								processing_time: `${processingTime}ms`,
								processed_at: new Date().toISOString(),
								success: false
							},
							updatedAt: new Date()
						})
						.where(eq(evidence.id, data.documentId));
				} catch (dbError) {
					console.error(`❌ Failed to update evidence with error status:`, dbError);
				}

				throw error;
			}
		},
		{
			connection: {
				host: 'localhost',
				port: 6379,
				// Parse Redis URL if provided
				...(REDIS_URL.startsWith('redis://') && {
					host: new URL(REDIS_URL).hostname,
					port: parseInt(new URL(REDIS_URL).port) || 6379
				})
			},
			concurrency: 2, // Process 2 documents simultaneously
			removeOnComplete: { count: 50 }, // Keep last 50 completed jobs
			removeOnFail: { count: 25 }, // Keep last 25 failed jobs
		}
	);

	// Event handlers
	worker.on('ready', () => {
		console.log('🟢 Legal AI Worker is ready and waiting for jobs');
	});

	worker.on('active', (job) => {
		console.log(`🔄 Legal AI Worker processing job: ${job.id}`);
	});

	worker.on('completed', (job, result) => {
		console.log(`✅ Legal AI Worker completed job: ${job.id}`);
		console.log(`📊 Results: ${JSON.stringify(result.summary, null, 2)}`);
	});

	worker.on('failed', (job, error) => {
		console.error(`❌ Legal AI Worker failed job: ${job?.id}:`, error);
	});

	worker.on('error', (error) => {
		console.error('❌ Legal AI Worker error:', error);
	});

	return worker;
}

/**
 * Add a document processing job to the queue
 */
export async function addLegalAIJob(
	jobData: LegalAIJobData,
	options?: {
		priority?: number;
		delay?: number;
		attempts?: number;
	}
): Promise<string> {
	const { Queue } = await import('bullmq');
	
	const queue = new Queue('legal-ai-processing', {
		connection: {
			host: 'localhost',
			port: 6379,
			...(REDIS_URL.startsWith('redis://') && {
				host: new URL(REDIS_URL).hostname,
				port: parseInt(new URL(REDIS_URL).port) || 6379
			})
		}
	});

	const job = await queue.add(
		'process-document',
		jobData,
		{
			priority: options?.priority || 0,
			delay: options?.delay || 0,
			attempts: options?.attempts || 3,
			backoff: {
				type: 'exponential',
				delay: 5000, // Start with 5 seconds
			},
			removeOnComplete: 50,
			removeOnFail: 25,
		}
	);

	console.log(`📋 Legal AI job queued: ${job.id} for document: ${jobData.documentId}`);
	return job.id!;
}

/**
 * Get job status
 */
export async function getLegalAIJobStatus(jobId: string): Promise<{
	status: string;
	progress: number;
	result?: unknown;
	error?: string;
}> {
	const { Queue } = await import('bullmq');
	
	const queue = new Queue('legal-ai-processing', {
		connection: {
			host: 'localhost',
			port: 6379,
			...(REDIS_URL.startsWith('redis://') && {
				host: new URL(REDIS_URL).hostname,
				port: parseInt(new URL(REDIS_URL).port) || 6379
			})
		}
	});

	const job = await queue.getJob(jobId);
	
	if (!job) {
		return { status: 'not_found', progress: 0 };
	}

	const state = await job.getState();
	const progress = job.progress || 0;

	return {
		status: state,
		progress: typeof progress === 'number' ? progress : 0,
		result: job.returnvalue,
		error: job.failedReason
	};
}

// Export for use in startup scripts
export { GO_SERVER_URL, REDIS_URL };