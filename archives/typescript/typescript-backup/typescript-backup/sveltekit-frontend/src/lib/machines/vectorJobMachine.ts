// XState Machine for Vector Job Status Tracking
// Manages lifecycle of vector processing jobs through Redis Streams + CUDA worker
import { createMachine, assign, fromPromise, type ActorRefFrom } from 'xstate';
import type { VectorJob, VectorJobResult, CUDAProcessingStatus } from '$lib/types/vector-jobs';

export interface VectorJobContext {
	jobId: string | null;
	ownerType: 'evidence' | 'report' | 'case' | 'document' | null;
	ownerId: string | null;
	operation: 'embedding' | 'similarity' | 'autoindex' | 'clustering' | null;
	priority: 'high' | 'medium' | 'low';
	
	// Job data
	inputData?: any;
	payload?: Record<string, any>;
	vector?: number[];
	
	// Results
	result?: VectorJobResult;
	cudaResponse?: any;
	error?: string;
	
	// Timing
	startTime?: Date;
	endTime?: Date;
	processingTimeMs?: number;
	
	// Retry logic
	attempts: number;
	maxAttempts: number;
	
	// WebGPU fallback
	useWebGPU: boolean;
	webGPUAvailable: boolean;
}

export type VectorJobEvent =
	| { type: 'SUBMIT_JOB'; jobId: string; ownerType: string; ownerId: string; operation: string; data?: any; priority?: string }
	| { type: 'JOB_QUEUED'; jobId: string }
	| { type: 'PROCESSING_STARTED' }
	| { type: 'CUDA_PROCESSING'; progress?: number }
	| { type: 'WEBGPU_FALLBACK' }
	| { type: 'PROCESSING_COMPLETED'; result: VectorJobResult }
	| { type: 'PROCESSING_FAILED'; error: string }
	| { type: 'RETRY' }
	| { type: 'CANCEL' }
	| { type: 'RESET' };

// Services for external API calls
const vectorJobServices = {
	submitToAPI: fromPromise(async ({ input }: { input: { context: VectorJobContext; event: any } }) => {
		const { context, event } = input;
		
		const jobData = {
			owner_type: context.ownerType,
			owner_id: context.ownerId,
			event: context.operation,
			vector: context.vector,
			payload: {
				operation: context.operation,
				data: context.inputData,
				...context.payload
			},
			priority: context.priority,
			use_webgpu_fallback: context.useWebGPU
		};

		const response = await fetch('/api/v1/vector/jobs', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(jobData)
		});

		if (!response.ok) {
			throw new Error(`Failed to submit job: ${response.statusText}`);
		}

		return await response.json();
	}),

	checkJobStatus: fromPromise(async ({ input }: { input: { jobId: string } }) => {
		const response = await fetch(`/api/v1/vector/jobs/${input.jobId}/status`);
		
		if (!response.ok) {
			throw new Error(`Failed to check job status: ${response.statusText}`);
		}

		return await response.json();
	}),

	processWithWebGPU: fromPromise(async ({ input }: { input: { context: VectorJobContext } }) => {
		// WebGPU fallback processing
		const { context } = input;
		
		if (!context.webGPUAvailable) {
			throw new Error('WebGPU not available for fallback processing');
		}

		// Use WebGPU for client-side vector processing
		const webGPUResponse = await fetch('/api/v1/webgpu/process', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				operation: context.operation,
				data: context.inputData,
				vector: context.vector,
				payload: context.payload
			})
		});

		if (!webGPUResponse.ok) {
			throw new Error(`WebGPU processing failed: ${webGPUResponse.statusText}`);
		}

		return await webGPUResponse.json();
	}),

	pollJobProgress: fromPromise(async ({ input }: { input: { jobId: string; signal: AbortSignal } }) => {
		let attempts = 0;
		const maxAttempts = 60; // 5 minutes with 5-second intervals
		
		while (attempts < maxAttempts) {
			if (input.signal.aborted) {
				throw new Error('Polling cancelled');
			}

			const response = await fetch(`/api/v1/vector/jobs/${input.jobId}/status`);
			const status = await response.json();

			if (status.status === 'completed') {
				return status;
			} else if (status.status === 'failed') {
				throw new Error(status.error || 'Job processing failed');
			}

			// Wait 5 seconds before next poll
			await new Promise(resolve => setTimeout(resolve, 5000));
			attempts++;
		}

		throw new Error('Job processing timeout');
	})
};

export const vectorJobMachine = createMachine({
	types: {
		context: {} as VectorJobContext,
		events: {} as VectorJobEvent,
	},
	
	id: 'vectorJob',
	
	initial: 'idle',
	
	context: {
		jobId: null,
		ownerType: null,
		ownerId: null,
		operation: null,
		priority: 'medium',
		attempts: 0,
		maxAttempts: 3,
		useWebGPU: false,
		webGPUAvailable: false
	},

	states: {
		idle: {
			on: {
				SUBMIT_JOB: {
					target: 'submitting',
					actions: assign(({ event }) => ({
						jobId: event.jobId,
						ownerType: event.ownerType as any,
						ownerId: event.ownerId,
						operation: event.operation as any,
						inputData: event.data,
						priority: (event.priority as any) || 'medium',
						startTime: new Date(),
						attempts: 0,
						error: undefined,
						result: undefined
					}))
				}
			}
		},

		submitting: {
			invoke: {
				id: 'submitJob',
				src: vectorJobServices.submitToAPI,
				input: ({ context, event }) => ({ context, event }),
				onDone: {
					target: 'queued',
					actions: assign(({ event }) => ({
						jobId: event.output.job_id,
					}))
				},
				onError: {
					target: 'failed',
					actions: assign(({ event }) => ({
						error: (event as any).error?.message || 'Failed to submit job'
					}))
				}
			}
		},

		queued: {
			entry: [
				// Check WebGPU availability
				assign(() => ({
					webGPUAvailable: typeof navigator !== 'undefined' && 'gpu' in navigator
				}))
			],
			
			invoke: {
				id: 'pollProgress',
				src: vectorJobServices.pollJobProgress,
				input: ({ context }) => ({ jobId: context.jobId! }),
				onDone: {
					target: 'completed',
					actions: assign(({ event }) => ({
						result: event.output,
						endTime: new Date(),
						processingTimeMs: Date.now() - (new Date().getTime())
					}))
				},
				onError: [
					{
						target: 'webgpuFallback',
						guard: ({ context }) => context.webGPUAvailable && !context.useWebGPU,
						actions: assign(() => ({ useWebGPU: true }))
					},
					{
						target: 'retrying',
						guard: ({ context }) => context.attempts < context.maxAttempts,
						actions: assign(({ context, event }) => ({
							attempts: context.attempts + 1,
							error: (event as any).error?.message || 'Processing failed'
						}))
					},
					{
						target: 'failed',
						actions: assign(({ event }) => ({
							error: (event as any).error?.message || 'Job processing failed after max retries',
							endTime: new Date()
						}))
					}
				]
			},

			on: {
				PROCESSING_STARTED: 'processing',
				CANCEL: 'cancelled'
			}
		},

		processing: {
			on: {
				CUDA_PROCESSING: {
					actions: assign(({ event }) => ({
						// Store CUDA processing updates
						cudaResponse: event
					}))
				},
				WEBGPU_FALLBACK: 'webgpuFallback',
				PROCESSING_COMPLETED: {
					target: 'completed',
					actions: assign(({ event }) => ({
						result: event.result,
						endTime: new Date(),
						processingTimeMs: Date.now() - (new Date().getTime())
					}))
				},
				PROCESSING_FAILED: {
					target: 'retrying',
					guard: ({ context }) => context.attempts < context.maxAttempts,
					actions: assign(({ context, event }) => ({
						attempts: context.attempts + 1,
						error: event.error
					}))
				},
				CANCEL: 'cancelled'
			}
		},

		webgpuFallback: {
			entry: assign(() => ({ useWebGPU: true })),
			
			invoke: {
				id: 'webgpuProcess',
				src: vectorJobServices.processWithWebGPU,
				input: ({ context }) => ({ context }),
				onDone: {
					target: 'completed',
					actions: assign(({ event }) => ({
						result: event.output,
						endTime: new Date(),
						processingTimeMs: Date.now() - (new Date().getTime())
					}))
				},
				onError: {
					target: 'failed',
					actions: assign(({ event }) => ({
						error: `WebGPU fallback failed: ${(event as any).error?.message}`,
						endTime: new Date()
					}))
				}
			},

			on: {
				CANCEL: 'cancelled'
			}
		},

		retrying: {
			after: {
				2000: {
					target: 'submitting',
					actions: assign(() => ({
						error: undefined
					}))
				}
			},

			on: {
				RETRY: 'submitting',
				CANCEL: 'cancelled'
			}
		},

		completed: {
			type: 'final',
			entry: [
				// Store result in local cache or IndexedDB if needed
				({ context }) => {
					console.log(`✅ Vector job ${context.jobId} completed in ${context.processingTimeMs}ms`);
					
					// Emit completion event for other parts of the app
					if (typeof window !== 'undefined') {
						window.dispatchEvent(new CustomEvent('vectorJobCompleted', {
							detail: {
								jobId: context.jobId,
								ownerType: context.ownerType,
								ownerId: context.ownerId,
								result: context.result,
								processingTime: context.processingTimeMs,
								usedWebGPU: context.useWebGPU
							}
						}));
					}
				}
			],

			on: {
				RESET: 'idle'
			}
		},

		failed: {
			entry: [
				({ context }) => {
					console.error(`❌ Vector job ${context.jobId} failed: ${context.error}`);
					
					// Emit failure event
					if (typeof window !== 'undefined') {
						window.dispatchEvent(new CustomEvent('vectorJobFailed', {
							detail: {
								jobId: context.jobId,
								error: context.error,
								attempts: context.attempts,
								usedWebGPU: context.useWebGPU
							}
						}));
					}
				}
			],

			on: {
				RETRY: {
					target: 'submitting',
					guard: ({ context }) => context.attempts < context.maxAttempts,
					actions: assign(() => ({ error: undefined }))
				},
				RESET: 'idle'
			}
		},

		cancelled: {
			entry: [
				({ context }) => {
					console.log(`🚫 Vector job ${context.jobId} cancelled`);
				}
			],
			
			on: {
				RESET: 'idle'
			}
		}
	}
});

export type VectorJobMachine = typeof vectorJobMachine;
export type VectorJobActor = ActorRefFrom<VectorJobMachine>;

// Helper function to create and start a vector job
export async function createVectorJob(
	ownerType: string,
	ownerId: string,
	operation: string,
	data?: any,
	priority: string = 'medium'
): Promise<any> {
	const { createActor } = await import('xstate');
	
	const actor = createActor(vectorJobMachine);
	actor.start();
	
	const jobId = `${ownerType}_${ownerId}_${operation}_${Date.now()}`;
	
	actor.send({
		type: 'SUBMIT_JOB',
		jobId,
		ownerType,
		ownerId,
		operation,
		data,
		priority
	});
	
	return actor;
}

// Utility for batch vector job processing
export async function processBatchVectorJobs(
	jobs: Array<{
		ownerType: string;
		ownerId: string;
		operation: string;
		data?: any;
		priority?: string;
	}>
): Promise<VectorJobActor[]> {
	const actors = await Promise.all(
		jobs.map(job => 
			createVectorJob(
				job.ownerType,
				job.ownerId,
				job.operation,
				job.data,
				job.priority || 'medium'
			)
		)
	);
	
	return actors;
}