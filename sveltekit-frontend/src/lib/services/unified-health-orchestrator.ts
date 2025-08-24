// Unified Health Monitoring and Job Orchestration Service
// Coordinates CUDA, WebGPU, WASM LLM, Redis Streams, and PostgreSQL components

import { createMachine, assign, fromPromise, type ActorRefFrom } from 'xstate';
import type { 
	VectorServiceMetrics, 
	VectorHealthStatus, 
	VectorProcessingError 
} from '$lib/types/vector-jobs';

import { webgpuPolyfill } from '$lib/webgpu/webgpu-polyfill';
import { wasmLLMService } from '$lib/wasm/wasm-llm-service';

export interface HealthOrchestratorContext {
	services: {
		redis: 'connected' | 'disconnected' | 'error';
		postgres: 'connected' | 'disconnected' | 'error';
		rabbitmq: 'connected' | 'disconnected' | 'error';
		cuda: 'available' | 'unavailable' | 'error';
		webgpu: 'available' | 'unavailable' | 'not_supported';
		wasmLLM: 'loaded' | 'unloaded' | 'error';
		vectorService: 'healthy' | 'degraded' | 'unhealthy';
	};
	
	metrics: VectorServiceMetrics | null;
	healthStatus: VectorHealthStatus | null;
	
	jobQueue: {
		pending: number;
		processing: number;
		completed: number;
		failed: number;
	};
	
	performance: {
		totalThroughput: number;
		averageLatency: number;
		errorRate: number;
		uptime: number;
	};
	
	errors: VectorProcessingError[];
	lastHealthCheck: Date | null;
	healthCheckInterval: number; // milliseconds
	autoRecovery: boolean;
	alertThresholds: {
		errorRate: number;
		latency: number;
		queueDepth: number;
	};
}

export type HealthOrchestratorEvent =
	| { type: 'START_MONITORING' }
	| { type: 'STOP_MONITORING' }
	| { type: 'HEALTH_CHECK' }
	| { type: 'SERVICE_UP'; service: string }
	| { type: 'SERVICE_DOWN'; service: string; error?: string }
	| { type: 'METRICS_UPDATE'; metrics: VectorServiceMetrics }
	| { type: 'ERROR_OCCURRED'; error: VectorProcessingError }
	| { type: 'RECOVERY_ATTEMPT'; service: string }
	| { type: 'ALERT_THRESHOLD_EXCEEDED'; threshold: string; value: number }
	| { type: 'RESET_ERRORS' }
	| { type: 'UPDATE_THRESHOLDS'; thresholds: any };

const healthOrchestratorServices = {
	checkServiceHealth: fromPromise(async ({ input }: { input: { context: HealthOrchestratorContext } }) => {
		const healthChecks = await Promise.allSettled([
			checkRedisHealth(),
			checkPostgresHealth(),
			checkRabbitMQHealth(),
			checkCUDAHealth(),
			checkWebGPUHealth(),
			checkWASMLLMHealth(),
			checkVectorServiceHealth()
		]);

		const services = {
			redis: resolveHealthStatus(healthChecks[0]),
			postgres: resolveHealthStatus(healthChecks[1]),
			rabbitmq: resolveHealthStatus(healthChecks[2]),
			cuda: resolveHealthStatus(healthChecks[3]),
			webgpu: resolveHealthStatus(healthChecks[4]),
			wasmLLM: resolveHealthStatus(healthChecks[5]),
			vectorService: resolveHealthStatus(healthChecks[6])
		};

		return { services, timestamp: new Date() };
	}),

	collectMetrics: fromPromise(async ({ input }: { input: { context: HealthOrchestratorContext } }) => {
		const [
			queueMetrics,
			performanceMetrics,
			resourceMetrics,
			jobMetrics
		] = await Promise.allSettled([
			fetchQueueMetrics(),
			fetchPerformanceMetrics(),
			fetchResourceMetrics(),
			fetchJobMetrics()
		]);

		const metrics: VectorServiceMetrics = {
			queueDepth: queueMetrics.status === 'fulfilled' ? queueMetrics.value : {
				embeddings: 0,
				similarities: 0,
				indexing: 0,
				clustering: 0
			},
			processingStats: performanceMetrics.status === 'fulfilled' ? performanceMetrics.value : {
				totalProcessed: 0,
				averageProcessingTimeMs: 0,
				successRate: 0,
				errorRate: 0
			},
			resourceUsage: resourceMetrics.status === 'fulfilled' ? resourceMetrics.value : {
				cudaUtilization: 0,
				webgpuUtilization: 0,
				memoryUsage: 0,
				redisConnections: 0
			},
			performance: {
				cudaOpsPerSecond: 0,
				webgpuOpsPerSecond: 0,
				vectorsPerSecond: 0,
				throughputMBps: 0,
				...((resourceMetrics.status === 'fulfilled' ? resourceMetrics.value : {}) as any)
			}
		};

		return metrics;
	}),

	attemptServiceRecovery: fromPromise(async ({ input }: { input: { service: string } }) => {
		const { service } = input;
		
		console.log(`🔄 Attempting recovery for service: ${service}`);
		
		switch (service) {
			case 'webgpu':
				return await recoverWebGPU();
			case 'wasmLLM':
				return await recoverWASMLLM();
			case 'redis':
				return await recoverRedis();
			case 'vectorService':
				return await recoverVectorService();
			default:
				throw new Error(`Unknown service: ${service}`);
		}
	}),

	sendAlert: fromPromise(async ({ input }: { input: { alert: any } }) => {
		// Send alert to monitoring systems, webhooks, etc.
		console.warn('🚨 Health Alert:', input.alert);
		
		// In production, this would integrate with:
		// - Slack/Teams webhooks
		// - Email notifications
		// - PagerDuty/OpsGenie
		// - Custom monitoring dashboards
		
		return { sent: true, timestamp: new Date() };
	})
};

export const healthOrchestratorMachine = createMachine({
	types: {
		context: {} as HealthOrchestratorContext,
		events: {} as HealthOrchestratorEvent,
	},

	id: 'healthOrchestrator',

	initial: 'idle',

	context: {
		services: {
			redis: 'disconnected',
			postgres: 'disconnected',
			rabbitmq: 'disconnected',
			cuda: 'unavailable',
			webgpu: 'unavailable',
			wasmLLM: 'unloaded',
			vectorService: 'unhealthy'
		},
		metrics: null,
		healthStatus: null,
		jobQueue: {
			pending: 0,
			processing: 0,
			completed: 0,
			failed: 0
		},
		performance: {
			totalThroughput: 0,
			averageLatency: 0,
			errorRate: 0,
			uptime: 0
		},
		errors: [],
		lastHealthCheck: null,
		healthCheckInterval: 30000, // 30 seconds
		autoRecovery: true,
		alertThresholds: {
			errorRate: 0.1, // 10%
			latency: 5000,  // 5 seconds
			queueDepth: 1000
		}
	},

	states: {
		idle: {
			on: {
				START_MONITORING: 'initializing'
			}
		},

		initializing: {
			invoke: {
				id: 'initializeServices',
				src: healthOrchestratorServices.checkServiceHealth,
				input: ({ context }) => ({ context }),
				onDone: {
					target: 'monitoring',
					actions: assign(({ event }) => ({
						services: event.output.services,
						lastHealthCheck: event.output.timestamp
					}))
				},
				onError: {
					target: 'error',
					actions: assign(({ event }) => ({
						errors: [{
							message: `Initialization failed: ${event.error}`,
							jobId: 'system',
							operation: 'initialization',
							stage: 'startup',
							retryable: true
						} as VectorProcessingError]
					}))
				}
			}
		},

		monitoring: {
			entry: [
				() => console.log('🔍 Health monitoring started'),
				// Initialize WebGPU and WASM services
				async () => {
					await webgpuPolyfill.initialize();
					await wasmLLMService.initialize();
				}
			],

			invoke: {
				id: 'healthCheckTimer',
				src: fromPromise(async ({ input }: { input: { interval: number } }) => {
					return new Promise((resolve) => {
						const timer = setInterval(() => {
							resolve({ type: 'HEALTH_CHECK' });
						}, input.interval);
						
						// Cleanup function
						return () => clearInterval(timer);
					});
				}),
				input: ({ context }) => ({ interval: context.healthCheckInterval })
			},

			on: {
				HEALTH_CHECK: {
					target: 'checkingHealth',
					actions: [
						() => console.log('🔍 Performing health check...')
					]
				},

				SERVICE_UP: {
					actions: assign(({ context, event }) => ({
						services: {
							...context.services,
							[event.service]: 'connected'
						}
					}))
				},

				SERVICE_DOWN: {
					target: 'degraded',
					actions: assign(({ context, event }) => ({
						services: {
							...context.services,
							[event.service]: 'error'
						},
						errors: [
							...context.errors,
							{
								message: `Service ${event.service} is down: ${event.error || 'Unknown error'}`,
								jobId: 'system',
								operation: 'service_check',
								stage: 'monitoring',
								retryable: true
							} as VectorProcessingError
						]
					}))
				},

				METRICS_UPDATE: {
					actions: [
						assign(({ event }) => ({ metrics: event.metrics })),
						// Check alert thresholds
						({ context, event }) => {
							const metrics = event.metrics;
							const thresholds = context.alertThresholds;
							
							if (metrics.processingStats.errorRate > thresholds.errorRate) {
								// Trigger alert
								console.warn(`🚨 Error rate threshold exceeded: ${metrics.processingStats.errorRate}`);
							}
							
							if (metrics.processingStats.averageProcessingTimeMs > thresholds.latency) {
								console.warn(`🚨 Latency threshold exceeded: ${metrics.processingStats.averageProcessingTimeMs}ms`);
							}
						}
					]
				},

				ERROR_OCCURRED: {
					actions: assign(({ context, event }) => ({
						errors: [...context.errors.slice(-99), event.error] // Keep last 100 errors
					}))
				},

				STOP_MONITORING: 'idle'
			}
		},

		checkingHealth: {
			invoke: {
				id: 'performHealthCheck',
				src: healthOrchestratorServices.checkServiceHealth,
				input: ({ context }) => ({ context }),
				onDone: {
					target: 'collectingMetrics',
					actions: assign(({ event }) => ({
						services: event.output.services,
						lastHealthCheck: event.output.timestamp
					}))
				},
				onError: {
					target: 'monitoring',
					actions: assign(({ context, event }) => ({
						errors: [
							...context.errors,
							{
								message: `Health check failed: ${event.error}`,
								jobId: 'system',
								operation: 'health_check',
								stage: 'monitoring',
								retryable: true
							} as VectorProcessingError
						]
					}))
				}
			}
		},

		collectingMetrics: {
			invoke: {
				id: 'collectSystemMetrics',
				src: healthOrchestratorServices.collectMetrics,
				input: ({ context }) => ({ context }),
				onDone: {
					target: 'monitoring',
					actions: [
						assign(({ event }) => ({ metrics: event.output })),
						// Emit metrics update
						({ event }) => {
							if (typeof window !== 'undefined') {
								window.dispatchEvent(new CustomEvent('systemMetricsUpdate', {
									detail: event.output
								}));
							}
						}
					]
				},
				onError: {
					target: 'monitoring',
					actions: assign(({ context, event }) => ({
						errors: [
							...context.errors,
							{
								message: `Metrics collection failed: ${event.error}`,
								jobId: 'system',
								operation: 'metrics_collection',
								stage: 'monitoring',
								retryable: true
							} as VectorProcessingError
						]
					}))
				}
			}
		},

		degraded: {
			entry: [
				() => console.warn('⚠️ System degraded - some services are unavailable'),
				// Attempt auto-recovery if enabled
				({ context }) => {
					if (context.autoRecovery) {
						const degradedServices = Object.entries(context.services)
							.filter(([_, status]) => status === 'error')
							.map(([service]) => service);
						
						if (degradedServices.length > 0) {
							console.log(`🔄 Auto-recovery triggered for services: ${degradedServices.join(', ')}`);
						}
					}
				}
			],

			on: {
				RECOVERY_ATTEMPT: {
					target: 'recovering',
					actions: assign(({ event }) => ({
						// Mark service as attempting recovery
					}))
				},

				HEALTH_CHECK: 'checkingHealth',
				STOP_MONITORING: 'idle'
			},

			after: {
				60000: { // Try recovery after 1 minute
					target: 'recovering',
					guard: ({ context }) => context.autoRecovery
				}
			}
		},

		recovering: {
			entry: () => console.log('🔧 Attempting service recovery...'),

			invoke: {
				id: 'recoverServices',
				src: healthOrchestratorServices.attemptServiceRecovery,
				input: ({ context }) => {
					// Find first degraded service to recover
					const degradedService = Object.entries(context.services)
						.find(([_, status]) => status === 'error')?.[0];
					
					return { service: degradedService || 'webgpu' };
				},
				onDone: {
					target: 'monitoring',
					actions: [
						() => console.log('✅ Service recovery successful'),
						// Re-check health after recovery
						({ self }) => self.send({ type: 'HEALTH_CHECK' })
					]
				},
				onError: {
					target: 'degraded',
					actions: assign(({ context, event }) => ({
						errors: [
							...context.errors,
							{
								message: `Service recovery failed: ${event.error}`,
								jobId: 'system',
								operation: 'service_recovery',
								stage: 'recovery',
								retryable: true
							} as VectorProcessingError
						]
					}))
				}
			}
		},

		error: {
			entry: () => console.error('❌ Health orchestrator encountered critical error'),
			
			on: {
				START_MONITORING: 'initializing',
				RESET_ERRORS: {
					target: 'idle',
					actions: assign(() => ({ errors: [] }))
				}
			}
		}
	}
});

// Health check implementations
async function checkRedisHealth(): Promise<boolean> {
	try {
		const response = await fetch('/api/v1/health/redis');
		return response.ok;
	} catch (error) {
		return false;
	}
}

async function checkPostgresHealth(): Promise<boolean> {
	try {
		const response = await fetch('/api/v1/health/postgres');
		return response.ok;
	} catch (error) {
		return false;
	}
}

async function checkRabbitMQHealth(): Promise<boolean> {
	try {
		const response = await fetch('/api/v1/health/rabbitmq');
		return response.ok;
	} catch (error) {
		return false;
	}
}

async function checkCUDAHealth(): Promise<boolean> {
	try {
		const response = await fetch('/api/v1/health/cuda');
		return response.ok;
	} catch (error) {
		return false;
	}
}

async function checkWebGPUHealth(): Promise<boolean> {
	return webgpuPolyfill.getDeviceInfo().isAvailable;
}

async function checkWASMLLMHealth(): Promise<boolean> {
	return wasmLLMService.getStats().isInitialized;
}

async function checkVectorServiceHealth(): Promise<boolean> {
	try {
		const response = await fetch('/api/v1/vector/health');
		return response.ok;
	} catch (error) {
		return false;
	}
}

// Metrics collection implementations
async function fetchQueueMetrics() {
	const response = await fetch('/api/v1/metrics/queues');
	return await response.json();
}

async function fetchPerformanceMetrics() {
	const response = await fetch('/api/v1/metrics/performance');
	return await response.json();
}

async function fetchResourceMetrics() {
	const webgpuStats = webgpuPolyfill.getPerformanceStats();
	const wasmStats = wasmLLMService.getStats();
	
	return {
		cudaUtilization: 0, // Would come from CUDA service
		webgpuUtilization: webgpuStats.webgpuPercentage || 0,
		memoryUsage: 0, // Would come from system metrics
		redisConnections: 0, // Would come from Redis service
		webgpuOpsPerSecond: webgpuStats.operationsCompleted / (webgpuStats.totalProcessingTime / 1000),
		wasmOperations: wasmStats.totalGenerations,
		wasmTokensPerSecond: wasmStats.averageTokensPerSecond
	};
}

async function fetchJobMetrics() {
	const response = await fetch('/api/v1/metrics/jobs');
	return await response.json();
}

// Recovery implementations
async function recoverWebGPU(): Promise<boolean> {
	try {
		webgpuPolyfill.dispose();
		return await webgpuPolyfill.initialize();
	} catch (error) {
		console.error('WebGPU recovery failed:', error);
		return false;
	}
}

async function recoverWASMLLM(): Promise<boolean> {
	try {
		wasmLLMService.dispose();
		return await wasmLLMService.initialize();
	} catch (error) {
		console.error('WASM LLM recovery failed:', error);
		return false;
	}
}

async function recoverRedis(): Promise<boolean> {
	// Would implement Redis reconnection logic
	return true;
}

async function recoverVectorService(): Promise<boolean> {
	// Would implement vector service restart logic
	return true;
}

// Utility functions
function resolveHealthStatus(result: PromiseSettledResult<any>): any {
	if (result.status === 'fulfilled') {
		return result.value ? 'connected' : 'disconnected';
	} else {
		return 'error';
	}
}

export type HealthOrchestratorMachine = typeof healthOrchestratorMachine;
export type HealthOrchestratorActor = ActorRefFrom<HealthOrchestratorMachine>;

// Singleton orchestrator instance
export class UnifiedHealthOrchestrator {
	private actor: HealthOrchestratorActor | null = null;
	private subscribers = new Set<(status: VectorHealthStatus) => void>();

	async initialize(): Promise<void> {
		if (this.actor) return;

		const { createActor } = await import('xstate');
		this.actor = createActor(healthOrchestratorMachine);
		
		this.actor.subscribe((state) => {
			const status: VectorHealthStatus = {
				overall: this.calculateOverallHealth(state.context.services),
				services: state.context.services,
				queues: this.formatQueueInfo(state.context.metrics),
				lastHealthCheck: state.context.lastHealthCheck || new Date()
			};

			// Notify subscribers
			this.subscribers.forEach(callback => callback(status));
		});

		this.actor.start();
	}

	start(): void {
		this.actor?.send({ type: 'START_MONITORING' });
	}

	stop(): void {
		this.actor?.send({ type: 'STOP_MONITORING' });
	}

	subscribe(callback: (status: VectorHealthStatus) => void): () => void {
		this.subscribers.add(callback);
		return () => this.subscribers.delete(callback);
	}

	getSnapshot() {
		return this.actor?.getSnapshot();
	}

	private calculateOverallHealth(services: any): 'healthy' | 'degraded' | 'unhealthy' {
		const serviceValues = Object.values(services);
		const healthyCount = serviceValues.filter(s => s === 'connected' || s === 'available' || s === 'loaded').length;
		const totalCount = serviceValues.length;

		if (healthyCount === totalCount) return 'healthy';
		if (healthyCount >= totalCount * 0.7) return 'degraded';
		return 'unhealthy';
	}

	private formatQueueInfo(metrics: VectorServiceMetrics | null): any {
		if (!metrics) return {};

		return {
			embeddings: {
				depth: metrics.queueDepth.embeddings,
				consumers: 1,
				processingRate: metrics.performance.vectorsPerSecond
			},
			similarities: {
				depth: metrics.queueDepth.similarities,
				consumers: 1,
				processingRate: metrics.performance.vectorsPerSecond
			}
		};
	}

	dispose(): void {
		this.actor?.stop();
		this.actor = null;
		this.subscribers.clear();
	}
}

// Global orchestrator instance
export const healthOrchestrator = new UnifiedHealthOrchestrator();