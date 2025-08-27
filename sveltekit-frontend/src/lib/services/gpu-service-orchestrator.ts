/**
 * GPU Service Orchestrator - Production Integration Hub
 * Coordinates all GPU/WASM/AI services with document ingestion system
 * Integrates with existing evidence processor, ingestion store, and summarizer service
 */

import { nvidiaLlamaService, type LlamaRequest } from './nvidiaLlamaService';
import { gpuServiceIntegration } from './gpu-service-integration';
import { llvmWasmBridge } from '$lib/wasm/llvm-wasm-bridge';
import { unifiedWASMGPUOrchestrator } from './unified-wasm-gpu-orchestrator';
import { writable, derived, type Writable } from 'svelte/store';
import { browser } from '$app/environment';

// Integration with existing services
interface DocumentIngestionTask {
	documentId: string;
	documentType: 'pdf' | 'docx' | 'txt' | 'image' | 'legal_brief' | 'contract' | 'evidence';
	content: string;
	metadata: {
		caseId?: string;
		jurisdiction?: string;
		dateCreated?: string;
		parties?: string[];
		documentClass?: string;
		priority?: 'low' | 'medium' | 'high' | 'urgent';
	};
	processingRequirements: {
		needsOCR?: boolean;
		needsSummary?: boolean;
		needsEmbedding?: boolean;
		needsCitationExtraction?: boolean;
		needsEntityExtraction?: boolean;
		needsRiskAssessment?: boolean;
	};
}

interface GPUOrchestrationResult {
	documentId: string;
	success: boolean;
	processedContent?: {
		summary?: string;
		extractedText?: string;
		citations?: string[];
		entities?: string[];
		riskLevel?: 'low' | 'medium' | 'high' | 'critical';
		embedding?: number[];
	};
	performance: {
		totalProcessingTime: number;
		serviceBreakdown: Record<string, number>;
		memoryPeakUsage: number;
		gpuUtilization: number;
	};
	servicesUsed: string[];
	errors?: string[];
	warnings?: string[];
}

interface ServiceHealth {
	serviceName: string;
	status: 'healthy' | 'degraded' | 'unhealthy' | 'offline';
	responseTime: number;
	errorRate: number;
	capabilities: string[];
	lastChecked: number;
}

/**
 * Main GPU Service Orchestrator
 * Provides unified interface for all GPU/WASM/AI operations
 */
export class GPUServiceOrchestrator {
	private services: Map<string, any> = new Map();
	private serviceHealth: Map<string, ServiceHealth> = new Map();
	private taskQueue: DocumentIngestionTask[] = [];
	private activeProcessing: Map<string, DocumentIngestionTask> = new Map();
	private processingResults: Map<string, GPUOrchestrationResult> = new Map();
	private isInitialized = false;
	private healthCheckInterval: NodeJS.Timeout | null = null;
	private metrics = {
		totalTasksProcessed: 0,
		successfulTasks: 0,
		failedTasks: 0,
		averageProcessingTime: 0,
		totalProcessingTime: 0
	};

	// Store for reactive updates
	private healthStore: Writable<Record<string, ServiceHealth>>;
	private metricsStore: Writable<typeof this.metrics>;
	private queueStore: Writable<{ queued: number; active: number; completed: number }>;

	constructor() {
		this.healthStore = writable({});
		this.metricsStore = writable(this.metrics);
		this.queueStore = writable({ queued: 0, active: 0, completed: 0 });
	}

	/**
	 * Initialize all GPU services and start health monitoring
	 */
	async initialize(): Promise<void> {
		if (this.isInitialized) return;

		console.log('🚀 Initializing GPU Service Orchestrator...');

		try {
			// Initialize NVIDIA LLaMA service
			await nvidiaLlamaService.initialize();
			this.services.set('nvidia_llama', nvidiaLlamaService);
			console.log('✅ NVIDIA LLaMA service initialized');

			// Initialize GPU Service Integration
			await gpuServiceIntegration.initialize();
			this.services.set('gpu_integration', gpuServiceIntegration);
			console.log('✅ GPU Service Integration initialized');

			// Initialize LLVM-WASM Bridge
			await llvmWasmBridge.initialize();
			this.services.set('llvm_wasm', llvmWasmBridge);
			console.log('✅ LLVM-WASM Bridge initialized');

			// Initialize Unified WASM-GPU Orchestrator
			await unifiedWASMGPUOrchestrator.initialize();
			this.services.set('unified_orchestrator', unifiedWASMGPUOrchestrator);
			console.log('✅ Unified WASM-GPU Orchestrator initialized');

			// Check external services
			await this.checkExternalServices();

			// Start health monitoring
			this.startHealthMonitoring();

			this.isInitialized = true;
			console.log('🎯 GPU Service Orchestrator fully initialized');

		} catch (error) {
			console.error('❌ Failed to initialize GPU Service Orchestrator:', error);
			throw error;
		}
	}

	/**
	 * Process document with intelligent service routing
	 */
	async processDocument(task: DocumentIngestionTask): Promise<GPUOrchestrationResult> {
		if (!this.isInitialized) {
			await this.initialize();
		}

		console.log(`📄 Processing document ${task.documentId} (${task.documentType})`);
		const startTime = Date.now();

		// Add to processing queue
		this.taskQueue.push(task);
		this.activeProcessing.set(task.documentId, task);
		this.updateQueueStore();

		const result: GPUOrchestrationResult = {
			documentId: task.documentId,
			success: false,
			processedContent: {},
			performance: {
				totalProcessingTime: 0,
				serviceBreakdown: {},
				memoryPeakUsage: 0,
				gpuUtilization: 0
			},
			servicesUsed: [],
			errors: [],
			warnings: []
		};

		try {
			// Route processing based on document type and requirements
			const processingPlan = this.createProcessingPlan(task);
			console.log(`📋 Processing plan:`, processingPlan);

			// Execute processing pipeline
			for (const step of processingPlan) {
				const stepStartTime = Date.now();
				
				try {
					const stepResult = await this.executeProcessingStep(step, task);
					
					// Merge results
					if (stepResult.summary) result.processedContent!.summary = stepResult.summary;
					if (stepResult.extractedText) result.processedContent!.extractedText = stepResult.extractedText;
					if (stepResult.citations) result.processedContent!.citations = stepResult.citations;
					if (stepResult.entities) result.processedContent!.entities = stepResult.entities;
					if (stepResult.riskLevel) result.processedContent!.riskLevel = stepResult.riskLevel;
					if (stepResult.embedding) result.processedContent!.embedding = stepResult.embedding;

					// Track performance
					const stepTime = Date.now() - stepStartTime;
					result.performance.serviceBreakdown[step.service] = stepTime;
					result.servicesUsed.push(step.service);
					
					console.log(`✅ Step ${step.operation} completed in ${stepTime}ms`);

				} catch (stepError) {
					const errorMsg = `Step ${step.operation} failed: ${stepError instanceof Error ? stepError.message : 'Unknown error'}`;
					result.errors!.push(errorMsg);
					console.warn(`⚠️ ${errorMsg}`);

					// Try fallback if available
					if (step.fallback) {
						try {
							console.log(`🔄 Attempting fallback: ${step.fallback.service}`);
							const fallbackResult = await this.executeProcessingStep(step.fallback, task);
							
							// Use fallback results
							Object.assign(result.processedContent!, fallbackResult);
							result.servicesUsed.push(step.fallback.service);
							result.warnings!.push(`Used fallback service for ${step.operation}`);
							
						} catch (fallbackError) {
							result.errors!.push(`Fallback also failed: ${fallbackError instanceof Error ? fallbackError.message : 'Unknown error'}`);
						}
					}
				}
			}

			// Calculate final performance metrics
			result.performance.totalProcessingTime = Date.now() - startTime;
			result.performance.gpuUtilization = this.calculateGPUUtilization(result.servicesUsed);
			result.success = result.errors!.length === 0 || result.processedContent?.summary !== undefined;

			// Update metrics
			this.updateMetrics(result);

		} catch (error) {
			result.errors!.push(error instanceof Error ? error.message : 'Unknown processing error');
			result.performance.totalProcessingTime = Date.now() - startTime;
		} finally {
			// Clean up
			this.activeProcessing.delete(task.documentId);
			this.processingResults.set(task.documentId, result);
			this.updateQueueStore();
		}

		console.log(`📊 Document ${task.documentId} processed in ${result.performance.totalProcessingTime}ms (success: ${result.success})`);
		return result;
	}

	/**
	 * Create intelligent processing plan based on document and requirements
	 */
	private createProcessingPlan(task: DocumentIngestionTask): ProcessingStep[] {
		const plan: ProcessingStep[] = [];

		// Text extraction step (if needed)
		if (task.documentType === 'pdf' || task.documentType === 'image' || task.processingRequirements.needsOCR) {
			plan.push({
				operation: 'text_extraction',
				service: 'external_ocr_service', // Evidence processor
				priority: 'high',
				fallback: {
					operation: 'text_extraction_fallback',
					service: 'llvm_wasm',
					priority: 'medium'
				}
			});
		}

		// Summary generation
		if (task.processingRequirements.needsSummary) {
			plan.push({
				operation: 'summarization',
				service: 'nvidia_llama',
				priority: 'high',
				fallback: {
					operation: 'summarization_fallback',
					service: 'external_summarizer_service', // Summarizer service
					priority: 'medium'
				}
			});
		}

		// Citation extraction
		if (task.processingRequirements.needsCitationExtraction) {
			plan.push({
				operation: 'citation_extraction',
				service: 'llvm_wasm',
				priority: 'medium',
				fallback: {
					operation: 'citation_extraction_fallback',
					service: 'nvidia_llama',
					priority: 'low'
				}
			});
		}

		// Entity extraction
		if (task.processingRequirements.needsEntityExtraction) {
			plan.push({
				operation: 'entity_extraction',
				service: 'nvidia_llama',
				priority: 'medium'
			});
		}

		// Vector embedding
		if (task.processingRequirements.needsEmbedding) {
			plan.push({
				operation: 'embedding_generation',
				service: 'gpu_integration',
				priority: 'medium',
				fallback: {
					operation: 'embedding_fallback',
					service: 'llvm_wasm',
					priority: 'low'
				}
			});
		}

		// Risk assessment
		if (task.processingRequirements.needsRiskAssessment) {
			plan.push({
				operation: 'risk_assessment',
				service: 'nvidia_llama',
				priority: 'low'
			});
		}

		return plan;
	}

	/**
	 * Execute individual processing step
	 */
	private async executeProcessingStep(step: ProcessingStep, task: DocumentIngestionTask): Promise<any> {
		const service = this.services.get(step.service);
		if (!service) {
			throw new Error(`Service ${step.service} not available`);
		}

		switch (step.operation) {
			case 'summarization':
				if (step.service === 'nvidia_llama') {
					const response = await nvidiaLlamaService.generateText({
						prompt: `Summarize this legal document:\n\n${task.content}`,
						max_tokens: 500,
						temperature: 0.3,
						priority: step.priority === 'high' ? 'urgent' : 'normal'
					});
					return { summary: response.text };
				} else if (step.service === 'external_summarizer_service') {
					// Call external Go summarizer service
					try {
						const response = await fetch('http://localhost:8095/summarize', {
							method: 'POST',
							headers: { 'Content-Type': 'application/json' },
							body: JSON.stringify({
								content: task.content,
								document_type: task.documentType,
								max_length: 500
							})
						});
						if (response.ok) {
							const result = await response.json();
							return { summary: result.summary };
						}
						throw new Error(`Summarizer service error: ${response.statusText}`);
					} catch (error) {
						throw new Error(`External summarizer failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
					}
				}
				break;

			case 'citation_extraction':
				if (step.service === 'llvm_wasm') {
					const result = await llvmWasmBridge.processLegalText(task.content, {
						extractCitations: true
					});
					return { citations: result.citations || [] };
				}
				break;

			case 'entity_extraction':
				if (step.service === 'nvidia_llama') {
					const response = await nvidiaLlamaService.generateText({
						prompt: `Extract legal entities (people, organizations, dates, amounts) from this document:\n\n${task.content.substring(0, 2000)}`,
						max_tokens: 300,
						temperature: 0.1
					});
					// Parse entities from response (simplified)
					const entities = response.text.split(/[,\n]/).map(e => e.trim()).filter(e => e.length > 0);
					return { entities };
				}
				break;

			case 'embedding_generation':
				if (step.service === 'gpu_integration') {
					const embeddings = await gpuServiceIntegration.generateEmbeddings([task.content.substring(0, 1000)]);
					return { embedding: Array.from(embeddings[0] || []) };
				} else if (step.service === 'llvm_wasm') {
					const result = await llvmWasmBridge.computeEmbedding(
						task.content.split(' ').map((_, i) => i % 100), // Simple tokenization
						384
					);
					return { embedding: result.embedding };
				}
				break;

			case 'risk_assessment':
				if (step.service === 'nvidia_llama') {
					const response = await nvidiaLlamaService.generateText({
						prompt: `Assess the legal risk level (low/medium/high/critical) of this document and explain why:\n\n${task.content.substring(0, 1500)}`,
						max_tokens: 150,
						temperature: 0.2
					});
					
					const riskText = response.text.toLowerCase();
					let riskLevel: 'low' | 'medium' | 'high' | 'critical' = 'medium';
					
					if (riskText.includes('critical')) riskLevel = 'critical';
					else if (riskText.includes('high')) riskLevel = 'high';
					else if (riskText.includes('low')) riskLevel = 'low';
					
					return { riskLevel };
				}
				break;

			case 'text_extraction':
				if (step.service === 'external_ocr_service') {
					// Call evidence processor for OCR
					try {
						const response = await fetch('http://localhost:8092/process-evidence', {
							method: 'POST',
							headers: { 'Content-Type': 'application/json' },
							body: JSON.stringify({
								type: 'ocr',
								content: task.content,
								metadata: task.metadata
							})
						});
						if (response.ok) {
							const result = await response.json();
							return { extractedText: result.text };
						}
						throw new Error(`OCR service error: ${response.statusText}`);
					} catch (error) {
						throw new Error(`External OCR failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
					}
				}
				break;

			default:
				throw new Error(`Unknown operation: ${step.operation}`);
		}

		throw new Error(`Operation ${step.operation} not implemented for service ${step.service}`);
	}

	/**
	 * Check health of external services
	 */
	private async checkExternalServices(): Promise<void> {
		const services = [
			{ name: 'wasm_llvm_service', url: 'http://localhost:8225/health' },
			{ name: 'summarizer_service', url: 'http://localhost:8095/health' },
			{ name: 'evidence_processor', url: 'http://localhost:8092/health' },
			{ name: 'ingestion_service', url: 'http://localhost:8091/health' }
		];

		for (const service of services) {
			try {
				const startTime = Date.now();
				const response = await fetch(service.url, { 
					method: 'GET',
					signal: AbortSignal.timeout(5000) // 5 second timeout
				});
				const responseTime = Date.now() - startTime;

				if (response.ok) {
					const healthData = await response.json();
					this.serviceHealth.set(service.name, {
						serviceName: service.name,
						status: 'healthy',
						responseTime,
						errorRate: 0,
						capabilities: healthData.capabilities || [],
						lastChecked: Date.now()
					});
					console.log(`✅ ${service.name} is healthy (${responseTime}ms)`);
				} else {
					throw new Error(`HTTP ${response.status}: ${response.statusText}`);
				}
			} catch (error) {
				this.serviceHealth.set(service.name, {
					serviceName: service.name,
					status: 'offline',
					responseTime: -1,
					errorRate: 1,
					capabilities: [],
					lastChecked: Date.now()
				});
				console.warn(`⚠️ ${service.name} is offline:`, error instanceof Error ? error.message : 'Unknown error');
			}
		}

		this.updateHealthStore();
	}

	/**
	 * Start periodic health monitoring
	 */
	private startHealthMonitoring(): void {
		if (this.healthCheckInterval) {
			clearInterval(this.healthCheckInterval);
		}

		this.healthCheckInterval = setInterval(async () => {
			await this.checkExternalServices();
		}, 30000); // Check every 30 seconds

		console.log('🔄 Health monitoring started');
	}

	/**
	 * Calculate GPU utilization based on services used
	 */
	private calculateGPUUtilization(servicesUsed: string[]): number {
		let utilization = 0;
		
		if (servicesUsed.includes('nvidia_llama')) utilization += 0.7;
		if (servicesUsed.includes('gpu_integration')) utilization += 0.5;
		if (servicesUsed.includes('unified_orchestrator')) utilization += 0.3;
		
		return Math.min(utilization, 1.0);
	}

	/**
	 * Update metrics after processing
	 */
	private updateMetrics(result: GPUOrchestrationResult): void {
		this.metrics.totalTasksProcessed++;
		this.metrics.totalProcessingTime += result.performance.totalProcessingTime;
		
		if (result.success) {
			this.metrics.successfulTasks++;
		} else {
			this.metrics.failedTasks++;
		}

		this.metrics.averageProcessingTime = this.metrics.totalProcessingTime / this.metrics.totalTasksProcessed;
		this.metricsStore.set({ ...this.metrics });
	}

	private updateQueueStore(): void {
		this.queueStore.set({
			queued: this.taskQueue.length,
			active: this.activeProcessing.size,
			completed: this.processingResults.size
		});
	}

	private updateHealthStore(): void {
		const healthObj: Record<string, ServiceHealth> = {};
		for (const [name, health] of this.serviceHealth.entries()) {
			healthObj[name] = health;
		}
		this.healthStore.set(healthObj);
	}

	// Public API methods
	public getHealth() { return this.healthStore; }
	public getMetrics() { return this.metricsStore; }
	public getQueue() { return this.queueStore; }
	public getResult(documentId: string): GPUOrchestrationResult | undefined {
		return this.processingResults.get(documentId);
	}

	public async dispose(): Promise<void> {
		if (this.healthCheckInterval) {
			clearInterval(this.healthCheckInterval);
			this.healthCheckInterval = null;
		}

		// Dispose of all services
		for (const service of this.services.values()) {
			if (service.dispose) {
				await service.dispose();
			}
		}

		this.services.clear();
		this.serviceHealth.clear();
		this.activeProcessing.clear();
		
		console.log('🧹 GPU Service Orchestrator disposed');
	}
}

interface ProcessingStep {
	operation: string;
	service: string;
	priority: 'low' | 'medium' | 'high';
	fallback?: ProcessingStep;
}

// Export singleton instance
export const gpuServiceOrchestrator = new GPUServiceOrchestrator();

// Reactive stores for UI integration
export const gpuHealthStore = gpuServiceOrchestrator.getHealth();
export const gpuMetricsStore = gpuServiceOrchestrator.getMetrics();
export const gpuQueueStore = gpuServiceOrchestrator.getQueue();