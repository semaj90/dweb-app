// @ts-nocheck
/**
 * AI Service Worker Manager
 * Handles multi-threaded AI processing with load balancing and task distribution
 */

import { writable, type Writable } from 'svelte/store';
// Orphaned content: import type { LLMProvider

// AI Task Types
export interface AITask {
	id: string;
	type: 'embedding' | 'generation' | 'analysis' | 'synthesis' | 'vector-search';
	priority: 'low' | 'medium' | 'high' | 'critical';
	provider: LLMProvider;
	payload: any;
	metadata?: {
		userId?: string;
		sessionId?: string;
		timestamp: number;
		estimatedDuration?: number;
	};
}

export interface AITaskResult {
	taskId: string;
	success: boolean;
	result?: any;
	error?: string;
	duration: number;
	metrics?: {
		tokensProcessed?: number;
		memoryUsed?: string;
		throughput?: number;
	};
}

export interface WorkerStatus {
	id: string;
	type: string;
	status: 'idle' | 'busy' | 'error' | 'offline';
	currentTask?: string;
	tasksCompleted: number;
	averageTaskTime: number;
	load: number; // 0-100%
}

// Service Worker Manager Class
export class AIServiceWorkerManager {
	private workers: Map<string, Worker> = new Map();
	private taskQueue: AITask[] = [];
	private activeTasksMap: Map<string, AITask> = new Map();
	private workerStatusMap: Map<string, WorkerStatus> = new Map();
	
	// Reactive stores for UI integration
	public taskQueue$: Writable<AITask[]> = writable([]);
	public workerStatus$: Writable<WorkerStatus[]> = writable([]);
	public systemMetrics$: Writable<AISystemMetrics> = writable({
		totalTasksProcessed: 0,
		averageResponseTime: 0,
		currentLoad: 0,
		availableWorkers: 0,
		queueLength: 0
	});

	private maxWorkers = navigator.hardwareConcurrency || 4;
	private isInitialized = false;

	constructor() {
		this.initializeWorkers();
	}

	private async initializeWorkers(): Promise<void> {
		if (this.isInitialized) return;

		try {
			// Initialize different types of workers
			const workerTypes = [
				{ type: 'embedding', count: Math.max(1, Math.floor(this.maxWorkers * 0.3)) },
				{ type: 'generation', count: Math.max(1, Math.floor(this.maxWorkers * 0.4)) },
				{ type: 'analysis', count: Math.max(1, Math.floor(this.maxWorkers * 0.2)) },
				{ type: 'general', count: Math.max(1, Math.floor(this.maxWorkers * 0.1)) }
			];

			for (const { type, count } of workerTypes) {
				for (let i = 0; i < count; i++) {
					await this.createWorker(`${type}-${i}`, type);
				}
			}

			this.isInitialized = true;
			this.updateSystemMetrics();
			
			console.log(`🧵 AI Service Worker Manager initialized with ${this.workers.size} workers`);
		} catch (error) {
			console.error('Failed to initialize AI Service Workers:', error);
		}
	}

	private async createWorker(workerId: string, type: string): Promise<void> {
		try {
			// Create worker from dedicated worker file
			const worker = new Worker(
				new URL('../workers/aiProcessingWorker.js', import.meta.url),
				{ type: 'module' }
			);

			// Set up worker communication
			worker.postMessage({ 
				type: 'INIT', 
				workerId,
				workerType: type,
				config: this.getWorkerConfig(type)
			});

			worker.onmessage = (event) => this.handleWorkerMessage(workerId, event);
			worker.onerror = (error) => this.handleWorkerError(workerId, error);

			this.workers.set(workerId, worker);
			this.workerStatusMap.set(workerId, {
				id: workerId,
				type,
				status: 'idle',
				tasksCompleted: 0,
				averageTaskTime: 0,
				load: 0
			});

		} catch (error) {
			console.error(`Failed to create worker ${workerId}:`, error);
		}
	}

	private getWorkerConfig(type: string) {
		return {
			maxConcurrentTasks: type === 'embedding' ? 3 : 1,
			preferredProviders: this.getPreferredProvidersForType(type),
			capabilities: this.getCapabilitiesForType(type)
		};
	}

	private getPreferredProvidersForType(type: string): string[] {
		switch (type) {
			case 'embedding':
				return ['ollama-local']; // nomic-embed-text
			case 'generation':
				return ['ollama-local', 'vllm-server']; // gemma3-legal
			case 'analysis':
				return ['autogen-framework', 'crewai-team'];
			default:
				return ['ollama-local'];
		}
	}

	private getCapabilitiesForType(type: string): string[] {
		switch (type) {
			case 'embedding':
				return ['text-embedding', 'similarity-search', 'clustering'];
			case 'generation':
				return ['text-generation', 'chat', 'completion'];
			case 'analysis':
				return ['document-analysis', 'sentiment', 'entity-extraction'];
			default:
				return ['general-purpose'];
		}
	}

	// Public API Methods
	public async queueTask(task: Omit<AITask, 'id'>): Promise<string> {
		const taskWithId: AITask = {
			...task,
			id: `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
			metadata: {
				timestamp: Date.now(),
				...task.metadata
			}
		};

		// Add to queue based on priority
		this.insertTaskByPriority(taskWithId);
		this.taskQueue$.update((queue: any) => [...queue]);
		
		// Try to process immediately if workers are available
		this.processQueue();
		
		return taskWithId.id;
	}

	public async processParallel(tasks: Omit<AITask, 'id'>[]): Promise<AITaskResult[]> {
		const taskIds = await Promise.all(
			tasks.map((task: any) => this.queueTask(task))
		);

		// Wait for all tasks to complete
		return new Promise((resolve) => {
			const results: AITaskResult[] = [];
			const checkCompletion = () => {
				if (results.length === taskIds.length) {
					resolve(results);
				}
			};

			// Listen for task completions
			taskIds.forEach((taskId: any) => {
				this.onTaskComplete(taskId, (result) => {
					results.push(result);
					checkCompletion();
				});
			});
		});
	}

	public getSystemHealth(): AISystemHealth {
		const workers = Array.from(this.workerStatusMap.values());
		const activeWorkers = workers.filter((w: any) => w.status !== 'offline').length;
		const busyWorkers = workers.filter((w: any) => w.status === 'busy').length;
		const errorWorkers = workers.filter((w: any) => w.status === 'error').length;

		return {
			totalWorkers: workers.length,
			activeWorkers,
			busyWorkers,
			errorWorkers,
			queueLength: this.taskQueue.length,
			averageLoad: workers.reduce((sum, w) => sum + w.load, 0) / workers.length,
			status: errorWorkers > workers.length * 0.5 ? 'critical' : 
					busyWorkers > workers.length * 0.8 ? 'busy' : 'healthy'
		};
	}

	// Private Methods
	private insertTaskByPriority(task: AITask): void {
		const priorityOrder = { 'critical': 0, 'high': 1, 'medium': 2, 'low': 3 };
		const insertIndex = this.taskQueue.findIndex(
			(queuedTask: any) => priorityOrder[task.priority] < priorityOrder[queuedTask.priority]
		);
		
		if (insertIndex === -1) {
			this.taskQueue.push(task);
		} else {
			this.taskQueue.splice(insertIndex, 0, task);
		}
	}

	private processQueue(): void {
		if (this.taskQueue.length === 0) return;

		// Find available workers
		const availableWorkers = Array.from(this.workerStatusMap.entries())
			.filter(([_, status]) => status.status === 'idle')
			.map(([workerId, _]) => workerId);

		if (availableWorkers.length === 0) return;

		// Assign tasks to available workers
		for (const workerId of availableWorkers) {
			if (this.taskQueue.length === 0) break;

			const workerStatus = this.workerStatusMap.get(workerId)!;
			const suitableTaskIndex = this.taskQueue.findIndex((task: any) => this.isWorkerSuitableForTask(workerStatus, task)
			);

			if (suitableTaskIndex !== -1) {
				const task = this.taskQueue.splice(suitableTaskIndex, 1)[0];
				this.assignTaskToWorker(workerId, task);
			}
		}

		this.taskQueue$.set([...this.taskQueue]);
		this.updateSystemMetrics();
	}

	private isWorkerSuitableForTask(workerStatus: WorkerStatus, task: AITask): boolean {
		const preferredProviders = this.getPreferredProvidersForType(workerStatus.type);
		return preferredProviders.includes(task.provider.id) || workerStatus.type === 'general';
	}

	private assignTaskToWorker(workerId: string, task: AITask): void {
		const worker = this.workers.get(workerId);
		if (!worker) return;

		// Update worker status
		const workerStatus = this.workerStatusMap.get(workerId)!;
		workerStatus.status = 'busy';
		workerStatus.currentTask = task.id;

		// Track active task
		this.activeTasksMap.set(task.id, task);

		// Send task to worker
		worker.postMessage({
			type: 'PROCESS_TASK',
			task
		});

		console.log(`🔄 Assigned task ${task.id} to worker ${workerId}`);
	}

	private handleWorkerMessage(workerId: string, event: MessageEvent): void {
		const { type, data } = event.data;

		switch (type) {
			case 'TASK_COMPLETE':
				this.handleTaskComplete(workerId, data);
				break;
			case 'TASK_ERROR':
				this.handleTaskError(workerId, data);
				break;
			case 'WORKER_STATUS':
				this.updateWorkerStatus(workerId, data);
				break;
			case 'WORKER_READY':
				console.log(`✅ Worker ${workerId} is ready`);
				break;
		}
	}

	private handleTaskComplete(workerId: string, result: AITaskResult): void {
		const workerStatus = this.workerStatusMap.get(workerId)!;
		workerStatus.status = 'idle';
		workerStatus.currentTask = undefined;
		workerStatus.tasksCompleted++;
		
		// Update average task time
		const oldAvg = workerStatus.averageTaskTime;
		const count = workerStatus.tasksCompleted;
		workerStatus.averageTaskTime = (oldAvg * (count - 1) + result.duration) / count;

		// Clean up
		this.activeTasksMap.delete(result.taskId);

		// Try to process more tasks
		this.processQueue();

		console.log(`✅ Task ${result.taskId} completed by worker ${workerId} in ${result.duration}ms`);
	}

	private handleTaskError(workerId: string, error: any): void {
		const workerStatus = this.workerStatusMap.get(workerId)!;
		workerStatus.status = 'error';
		
		console.error(`❌ Worker ${workerId} error:`, error);
		
		// Retry task on different worker or fail gracefully
		// Implementation depends on your retry strategy
	}

	private handleWorkerError(workerId: string, error: ErrorEvent): void {
		console.error(`❌ Worker ${workerId} encountered an error:`, error);
		this.workerStatusMap.get(workerId)!.status = 'error';
	}

	private updateWorkerStatus(workerId: string, statusUpdate: Partial<WorkerStatus>): void {
		const currentStatus = this.workerStatusMap.get(workerId)!;
		Object.assign(currentStatus, statusUpdate);
		this.workerStatus$.update((statuses: any) => statuses.map((s: any) => s.id === workerId ? currentStatus : s)
		);
	}

	private updateSystemMetrics(): void {
		const health = this.getSystemHealth();
		this.systemMetrics$.update((metrics: any) => ({
			...metrics,
			queueLength: health.queueLength,
			currentLoad: health.averageLoad,
			availableWorkers: health.activeWorkers - health.busyWorkers
		}));
	}

	private onTaskComplete(taskId: string, callback: (result: AITaskResult) => void): void {
		// Implementation for task completion callbacks
		// This would integrate with your event system
	}

	// Cleanup
	public destroy(): void {
		for (const worker of this.workers.values()) {
			worker.terminate();
		}
		this.workers.clear();
		this.workerStatusMap.clear();
		this.taskQueue = [];
		this.activeTasksMap.clear();
	}
}

// Singleton instance
export const aiServiceWorkerManager = new AIServiceWorkerManager();

// Types
export interface AISystemMetrics {
	totalTasksProcessed: number;
	averageResponseTime: number;
	currentLoad: number;
	availableWorkers: number;
	queueLength: number;
}

export interface AISystemHealth {
	totalWorkers: number;
	activeWorkers: number;
	busyWorkers: number;
	errorWorkers: number;
	queueLength: number;
	averageLoad: number;
	status: 'healthy' | 'busy' | 'critical';
}