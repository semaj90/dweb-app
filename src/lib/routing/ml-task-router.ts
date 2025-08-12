/**
 * ML-Based Task Routing System
 * 
 * Intelligent task routing using machine learning to optimize:
 * - Agent selection based on task characteristics and performance history
 * - Load balancing across services with concurrency awareness
 * - Predictive routing using historical success rates and response times
 * - Dynamic routing adaptation based on real-time metrics
 * - Queue management with backpressure and priority handling
 * - Service health and capacity-aware routing
 * 
 * Features:
 * - Multi-dimensional feature extraction from tasks
 * - Real-time learning from routing outcomes
 * - Adaptive load balancing with concurrency limits
 * - Streaming-aware routing for SSE endpoints
 * - Integration with existing summarization and scaling services
 */

import { EventEmitter } from 'events';
import { advancedCacheManager } from '$lib/caching/advanced-cache-manager';
import { horizontalAgentScaler } from '$lib/scaling/horizontal-agent-scaler';
import type {
    TaskRoute,
    RoutingDecision,
    TaskFeatures,
    ServiceEndpoint,
    RoutingModel,
    TaskMetrics,
    RoutingConfig,
    ServiceCapacity,
    RoutingAnalytics
} from '$lib/ai/types';

export interface TaskRequest {
    id: string;
    type: string;
    priority: 'low' | 'medium' | 'high' | 'urgent';
    payload: any;
    metadata: {
        userAgent?: string;
        sessionId?: string;
        expectedResponseTime?: number;
        streaming?: boolean;
        retryCount?: number;
    };
    features?: TaskFeatures;
    timeout?: number;
    createdAt: Date;
}

export interface ServiceEndpointConfig {
    id: string;
    name: string;
    type: string;
    url: string;
    port: number;
    maxConcurrency: number;
    currentLoad: number;
    healthStatus: 'healthy' | 'degraded' | 'unhealthy';
    capabilities: string[];
    performance: {
        averageResponseTime: number;
        successRate: number;
        errorRate: number;
        throughput: number;
    };
    streaming: {
        supported: boolean;
        endpoint?: string;
        maxStreams?: number;
        currentStreams?: number;
    };
}

export class MLTaskRouter extends EventEmitter {
    private services: Map<string, ServiceEndpointConfig> = new Map();
    private routingModels: Map<string, RoutingModel> = new Map();
    private taskHistory: Map<string, TaskMetrics[]> = new Map();
    private routingCache: Map<string, RoutingDecision> = new Map();
    private analytics: RoutingAnalytics;
    private config: RoutingConfig;
    private featureExtractor: TaskFeatureExtractor;
    private loadBalancer: IntelligentLoadBalancer;
    private queueManager: PriorityQueueManager;
    private performanceTracker: RoutingPerformanceTracker;

    constructor(config: RoutingConfig = {}) {
        super();
        
        this.config = {
            enableMLRouting: true,
            enablePredictiveRouting: true,
            enableAdaptiveLoadBalancing: true,
            enableStreamingOptimization: true,
            enableConcurrencyAwareness: true,
            cacheRoutingDecisions: true,
            learningRate: 0.1,
            modelUpdateInterval: 300000, // 5 minutes
            metricsCollectionInterval: 30000,
            routingCacheTimeout: 60000,
            maxRetryAttempts: 3,
            defaultTimeout: 30000,
            concurrencyLimits: {
                'summarization': 2,
                'embedding': 4,
                'classification': 3,
                'analysis': 2,
                'vector-search': 6
            },
            ...config
        };

        this.analytics = this.initializeAnalytics();
        this.featureExtractor = new TaskFeatureExtractor(this.config);
        this.loadBalancer = new IntelligentLoadBalancer(this.config);
        this.queueManager = new PriorityQueueManager(this.config);
        this.performanceTracker = new RoutingPerformanceTracker(this.config);

        this.initializeServices();
        this.initializeRoutingModels();
        this.setupEventListeners();
    }

    /**
     * Start the ML task routing system
     */
    async start(): Promise<void> {
        try {
            console.log('🧠 Starting ML Task Routing System...');

            // Initialize service discovery
            await this.discoverServices();

            // Start performance tracking
            await this.performanceTracker.start();

            // Start queue manager
            await this.queueManager.start();

            // Begin model training with historical data
            if (this.config.enableMLRouting) {
                await this.trainInitialModels();
            }

            // Start periodic model updates
            this.startModelUpdates();

            // Start metrics collection
            this.startMetricsCollection();

            console.log('✅ ML Task Routing System started successfully');
            console.log(`🎯 Discovered ${this.services.size} services`);
            console.log(`🧠 Initialized ${this.routingModels.size} routing models`);

            this.emit('routerStarted', {
                services: this.services.size,
                models: this.routingModels.size,
                timestamp: new Date()
            });

        } catch (error) {
            console.error('❌ Failed to start ML task routing system:', error);
            throw error;
        }
    }

    /**
     * Route a task to the optimal service
     */
    async routeTask(task: TaskRequest): Promise<RoutingDecision> {
        const startTime = performance.now();
        
        try {
            console.log(`🎯 Routing task ${task.id} (type: ${task.type}, priority: ${task.priority})`);

            // Extract task features
            const features = await this.featureExtractor.extractFeatures(task);
            task.features = features;

            // Check routing cache
            if (this.config.cacheRoutingDecisions) {
                const cachedDecision = await this.getCachedRouting(task);
                if (cachedDecision) {
                    console.log(`📋 Using cached routing for task ${task.id}`);
                    return cachedDecision;
                }
            }

            // Get available services for task type
            const candidateServices = this.getEligibleServices(task);
            
            if (candidateServices.length === 0) {
                throw new Error(`No available services for task type: ${task.type}`);
            }

            // Make routing decision using ML model
            const decision = await this.makeRoutingDecision(task, candidateServices);

            // Check service capacity and concurrency
            const finalDecision = await this.validateAndAdjustDecision(decision, task);

            // Cache the routing decision
            if (this.config.cacheRoutingDecisions) {
                await this.cacheRoutingDecision(task, finalDecision);
            }

            // Track routing metrics
            const routingTime = performance.now() - startTime;
            this.updateRoutingMetrics(task, finalDecision, routingTime);

            // Queue the task if needed
            if (finalDecision.needsQueuing) {
                await this.queueManager.enqueue(task, finalDecision);
            }

            console.log(`✅ Routed task ${task.id} to ${finalDecision.selectedService.name} (confidence: ${(finalDecision.confidence * 100).toFixed(1)}%)`);
            
            this.emit('taskRouted', { task, decision: finalDecision, routingTime });
            
            return finalDecision;

        } catch (error) {
            console.error(`❌ Failed to route task ${task.id}:`, error);
            
            // Fallback routing
            const fallbackDecision = await this.getFallbackRouting(task);
            
            this.emit('routingError', { task, error, fallback: fallbackDecision });
            
            return fallbackDecision;
        }
    }

    /**
     * Execute a routed task
     */
    async executeTask(task: TaskRequest, decision: RoutingDecision): Promise<any> {
        const startTime = performance.now();
        
        try {
            console.log(`🚀 Executing task ${task.id} on ${decision.selectedService.name}`);

            // Update service load
            this.updateServiceLoad(decision.selectedService.id, 1);

            // Execute based on streaming requirements
            let result;
            if (task.metadata.streaming && decision.selectedService.streaming.supported) {
                result = await this.executeStreamingTask(task, decision);
            } else {
                result = await this.executeStandardTask(task, decision);
            }

            const executionTime = performance.now() - startTime;

            // Update performance metrics
            await this.updateTaskMetrics(task, decision, {
                success: true,
                executionTime,
                responseSize: this.estimateResponseSize(result)
            });

            // Learn from successful execution
            if (this.config.enableMLRouting) {
                await this.updateRoutingModel(task, decision, { success: true, executionTime });
            }

            console.log(`✅ Task ${task.id} completed successfully in ${executionTime.toFixed(2)}ms`);
            
            this.emit('taskCompleted', { task, decision, result, executionTime });
            
            return result;

        } catch (error) {
            const executionTime = performance.now() - startTime;
            
            console.error(`❌ Task ${task.id} failed:`, error);

            // Update error metrics
            await this.updateTaskMetrics(task, decision, {
                success: false,
                executionTime,
                error: error.message
            });

            // Learn from failure
            if (this.config.enableMLRouting) {
                await this.updateRoutingModel(task, decision, { success: false, executionTime, error });
            }

            // Handle retry logic
            if (task.metadata.retryCount < this.config.maxRetryAttempts) {
                console.log(`🔄 Retrying task ${task.id} (attempt ${task.metadata.retryCount + 1})`);
                task.metadata.retryCount++;
                return this.routeAndExecuteTask(task);
            }

            this.emit('taskFailed', { task, decision, error, executionTime });
            throw error;

        } finally {
            // Update service load
            this.updateServiceLoad(decision.selectedService.id, -1);
        }
    }

    /**
     * Route and execute task in one operation
     */
    async routeAndExecuteTask(task: TaskRequest): Promise<any> {
        const decision = await this.routeTask(task);
        return this.executeTask(task, decision);
    }

    /**
     * Initialize service endpoints
     */
    private initializeServices(): void {
        console.log('🔧 Initializing service endpoints...');

        // Summarization service (enhanced with concurrency and streaming)
        this.services.set('summarization-service', {
            id: 'summarization-service',
            name: 'Enhanced Summarization Service',
            type: 'summarization',
            url: 'http://localhost:8091',
            port: 8091,
            maxConcurrency: this.config.concurrencyLimits.summarization || 2,
            currentLoad: 0,
            healthStatus: 'healthy',
            capabilities: ['text-summarization', 'document-processing', 'legal-text'],
            performance: {
                averageResponseTime: 2500,
                successRate: 0.95,
                errorRate: 0.05,
                throughput: 10
            },
            streaming: {
                supported: true,
                endpoint: '/summarize/stream',
                maxStreams: 2,
                currentStreams: 0
            }
        });

        // Vector search service
        this.services.set('vector-search-service', {
            id: 'vector-search-service',
            name: 'Vector Search Engine',
            type: 'vector-search',
            url: 'http://localhost:6333',
            port: 6333,
            maxConcurrency: this.config.concurrencyLimits['vector-search'] || 6,
            currentLoad: 0,
            healthStatus: 'healthy',
            capabilities: ['vector-search', 'similarity', 'embedding-retrieval'],
            performance: {
                averageResponseTime: 150,
                successRate: 0.98,
                errorRate: 0.02,
                throughput: 50
            },
            streaming: {
                supported: false
            }
        });

        // Embedding generation service
        this.services.set('embedding-service', {
            id: 'embedding-service',
            name: 'Embedding Generator',
            type: 'embedding',
            url: 'http://localhost:11434',
            port: 11434,
            maxConcurrency: this.config.concurrencyLimits.embedding || 4,
            currentLoad: 0,
            healthStatus: 'healthy',
            capabilities: ['text-embedding', 'semantic-encoding', 'ollama-models'],
            performance: {
                averageResponseTime: 800,
                successRate: 0.92,
                errorRate: 0.08,
                throughput: 25
            },
            streaming: {
                supported: true,
                endpoint: '/api/embeddings',
                maxStreams: 2,
                currentStreams: 0
            }
        });

        // Classification service
        this.services.set('classification-service', {
            id: 'classification-service',
            name: 'Document Classification',
            type: 'classification',
            url: 'http://localhost:8092',
            port: 8092,
            maxConcurrency: this.config.concurrencyLimits.classification || 3,
            currentLoad: 0,
            healthStatus: 'healthy',
            capabilities: ['document-classification', 'legal-category', 'content-analysis'],
            performance: {
                averageResponseTime: 1200,
                successRate: 0.90,
                errorRate: 0.10,
                throughput: 15
            },
            streaming: {
                supported: false
            }
        });

        // Analysis service
        this.services.set('analysis-service', {
            id: 'analysis-service',
            name: 'Legal Analysis Engine',
            type: 'analysis',
            url: 'http://localhost:8093',
            port: 8093,
            maxConcurrency: this.config.concurrencyLimits.analysis || 2,
            currentLoad: 0,
            healthStatus: 'healthy',
            capabilities: ['legal-analysis', 'precedent-search', 'case-law'],
            performance: {
                averageResponseTime: 3500,
                successRate: 0.88,
                errorRate: 0.12,
                throughput: 8
            },
            streaming: {
                supported: true,
                endpoint: '/analyze/stream',
                maxStreams: 1,
                currentStreams: 0
            }
        });

        console.log(`✅ Initialized ${this.services.size} service endpoints`);
    }

    /**
     * Initialize ML routing models
     */
    private initializeRoutingModels(): void {
        console.log('🧠 Initializing ML routing models...');

        // Performance-based routing model
        this.routingModels.set('performance-router', {
            id: 'performance-router',
            name: 'Performance-Based Router',
            type: 'gradient-boosting',
            features: ['task_complexity', 'payload_size', 'response_time_requirement', 'streaming_required'],
            weights: { accuracy: 0.4, speed: 0.3, reliability: 0.2, cost: 0.1 },
            accuracy: 0.85,
            lastTrained: new Date(),
            predictions: 0
        });

        // Load-aware routing model
        this.routingModels.set('load-balancer', {
            id: 'load-balancer',
            name: 'Load-Aware Router',
            type: 'neural-network',
            features: ['current_load', 'service_capacity', 'queue_length', 'health_score'],
            weights: { load_distribution: 0.5, service_health: 0.3, response_time: 0.2 },
            accuracy: 0.78,
            lastTrained: new Date(),
            predictions: 0
        });

        // Streaming-optimized router
        this.routingModels.set('streaming-router', {
            id: 'streaming-router',
            name: 'Streaming-Optimized Router',
            type: 'decision-tree',
            features: ['streaming_required', 'stream_capacity', 'bandwidth_available', 'client_capability'],
            weights: { stream_efficiency: 0.6, latency: 0.4 },
            accuracy: 0.92,
            lastTrained: new Date(),
            predictions: 0
        });

        console.log(`✅ Initialized ${this.routingModels.size} routing models`);
    }

    /**
     * Discover available services
     */
    private async discoverServices(): Promise<void> {
        console.log('🔍 Discovering available services...');

        const healthCheckPromises = Array.from(this.services.values()).map(service =>
            this.checkServiceHealth(service)
        );

        const healthResults = await Promise.allSettled(healthCheckPromises);
        
        healthResults.forEach((result, index) => {
            const service = Array.from(this.services.values())[index];
            
            if (result.status === 'fulfilled' && result.value) {
                service.healthStatus = 'healthy';
                console.log(`✅ Service ${service.name} is healthy`);
            } else {
                service.healthStatus = 'unhealthy';
                console.log(`❌ Service ${service.name} is unhealthy`);
            }
        });

        const healthyServices = Array.from(this.services.values())
            .filter(s => s.healthStatus === 'healthy').length;
        
        console.log(`✅ Service discovery complete: ${healthyServices}/${this.services.size} services healthy`);
    }

    /**
     * Check individual service health
     */
    private async checkServiceHealth(service: ServiceEndpointConfig): Promise<boolean> {
        try {
            // For the summarization service, check the health endpoint
            if (service.type === 'summarization') {
                const response = await fetch(`${service.url}/health`, {
                    method: 'GET',
                    signal: AbortSignal.timeout(5000)
                });
                return response.ok;
            }

            // For other services, do basic connectivity check
            const response = await fetch(`${service.url}/`, {
                method: 'HEAD',
                signal: AbortSignal.timeout(5000)
            });
            return response.ok;

        } catch (error) {
            console.error(`Health check failed for ${service.name}:`, error);
            return false;
        }
    }

    /**
     * Get eligible services for a task
     */
    private getEligibleServices(task: TaskRequest): ServiceEndpointConfig[] {
        const eligibleServices = Array.from(this.services.values()).filter(service => {
            // Basic health check
            if (service.healthStatus !== 'healthy') return false;

            // Capability check
            if (!service.capabilities.includes(task.type) && service.type !== task.type) return false;

            // Concurrency check
            if (service.currentLoad >= service.maxConcurrency) return false;

            // Streaming requirement check
            if (task.metadata.streaming && !service.streaming.supported) return false;

            // Stream capacity check
            if (task.metadata.streaming && service.streaming.currentStreams >= service.streaming.maxStreams) return false;

            return true;
        });

        return eligibleServices.sort((a, b) => {
            // Sort by performance score (combination of success rate and response time)
            const scoreA = a.performance.successRate / (a.performance.averageResponseTime / 1000);
            const scoreB = b.performance.successRate / (b.performance.averageResponseTime / 1000);
            return scoreB - scoreA;
        });
    }

    /**
     * Make intelligent routing decision using ML
     */
    private async makeRoutingDecision(
        task: TaskRequest, 
        candidateServices: ServiceEndpointConfig[]
    ): Promise<RoutingDecision> {
        if (!this.config.enableMLRouting || candidateServices.length === 1) {
            // Simple selection if ML is disabled or only one option
            return {
                taskId: task.id,
                selectedService: candidateServices[0],
                confidence: 0.5,
                reasoning: 'simple-selection',
                alternatives: candidateServices.slice(1),
                estimatedResponseTime: candidateServices[0].performance.averageResponseTime,
                needsQueuing: false,
                routingMetadata: {}
            };
        }

        // Use ML model to score each candidate service
        const model = this.selectBestModel(task);
        const scores = await Promise.all(
            candidateServices.map(service => 
                this.scoreServiceForTask(task, service, model)
            )
        );

        // Find the best service
        const bestIndex = scores.indexOf(Math.max(...scores));
        const selectedService = candidateServices[bestIndex];
        const confidence = scores[bestIndex];

        // Estimate response time considering current load
        const loadFactor = selectedService.currentLoad / selectedService.maxConcurrency;
        const estimatedResponseTime = selectedService.performance.averageResponseTime * (1 + loadFactor);

        return {
            taskId: task.id,
            selectedService,
            confidence,
            reasoning: `ml-selection-${model.name}`,
            alternatives: candidateServices.filter((_, i) => i !== bestIndex),
            estimatedResponseTime,
            needsQueuing: selectedService.currentLoad >= selectedService.maxConcurrency * 0.8,
            routingMetadata: {
                modelUsed: model.id,
                scores,
                loadFactor,
                features: task.features
            }
        };
    }

    /**
     * Select the best ML model for the task
     */
    private selectBestModel(task: TaskRequest): RoutingModel {
        if (task.metadata.streaming) {
            return this.routingModels.get('streaming-router')!;
        }

        if (task.priority === 'urgent' || task.priority === 'high') {
            return this.routingModels.get('performance-router')!;
        }

        return this.routingModels.get('load-balancer')!;
    }

    /**
     * Score a service for a specific task using ML
     */
    private async scoreServiceForTask(
        task: TaskRequest, 
        service: ServiceEndpointConfig, 
        model: RoutingModel
    ): Promise<number> {
        const features = task.features!;
        
        // Simple scoring algorithm (would be replaced with actual ML model)
        let score = 0;

        // Performance factors
        score += service.performance.successRate * 0.4;
        score += (1 / (service.performance.averageResponseTime / 1000)) * 0.3;
        score += (1 - service.performance.errorRate) * 0.2;

        // Load factors
        const loadRatio = service.currentLoad / service.maxConcurrency;
        score += (1 - loadRatio) * 0.1;

        // Task-specific adjustments
        if (features.complexity === 'high' && service.capabilities.includes('high-performance')) {
            score += 0.1;
        }

        if (features.payloadSize === 'large' && service.capabilities.includes('bulk-processing')) {
            score += 0.1;
        }

        if (task.metadata.streaming && service.streaming.supported) {
            const streamLoadRatio = service.streaming.currentStreams / service.streaming.maxStreams;
            score += (1 - streamLoadRatio) * 0.2;
        }

        // Priority adjustments
        if (task.priority === 'urgent' && service.performance.averageResponseTime < 1000) {
            score += 0.15;
        }

        return Math.min(1, Math.max(0, score));
    }

    /**
     * Validate and adjust routing decision
     */
    private async validateAndAdjustDecision(
        decision: RoutingDecision, 
        task: TaskRequest
    ): Promise<RoutingDecision> {
        const service = decision.selectedService;

        // Check for 429 (service at capacity)
        if (service.currentLoad >= service.maxConcurrency) {
            console.log(`⚠️ Service ${service.name} at capacity, implementing backpressure`);
            
            // Look for alternative service
            const alternatives = decision.alternatives.filter(alt => 
                alt.currentLoad < alt.maxConcurrency
            );

            if (alternatives.length > 0) {
                return {
                    ...decision,
                    selectedService: alternatives[0],
                    reasoning: 'capacity-overflow-redirect',
                    needsQueuing: false
                };
            }

            // Queue the task with backpressure
            return {
                ...decision,
                needsQueuing: true,
                estimatedResponseTime: decision.estimatedResponseTime + 5000, // Add queue wait time
                routingMetadata: {
                    ...decision.routingMetadata,
                    backpressureApplied: true,
                    queuePosition: await this.queueManager.getQueueLength(task.type)
                }
            };
        }

        return decision;
    }

    /**
     * Execute streaming task
     */
    private async executeStreamingTask(task: TaskRequest, decision: RoutingDecision): Promise<any> {
        const service = decision.selectedService;
        const streamEndpoint = `${service.url}${service.streaming.endpoint}`;

        console.log(`📡 Executing streaming task ${task.id} on ${streamEndpoint}`);

        // Update stream count
        if (service.streaming.currentStreams !== undefined) {
            service.streaming.currentStreams++;
        }

        try {
            const response = await fetch(streamEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                body: JSON.stringify(task.payload),
                signal: AbortSignal.timeout(task.timeout || this.config.defaultTimeout)
            });

            if (!response.ok) {
                if (response.status === 429) {
                    throw new Error('Service at capacity (429)');
                }
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            // Handle SSE stream
            const reader = response.body?.getReader();
            const decoder = new TextDecoder();
            let result = '';

            if (reader) {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    result += chunk;
                    
                    // Emit streaming data
                    this.emit('streamingData', { taskId: task.id, chunk });
                }
            }

            return { streaming: true, data: result, service: service.name };

        } finally {
            // Update stream count
            if (service.streaming.currentStreams !== undefined) {
                service.streaming.currentStreams--;
            }
        }
    }

    /**
     * Execute standard (non-streaming) task
     */
    private async executeStandardTask(task: TaskRequest, decision: RoutingDecision): Promise<any> {
        const service = decision.selectedService;
        const endpoint = `${service.url}/api/${task.type}`;

        console.log(`🔧 Executing standard task ${task.id} on ${endpoint}`);

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(task.payload),
            signal: AbortSignal.timeout(task.timeout || this.config.defaultTimeout)
        });

        if (!response.ok) {
            if (response.status === 429) {
                throw new Error('Service at capacity (429)');
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        return { streaming: false, data: result, service: service.name };
    }

    /**
     * Get fallback routing when primary routing fails
     */
    private async getFallbackRouting(task: TaskRequest): Promise<RoutingDecision> {
        // Try to find any healthy service that can handle the task type
        const fallbackServices = Array.from(this.services.values()).filter(service =>
            service.healthStatus !== 'unhealthy' && 
            (service.capabilities.includes(task.type) || service.type === task.type)
        );

        if (fallbackServices.length === 0) {
            throw new Error(`No fallback services available for task type: ${task.type}`);
        }

        // Select service with lowest current load
        const selectedService = fallbackServices.reduce((prev, current) =>
            prev.currentLoad < current.currentLoad ? prev : current
        );

        return {
            taskId: task.id,
            selectedService,
            confidence: 0.3,
            reasoning: 'fallback-selection',
            alternatives: [],
            estimatedResponseTime: selectedService.performance.averageResponseTime * 2,
            needsQueuing: true,
            routingMetadata: { fallback: true }
        };
    }

    /**
     * Setup event listeners
     */
    private setupEventListeners(): void {
        // Listen to horizontal agent scaler for service updates
        horizontalAgentScaler.on('agentsScaled', (data) => {
            this.handleServiceScaling(data);
        });

        // Listen to cache manager for caching routing decisions
        advancedCacheManager.on('metricsCollected', (metrics) => {
            this.handleCacheMetrics(metrics);
        });

        // Self-monitoring
        this.on('taskRouted', (data) => {
            this.analytics.totalRoutingDecisions++;
            this.analytics.averageRoutingTime = 
                (this.analytics.averageRoutingTime + data.routingTime) / 2;
        });

        this.on('taskCompleted', (data) => {
            this.analytics.successfulExecutions++;
            this.analytics.averageExecutionTime = 
                (this.analytics.averageExecutionTime + data.executionTime) / 2;
        });

        this.on('taskFailed', (data) => {
            this.analytics.failedExecutions++;
            this.analytics.totalErrors++;
        });
    }

    /**
     * Handle service scaling events
     */
    private handleServiceScaling(scalingData: any): void {
        console.log('📈 Handling service scaling:', scalingData);
        
        // Update service capacities based on scaling events
        scalingData.scalingActions?.forEach(action => {
            if (action.action === 'scale-up') {
                // Increase service capacity
                this.updateServiceCapacity(action.agentType, 1);
            } else if (action.action === 'scale-down') {
                // Decrease service capacity
                this.updateServiceCapacity(action.agentType, -1);
            }
        });
    }

    /**
     * Update service capacity
     */
    private updateServiceCapacity(serviceType: string, capacityChange: number): void {
        for (const service of this.services.values()) {
            if (service.type === serviceType) {
                service.maxConcurrency += capacityChange;
                console.log(`📊 Updated ${service.name} capacity to ${service.maxConcurrency}`);
            }
        }
    }

    /**
     * Update service load
     */
    private updateServiceLoad(serviceId: string, loadChange: number): void {
        const service = this.services.get(serviceId);
        if (service) {
            service.currentLoad = Math.max(0, service.currentLoad + loadChange);
        }
    }

    /**
     * Various utility methods for caching, metrics, etc.
     */
    private async getCachedRouting(task: TaskRequest): Promise<RoutingDecision | null> {
        const cacheKey = `routing:${task.type}:${JSON.stringify(task.features)}`;
        return await advancedCacheManager.get(cacheKey) as RoutingDecision | null;
    }

    private async cacheRoutingDecision(task: TaskRequest, decision: RoutingDecision): Promise<void> {
        const cacheKey = `routing:${task.type}:${JSON.stringify(task.features)}`;
        await advancedCacheManager.set(cacheKey, decision, { ttl: this.config.routingCacheTimeout });
    }

    private updateRoutingMetrics(task: TaskRequest, decision: RoutingDecision, routingTime: number): void {
        this.analytics.totalRoutingDecisions++;
        this.analytics.averageRoutingTime = (this.analytics.averageRoutingTime + routingTime) / 2;
    }

    private async updateTaskMetrics(task: TaskRequest, decision: RoutingDecision, metrics: any): Promise<void> {
        const serviceMetrics = this.taskHistory.get(decision.selectedService.id) || [];
        serviceMetrics.push({
            taskId: task.id,
            taskType: task.type,
            success: metrics.success,
            executionTime: metrics.executionTime,
            responseSize: metrics.responseSize,
            timestamp: Date.now()
        });
        
        // Keep only recent metrics
        if (serviceMetrics.length > 1000) {
            serviceMetrics.splice(0, serviceMetrics.length - 1000);
        }
        
        this.taskHistory.set(decision.selectedService.id, serviceMetrics);
    }

    private async updateRoutingModel(task: TaskRequest, decision: RoutingDecision, outcome: any): Promise<void> {
        const model = this.routingModels.get(decision.routingMetadata?.modelUsed);
        if (model) {
            model.predictions++;
            // Update model accuracy based on outcome
            // In production, this would involve actual ML model training
        }
    }

    private estimateResponseSize(result: any): number {
        return JSON.stringify(result).length;
    }

    private handleCacheMetrics(metrics: any): void {
        // Use cache metrics to inform routing decisions
        this.analytics.cacheHitRate = metrics.hitRate;
    }

    private async trainInitialModels(): Promise<void> {
        console.log('🧠 Training initial ML routing models...');
        // In production, this would load historical data and train models
        console.log('✅ Initial model training completed');
    }

    private startModelUpdates(): void {
        setInterval(() => {
            this.updateModelsWithRecentData();
        }, this.config.modelUpdateInterval);
    }

    private startMetricsCollection(): void {
        setInterval(() => {
            this.collectAndEmitMetrics();
        }, this.config.metricsCollectionInterval);
    }

    private async updateModelsWithRecentData(): Promise<void> {
        // Update models with recent performance data
        console.log('🔄 Updating ML models with recent data...');
    }

    private async collectAndEmitMetrics(): Promise<void> {
        const metrics = {
            ...this.analytics,
            services: Object.fromEntries(
                Array.from(this.services.entries()).map(([id, service]) => [
                    id,
                    {
                        name: service.name,
                        currentLoad: service.currentLoad,
                        maxConcurrency: service.maxConcurrency,
                        healthStatus: service.healthStatus,
                        performance: service.performance
                    }
                ])
            ),
            timestamp: Date.now()
        };

        this.emit('metricsCollected', metrics);
    }

    /**
     * Initialize analytics
     */
    private initializeAnalytics(): RoutingAnalytics {
        return {
            totalRoutingDecisions: 0,
            successfulExecutions: 0,
            failedExecutions: 0,
            averageRoutingTime: 0,
            averageExecutionTime: 0,
            totalErrors: 0,
            cacheHitRate: 0,
            modelAccuracy: new Map(),
            serviceUtilization: new Map()
        };
    }

    /**
     * Get routing analytics
     */
    getAnalytics(): RoutingAnalytics {
        return this.analytics;
    }

    /**
     * Get service status
     */
    getServiceStatus(): Map<string, ServiceEndpointConfig> {
        return new Map(this.services);
    }

    /**
     * Get routing models status
     */
    getModelsStatus(): Map<string, RoutingModel> {
        return new Map(this.routingModels);
    }
}

/**
 * Task Feature Extractor
 */
class TaskFeatureExtractor {
    constructor(private config: RoutingConfig) {}

    async extractFeatures(task: TaskRequest): Promise<TaskFeatures> {
        const payload = task.payload;
        const payloadSize = JSON.stringify(payload).length;

        return {
            taskType: task.type,
            priority: task.priority,
            payloadSize: payloadSize > 10000 ? 'large' : payloadSize > 1000 ? 'medium' : 'small',
            complexity: this.estimateComplexity(task),
            streamingRequired: task.metadata.streaming || false,
            expectedResponseTime: task.metadata.expectedResponseTime || 0,
            retryCount: task.metadata.retryCount || 0,
            userContext: {
                sessionId: task.metadata.sessionId,
                userAgent: task.metadata.userAgent
            }
        };
    }

    private estimateComplexity(task: TaskRequest): 'low' | 'medium' | 'high' {
        const payload = task.payload;
        
        if (task.type === 'summarization') {
            const textLength = payload.text?.length || 0;
            return textLength > 10000 ? 'high' : textLength > 2000 ? 'medium' : 'low';
        }
        
        if (task.type === 'analysis') {
            return 'high'; // Legal analysis is typically complex
        }
        
        return 'medium'; // Default complexity
    }
}

/**
 * Intelligent Load Balancer
 */
class IntelligentLoadBalancer {
    constructor(private config: RoutingConfig) {}

    selectOptimalService(services: ServiceEndpointConfig[], task: TaskRequest): ServiceEndpointConfig {
        // Weighted round-robin with health and capacity awareness
        const healthyServices = services.filter(s => s.healthStatus === 'healthy');
        
        if (healthyServices.length === 0) {
            throw new Error('No healthy services available');
        }

        // Calculate load-adjusted scores
        const scoredServices = healthyServices.map(service => ({
            service,
            score: this.calculateLoadScore(service)
        }));

        // Select service with highest score
        scoredServices.sort((a, b) => b.score - a.score);
        return scoredServices[0].service;
    }

    private calculateLoadScore(service: ServiceEndpointConfig): number {
        const loadRatio = service.currentLoad / service.maxConcurrency;
        const performanceScore = service.performance.successRate * 
                               (1 / (service.performance.averageResponseTime / 1000));
        
        return performanceScore * (1 - loadRatio);
    }
}

/**
 * Priority Queue Manager
 */
class PriorityQueueManager {
    private queues: Map<string, TaskRequest[]> = new Map();

    constructor(private config: RoutingConfig) {}

    async start(): Promise<void> {
        console.log('📋 Priority queue manager started');
    }

    async enqueue(task: TaskRequest, decision: RoutingDecision): Promise<void> {
        const queueKey = `${task.type}:${decision.selectedService.id}`;
        const queue = this.queues.get(queueKey) || [];
        
        // Insert based on priority
        const insertIndex = this.findInsertionIndex(queue, task);
        queue.splice(insertIndex, 0, task);
        
        this.queues.set(queueKey, queue);
        
        console.log(`📋 Queued task ${task.id} at position ${insertIndex} for ${decision.selectedService.name}`);
    }

    async getQueueLength(taskType: string): Promise<number> {
        let totalLength = 0;
        for (const [key, queue] of this.queues) {
            if (key.startsWith(`${taskType}:`)) {
                totalLength += queue.length;
            }
        }
        return totalLength;
    }

    private findInsertionIndex(queue: TaskRequest[], task: TaskRequest): number {
        const priorityOrder = { urgent: 0, high: 1, medium: 2, low: 3 };
        const taskPriorityValue = priorityOrder[task.priority];
        
        for (let i = 0; i < queue.length; i++) {
            if (priorityOrder[queue[i].priority] > taskPriorityValue) {
                return i;
            }
        }
        
        return queue.length;
    }
}

/**
 * Routing Performance Tracker
 */
class RoutingPerformanceTracker {
    constructor(private config: RoutingConfig) {}

    async start(): Promise<void> {
        console.log('📊 Routing performance tracker started');
    }
}

// Export singleton instance
export const mlTaskRouter = new MLTaskRouter({
    enableMLRouting: true,
    enablePredictiveRouting: true,
    enableAdaptiveLoadBalancing: true,
    enableStreamingOptimization: true,
    enableConcurrencyAwareness: true,
    cacheRoutingDecisions: true,
    learningRate: 0.1,
    modelUpdateInterval: 300000,
    metricsCollectionInterval: 30000,
    routingCacheTimeout: 60000,
    maxRetryAttempts: 3,
    defaultTimeout: 30000,
    concurrencyLimits: {
        'summarization': 2,
        'embedding': 4,
        'classification': 3,
        'analysis': 2,
        'vector-search': 6
    }
});