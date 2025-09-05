/**
 * Automation Integration Service
 * 
 * Central service that integrates all automation components:
 * - Automated workflow triggers
 * - Real-time monitoring
 * - Performance metrics collection
 * - Event routing and coordination
 * - Service health monitoring
 * - Resource management
 * 
 * This service acts as the central coordinator for all automated operations
 */

import { EventEmitter } from 'events';
import { automatedWorkflowEngine } from '$lib/orchestration/automated-workflow-triggers';
import { automationMonitor } from '$lib/websockets/automation-monitor';
import { performanceOptimizer } from '$lib/optimization/performance-optimizer';
import { advancedCacheManager } from '$lib/caching/advanced-cache-manager';
import { cacheMonitoringService } from '$lib/websockets/cache-monitoring-service';
import { mlTaskRouter } from '$lib/routing/ml-task-router';
import type { 
    PerformanceMetrics, 
    SystemAlert, 
    WorkflowTrigger,
    AutomationConfig,
    ServiceHealth,
    ResourceMetrics,
    PerformanceAnalysis 
} from '$lib/ai/types';

export class AutomationIntegrationService extends EventEmitter {
    private isRunning = false;
    private metricsCollectionInterval: NodeJS.Timeout | null = null;
    private healthCheckInterval: NodeJS.Timeout | null = null;
    private resourceMonitoringInterval: NodeJS.Timeout | null = null;
    private lastMetrics: PerformanceMetrics | null = null;
    private serviceHealthCache: Map<string, ServiceHealth> = new Map();
    private config: AutomationConfig;

    constructor(config: AutomationConfig = {}) {
        super();
        
        this.config = {
            metricsInterval: 5000,
            healthCheckInterval: 30000,
            resourceMonitoringInterval: 10000,
            enableRealTimeMonitoring: true,
            enableAutomaticScaling: false,
            enablePredictiveAnalysis: true,
            enableAdvancedCaching: true,
            enableCacheMonitoring: true,
            enableMLTaskRouting: true,
            alertThresholds: {
                errorRate: 0.05,
                responseTime: 5000,
                cpuUsage: 0.8,
                memoryUsage: 0.85,
                cacheHitRate: 0.7,
                cacheResponseTime: 100
            },
            ...config
        };

        this.setupEventListeners();
    }

    /**
     * Start the automation integration service
     */
    async start(): Promise<void> {
        if (this.isRunning) {
            console.log('⚠️ Automation integration service is already running');
            return;
        }

        try {
            console.log('🚀 Starting Automation Integration Service...');

            // Start WebSocket monitoring if enabled
            if (this.config.enableRealTimeMonitoring) {
                await automationMonitor.startServer();
                console.log('📡 Real-time monitoring WebSocket server started');
            }

            // Start advanced caching system if enabled
            if (this.config.enableAdvancedCaching) {
                await advancedCacheManager.start();
                console.log('🗄️ Advanced cache manager started');
            }

            // Start cache monitoring if enabled
            if (this.config.enableCacheMonitoring) {
                await cacheMonitoringService.startServer();
                console.log('📊 Cache monitoring WebSocket server started');
            }

            // Start ML task routing if enabled
            if (this.config.enableMLTaskRouting) {
                await mlTaskRouter.start();
                console.log('🧠 ML task router started');
            }

            // Start metrics collection
            this.startMetricsCollection();
            
            // Start health monitoring
            this.startHealthMonitoring();
            
            // Start resource monitoring
            this.startResourceMonitoring();

            // Initialize default triggers and monitoring
            await this.initializeDefaultConfiguration();

            this.isRunning = true;
            
            console.log('✅ Automation Integration Service started successfully');
            console.log(`📊 Metrics collection interval: ${this.config.metricsInterval}ms`);
            console.log(`🏥 Health check interval: ${this.config.healthCheckInterval}ms`);
            console.log(`💻 Resource monitoring interval: ${this.config.resourceMonitoringInterval}ms`);
            
            this.emit('serviceStarted', {
                timestamp: new Date(),
                config: this.config
            });

        } catch (error) {
            console.error('❌ Failed to start automation integration service:', error);
            throw error;
        }
    }

    /**
     * Stop the automation integration service
     */
    async stop(): Promise<void> {
        if (!this.isRunning) {
            console.log('⚠️ Automation integration service is not running');
            return;
        }

        try {
            console.log('🛑 Stopping Automation Integration Service...');

            // Stop intervals
            if (this.metricsCollectionInterval) {
                clearInterval(this.metricsCollectionInterval);
                this.metricsCollectionInterval = null;
            }

            if (this.healthCheckInterval) {
                clearInterval(this.healthCheckInterval);
                this.healthCheckInterval = null;
            }

            if (this.resourceMonitoringInterval) {
                clearInterval(this.resourceMonitoringInterval);
                this.resourceMonitoringInterval = null;
            }

            // Shutdown WebSocket server
            if (this.config.enableRealTimeMonitoring) {
                await automationMonitor.shutdown();
                console.log('📡 Real-time monitoring WebSocket server stopped');
            }

            // Shutdown cache monitoring
            if (this.config.enableCacheMonitoring) {
                await cacheMonitoringService.shutdown();
                console.log('📊 Cache monitoring WebSocket server stopped');
            }

            this.isRunning = false;
            
            console.log('✅ Automation Integration Service stopped successfully');
            
            this.emit('serviceStopped', {
                timestamp: new Date()
            });

        } catch (error) {
            console.error('❌ Error stopping automation integration service:', error);
            throw error;
        }
    }

    /**
     * Setup event listeners for coordination
     */
    private setupEventListeners(): void {
        // Listen to automation engine events
        automatedWorkflowEngine.on('triggerExecuted', (data) => {
            this.handleTriggerExecution(data);
        });

        automatedWorkflowEngine.on('alert', (alert) => {
            this.handleSystemAlert(alert);
        });

        automatedWorkflowEngine.on('error', (error) => {
            this.handleSystemError(error);
        });

        // Performance optimizer events
        performanceOptimizer.on('performanceAnalyzed', (data) => {
            this.handlePerformanceAnalysis(data);
        });

        performanceOptimizer.on('optimizationCompleted', (data) => {
            this.handleOptimizationCompleted(data);
        });

        performanceOptimizer.on('optimizationFailed', (data) => {
            this.handleOptimizationFailed(data);
        });

        // Listen to monitoring events
        automationMonitor.on('clientConnected', (data) => {
            console.log(`📡 Monitoring client connected: ${data.clientId}`);
        });

        automationMonitor.on('clientDisconnected', (data) => {
            console.log(`📡 Monitoring client disconnected: ${data.clientId}`);
        });

        // Advanced cache manager events
        advancedCacheManager.on('cacheManagerStarted', (data) => {
            console.log('🗄️ Advanced cache manager started:', data);
        });

        advancedCacheManager.on('metricsCollected', (metrics) => {
            this.handleCacheMetrics(metrics);
        });

        advancedCacheManager.on('analyticsGenerated', (analytics) => {
            this.handleCacheAnalytics(analytics);
        });

        advancedCacheManager.on('lowHitRate', (data) => {
            this.handleCacheAlert({
                type: 'cache-performance',
                severity: 'warning',
                message: `Cache hit rate is low: ${(data.hitRate * 100).toFixed(1)}%`,
                data,
                timestamp: new Date()
            });
        });

        advancedCacheManager.on('highOperationTime', (data) => {
            this.handleCacheAlert({
                type: 'cache-performance',
                severity: 'warning',
                message: `Cache operation time is high: ${data.operationTime.toFixed(2)}ms`,
                data,
                timestamp: new Date()
            });
        });

        advancedCacheManager.on('cacheInconsistency', (data) => {
            this.handleCacheAlert({
                type: 'cache-coherence',
                severity: 'error',
                message: 'Cache inconsistency detected across layers',
                data,
                timestamp: new Date()
            });
        });

        // Cache monitoring service events
        cacheMonitoringService.on('clientConnected', (data) => {
            console.log(`📊 Cache monitoring client connected: ${data.clientId}`);
        });

        cacheMonitoringService.on('clientDisconnected', (data) => {
            console.log(`📊 Cache monitoring client disconnected: ${data.clientId}`);
        });

        // ML task router events
        mlTaskRouter.on('routerStarted', (data) => {
            console.log('🧠 ML task router started:', data);
        });

        mlTaskRouter.on('taskRouted', (data) => {
            this.handleTaskRouting(data);
        });

        mlTaskRouter.on('taskCompleted', (data) => {
            this.handleTaskCompletion(data);
        });

        mlTaskRouter.on('taskFailed', (data) => {
            this.handleTaskFailure(data);
        });

        mlTaskRouter.on('routingError', (data) => {
            this.handleRoutingError(data);
        });

        mlTaskRouter.on('metricsCollected', (data) => {
            this.handleRoutingMetrics(data);
        });

        mlTaskRouter.on('streamingData', (data) => {
            this.handleStreamingData(data);
        });

        // Self-monitoring
        this.on('metricsCollected', (metrics) => {
            this.analyzeMetrics(metrics);
        });

        this.on('healthStatusChanged', (healthData) => {
            this.handleHealthStatusChange(healthData);
        });
    }

    /**
     * Initialize default configuration and triggers
     */
    private async initializeDefaultConfiguration(): Promise<void> {
        console.log('⚙️ Initializing default automation configuration...');

        // Register system-level triggers
        this.registerSystemLevelTriggers();

        // Setup predictive analysis if enabled
        if (this.config.enablePredictiveAnalysis) {
            this.setupPredictiveAnalysis();
        }

        // Register custom legal AI specific triggers
        this.registerLegalAITriggers();

        console.log('✅ Default automation configuration initialized');
    }

    /**
     * Start metrics collection
     */
    private startMetricsCollection(): void {
        this.metricsCollectionInterval = setInterval(async () => {
            try {
                const metrics = await this.collectSystemMetrics();
                
                // Process through automation engine
                await automatedWorkflowEngine.processMetrics(metrics);
                
                // Process through performance optimizer
                const performanceAnalysis = await performanceOptimizer.processMetrics(metrics);
                
                // Send to WebSocket clients
                if (this.config.enableRealTimeMonitoring) {
                    automationMonitor.addMetrics(metrics);
                }
                
                this.lastMetrics = metrics;
                this.emit('metricsCollected', { metrics, performanceAnalysis });
                
            } catch (error) {
                console.error('❌ Error collecting metrics:', error);
            }
        }, this.config.metricsInterval);
    }

    /**
     * Start health monitoring
     */
    private startHealthMonitoring(): void {
        this.healthCheckInterval = setInterval(async () => {
            try {
                await this.performHealthChecks();
            } catch (error) {
                console.error('❌ Error during health checks:', error);
            }
        }, this.config.healthCheckInterval);
    }

    /**
     * Start resource monitoring
     */
    private startResourceMonitoring(): void {
        this.resourceMonitoringInterval = setInterval(async () => {
            try {
                const resourceMetrics = await this.collectResourceMetrics();
                this.analyzeResourceUsage(resourceMetrics);
            } catch (error) {
                console.error('❌ Error monitoring resources:', error);
            }
        }, this.config.resourceMonitoringInterval);
    }

    /**
     * Collect comprehensive system metrics
     */
    private async collectSystemMetrics(): Promise<PerformanceMetrics> {
        const startTime = performance.now();
        
        // Simulate collecting various metrics
        // In production, these would come from actual monitoring systems
        const metrics: PerformanceMetrics = {
            timestamp: Date.now(),
            
            // Performance metrics
            responseTime: Math.random() * 1000 + 200,
            errorRate: Math.random() * 0.1,
            throughput: Math.random() * 1000 + 500,
            
            // System metrics
            cpuUsage: Math.random() * 0.6 + 0.2,
            memoryUsage: Math.random() * 0.4 + 0.4,
            diskUsage: Math.random() * 0.3 + 0.3,
            
            // Application metrics
            activeConnections: Math.floor(Math.random() * 100) + 10,
            queueLength: Math.floor(Math.random() * 50),
            cacheHitRate: Math.random() * 0.3 + 0.7,
            
            // Agent metrics
            agentFailureRate: Math.random() * 0.1,
            agentResponseTime: Math.random() * 2000 + 500,
            activeWorkflows: Math.floor(Math.random() * 10),
            
            // Database metrics
            dbConnectionCount: Math.floor(Math.random() * 50) + 10,
            dbQueryTime: Math.random() * 500 + 100,
            dbLockWaitTime: Math.random() * 100,
            
            // Custom metrics
            documentsProcessed: Math.floor(Math.random() * 100),
            vectorSearchLatency: Math.random() * 200 + 50,
            embeddingGenerationTime: Math.random() * 1000 + 200,
            
            // Collection metadata
            collectionTime: performance.now() - startTime,
            source: 'automation-integration-service'
        };

        return metrics;
    }

    /**
     * Collect resource metrics
     */
    private async collectResourceMetrics(): Promise<ResourceMetrics> {
        return {
            timestamp: Date.now(),
            
            // CPU metrics
            cpu: {
                usage: Math.random() * 0.6 + 0.2,
                loadAverage: [Math.random() * 2, Math.random() * 2, Math.random() * 2],
                cores: 8
            },
            
            // Memory metrics
            memory: {
                total: 16 * 1024 * 1024 * 1024, // 16GB
                used: Math.random() * 8 * 1024 * 1024 * 1024 + 4 * 1024 * 1024 * 1024,
                free: Math.random() * 4 * 1024 * 1024 * 1024 + 2 * 1024 * 1024 * 1024,
                cached: Math.random() * 2 * 1024 * 1024 * 1024,
                buffers: Math.random() * 1024 * 1024 * 1024
            },
            
            // Disk metrics
            disk: {
                usage: Math.random() * 0.5 + 0.3,
                readIops: Math.random() * 1000,
                writeIops: Math.random() * 500,
                readLatency: Math.random() * 10 + 1,
                writeLatency: Math.random() * 20 + 2
            },
            
            // Network metrics
            network: {
                bytesIn: Math.random() * 1000000,
                bytesOut: Math.random() * 1000000,
                packetsIn: Math.random() * 10000,
                packetsOut: Math.random() * 10000,
                errorsIn: Math.random() * 10,
                errorsOut: Math.random() * 5
            },
            
            // Process metrics
            processes: {
                total: Math.floor(Math.random() * 500) + 200,
                running: Math.floor(Math.random() * 20) + 5,
                sleeping: Math.floor(Math.random() * 400) + 150,
                zombies: Math.floor(Math.random() * 3)
            }
        };
    }

    /**
     * Perform health checks on all services
     */
    private async performHealthChecks(): Promise<void> {
        const services = [
            'automation-engine',
            'websocket-monitor', 
            'database',
            'redis-cache',
            'vector-store',
            'llm-service',
            'document-processor'
        ];

        const healthPromises = services.map(service => this.checkServiceHealth(service));
        const healthResults = await Promise.allSettled(healthPromises);

        healthResults.forEach((result, index) => {
            const serviceName = services[index];
            
            if (result.status === 'fulfilled') {
                this.serviceHealthCache.set(serviceName, result.value);
            } else {
                const unhealthyStatus: ServiceHealth = {
                    serviceName,
                    status: 'unhealthy',
                    responseTime: -1,
                    lastCheck: new Date(),
                    error: result.reason?.message || 'Health check failed'
                };
                
                this.serviceHealthCache.set(serviceName, unhealthyStatus);
                this.emit('serviceUnhealthy', { serviceName, error: result.reason });
            }
        });

        // Emit overall health status
        const overallHealth = this.calculateOverallHealth();
        this.emit('healthStatusChanged', {
            overall: overallHealth,
            services: Object.fromEntries(this.serviceHealthCache),
            timestamp: new Date()
        });
    }

    /**
     * Check individual service health
     */
    private async checkServiceHealth(serviceName: string): Promise<ServiceHealth> {
        const startTime = performance.now();
        
        // Simulate health check - in production, these would be actual health endpoints
        const isHealthy = Math.random() > 0.05; // 95% healthy
        const responseTime = performance.now() - startTime + (Math.random() * 100);
        
        return {
            serviceName,
            status: isHealthy ? 'healthy' : 'unhealthy',
            responseTime,
            lastCheck: new Date(),
            metadata: {
                endpoint: `/health/${serviceName}`,
                version: '1.0.0',
                uptime: Math.random() * 86400000 // Random uptime in ms
            }
        };
    }

    /**
     * Calculate overall system health
     */
    private calculateOverallHealth(): { status: string; score: number; details: any } {
        const services = Array.from(this.serviceHealthCache.values());
        
        if (services.length === 0) {
            return { status: 'unknown', score: 0, details: {} };
        }

        const healthyServices = services.filter(s => s.status === 'healthy').length;
        const score = (healthyServices / services.length) * 100;
        
        const status = score >= 90 ? 'excellent' :
                      score >= 75 ? 'good' :
                      score >= 50 ? 'degraded' : 'critical';

        return {
            status,
            score: Math.round(score),
            details: {
                totalServices: services.length,
                healthyServices,
                unhealthyServices: services.length - healthyServices,
                averageResponseTime: services.reduce((sum, s) => sum + s.responseTime, 0) / services.length
            }
        };
    }

    /**
     * Analyze collected metrics for patterns and anomalies
     */
    private analyzeMetrics(metrics: PerformanceMetrics): void {
        // Check against thresholds
        const alerts: SystemAlert[] = [];

        if (metrics.errorRate > this.config.alertThresholds.errorRate) {
            alerts.push({
                id: `error-rate-${Date.now()}`,
                type: 'performance',
                severity: 'high',
                message: `Error rate exceeded threshold: ${metrics.errorRate.toFixed(3)} > ${this.config.alertThresholds.errorRate}`,
                timestamp: new Date(),
                metrics
            });
        }

        if (metrics.responseTime > this.config.alertThresholds.responseTime) {
            alerts.push({
                id: `response-time-${Date.now()}`,
                type: 'performance',
                severity: 'medium',
                message: `Response time exceeded threshold: ${metrics.responseTime.toFixed(2)}ms > ${this.config.alertThresholds.responseTime}ms`,
                timestamp: new Date(),
                metrics
            });
        }

        // Send alerts through automation engine
        alerts.forEach(alert => {
            automatedWorkflowEngine.emit('alert', alert);
        });

        // Predictive analysis
        if (this.config.enablePredictiveAnalysis) {
            this.performPredictiveAnalysis(metrics);
        }
    }

    /**
     * Analyze resource usage
     */
    private analyzeResourceUsage(resourceMetrics: ResourceMetrics): void {
        const alerts: SystemAlert[] = [];

        // CPU usage alert
        if (resourceMetrics.cpu.usage > this.config.alertThresholds.cpuUsage) {
            alerts.push({
                id: `cpu-usage-${Date.now()}`,
                type: 'resource',
                severity: 'high',
                message: `CPU usage exceeded threshold: ${(resourceMetrics.cpu.usage * 100).toFixed(1)}% > ${(this.config.alertThresholds.cpuUsage * 100)}%`,
                timestamp: new Date()
            });
        }

        // Memory usage alert
        const memoryUsageRatio = resourceMetrics.memory.used / resourceMetrics.memory.total;
        if (memoryUsageRatio > this.config.alertThresholds.memoryUsage) {
            alerts.push({
                id: `memory-usage-${Date.now()}`,
                type: 'resource',
                severity: 'high',
                message: `Memory usage exceeded threshold: ${(memoryUsageRatio * 100).toFixed(1)}% > ${(this.config.alertThresholds.memoryUsage * 100)}%`,
                timestamp: new Date()
            });
        }

        // Send resource alerts
        alerts.forEach(alert => {
            automatedWorkflowEngine.emit('alert', alert);
        });

        // Auto-scaling logic
        if (this.config.enableAutomaticScaling) {
            this.evaluateAutoScaling(resourceMetrics);
        }
    }

    /**
     * Register system-level triggers
     */
    private registerSystemLevelTriggers(): void {
        // Service health trigger
        automatedWorkflowEngine.registerTrigger({
            id: 'service-health-degradation',
            name: 'Service Health Degradation',
            type: 'system',
            conditions: [
                {
                    metric: 'serviceHealthScore',
                    operator: 'lessThan',
                    threshold: 75,
                    windowMs: 60000
                }
            ],
            actions: [
                {
                    type: 'workflow',
                    workflow: 'service-recovery',
                    priority: 'high'
                }
            ],
            cooldownMs: 120000,
            enabled: true
        });

        // Resource exhaustion prevention
        automatedWorkflowEngine.registerTrigger({
            id: 'resource-exhaustion-prevention',
            name: 'Resource Exhaustion Prevention',
            type: 'resource',
            conditions: [
                {
                    metric: 'cpuUsage',
                    operator: 'greaterThan',
                    threshold: 0.9,
                    windowMs: 30000
                },
                {
                    metric: 'memoryUsage',
                    operator: 'greaterThan',
                    threshold: 0.9,
                    windowMs: 30000
                }
            ],
            conditionLogic: 'OR',
            actions: [
                {
                    type: 'workflow',
                    workflow: 'emergency-resource-management',
                    priority: 'urgent'
                }
            ],
            cooldownMs: 60000,
            enabled: true
        });
    }

    /**
     * Register Legal AI specific triggers
     */
    private registerLegalAITriggers(): void {
        // Document processing backup trigger
        automatedWorkflowEngine.registerTrigger({
            id: 'document-processing-backup',
            name: 'Document Processing Backup',
            type: 'legal-ai',
            conditions: [
                {
                    metric: 'documentsProcessed',
                    operator: 'lessThan',
                    threshold: 10,
                    windowMs: 300000 // 5 minutes
                }
            ],
            actions: [
                {
                    type: 'workflow',
                    workflow: 'document-processing-recovery',
                    priority: 'medium'
                }
            ],
            cooldownMs: 180000,
            enabled: true
        });

        // Vector search performance trigger
        automatedWorkflowEngine.registerTrigger({
            id: 'vector-search-performance',
            name: 'Vector Search Performance Degradation',
            type: 'legal-ai',
            conditions: [
                {
                    metric: 'vectorSearchLatency',
                    operator: 'greaterThan',
                    threshold: 500,
                    windowMs: 120000
                }
            ],
            actions: [
                {
                    type: 'workflow',
                    workflow: 'vector-store-optimization',
                    priority: 'medium'
                }
            ],
            cooldownMs: 300000,
            enabled: true
        });
    }

    /**
     * Setup predictive analysis
     */
    private setupPredictiveAnalysis(): void {
        console.log('🔮 Setting up predictive analysis for automation...');
        // This would integrate with ML models for predictive analysis
        // For now, we'll use simple statistical methods
    }

    /**
     * Perform predictive analysis
     */
    private performPredictiveAnalysis(metrics: PerformanceMetrics): void {
        // Simple trend analysis - in production, this would use ML models
        if (this.lastMetrics) {
            const trends = this.calculateTrends(this.lastMetrics, metrics);
            
            if (trends.errorRateTrend === 'increasing' && trends.errorRateChange > 50) {
                automatedWorkflowEngine.emit('alert', {
                    id: `predictive-error-rate-${Date.now()}`,
                    type: 'predictive',
                    severity: 'medium',
                    message: `Predictive analysis: Error rate trending upward (${trends.errorRateChange.toFixed(1)}% increase)`,
                    timestamp: new Date(),
                    metrics
                });
            }
        }
    }

    /**
     * Calculate trends between metrics
     */
    private calculateTrends(previous: PerformanceMetrics, current: PerformanceMetrics): any {
        const errorRateChange = ((current.errorRate - previous.errorRate) / previous.errorRate) * 100;
        const responseTimeChange = ((current.responseTime - previous.responseTime) / previous.responseTime) * 100;
        
        return {
            errorRateTrend: errorRateChange > 5 ? 'increasing' : errorRateChange < -5 ? 'decreasing' : 'stable',
            errorRateChange: Math.abs(errorRateChange),
            responseTimeTrend: responseTimeChange > 10 ? 'increasing' : responseTimeChange < -10 ? 'decreasing' : 'stable',
            responseTimeChange: Math.abs(responseTimeChange)
        };
    }

    /**
     * Evaluate auto-scaling decisions
     */
    private evaluateAutoScaling(resourceMetrics: ResourceMetrics): void {
        const cpuUsage = resourceMetrics.cpu.usage;
        const memoryUsageRatio = resourceMetrics.memory.used / resourceMetrics.memory.total;
        
        if (cpuUsage > 0.8 || memoryUsageRatio > 0.85) {
            console.log('📈 Auto-scaling: High resource usage detected, considering scale-up');
            // Implement scaling logic here
        } else if (cpuUsage < 0.3 && memoryUsageRatio < 0.5) {
            console.log('📉 Auto-scaling: Low resource usage detected, considering scale-down');
            // Implement scaling logic here
        }
    }

    /**
     * Event handlers
     */
    private handleTriggerExecution(data: any): void {
        console.log(`⚡ Trigger executed: ${data.triggerId} - ${data.trigger.name}`);
        
        // Log to monitoring system
        this.emit('triggerExecutionLogged', {
            triggerId: data.triggerId,
            triggerName: data.trigger.name,
            timestamp: new Date(),
            successful: data.successful,
            failed: data.failed
        });
    }

    private handleSystemAlert(alert: SystemAlert): void {
        console.log(`🚨 System alert: ${alert.severity} - ${alert.message}`);
        
        // Forward to monitoring clients
        if (this.config.enableRealTimeMonitoring) {
            automationMonitor.emit('systemAlert', alert);
        }
    }

    private handleSystemError(error: any): void {
        console.error('❌ System error in automation engine:', error);
        
        // Create critical alert
        const criticalAlert: SystemAlert = {
            id: `system-error-${Date.now()}`,
            type: 'system',
            severity: 'critical',
            message: `System error in automation engine: ${error.message || error.type}`,
            timestamp: new Date(),
            error
        };
        
        automatedWorkflowEngine.emit('alert', criticalAlert);
    }

    private handleHealthStatusChange(healthData: any): void {
        console.log(`🏥 Health status changed: ${healthData.overall.status} (${healthData.overall.score}%)`);
        
        // Broadcast to monitoring clients
        if (this.config.enableRealTimeMonitoring) {
            automationMonitor.broadcastToChannel('health', {
                type: 'health-status-update',
                data: healthData,
                timestamp: new Date().toISOString()
            });
        }
    }

    private handlePerformanceAnalysis(data: any): void {
        console.log(`📊 Performance analysis completed: Health Score ${data.analysis.healthScore}`);
        
        // Forward to monitoring clients
        if (this.config.enableRealTimeMonitoring) {
            automationMonitor.broadcastToChannel('performance', {
                type: 'performance-analysis',
                data: data.analysis,
                timestamp: new Date().toISOString()
            });
        }

        // Create alerts for poor performance
        if (data.analysis.healthScore < 60) {
            const performanceAlert: SystemAlert = {
                id: `performance-${Date.now()}`,
                type: 'performance',
                severity: 'medium',
                message: `Performance health score dropped to ${data.analysis.healthScore}`,
                timestamp: new Date(),
                analysis: data.analysis
            };
            
            automatedWorkflowEngine.emit('alert', performanceAlert);
        }
    }

    private handleOptimizationCompleted(data: any): void {
        console.log(`✅ Optimization completed: ${data.recommendation.title} - Duration: ${data.duration}ms`);
        
        // Broadcast to monitoring clients
        if (this.config.enableRealTimeMonitoring) {
            automationMonitor.broadcastToChannel('optimization', {
                type: 'optimization-completed',
                data: {
                    optimizationId: data.optimizationId,
                    title: data.recommendation.title,
                    type: data.recommendation.type,
                    duration: data.duration,
                    result: data.result
                },
                timestamp: new Date().toISOString()
            });
        }
    }

    private handleOptimizationFailed(data: any): void {
        console.error(`❌ Optimization failed: ${data.recommendation.title} - ${data.error.message}`);
        
        // Create critical alert for failed optimization
        const optimizationAlert: SystemAlert = {
            id: `optimization-failed-${Date.now()}`,
            type: 'optimization',
            severity: 'high',
            message: `Optimization failed: ${data.recommendation.title} - ${data.error.message}`,
            timestamp: new Date(),
            optimizationId: data.optimizationId,
            error: data.error
        };
        
        automatedWorkflowEngine.emit('alert', optimizationAlert);
        
        // Broadcast to monitoring clients
        if (this.config.enableRealTimeMonitoring) {
            automationMonitor.broadcastToChannel('optimization', {
                type: 'optimization-failed',
                data: {
                    optimizationId: data.optimizationId,
                    title: data.recommendation.title,
                    error: data.error.message
                },
                timestamp: new Date().toISOString()
            });
        }
    }

    /**
     * Cache event handlers
     */
    private handleCacheMetrics(metrics: any): void {
        console.log('📊 Cache metrics received:', {
            hitRate: (metrics.hitRate * 100).toFixed(1) + '%',
            avgResponseTime: metrics.averageOperationTime?.toFixed(2) + 'ms'
        });

        // Check cache performance thresholds
        if (metrics.hitRate < this.config.alertThresholds.cacheHitRate) {
            this.handleCacheAlert({
                type: 'cache-performance',
                severity: 'warning',
                message: `Cache hit rate below threshold: ${(metrics.hitRate * 100).toFixed(1)}% < ${(this.config.alertThresholds.cacheHitRate * 100)}%`,
                data: metrics,
                timestamp: new Date()
            });
        }

        if (metrics.averageOperationTime > this.config.alertThresholds.cacheResponseTime) {
            this.handleCacheAlert({
                type: 'cache-performance',
                severity: 'warning',
                message: `Cache response time above threshold: ${metrics.averageOperationTime.toFixed(2)}ms > ${this.config.alertThresholds.cacheResponseTime}ms`,
                data: metrics,
                timestamp: new Date()
            });
        }

        // Forward to automation monitoring if enabled
        if (this.config.enableRealTimeMonitoring) {
            automationMonitor.broadcastToChannel('performance', {
                type: 'cache-metrics-update',
                data: metrics,
                timestamp: new Date().toISOString()
            });
        }
    }

    private handleCacheAnalytics(analytics: any): void {
        console.log('📈 Cache analytics generated:', {
            hotKeys: analytics.hotKeys?.length || 0,
            coldKeys: analytics.coldKeys?.length || 0,
            recommendations: analytics.recommendations?.length || 0
        });

        // Process recommendations
        if (analytics.recommendations && analytics.recommendations.length > 0) {
            analytics.recommendations.forEach(recommendation => {
                if (recommendation.priority === 'high') {
                    this.handleCacheAlert({
                        type: 'cache-optimization',
                        severity: 'medium',
                        message: `High priority cache optimization: ${recommendation.type}`,
                        data: recommendation,
                        timestamp: new Date()
                    });
                }
            });
        }

        // Forward to monitoring
        if (this.config.enableRealTimeMonitoring) {
            automationMonitor.broadcastToChannel('analytics', {
                type: 'cache-analytics-update',
                data: analytics,
                timestamp: new Date().toISOString()
            });
        }
    }

    private handleCacheAlert(alert: SystemAlert): void {
        console.log(`🚨 Cache alert: ${alert.severity} - ${alert.message}`);

        // Forward to automation engine for potential workflow triggers
        automatedWorkflowEngine.emit('alert', alert);

        // Forward to monitoring clients
        if (this.config.enableRealTimeMonitoring) {
            automationMonitor.broadcastToChannel('alerts', {
                type: 'cache-alert',
                data: alert,
                timestamp: new Date().toISOString()
            });
        }
    }

    /**
     * Get current service status
     */
    getStatus(): any {
        const baseStatus = {
            isRunning: this.isRunning,
            config: this.config,
            lastMetrics: this.lastMetrics,
            serviceHealth: Object.fromEntries(this.serviceHealthCache),
            automationEngine: automatedWorkflowEngine.getTriggerStats(),
            monitoring: automationMonitor.getServerStats(),
            uptime: process.uptime()
        };

        // Add cache status if enabled
        if (this.config.enableAdvancedCaching) {
            baseStatus.caching = {
                status: advancedCacheManager.getStatus(),
                monitoring: this.config.enableCacheMonitoring ? cacheMonitoringService.getServerStats() : null
            };
        }

        return baseStatus;
    }
}

// Export singleton instance
export const automationIntegration = new AutomationIntegrationService({
    metricsInterval: 5000,
    healthCheckInterval: 30000,
    resourceMonitoringInterval: 10000,
    enableRealTimeMonitoring: true,
    enableAutomaticScaling: false,
    enablePredictiveAnalysis: true,
    alertThresholds: {
        errorRate: 0.05,
        responseTime: 5000,
        cpuUsage: 0.8,
        memoryUsage: 0.85
    }
});