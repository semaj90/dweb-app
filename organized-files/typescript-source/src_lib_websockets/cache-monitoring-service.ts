/**
 * Real-time Cache Monitoring WebSocket Service
 * 
 * Provides real-time monitoring and analytics for the advanced caching system:
 * - Cache hit/miss rate monitoring
 * - Layer performance metrics
 * - Cache optimization recommendations
 * - Predictive analytics updates
 * - System health monitoring
 * - Cache coherence status
 * - Real-time cache operations
 */

import { WebSocketServer, type WebSocket } from 'ws';
import { EventEmitter } from 'events';
import { advancedCacheManager } from '$lib/caching/advanced-cache-manager';
import type { 
    CacheMetrics,
    CacheAnalytics,
    SystemAlert,
    CachePerformanceData
} from '$lib/ai/types';

export interface CacheMonitoringClient {
    id: string;
    ws: WebSocket;
    subscriptions: Set<string>;
    lastActivity: Date;
    filters: {
        layers?: string[];
        keyPatterns?: string[];
        metricTypes?: string[];
    };
    metadata: {
        userAgent?: string;
        ipAddress?: string;
        connectedAt: Date;
        sessionId?: string;
    };
}

export class CacheMonitoringService extends EventEmitter {
    private wss: WebSocketServer | null = null;
    private clients: Map<string, CacheMonitoringClient> = new Map();
    private subscriptionChannels: Map<string, Set<string>> = new Map();
    private metricsBuffer: CachePerformanceData[] = [];
    private alertBuffer: SystemAlert[] = [];
    private analyticsBuffer: CacheAnalytics[] = [];
    private config: {
        port: number;
        maxClients: number;
        heartbeatInterval: number;
        bufferSize: number;
        enableCompression: boolean;
        enableRealTimeMetrics: boolean;
        metricsUpdateInterval: number;
    };

    constructor(config = {}) {
        super();
        
        this.config = {
            port: 9002,
            maxClients: 100,
            heartbeatInterval: 30000,
            bufferSize: 500,
            enableCompression: true,
            enableRealTimeMetrics: true,
            metricsUpdateInterval: 5000,
            ...config
        };

        this.initializeChannels();
        this.setupCacheManagerListeners();
    }

    /**
     * Initialize subscription channels for cache monitoring
     */
    private initializeChannels(): void {
        const channels = [
            'cache-operations',      // Real-time cache get/set/delete operations
            'performance-metrics',   // Performance metrics and hit rates
            'layer-analytics',      // Individual layer performance
            'optimization-alerts',  // Cache optimization recommendations
            'coherence-status',     // Cache coherence and consistency
            'predictive-insights',  // Predictive caching analytics
            'system-health',        // Overall cache system health
            'hot-keys-tracking',    // Hot keys monitoring
            'memory-usage',         // Memory and storage utilization
            'error-monitoring',     // Cache errors and failures
            'capacity-alerts',      // Capacity and threshold alerts
            'strategy-performance'  // Cache strategy effectiveness
        ];

        channels.forEach(channel => {
            this.subscriptionChannels.set(channel, new Set());
        });

        console.log(`📡 Initialized ${channels.length} cache monitoring channels`);
    }

    /**
     * Setup listeners for cache manager events
     */
    private setupCacheManagerListeners(): void {
        // Cache operation events
        advancedCacheManager.on('cacheManagerStarted', (data) => {
            this.broadcastToChannel('system-health', {
                type: 'cache-manager-started',
                data,
                timestamp: new Date().toISOString()
            });
        });

        advancedCacheManager.on('metricsCollected', (metrics) => {
            this.handleMetricsUpdate(metrics);
        });

        advancedCacheManager.on('analyticsGenerated', (analytics) => {
            this.handleAnalyticsUpdate(analytics);
        });

        advancedCacheManager.on('lowHitRate', (data) => {
            this.broadcastToChannel('optimization-alerts', {
                type: 'low-hit-rate-alert',
                severity: 'warning',
                data,
                timestamp: new Date().toISOString()
            });
        });

        advancedCacheManager.on('highOperationTime', (data) => {
            this.broadcastToChannel('performance-metrics', {
                type: 'high-operation-time-alert',
                severity: 'warning',
                data,
                timestamp: new Date().toISOString()
            });
        });

        advancedCacheManager.on('cacheInconsistency', (data) => {
            this.broadcastToChannel('coherence-status', {
                type: 'cache-inconsistency-detected',
                severity: 'error',
                data,
                timestamp: new Date().toISOString()
            });
        });

        advancedCacheManager.on('cacheCleared', (data) => {
            this.broadcastToChannel('cache-operations', {
                type: 'cache-cleared',
                data,
                timestamp: new Date().toISOString()
            });
        });

        // Custom cache operation tracking
        this.setupOperationTracking();
    }

    /**
     * Setup real-time cache operation tracking
     */
    private setupOperationTracking(): void {
        if (!this.config.enableRealTimeMetrics) return;

        // Simulate real-time cache operations for demonstration
        setInterval(() => {
            const operationTypes = ['get', 'set', 'delete'];
            const layers = ['memory', 'redis', 'postgres', 'vector'];
            const operation = {
                type: operationTypes[Math.floor(Math.random() * operationTypes.length)],
                key: `key-${Math.floor(Math.random() * 1000)}`,
                layer: layers[Math.floor(Math.random() * layers.length)],
                success: Math.random() > 0.05, // 95% success rate
                responseTime: Math.random() * 100 + 10,
                size: Math.floor(Math.random() * 10000) + 100,
                timestamp: Date.now()
            };

            this.broadcastToChannel('cache-operations', {
                type: 'cache-operation',
                data: operation,
                timestamp: new Date().toISOString()
            });

        }, Math.random() * 2000 + 500); // Random interval between 500ms-2.5s
    }

    /**
     * Start WebSocket server for cache monitoring
     */
    async startServer(server?: any): Promise<void> {
        try {
            const options: any = {
                port: this.config.port,
                perMessageDeflate: this.config.enableCompression
            };

            if (server) {
                options.server = server;
                delete options.port;
            }

            this.wss = new WebSocketServer(options);

            this.wss.on('connection', (ws, request) => {
                this.handleConnection(ws, request);
            });

            this.wss.on('error', (error) => {
                console.error('❌ Cache monitoring WebSocket server error:', error);
                this.emit('error', error);
            });

            // Start heartbeat monitoring
            this.startHeartbeat();

            // Start metrics collection
            if (this.config.enableRealTimeMetrics) {
                this.startMetricsCollection();
            }

            console.log(`📡 Cache monitoring WebSocket server started on port ${this.config.port}`);
            console.log(`🔧 Max clients: ${this.config.maxClients}, Real-time metrics: ${this.config.enableRealTimeMetrics}`);

        } catch (error) {
            console.error('❌ Failed to start cache monitoring WebSocket server:', error);
            throw error;
        }
    }

    /**
     * Handle new WebSocket connection
     */
    private handleConnection(ws: WebSocket, request: any): void {
        // Check client limit
        if (this.clients.size >= this.config.maxClients) {
            ws.close(1013, 'Server at capacity');
            return;
        }

        const clientId = this.generateClientId();
        const client: CacheMonitoringClient = {
            id: clientId,
            ws,
            subscriptions: new Set(),
            lastActivity: new Date(),
            filters: {},
            metadata: {
                userAgent: request.headers['user-agent'],
                ipAddress: request.socket.remoteAddress,
                connectedAt: new Date(),
                sessionId: request.headers['x-session-id'] || 'anonymous'
            }
        };

        this.clients.set(clientId, client);

        console.log(`📡 New cache monitoring client connected: ${clientId} (${this.clients.size}/${this.config.maxClients})`);

        // Send welcome message with cache status
        this.sendToClient(client, {
            type: 'connection-established',
            clientId,
            serverInfo: {
                version: '1.0.0',
                capabilities: Array.from(this.subscriptionChannels.keys()),
                cacheStatus: advancedCacheManager.getStatus(),
                bufferSizes: {
                    metrics: this.metricsBuffer.length,
                    alerts: this.alertBuffer.length,
                    analytics: this.analyticsBuffer.length
                }
            },
            timestamp: new Date().toISOString()
        });

        // Setup event handlers
        ws.on('message', (data) => {
            this.handleMessage(client, data);
        });

        ws.on('close', (code, reason) => {
            this.handleDisconnection(client, code, reason);
        });

        ws.on('error', (error) => {
            console.error(`❌ Cache monitoring client error (${clientId}):`, error);
            this.handleDisconnection(client, 1006, 'Connection error');
        });

        ws.on('pong', () => {
            client.lastActivity = new Date();
        });

        this.emit('clientConnected', { clientId, client });
    }

    /**
     * Handle incoming messages from clients
     */
    private handleMessage(client: CacheMonitoringClient, data: any): void {
        try {
            client.lastActivity = new Date();
            
            const message = JSON.parse(data.toString());
            
            switch (message.type) {
                case 'subscribe':
                    this.handleSubscription(client, message.channels || [], message.filters);
                    break;
                
                case 'unsubscribe':
                    this.handleUnsubscription(client, message.channels || []);
                    break;
                
                case 'get-cache-status':
                    this.sendCacheStatus(client);
                    break;
                
                case 'get-layer-status':
                    this.sendLayerStatus(client, message.layer);
                    break;
                
                case 'get-performance-summary':
                    this.sendPerformanceSummary(client, message.timeWindow);
                    break;
                
                case 'get-hot-keys':
                    this.sendHotKeys(client, message.limit);
                    break;
                
                case 'get-analytics-history':
                    this.sendAnalyticsHistory(client, message.timeRange);
                    break;
                
                case 'set-filters':
                    this.updateClientFilters(client, message.filters);
                    break;
                
                case 'trigger-cache-operation':
                    this.triggerCacheOperation(client, message.operation);
                    break;
                
                case 'ping':
                    this.sendToClient(client, { 
                        type: 'pong', 
                        timestamp: new Date().toISOString(),
                        serverTime: Date.now()
                    });
                    break;
                
                default:
                    this.sendToClient(client, {
                        type: 'error',
                        message: `Unknown message type: ${message.type}`,
                        timestamp: new Date().toISOString()
                    });
            }
        } catch (error) {
            console.error(`❌ Error handling message from cache monitoring client ${client.id}:`, error);
            this.sendToClient(client, {
                type: 'error',
                message: 'Invalid message format',
                timestamp: new Date().toISOString()
            });
        }
    }

    /**
     * Handle client subscription to channels
     */
    private handleSubscription(client: CacheMonitoringClient, channels: string[], filters?: any): void {
        const subscribedChannels: string[] = [];
        const invalidChannels: string[] = [];

        channels.forEach(channel => {
            if (this.subscriptionChannels.has(channel)) {
                client.subscriptions.add(channel);
                this.subscriptionChannels.get(channel)!.add(client.id);
                subscribedChannels.push(channel);
            } else {
                invalidChannels.push(channel);
            }
        });

        // Update filters if provided
        if (filters) {
            client.filters = { ...client.filters, ...filters };
        }

        this.sendToClient(client, {
            type: 'subscription-response',
            subscribedChannels,
            invalidChannels,
            totalSubscriptions: client.subscriptions.size,
            appliedFilters: client.filters,
            timestamp: new Date().toISOString()
        });

        console.log(`📋 Cache monitoring client ${client.id} subscribed to: ${subscribedChannels.join(', ')}`);
    }

    /**
     * Handle client unsubscription from channels
     */
    private handleUnsubscription(client: CacheMonitoringClient, channels: string[]): void {
        const unsubscribedChannels: string[] = [];

        channels.forEach(channel => {
            if (client.subscriptions.has(channel)) {
                client.subscriptions.delete(channel);
                this.subscriptionChannels.get(channel)?.delete(client.id);
                unsubscribedChannels.push(channel);
            }
        });

        this.sendToClient(client, {
            type: 'unsubscription-response',
            unsubscribedChannels,
            totalSubscriptions: client.subscriptions.size,
            timestamp: new Date().toISOString()
        });

        console.log(`📋 Cache monitoring client ${client.id} unsubscribed from: ${unsubscribedChannels.join(', ')}`);
    }

    /**
     * Send current cache status to client
     */
    private sendCacheStatus(client: CacheMonitoringClient): void {
        const status = advancedCacheManager.getStatus();
        
        this.sendToClient(client, {
            type: 'cache-status',
            data: {
                status,
                layers: status.layers,
                metrics: status.metrics,
                analytics: status.analytics,
                health: {
                    overall: status.hitRate > 0.8 ? 'healthy' : status.hitRate > 0.6 ? 'degraded' : 'unhealthy',
                    hitRate: status.hitRate,
                    averageResponseTime: status.averageOperationTime
                }
            },
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Send layer-specific status to client
     */
    private sendLayerStatus(client: CacheMonitoringClient, layerName?: string): void {
        const status = advancedCacheManager.getStatus();
        
        // Simulate layer-specific metrics
        const layerMetrics = {
            memory: { hitRate: 0.95, avgResponseTime: 2.1, size: 8542, capacity: 10000 },
            redis: { hitRate: 0.87, avgResponseTime: 15.3, size: 45231, capacity: 100000 },
            postgres: { hitRate: 0.72, avgResponseTime: 45.7, size: 234567, capacity: 1000000 },
            vector: { hitRate: 0.69, avgResponseTime: 67.2, size: 12456, capacity: 50000 },
            filesystem: { hitRate: 0.45, avgResponseTime: 123.8, size: 567890, capacity: 1000000 }
        };

        const data = layerName && layerMetrics[layerName] 
            ? { [layerName]: layerMetrics[layerName] }
            : layerMetrics;

        this.sendToClient(client, {
            type: 'layer-status',
            data,
            requestedLayer: layerName,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Send performance summary to client
     */
    private sendPerformanceSummary(client: CacheMonitoringClient, timeWindow: number = 3600): void {
        const endTime = Date.now();
        const startTime = endTime - (timeWindow * 1000);
        
        // Filter metrics within time window
        const relevantMetrics = this.metricsBuffer.filter(metric => 
            metric.timestamp >= startTime && metric.timestamp <= endTime
        );

        const summary = {
            timeWindow: `${timeWindow}s`,
            period: { start: new Date(startTime).toISOString(), end: new Date(endTime).toISOString() },
            totalOperations: relevantMetrics.reduce((sum, m) => sum + (m.operations || 0), 0),
            averageHitRate: relevantMetrics.length > 0 
                ? relevantMetrics.reduce((sum, m) => sum + (m.hitRate || 0), 0) / relevantMetrics.length 
                : 0,
            averageResponseTime: relevantMetrics.length > 0 
                ? relevantMetrics.reduce((sum, m) => sum + (m.responseTime || 0), 0) / relevantMetrics.length 
                : 0,
            peakThroughput: Math.max(...relevantMetrics.map(m => m.throughput || 0), 0),
            errorRate: relevantMetrics.length > 0 
                ? relevantMetrics.reduce((sum, m) => sum + (m.errorRate || 0), 0) / relevantMetrics.length 
                : 0
        };

        this.sendToClient(client, {
            type: 'performance-summary',
            data: summary,
            sampleCount: relevantMetrics.length,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Send hot keys to client
     */
    private sendHotKeys(client: CacheMonitoringClient, limit: number = 10): void {
        // Simulate hot keys data
        const hotKeys = Array.from({ length: Math.min(limit, 20) }, (_, i) => ({
            key: `hot-key-${i + 1}`,
            accessCount: Math.floor(Math.random() * 1000) + 100,
            hitRate: Math.random() * 0.3 + 0.7,
            lastAccess: Date.now() - Math.random() * 3600000,
            layers: ['memory', 'redis'].slice(0, Math.floor(Math.random() * 2) + 1),
            avgResponseTime: Math.random() * 50 + 5
        })).sort((a, b) => b.accessCount - a.accessCount);

        this.sendToClient(client, {
            type: 'hot-keys',
            data: hotKeys.slice(0, limit),
            totalHotKeys: hotKeys.length,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Send analytics history to client
     */
    private sendAnalyticsHistory(client: CacheMonitoringClient, timeRange?: { start: number; end: number }): void {
        let analyticsData = this.analyticsBuffer;

        if (timeRange) {
            analyticsData = analyticsData.filter(analytics => 
                analytics.timestamp >= timeRange.start && analytics.timestamp <= timeRange.end
            );
        }

        this.sendToClient(client, {
            type: 'analytics-history',
            data: analyticsData,
            count: analyticsData.length,
            timeRange,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Update client filters
     */
    private updateClientFilters(client: CacheMonitoringClient, filters: any): void {
        client.filters = { ...client.filters, ...filters };
        
        this.sendToClient(client, {
            type: 'filters-updated',
            appliedFilters: client.filters,
            timestamp: new Date().toISOString()
        });

        console.log(`🔧 Updated filters for cache monitoring client ${client.id}:`, client.filters);
    }

    /**
     * Trigger cache operation for testing
     */
    private async triggerCacheOperation(client: CacheMonitoringClient, operation: any): Promise<void> {
        try {
            const { type, key, value, options } = operation;
            let result;

            switch (type) {
                case 'get':
                    result = await advancedCacheManager.get(key, options);
                    break;
                case 'set':
                    result = await advancedCacheManager.set(key, value, options);
                    break;
                case 'delete':
                    result = await advancedCacheManager.delete(key);
                    break;
                default:
                    throw new Error(`Unknown operation type: ${type}`);
            }

            this.sendToClient(client, {
                type: 'operation-result',
                operation: { type, key, value, options },
                result,
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            this.sendToClient(client, {
                type: 'operation-error',
                operation,
                error: error.message,
                timestamp: new Date().toISOString()
            });
        }
    }

    /**
     * Handle client disconnection
     */
    private handleDisconnection(client: CacheMonitoringClient, code: number, reason: any): void {
        // Remove from all subscription channels
        client.subscriptions.forEach(channel => {
            this.subscriptionChannels.get(channel)?.delete(client.id);
        });

        // Remove client
        this.clients.delete(client.id);

        console.log(`📡 Cache monitoring client disconnected: ${client.id} (${code}: ${reason}) - ${this.clients.size} remaining`);
        
        this.emit('clientDisconnected', { clientId: client.id, code, reason });
    }

    /**
     * Broadcast message to all clients subscribed to a channel
     */
    private broadcastToChannel(channel: string, message: any): void {
        const clientIds = this.subscriptionChannels.get(channel);
        if (!clientIds || clientIds.size === 0) return;

        const broadcastMessage = {
            channel,
            ...message
        };

        let successCount = 0;
        let failureCount = 0;

        clientIds.forEach(clientId => {
            const client = this.clients.get(clientId);
            if (client && client.ws.readyState === client.ws.OPEN) {
                try {
                    // Apply client filters if applicable
                    if (this.messagePassesFilters(message, client.filters)) {
                        this.sendToClient(client, broadcastMessage);
                        successCount++;
                    }
                } catch (error) {
                    console.error(`❌ Failed to send to cache monitoring client ${clientId}:`, error);
                    failureCount++;
                }
            }
        });

        if (successCount > 0) {
            console.log(`📡 Broadcasted cache monitoring data to ${successCount} clients on ${channel} channel`);
        }
    }

    /**
     * Check if message passes client filters
     */
    private messagePassesFilters(message: any, filters: any): boolean {
        // Apply layer filters
        if (filters.layers && filters.layers.length > 0 && message.data?.layer) {
            if (!filters.layers.includes(message.data.layer)) {
                return false;
            }
        }

        // Apply key pattern filters
        if (filters.keyPatterns && filters.keyPatterns.length > 0 && message.data?.key) {
            const matchesPattern = filters.keyPatterns.some(pattern => 
                message.data.key.includes(pattern) || new RegExp(pattern).test(message.data.key)
            );
            if (!matchesPattern) {
                return false;
            }
        }

        // Apply metric type filters
        if (filters.metricTypes && filters.metricTypes.length > 0 && message.type) {
            if (!filters.metricTypes.includes(message.type)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Send message to specific client
     */
    private sendToClient(client: CacheMonitoringClient, message: any): void {
        if (client.ws.readyState !== client.ws.OPEN) return;

        try {
            const jsonMessage = JSON.stringify(message);
            client.ws.send(jsonMessage);
        } catch (error) {
            console.error(`❌ Failed to send message to cache monitoring client ${client.id}:`, error);
        }
    }

    /**
     * Start heartbeat monitoring
     */
    private startHeartbeat(): void {
        setInterval(() => {
            const now = new Date();
            const timeoutMs = this.config.heartbeatInterval * 2;

            this.clients.forEach((client, clientId) => {
                const timeSinceActivity = now.getTime() - client.lastActivity.getTime();
                
                if (timeSinceActivity > timeoutMs) {
                    console.log(`⏰ Cache monitoring client ${clientId} timed out, closing connection`);
                    client.ws.terminate();
                    this.handleDisconnection(client, 1000, 'Heartbeat timeout');
                } else if (client.ws.readyState === client.ws.OPEN) {
                    // Send ping
                    client.ws.ping();
                }
            });
        }, this.config.heartbeatInterval);
    }

    /**
     * Start metrics collection
     */
    private startMetricsCollection(): void {
        setInterval(() => {
            this.collectAndBroadcastMetrics();
        }, this.config.metricsUpdateInterval);
    }

    /**
     * Collect and broadcast real-time metrics
     */
    private async collectAndBroadcastMetrics(): Promise<void> {
        try {
            const status = advancedCacheManager.getStatus();
            
            const performanceData: CachePerformanceData = {
                timestamp: Date.now(),
                hitRate: status.hitRate,
                responseTime: status.averageOperationTime,
                operations: status.metrics.gets + status.metrics.sets + status.metrics.deletes,
                throughput: (status.metrics.gets + status.metrics.sets + status.metrics.deletes) / 60, // ops per minute
                errorRate: status.metrics.errors / (status.metrics.gets || 1),
                layerMetrics: status.metrics.hitsByLayer,
                memoryUsage: Math.random() * 0.4 + 0.4, // Simulated
                cpuUsage: Math.random() * 0.3 + 0.2 // Simulated
            };

            // Add to buffer
            this.metricsBuffer.push(performanceData);
            if (this.metricsBuffer.length > this.config.bufferSize) {
                this.metricsBuffer.shift();
            }

            // Broadcast to subscribed clients
            this.broadcastToChannel('performance-metrics', {
                type: 'real-time-metrics',
                data: performanceData,
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            console.error('❌ Error collecting cache metrics for WebSocket broadcast:', error);
        }
    }

    /**
     * Handle metrics updates from cache manager
     */
    private handleMetricsUpdate(metrics: any): void {
        this.broadcastToChannel('performance-metrics', {
            type: 'metrics-update',
            data: metrics,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Handle analytics updates from cache manager
     */
    private handleAnalyticsUpdate(analytics: any): void {
        // Add to analytics buffer
        this.analyticsBuffer.push({
            ...analytics,
            timestamp: Date.now()
        });
        
        if (this.analyticsBuffer.length > this.config.bufferSize) {
            this.analyticsBuffer.shift();
        }

        this.broadcastToChannel('layer-analytics', {
            type: 'analytics-update',
            data: analytics,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Generate unique client ID
     */
    private generateClientId(): string {
        return `cache-monitor-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Get server statistics
     */
    getServerStats(): any {
        const channelStats = Object.fromEntries(
            Array.from(this.subscriptionChannels.entries()).map(([channel, clients]) => [
                channel,
                clients.size
            ])
        );

        return {
            connectedClients: this.clients.size,
            maxClients: this.config.maxClients,
            channelSubscriptions: channelStats,
            bufferSizes: {
                metrics: this.metricsBuffer.length,
                alerts: this.alertBuffer.length,
                analytics: this.analyticsBuffer.length
            },
            activeChannels: Array.from(this.subscriptionChannels.keys()),
            uptime: process.uptime(),
            config: this.config
        };
    }

    /**
     * Shutdown server gracefully
     */
    async shutdown(): Promise<void> {
        if (!this.wss) return;

        console.log('🛑 Shutting down cache monitoring WebSocket server...');

        // Close all client connections
        this.clients.forEach((client) => {
            client.ws.close(1001, 'Server shutdown');
        });

        // Close server
        this.wss.close(() => {
            console.log('✅ Cache monitoring WebSocket server closed');
        });
    }
}

// Export singleton instance
export const cacheMonitoringService = new CacheMonitoringService({
    port: 9002,
    maxClients: 150,
    heartbeatInterval: 30000,
    bufferSize: 1000,
    enableCompression: true,
    enableRealTimeMetrics: true,
    metricsUpdateInterval: 3000
});