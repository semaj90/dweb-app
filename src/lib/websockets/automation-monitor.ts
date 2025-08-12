/**
 * Real-time Automation Monitoring WebSocket Service
 * 
 * Provides real-time updates for:
 * - Trigger executions and status changes
 * - Workflow progress and completion
 * - System alerts and notifications
 * - Performance metrics and anomalies
 * - Health status changes
 */

import { WebSocketServer, type WebSocket } from 'ws';
import { EventEmitter } from 'events';
import { automatedWorkflowEngine } from '$lib/orchestration/automated-workflow-triggers';
import type { 
    TriggerEvent,
    WorkflowExecution,
    SystemAlert,
    PerformanceMetrics 
} from '$lib/ai/types';

export interface MonitoringClient {
    id: string;
    ws: WebSocket;
    subscriptions: Set<string>;
    lastActivity: Date;
    metadata: {
        userAgent?: string;
        ipAddress?: string;
        connectedAt: Date;
    };
}

export class AutomationMonitoringService extends EventEmitter {
    private wss: WebSocketServer | null = null;
    private clients: Map<string, MonitoringClient> = new Map();
    private subscriptionChannels: Map<string, Set<string>> = new Map();
    private metricsBuffer: PerformanceMetrics[] = [];
    private alertBuffer: SystemAlert[] = [];
    private config: {
        port: number;
        maxClients: number;
        heartbeatInterval: number;
        bufferSize: number;
        enableCompression: boolean;
    };

    constructor(config = {}) {
        super();
        
        this.config = {
            port: 9001,
            maxClients: 50,
            heartbeatInterval: 30000,
            bufferSize: 100,
            enableCompression: true,
            ...config
        };

        this.initializeChannels();
        this.setupAutomationEngineListeners();
    }

    /**
     * Initialize subscription channels
     */
    private initializeChannels(): void {
        const channels = [
            'triggers',
            'workflows', 
            'alerts',
            'metrics',
            'health',
            'system-events',
            'performance',
            'anomalies',
            'optimization'
        ];

        channels.forEach(channel => {
            this.subscriptionChannels.set(channel, new Set());
        });
    }

    /**
     * Setup listeners for automation engine events
     */
    private setupAutomationEngineListeners(): void {
        // Trigger events
        automatedWorkflowEngine.on('triggerRegistered', (data) => {
            this.broadcastToChannel('triggers', {
                type: 'trigger-registered',
                data,
                timestamp: new Date().toISOString()
            });
        });

        automatedWorkflowEngine.on('triggerExecuted', (data) => {
            this.broadcastToChannel('triggers', {
                type: 'trigger-executed',
                data,
                timestamp: new Date().toISOString()
            });
            
            // Also broadcast to workflows channel if workflow was started
            this.broadcastToChannel('workflows', {
                type: 'workflow-triggered',
                data,
                timestamp: new Date().toISOString()
            });
        });

        automatedWorkflowEngine.on('triggerError', (data) => {
            this.broadcastToChannel('triggers', {
                type: 'trigger-error',
                data,
                timestamp: new Date().toISOString()
            });
            
            // Send as alert
            this.broadcastToChannel('alerts', {
                type: 'trigger-alert',
                severity: 'error',
                data,
                timestamp: new Date().toISOString()
            });
        });

        // Alert events
        automatedWorkflowEngine.on('alert', (alert: SystemAlert) => {
            this.alertBuffer.push(alert);
            if (this.alertBuffer.length > this.config.bufferSize) {
                this.alertBuffer.shift();
            }

            this.broadcastToChannel('alerts', {
                type: 'system-alert',
                data: alert,
                timestamp: new Date().toISOString()
            });
        });

        // WebSocket alert events
        automatedWorkflowEngine.on('websocketAlert', (alert: SystemAlert) => {
            this.broadcastToChannel('alerts', {
                type: 'websocket-alert',
                data: alert,
                timestamp: new Date().toISOString()
            });
        });

        // System events
        automatedWorkflowEngine.on('error', (error) => {
            this.broadcastToChannel('system-events', {
                type: 'system-error',
                data: error,
                timestamp: new Date().toISOString()
            });
        });
    }

    /**
     * Start WebSocket server
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
                console.error('❌ WebSocket server error:', error);
                this.emit('error', error);
            });

            // Start heartbeat monitoring
            this.startHeartbeat();

            console.log(`📡 Automation monitoring WebSocket server started on port ${this.config.port}`);
            console.log(`🔧 Max clients: ${this.config.maxClients}, Compression: ${this.config.enableCompression}`);

        } catch (error) {
            console.error('❌ Failed to start WebSocket server:', error);
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
        const client: MonitoringClient = {
            id: clientId,
            ws,
            subscriptions: new Set(),
            lastActivity: new Date(),
            metadata: {
                userAgent: request.headers['user-agent'],
                ipAddress: request.socket.remoteAddress,
                connectedAt: new Date()
            }
        };

        this.clients.set(clientId, client);

        console.log(`📡 New monitoring client connected: ${clientId} (${this.clients.size}/${this.config.maxClients})`);

        // Send welcome message with initial data
        this.sendToClient(client, {
            type: 'connection-established',
            clientId,
            serverInfo: {
                version: '1.0.0',
                capabilities: Array.from(this.subscriptionChannels.keys()),
                bufferSizes: {
                    metrics: this.metricsBuffer.length,
                    alerts: this.alertBuffer.length
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
            console.error(`❌ WebSocket client error (${clientId}):`, error);
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
    private handleMessage(client: MonitoringClient, data: any): void {
        try {
            client.lastActivity = new Date();
            
            const message = JSON.parse(data.toString());
            
            switch (message.type) {
                case 'subscribe':
                    this.handleSubscription(client, message.channels || []);
                    break;
                
                case 'unsubscribe':
                    this.handleUnsubscription(client, message.channels || []);
                    break;
                
                case 'get-status':
                    this.sendStatusUpdate(client);
                    break;
                
                case 'get-history':
                    this.sendHistoryData(client, message.channel, message.limit);
                    break;
                
                case 'ping':
                    this.sendToClient(client, { type: 'pong', timestamp: new Date().toISOString() });
                    break;
                
                default:
                    this.sendToClient(client, {
                        type: 'error',
                        message: `Unknown message type: ${message.type}`,
                        timestamp: new Date().toISOString()
                    });
            }
        } catch (error) {
            console.error(`❌ Error handling message from ${client.id}:`, error);
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
    private handleSubscription(client: MonitoringClient, channels: string[]): void {
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

        this.sendToClient(client, {
            type: 'subscription-response',
            subscribedChannels,
            invalidChannels,
            totalSubscriptions: client.subscriptions.size,
            timestamp: new Date().toISOString()
        });

        console.log(`📋 Client ${client.id} subscribed to: ${subscribedChannels.join(', ')}`);
    }

    /**
     * Handle client unsubscription from channels
     */
    private handleUnsubscription(client: MonitoringClient, channels: string[]): void {
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

        console.log(`📋 Client ${client.id} unsubscribed from: ${unsubscribedChannels.join(', ')}`);
    }

    /**
     * Send current status to client
     */
    private sendStatusUpdate(client: MonitoringClient): void {
        const stats = automatedWorkflowEngine.getTriggerStats();
        const metrics = automatedWorkflowEngine.getSystemMetrics();

        this.sendToClient(client, {
            type: 'status-update',
            data: {
                triggers: stats,
                metrics: metrics,
                server: {
                    connectedClients: this.clients.size,
                    activeChannels: Array.from(this.subscriptionChannels.keys()),
                    bufferSizes: {
                        metrics: this.metricsBuffer.length,
                        alerts: this.alertBuffer.length
                    }
                }
            },
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Send historical data to client
     */
    private sendHistoryData(client: MonitoringClient, channel: string, limit = 50): void {
        let historyData: any[] = [];

        switch (channel) {
            case 'metrics':
                historyData = this.metricsBuffer.slice(-limit);
                break;
            case 'alerts':
                historyData = this.alertBuffer.slice(-limit);
                break;
            default:
                this.sendToClient(client, {
                    type: 'error',
                    message: `No history available for channel: ${channel}`,
                    timestamp: new Date().toISOString()
                });
                return;
        }

        this.sendToClient(client, {
            type: 'history-data',
            channel,
            data: historyData,
            count: historyData.length,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Handle client disconnection
     */
    private handleDisconnection(client: MonitoringClient, code: number, reason: any): void {
        // Remove from all subscription channels
        client.subscriptions.forEach(channel => {
            this.subscriptionChannels.get(channel)?.delete(client.id);
        });

        // Remove client
        this.clients.delete(client.id);

        console.log(`📡 Client disconnected: ${client.id} (${code}: ${reason}) - ${this.clients.size} remaining`);
        
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
                    this.sendToClient(client, broadcastMessage);
                    successCount++;
                } catch (error) {
                    console.error(`❌ Failed to send to client ${clientId}:`, error);
                    failureCount++;
                }
            }
        });

        if (successCount > 0 && this.config.enableCompression) {
            console.log(`📡 Broadcasted to ${successCount} clients on ${channel} channel`);
        }
    }

    /**
     * Send message to specific client
     */
    private sendToClient(client: MonitoringClient, message: any): void {
        if (client.ws.readyState !== client.ws.OPEN) return;

        try {
            const jsonMessage = JSON.stringify(message);
            client.ws.send(jsonMessage);
        } catch (error) {
            console.error(`❌ Failed to send message to client ${client.id}:`, error);
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
                    console.log(`⏰ Client ${clientId} timed out, closing connection`);
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
     * Add metrics to buffer and broadcast
     */
    addMetrics(metrics: PerformanceMetrics): void {
        this.metricsBuffer.push(metrics);
        if (this.metricsBuffer.length > this.config.bufferSize) {
            this.metricsBuffer.shift();
        }

        this.broadcastToChannel('metrics', {
            type: 'metrics-update',
            data: metrics,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Generate unique client ID
     */
    private generateClientId(): string {
        return `client-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
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
                alerts: this.alertBuffer.length
            },
            uptime: process.uptime(),
            config: this.config
        };
    }

    /**
     * Shutdown server gracefully
     */
    async shutdown(): Promise<void> {
        if (!this.wss) return;

        console.log('🛑 Shutting down automation monitoring WebSocket server...');

        // Close all client connections
        this.clients.forEach((client) => {
            client.ws.close(1001, 'Server shutdown');
        });

        // Close server
        this.wss.close(() => {
            console.log('✅ Automation monitoring WebSocket server closed');
        });
    }
}

// Export singleton instance
export const automationMonitor = new AutomationMonitoringService({
    port: 9001,
    maxClients: 100,
    heartbeatInterval: 30000,
    bufferSize: 200,
    enableCompression: true
});