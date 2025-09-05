/**
 * Frontend Cache Monitoring WebSocket Client
 * 
 * Automatically connects to the cache monitoring WebSocket service
 * using dynamic port detection.
 */

import { browser } from '$app/environment';
import { writable } from 'svelte/store';

export interface CacheMonitoringConnection {
    ws: WebSocket | null;
    isConnected: boolean;
    port: number | null;
    url: string | null;
    lastError: string | null;
    reconnectAttempts: number;
    subscriptions: Set<string>;
}

// Connection state store
export const cacheMonitorConnection = writable<CacheMonitoringConnection>({
    ws: null,
    isConnected: false,
    port: null,
    url: null,
    lastError: null,
    reconnectAttempts: 0,
    subscriptions: new Set()
});

class CacheMonitoringClient {
    private connection: CacheMonitoringConnection = {
        ws: null,
        isConnected: false,
        port: null,
        url: null,
        lastError: null,
        reconnectAttempts: 0,
        subscriptions: new Set()
    };

    private maxReconnectAttempts = 10;
    private reconnectDelay = 2000;
    private heartbeatInterval: number | null = null;
    private eventListeners: Map<string, ((data: any) => void)[]> = new Map();

    constructor() {
        if (browser) {
            this.init();
        }
    }

    private async init(): Promise<any> {
        await this.connect();
        this.setupHeartbeat();
    }

    /**
     * Discover the WebSocket port dynamically
     */
    private async discoverPort(): Promise<number | null> {
        try {
            // Method 1: Try to get from environment variable (set by server)
            if (typeof window !== 'undefined' && (window as any).PUBLIC_CACHE_MONITOR_WS_PORT) {
                const port = parseInt((window as any).PUBLIC_CACHE_MONITOR_WS_PORT, 10);
                if (!isNaN(port)) {
                    console.log(`🔍 Found cache monitor WebSocket port from window: ${port}`);
                    return port;
                }
            }

            // Method 2: Try to fetch from API endpoint
            try {
                const response = await fetch('/api/cache/monitor/port');
                if (response.ok) {
                    const data = await response.json();
                    if (data.port) {
                        console.log(`🔍 Found cache monitor WebSocket port from API: ${data.port}`);
                        return data.port;
                    }
                }
            } catch (error: any) {
                console.log('Could not fetch port from API, trying port scanning...');
            }

            // Method 3: Port scanning (try common ports)
            const commonPorts = [9002, 9003, 9004, 9005, 9006];
            for (const port of commonPorts) {
                if (await this.testPort(port)) {
                    console.log(`🔍 Found cache monitor WebSocket port via scanning: ${port}`);
                    return port;
                }
            }

            return null;
        } catch (error: any) {
            console.error('Error discovering WebSocket port:', error);
            return null;
        }
    }

    /**
     * Test if a WebSocket port is available
     */
    private async testPort(port: number): Promise<boolean> {
        return new Promise((resolve) => {
            const ws = new WebSocket(`ws://localhost:${port}`);
            const timeout = setTimeout(() => {
                ws.close();
                resolve(false);
            }, 1000);

            ws.onopen = () => {
                clearTimeout(timeout);
                ws.close();
                resolve(true);
            };

            ws.onerror = () => {
                clearTimeout(timeout);
                resolve(false);
            };
        });
    }

    /**
     * Connect to the WebSocket server
     */
    public async connect(): Promise<boolean> {
        if (!browser) {
            console.log('Not in browser environment, skipping WebSocket connection');
            return false;
        }

        try {
            const port = await this.discoverPort();
            if (!port) {
                const error = 'Could not discover cache monitoring WebSocket port';
                console.error(error);
                this.connection.lastError = error;
                this.updateStore();
                return false;
            }

            const url = `ws://localhost:${port}`;
            console.log(`🔌 Connecting to cache monitoring WebSocket: ${url}`);

            const ws = new WebSocket(url);

            ws.onopen = () => {
                console.log('✅ Cache monitoring WebSocket connected');
                this.connection.ws = ws;
                this.connection.isConnected = true;
                this.connection.port = port;
                this.connection.url = url;
                this.connection.lastError = null;
                this.connection.reconnectAttempts = 0;
                this.updateStore();

                // Resubscribe to previous subscriptions
                this.connection.subscriptions.forEach(channel => {
                    this.subscribe(channel);
                });
            };

            ws.onmessage = (event: any) => {
                this.handleMessage(event);
            };

            ws.onclose = (event: any) => {
                console.log('🔌 Cache monitoring WebSocket disconnected:', event.code);
                this.connection.isConnected = false;
                this.connection.ws = null;
                this.updateStore();
                
                // Attempt to reconnect
                if (this.connection.reconnectAttempts < this.maxReconnectAttempts) {
                    this.connection.reconnectAttempts++;
                    console.log(`🔄 Reconnecting in ${this.reconnectDelay}ms... (attempt ${this.connection.reconnectAttempts})`);
                    setTimeout(() => this.connect(), this.reconnectDelay);
                }
            };

            ws.onerror = (error) => {
                console.error('❌ Cache monitoring WebSocket error:', error);
                this.connection.lastError = 'WebSocket connection error';
                this.updateStore();
            };

            return true;
        } catch (error: any) {
            console.error('Failed to connect to cache monitoring WebSocket:', error);
            this.connection.lastError = (error as Error).message;
            this.updateStore();
            return false;
        }
    }

    /**
     * Handle incoming WebSocket messages
     */
    private handleMessage(event: MessageEvent) {
        try {
            const data = JSON.parse(event.data);
            
            // Trigger event listeners
            const listeners = this.eventListeners.get(data.type) || [];
            listeners.forEach(listener => listener(data));

            // Also trigger generic message listeners
            const allListeners = this.eventListeners.get('*') || [];
            allListeners.forEach(listener => listener(data));

        } catch (error: any) {
            console.error('Error parsing WebSocket message:', error);
        }
    }

    /**
     * Subscribe to a cache monitoring channel
     */
    public subscribe(channel: string): void {
        if (!this.connection.ws || !this.connection.isConnected) {
            // Store subscription for when we reconnect
            this.connection.subscriptions.add(channel);
            return;
        }

        const message = {
            type: 'subscribe',
            channel: channel
        };

        this.connection.ws.send(JSON.stringify(message));
        this.connection.subscriptions.add(channel);
        console.log(`📡 Subscribed to cache monitoring channel: ${channel}`);
    }

    /**
     * Unsubscribe from a cache monitoring channel
     */
    public unsubscribe(channel: string): void {
        if (this.connection.ws && this.connection.isConnected) {
            const message = {
                type: 'unsubscribe',
                channel: channel
            };

            this.connection.ws.send(JSON.stringify(message));
        }

        this.connection.subscriptions.delete(channel);
        console.log(`📡 Unsubscribed from cache monitoring channel: ${channel}`);
    }

    /**
     * Add event listener for specific message types
     */
    public on(eventType: string, listener: (data: any) => void): void {
        if (!this.eventListeners.has(eventType)) {
            this.eventListeners.set(eventType, []);
        }
        this.eventListeners.get(eventType)!.push(listener);
    }

    /**
     * Remove event listener
     */
    public off(eventType: string, listener: (data: any) => void): void {
        const listeners = this.eventListeners.get(eventType);
        if (listeners) {
            const index = listeners.indexOf(listener);
            if (index > -1) {
                listeners.splice(index, 1);
            }
        }
    }

    /**
     * Setup heartbeat to keep connection alive
     */
    private setupHeartbeat(): void {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }

        this.heartbeatInterval = setInterval(() => {
            if (this.connection.ws && this.connection.isConnected) {
                this.connection.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000) as any;
    }

    /**
     * Update the svelte store
     */
    private updateStore(): void {
        cacheMonitorConnection.set({ ...this.connection });
    }

    /**
     * Disconnect from WebSocket
     */
    public disconnect(): void {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }

        if (this.connection.ws) {
            this.connection.ws.close();
        }

        this.connection.ws = null;
        this.connection.isConnected = false;
        this.updateStore();
    }

    /**
     * Get current connection status
     */
    public getConnectionStatus(): CacheMonitoringConnection {
        return { ...this.connection };
    }
}

// Export singleton instance
export const cacheMonitoringClient = new CacheMonitoringClient();

// Convenient reactive functions for Svelte components
export function subscribeToCacheChannel(channel: string, callback: (data: any) => void) {
    cacheMonitoringClient.subscribe(channel);
    cacheMonitoringClient.on(channel, callback);

    // Return unsubscribe function
    return () => {
        cacheMonitoringClient.unsubscribe(channel);
        cacheMonitoringClient.off(channel, callback);
    };
}

// Predefined channel subscriptions
export const CACHE_CHANNELS = {
    OPERATIONS: 'cache-operations',
    PERFORMANCE: 'performance-metrics',
    ANALYTICS: 'layer-analytics',
    ALERTS: 'optimization-alerts',
    HEALTH: 'system-health',
    HOT_KEYS: 'hot-keys-tracking',
    MEMORY: 'memory-usage',
    ERRORS: 'error-monitoring'
} as const;