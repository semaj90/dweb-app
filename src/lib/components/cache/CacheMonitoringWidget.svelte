<!--
  Cache Monitoring Widget - Example Component
  Shows how to use the dynamic WebSocket cache monitoring client
-->

<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { 
        cacheMonitorConnection, 
        cacheMonitoringClient, 
        subscribeToCacheChannel,
        CACHE_CHANNELS 
    } from '$lib/websockets/cache-monitoring-client';
    
    // Reactive connection state
    $: connection = $cacheMonitorConnection;
    
    // Cache metrics data
    let cacheMetrics = {
        hitRate: 0,
        operations: 0,
        memoryUsage: 0,
        activeClients: 0
    };
    
    let recentOperations: any[] = [];
    let systemAlerts: any[] = [];
    
    // Unsubscribe functions
    let unsubscribeFunctions: (() => void)[] = [];
    
    onMount(() => {
        // Subscribe to various cache monitoring channels
        
        // Performance metrics
        const unsubPerformance = subscribeToCacheChannel(
            CACHE_CHANNELS.PERFORMANCE, 
            (data) => {
                if (data.hitRate !== undefined) {
                    cacheMetrics.hitRate = data.hitRate;
                }
                if (data.operations !== undefined) {
                    cacheMetrics.operations = data.operations;
                }
                if (data.memoryUsage !== undefined) {
                    cacheMetrics.memoryUsage = data.memoryUsage;
                }
            }
        );
        
        // Cache operations
        const unsubOperations = subscribeToCacheChannel(
            CACHE_CHANNELS.OPERATIONS,
            (data) => {
                recentOperations = [data, ...recentOperations].slice(0, 10);
            }
        );
        
        // System alerts
        const unsubAlerts = subscribeToCacheChannel(
            CACHE_CHANNELS.ALERTS,
            (data) => {
                systemAlerts = [data, ...systemAlerts].slice(0, 5);
            }
        );
        
        // System health
        const unsubHealth = subscribeToCacheChannel(
            CACHE_CHANNELS.HEALTH,
            (data) => {
                if (data.activeClients !== undefined) {
                    cacheMetrics.activeClients = data.activeClients;
                }
            }
        );
        
        // Store unsubscribe functions
        unsubscribeFunctions = [
            unsubPerformance,
            unsubOperations, 
            unsubAlerts,
            unsubHealth
        ];
    });
    
    onDestroy(() => {
        // Clean up subscriptions
        unsubscribeFunctions.forEach(unsub => unsub());
    });
    
    function formatBytes(bytes: number): string {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    function reconnect() {
        cacheMonitoringClient.connect();
    }
</script>

<div class="cache-monitor-widget">
    <div class="header">
        <h3>Cache Monitoring</h3>
        <div class="connection-status" class:connected={connection.isConnected}>
            {#if connection.isConnected}
                <span class="status-dot connected"></span>
                Connected (Port: {connection.port})
            {:else}
                <span class="status-dot disconnected"></span>
                {#if connection.lastError}
                    Error: {connection.lastError}
                {:else}
                    Disconnected
                {/if}
            {/if}
        </div>
    </div>
    
    {#if connection.isConnected}
        <!-- Metrics Dashboard -->
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-label">Hit Rate</div>
                <div class="metric-value">{(cacheMetrics.hitRate * 100).toFixed(1)}%</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Operations</div>
                <div class="metric-value">{cacheMetrics.operations.toLocaleString()}</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Memory Usage</div>
                <div class="metric-value">{formatBytes(cacheMetrics.memoryUsage)}</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Active Clients</div>
                <div class="metric-value">{cacheMetrics.activeClients}</div>
            </div>
        </div>
        
        <!-- Recent Operations -->
        {#if recentOperations.length > 0}
            <div class="section">
                <h4>Recent Operations</h4>
                <div class="operations-list">
                    {#each recentOperations as operation, i (i)}
                        <div class="operation-item">
                            <span class="operation-type">{operation.type}</span>
                            <span class="operation-key">{operation.key}</span>
                            <span class="operation-time">{new Date(operation.timestamp).toLocaleTimeString()}</span>
                        </div>
                    {/each}
                </div>
            </div>
        {/if}
        
        <!-- System Alerts -->
        {#if systemAlerts.length > 0}
            <div class="section">
                <h4>System Alerts</h4>
                <div class="alerts-list">
                    {#each systemAlerts as alert, i (i)}
                        <div class="alert-item" class:warning={alert.severity === 'warning'} class:critical={alert.severity === 'critical'}>
                            <span class="alert-message">{alert.message}</span>
                            <span class="alert-time">{new Date(alert.timestamp).toLocaleTimeString()}</span>
                        </div>
                    {/each}
                </div>
            </div>
        {/if}
        
    {:else}
        <!-- Connection Error State -->
        <div class="error-state">
            <p>WebSocket connection not available</p>
            <button on:click={reconnect} class="reconnect-btn">
                Reconnect
            </button>
            {#if connection.reconnectAttempts > 0}
                <p class="reconnect-info">
                    Reconnection attempts: {connection.reconnectAttempts}
                </p>
            {/if}
        </div>
    {/if}
</div>

<style>
    .cache-monitor-widget {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .header h3 {
        margin: 0;
        color: #1a202c;
        font-size: 1.125rem;
    }
    
    .connection-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        color: #64748b;
    }
    
    .connection-status.connected {
        color: #059669;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    
    .status-dot.connected {
        background: #10b981;
    }
    
    .status-dot.disconnected {
        background: #ef4444;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric {
        text-align: center;
        padding: 0.75rem;
        background: #f8fafc;
        border-radius: 6px;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a202c;
    }
    
    .section {
        margin-bottom: 1rem;
    }
    
    .section h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.875rem;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .operations-list, .alerts-list {
        max-height: 200px;
        overflow-y: auto;
    }
    
    .operation-item, .alert-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        margin-bottom: 0.25rem;
        background: #f8fafc;
        border-radius: 4px;
        font-size: 0.875rem;
    }
    
    .operation-type {
        font-weight: 500;
        color: #3b82f6;
        min-width: 60px;
    }
    
    .operation-key {
        flex: 1;
        margin: 0 0.5rem;
        color: #374151;
        font-family: monospace;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .operation-time {
        color: #6b7280;
        font-size: 0.75rem;
        min-width: 80px;
        text-align: right;
    }
    
    .alert-item.warning {
        background: #fef3c7;
        border-left: 3px solid #f59e0b;
    }
    
    .alert-item.critical {
        background: #fee2e2;
        border-left: 3px solid #ef4444;
    }
    
    .alert-message {
        flex: 1;
        color: #374151;
    }
    
    .alert-time {
        color: #6b7280;
        font-size: 0.75rem;
        min-width: 80px;
        text-align: right;
    }
    
    .error-state {
        text-align: center;
        padding: 2rem;
        color: #6b7280;
    }
    
    .reconnect-btn {
        background: #3b82f6;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.875rem;
        margin-top: 1rem;
    }
    
    .reconnect-btn:hover {
        background: #2563eb;
    }
    
    .reconnect-info {
        font-size: 0.75rem;
        margin-top: 0.5rem;
        color: #9ca3af;
    }
</style>