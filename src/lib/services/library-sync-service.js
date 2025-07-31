/**
 * Library Sync Service for Enhanced RAG Integration
 * Manages synchronization between frontend and backend services
 */

export class LibrarySyncService {
    constructor(config = {}) {
        this.baseUrl = config.baseUrl || 'http://localhost:8000';
        this.timeout = config.timeout || 30000;
        this.retries = config.retries || 3;
        this.syncInterval = null;
        this.isConnected = false;
    }

    async initialize() {
        console.log('🔄 Initializing Library Sync Service...');
        
        try {
            await this.testConnection();
            this.startPeriodicSync();
            console.log('✅ Library Sync Service initialized');
            return true;
        } catch (error) {
            console.error('❌ Library Sync Service initialization failed:', error);
            return false;
        }
    }

    async testConnection() {
        try {
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                timeout: this.timeout
            });
            
            this.isConnected = response.ok;
            return response.ok;
        } catch (error) {
            this.isConnected = false;
            throw new Error(`Connection test failed: ${error.message}`);
        }
    }

    async syncLibraries() {
        if (!this.isConnected) {
            console.log('⚠️ Library sync skipped - not connected');
            return false;
        }

        try {
            const response = await fetch(`${this.baseUrl}/api/sync/libraries`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    timestamp: new Date().toISOString(),
                    source: 'frontend-sync'
                })
            });

            if (response.ok) {
                console.log('✅ Library sync completed');
                return true;
            } else {
                console.warn('⚠️ Library sync returned non-OK status:', response.status);
                return false;
            }
        } catch (error) {
            console.error('❌ Library sync failed:', error);
            return false;
        }
    }

    startPeriodicSync(intervalMs = 300000) { // 5 minutes
        if (this.syncInterval) {
            clearInterval(this.syncInterval);
        }

        this.syncInterval = setInterval(async () => {
            await this.syncLibraries();
        }, intervalMs);

        console.log(`🔄 Periodic library sync started (${intervalMs}ms interval)`);
    }

    stopPeriodicSync() {
        if (this.syncInterval) {
            clearInterval(this.syncInterval);
            this.syncInterval = null;
            console.log('⏹️ Periodic library sync stopped');
        }
    }

    async getStatus() {
        return {
            isConnected: this.isConnected,
            baseUrl: this.baseUrl,
            syncActive: !!this.syncInterval,
            lastSync: new Date().toISOString()
        };
    }

    dispose() {
        this.stopPeriodicSync();
        this.isConnected = false;
        console.log('🧹 Library Sync Service disposed');
    }
}

// Default export for ES modules
export default LibrarySyncService;