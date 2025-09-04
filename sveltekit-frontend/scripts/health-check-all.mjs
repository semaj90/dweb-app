#!/usr/bin/env node
import { promises as fs } from 'fs';

class HealthChecker {
    constructor() {
        this.services = [
            { name: 'Frontend', url: 'http://localhost:5173' },
            { name: 'Go Backend', url: 'http://localhost:8084/api/health' }
        ];
    }

    async check(service) {
        try {
            const response = await fetch(service.url, { 
                signal: AbortSignal.timeout(3000) 
            });
            return response.ok ? 'healthy' : 'unhealthy';
        } catch {
            return 'down';
        }
    }

    async checkAll() {
        console.log('Health Check Report');
        console.log('==================');
        
        for (const service of this.services) {
            const status = await this.check(service);
            const icon = status === 'healthy' ? '✅' : '❌';
            console.log(`${icon} ${service.name}: ${status}`);
        }
        
        console.log('==================');
    }
}

new HealthChecker().checkAll();
