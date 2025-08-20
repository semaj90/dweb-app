/**
 * Multi-Protocol Service Orchestrator
 * Native Windows Legal AI Platform - Production Ready
 * Manages HTTP, gRPC, QUIC, WebSocket, and NATS services
 * Uses ServiceRegistry for endpoint resolution
 */
import { ServiceRegistry } from './serviceRegistry';

export interface ServiceStatus {
  name: string;
  url: string;
  status: 'healthy' | 'unhealthy' | 'unreachable';
  details?: any;
}

export class ServiceOrchestrator {
  async checkAllHttpServices(): Promise<ServiceStatus[]> {
    const results: ServiceStatus[] = [];
    for (const [name, url] of Object.entries(ServiceRegistry.http)) {
      try {
        const res = await fetch(url + '/health');
        const status = res.ok ? 'healthy' : 'unhealthy';
        const details = res.ok ? await res.json() : undefined;
        results.push({ name, url, status, details });
      } catch {
        results.push({ name, url, status: 'unreachable' });
      }
    }
    return results;
  }

  // Example: gRPC client stub (requires generated proto code)
  getGrpcClient(serviceName: keyof typeof ServiceRegistry.grpc, ClientClass: any) {
    const endpoint = ServiceRegistry.grpc[serviceName];
    return new ClientClass(endpoint, /* credentials */);
  }

  // Example: QUIC endpoint usage (pseudo-code)
  getQuicEndpoint(serviceName: keyof typeof ServiceRegistry.quic) {
    return ServiceRegistry.quic[serviceName];
  }

  // Example: WebSocket connection
  connectWebSocket(serviceName: keyof typeof ServiceRegistry.ws) {
    const wsUrl = ServiceRegistry.ws[serviceName];
    return new WebSocket(wsUrl);
  }

  // Example: NATS connection (pseudo-code)
  getNatsEndpoints() {
    return Object.values(ServiceRegistry.nats);
  }

  // Dynamic API call
  async callApi(endpointKey: keyof typeof ServiceRegistry.endpoints, method: string = 'GET', body?: any) {
    const url = ServiceRegistry.endpoints[endpointKey];
    const options: RequestInit = { method };
    if (body) {
      options.headers = { 'Content-Type': 'application/json' };
      options.body = JSON.stringify(body);
    }
    const res = await fetch(url, options);
    return res.json();
  }
}

export const serviceOrchestrator = new ServiceOrchestrator();
