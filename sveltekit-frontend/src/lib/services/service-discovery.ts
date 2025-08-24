// Service Discovery for Legal AI Platform
export class LegalAIServiceDiscovery {
  async discoverServices() {
    console.log('🔍 Discovering services...');
    
    const services = {
      enhancedRAG: await this.checkService('http://localhost:8094', ['/health']),
      uploadService: await this.checkService('http://localhost:8093', ['/health']),
      kratosService: { status: 'unknown', capabilities: ['grpc_service'] }
    };
    
    return services;
  }

  private async checkService(baseUrl: string, endpoints: string[]) {
    for (const endpoint of endpoints) {
      try {
        const response = await fetch(`${baseUrl}${endpoint}`);
        if (response.ok) {
          return { status: 'online', url: baseUrl, capabilities: ['health_check'] };
        }
      } catch (error) {
        continue;
      }
    }
    return { status: 'offline', url: baseUrl, capabilities: [] };
  }
}

export const serviceDiscovery = new LegalAIServiceDiscovery();