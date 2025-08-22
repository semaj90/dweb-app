/**
 * Cluster Services Management API
 * Service discovery, management, and orchestration for all Go binaries
 */

import { type RequestHandler,  json } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { productionServiceRegistry, GO_SERVICES_REGISTRY, type ServiceDefinition } from '$lib/../../../../lib/services/production-service-registry.js';
import { context7OrchestrationService } from '$lib/../../../../lib/services/context7-orchestration-integration.js';

export const GET: RequestHandler = async ({ url }) => {
  const category = url.searchParams.get('category');
  const tier = url.searchParams.get('tier');
  const includeHealth = url.searchParams.get('health') === 'true';
  
  try {
    let services: ServiceDefinition[] = [];
    
    if (category) {
      services = productionServiceRegistry.getServicesByCategory(category as any);
    } else if (tier) {
      services = productionServiceRegistry.getServicesByTier(tier as any);
    } else {
      services = Object.values(GO_SERVICES_REGISTRY);
    }

    // Add health status if requested
    let servicesWithHealth: any[] = services;
    if (includeHealth) {
      const healthChecks = await Promise.all(
        services.map(async (service) => ({
          ...service,
          healthy: await productionServiceRegistry.checkServiceHealth(service.name),
          lastHealthCheck: new Date().toISOString()
        }))
      );
      servicesWithHealth = healthChecks;
    }

    const orchestrationPlan = context7OrchestrationService.getOrchestrationPlan();

    const response = {
      timestamp: new Date().toISOString(),
      services: servicesWithHealth,
      summary: {
        total: services.length,
        byCategory: getCategoryBreakdown(services),
        byTier: getTierBreakdown(services),
        protocols: getProtocolBreakdown(services)
      },
      orchestration: {
        startupOrder: orchestrationPlan.startupSequence.length,
        healthEndpoints: orchestrationPlan.healthChecks.length,
        protocolRoutes: Object.keys(orchestrationPlan.protocolRouting).length
      }
    };

    return json(response);
  } catch (error) {
    console.error('Services query failed:', error);
    return json(
      {
        error: 'Services query failed',
        message: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
};

export const POST: RequestHandler = async ({ request }) => {
  const { action, services, options } = await request.json();
  
  switch (action) {
    case 'start_services':
      return handleStartServices(services, options);
    case 'stop_services':
      return handleStopServices(services);
    case 'restart_tier':
      return handleRestartTier(options?.tier);
    case 'generate_startup_script':
      return handleGenerateStartupScript();
    case 'update_orchestration':
      return handleUpdateOrchestration(options);
    default:
      return json({ error: 'Invalid action' }, { status: 400 });
  }
};

function getCategoryBreakdown(services: ServiceDefinition[]): Record<string, number> {
  const breakdown: Record<string, number> = {};
  services.forEach(service => {
    breakdown[service.category] = (breakdown[service.category] || 0) + 1;
  });
  return breakdown;
}

function getTierBreakdown(services: ServiceDefinition[]): Record<string, number> {
  const breakdown: Record<string, number> = {};
  services.forEach(service => {
    breakdown[service.tier] = (breakdown[service.tier] || 0) + 1;
  });
  return breakdown;
}

function getProtocolBreakdown(services: ServiceDefinition[]): Record<string, number> {
  const breakdown: Record<string, number> = {};
  services.forEach(service => {
    service.protocols.forEach(protocol => {
      breakdown[protocol] = (breakdown[protocol] || 0) + 1;
    });
  });
  return breakdown;
}

async function handleStartServices(serviceNames: string[], options?: unknown): Promise<Response> {
  const results: Record<string, { success: boolean; message: string }> = {};
  
  for (const serviceName of serviceNames) {
    const service = productionServiceRegistry.getServiceByName(serviceName);
    if (!service) {
      results[serviceName] = {
        success: false,
        message: 'Service not found in registry'
      };
      continue;
    }

    try {
      // In production, this would use Windows service manager or process spawning
      // For now, check if service is already running
      const healthy = await productionServiceRegistry.checkServiceHealth(serviceName);
      results[serviceName] = {
        success: healthy,
        message: healthy ? 'Service is running' : 'Service failed to start'
      };
    } catch (error) {
      results[serviceName] = {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  return json({
    action: 'start_services',
    results,
    timestamp: new Date().toISOString()
  });
}

async function handleStopServices(serviceNames: string[]): Promise<Response> {
  const results: Record<string, { success: boolean; message: string }> = {};
  
  for (const serviceName of serviceNames) {
    // In production, this would actually stop the service
    results[serviceName] = {
      success: true,
      message: 'Stop command sent (simulation)'
    };
  }

  return json({
    action: 'stop_services',
    results,
    timestamp: new Date().toISOString()
  });
}

async function handleRestartTier(tier: string): Promise<Response> {
  if (!['tier1', 'tier2', 'tier3', 'tier4'].includes(tier)) {
    return json({ error: 'Invalid tier' }, { status: 400 });
  }

  const tierServices = productionServiceRegistry.getServicesByTier(tier as any);
  const results: Record<string, boolean> = {};
  
  for (const service of tierServices) {
    try {
      // Simulate tier restart
      const healthy = await productionServiceRegistry.checkServiceHealth(service.name);
      results[service.name] = healthy;
    } catch {
      results[service.name] = false;
    }
  }

  return json({
    action: 'restart_tier',
    tier,
    results,
    servicesAffected: tierServices.length,
    timestamp: new Date().toISOString()
  });
}

async function handleGenerateStartupScript(): Promise<Response> {
  const startupScript = await context7OrchestrationService.generateStartupScript();
  
  return json({
    action: 'generate_startup_script',
    script: startupScript,
    services: Object.keys(GO_SERVICES_REGISTRY).length,
    timestamp: new Date().toISOString()
  });
}

async function handleUpdateOrchestration(options: any): Promise<Response> {
  // Update orchestration configuration
  await context7OrchestrationService.updateServiceHealth();
  
  const metrics = context7OrchestrationService.getMetrics();
  const plan = context7OrchestrationService.getOrchestrationPlan();

  return json({
    action: 'update_orchestration',
    metrics,
    plan,
    options,
    timestamp: new Date().toISOString()
  });
}