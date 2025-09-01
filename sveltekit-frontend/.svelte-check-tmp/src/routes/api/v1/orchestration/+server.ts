import type { RequestHandler } from './$types';

// Comprehensive Service Orchestration System
// Manages all 37 Go binaries with intelligent routing and health monitoring

import { ServiceOrchestrator } from '$lib/services/service-orchestrator';
import type { 
	ServiceConfig, 
	OrchestrationRequest, 
	ServiceStatus,
	HealthCheckReport,
	ServiceTier 
} from '$lib/types/orchestration';

// Initialize the service orchestrator
const orchestrator = new ServiceOrchestrator();

/** POST /api/v1/orchestration - Orchestrate services */
export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json() as OrchestrationRequest;

		console.log(`🎼 Orchestrator: Processing ${body.action} request for ${body.services?.length || 'all'} services`);

		let result;
		switch (body.action) {
			case 'start':
				result = await orchestrator.startServices(body.services, body.options);
				break;
			case 'stop':
				result = await orchestrator.stopServices(body.services, body.options);
				break;
			case 'restart':
				result = await orchestrator.restartServices(body.services, body.options);
				break;
			case 'scale':
				result = await orchestrator.scaleServices(body.services, body.options);
				break;
			case 'health_check':
				result = await orchestrator.performHealthCheck(body.services);
				break;
			case 'deploy':
				result = await orchestrator.deployServices(body.services, body.options);
				break;
			default:
				throw new Error(`Unsupported orchestration action: ${body.action}`);
		}

		console.log(`✅ Orchestrator: ${body.action} completed successfully`);

		return json({
			success: true,
			action: body.action,
			result,
			timestamp: new Date().toISOString(),
			orchestration_id: generateOrchestrationId(),
		});

	} catch (error: any) {
		console.error('Orchestration Error:', error);
		
		return json({
			success: false,
			error: 'Service orchestration failed',
			details: error instanceof Error ? error.message : 'Unknown error',
			timestamp: new Date().toISOString(),
		}, { status: 500 });
	}
};

/** GET /api/v1/orchestration - Get orchestration status and capabilities */
export const GET: RequestHandler = async () => {
	try {
		const status = await orchestrator.getSystemStatus();
		const capabilities = orchestrator.getCapabilities();

		return json({
			service: 'Comprehensive Service Orchestrator',
			status: 'operational',
			version: '2.0.0',
			system_status: status,
			capabilities,
			managed_services: orchestrator.getManagedServices(),
			performance_metrics: await orchestrator.getPerformanceMetrics(),
			timestamp: new Date().toISOString(),
		});

	} catch (error: any) {
		return json({
			service: 'Comprehensive Service Orchestrator',
			status: 'degraded',
			error: 'Unable to get system status',
			details: error instanceof Error ? error.message : 'Unknown error',
			timestamp: new Date().toISOString(),
		}, { status: 503 });
	}
};

/** GET /api/v1/orchestration/health - Comprehensive health check */
export const GET_HEALTH: RequestHandler = async () => {
	try {
		const healthReport = await orchestrator.comprehensiveHealthCheck();

		return json(healthReport, {
			status: healthReport.overall_health === 'healthy' ? 200 : 
					healthReport.overall_health === 'degraded' ? 206 : 503
		});

	} catch (error: any) {
		return json({
			overall_health: 'critical',
			error: 'Health check system failure',
			details: error instanceof Error ? error.message : 'Unknown error',
			timestamp: new Date().toISOString(),
		}, { status: 503 });
	}
};

/** POST /api/v1/orchestration/emergency - Emergency service management */
export const POST_EMERGENCY: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json();
		
		console.log(`🚨 Emergency Orchestration: ${body.emergency_action}`);

		let result;
		switch (body.emergency_action) {
			case 'shutdown_all':
				result = await orchestrator.emergencyShutdown();
				break;
			case 'restart_critical':
				result = await orchestrator.restartCriticalServices();
				break;
			case 'enable_safe_mode':
				result = await orchestrator.enableSafeMode();
				break;
			case 'recover_from_failure':
				result = await orchestrator.recoverFromFailure(body.failure_context);
				break;
			default:
				throw new Error(`Unsupported emergency action: ${body.emergency_action}`);
		}

		return json({
			success: true,
			emergency_action: body.emergency_action,
			result,
			timestamp: new Date().toISOString(),
		});

	} catch (error: any) {
		console.error('Emergency Orchestration Error:', error);
		
		return json({
			success: false,
			error: 'Emergency orchestration failed',
			details: error instanceof Error ? error.message : 'Unknown error',
			timestamp: new Date().toISOString(),
		}, { status: 500 });
	}
};

// Helper functions
function generateOrchestrationId(): string {
	return `orch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Extend the GET handler to include health endpoint
export const GET_orchestration_health = GET_HEALTH;
export const POST_orchestration_emergency = POST_EMERGENCY;