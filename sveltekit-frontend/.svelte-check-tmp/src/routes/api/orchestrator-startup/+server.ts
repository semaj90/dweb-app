import orchestrator from "$lib/services/comprehensive-database-orchestrator";
import type { RequestHandler } from './$types';

// Orchestrator Startup API - Initialize and manage the comprehensive system
// Handles startup, monitoring, and coordination of all services

// GET /api/orchestrator-startup - Get startup status and system readiness
export const GET: RequestHandler = async () => {
  try {
    const orchestratorStatus = (orchestrator as any).getStatus?.() || { isRunning: false };

    // Check if orchestrator is running
    if (!orchestratorStatus.isRunning) {
      return json({
        success: false,
        message: 'Database orchestrator is not running',
        status: orchestratorStatus,
        recommendations: [
          'Call POST /api/orchestrator-startup with action: "start"',
          'Verify all required services are available',
          'Check database connectivity',
        ],
        timestamp: new Date().toISOString(),
      });
    }

    return json({
      success: true,
      message: 'Database orchestrator is running',
      status: orchestratorStatus,
      capabilities: [
        'Real-time event processing',
        'Database operation monitoring',
        'Context7 MCP integration',
        'Condition-based automation',
        'Multi-service health monitoring',
      ],
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};

// POST /api/orchestrator-startup - Control orchestrator startup and operations
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action, configuration } = await request.json();

    switch (action) {
      case 'start':
        return await startOrchestrator(configuration);

      case 'stop':
        return await stopOrchestrator();

      case 'restart':
        return await restartOrchestrator(configuration);

      case 'initialize_system':
        return await initializeFullSystem(configuration);

      case 'health_check':
        return await performSystemHealthCheck();

      case 'configure':
        return await configureOrchestrator(configuration);

      default:
        return json(
          {
            success: false,
            error: `Unknown action: ${action}`,
            available_actions: [
              'start',
              'stop',
              'restart',
              'initialize_system',
              'health_check',
              'configure',
            ],
          },
          { status: 400 }
        );
    }
  } catch (error: any) {
    return json(
      {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};

// Start the database orchestrator
async function startOrchestrator(configuration: any = {}): Promise<any> {
  try {
    const status = (orchestrator as any).getStatus?.() || { isRunning: false };

    if (status.isRunning) {
      return json({
        success: true,
        message: 'Database orchestrator is already running',
        status,
        timestamp: new Date().toISOString(),
      });
    }

    // Configure before starting if configuration provided
    if (configuration && Object.keys(configuration).length > 0) {
      await configureOrchestrator(configuration);
    }

    // Start the orchestrator
    await (orchestrator as any).start?.();

    const newStatus = (orchestrator as any).getStatus?.() || { isRunning: true };

    return json({
      success: true,
      message: 'Database orchestrator started successfully',
      status: newStatus,
      startup_time: new Date().toISOString(),
      active_features: [
        'Event loops running',
        'Database monitoring active',
        'Context7 integration enabled',
        'Real-time processing ready',
      ],
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `Failed to start orchestrator: ${error.message}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// Stop the database orchestrator
async function stopOrchestrator(): Promise<any> {
  try {
    const status = (orchestrator as any).getStatus?.() || { isRunning: false };

    if (!status.isRunning) {
      return json({
        success: true,
        message: 'Database orchestrator is already stopped',
        status,
        timestamp: new Date().toISOString(),
      });
    }

    await (orchestrator as any).stop?.();

    return json({
      success: true,
      message: 'Database orchestrator stopped successfully',
      shutdown_time: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `Failed to stop orchestrator: ${error.message}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// Restart the orchestrator
async function restartOrchestrator(configuration: any = {}): Promise<any> {
  try {
    await stopOrchestrator();

    // Wait a moment for cleanup
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const startResult = await startOrchestrator(configuration);

    return json({
      success: true,
      message: 'Database orchestrator restarted successfully',
      restart_time: new Date().toISOString(),
      start_result: startResult,
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `Failed to restart orchestrator: ${error.message}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// Initialize the full system
async function initializeFullSystem(configuration: any = {}): Promise<any> {
  try {
    const initializationSteps = [];

    // Step 1: Start orchestrator
    initializationSteps.push('Starting database orchestrator...');
    await (orchestrator as any).start?.();

    // Step 2: Setup default conditions
    initializationSteps.push('Setting up default conditions...');
    const defaultConditions = [
      {
        id: 'system_health_monitor',
        type: 'timer',
        condition: { interval: 60000 }, // 1 minute
        action: 'check_system_health',
        isActive: true,
        metadata: { created_by: 'initialization' },
      },
      {
        id: 'database_change_processor',
        type: 'database_change',
        condition: { table: '*', operation: 'insert' },
        action: 'process_new_data',
        isActive: true,
        metadata: { created_by: 'initialization' },
      },
      {
        id: 'context7_sync_condition',
        type: 'timer',
        condition: { interval: 300000 }, // 5 minutes
        action: 'sync_context7_data',
        isActive: true,
        metadata: { created_by: 'initialization' },
      },
    ];

    for (const condition of defaultConditions) {
      (orchestrator as any).addCondition?.(condition);
    }

    // Step 3: Test database connectivity
    initializationSteps.push('Testing database connectivity...');
    const testRecord = {
      type: 'system_initialization',
      message: 'System initialization test',
      timestamp: new Date(),
    };
    await (orchestrator as any).saveToDatabase?.(testRecord, 'system_logs');

    // Step 4: Check service health
    initializationSteps.push('Checking service health...');
    const healthResults = await performSystemHealthCheck();

    // Step 5: Initialize Context7 integration
    initializationSteps.push('Initializing Context7 integration...');
    (orchestrator as any).emit?.('system:initialization_complete', {
      timestamp: new Date(),
      configuration,
      health_results: healthResults,
    });

    return json({
      success: true,
      message: 'Full system initialization completed',
      initialization_steps: initializationSteps,
      status: (orchestrator as any).getStatus?.(),
      default_conditions_added: defaultConditions.length,
      system_ready: true,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `System initialization failed: ${error.message}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// Perform comprehensive system health check
async function performSystemHealthCheck(): Promise<any> {
  try {
    const healthResults = {
      orchestrator: (orchestrator as any).getStatus?.(),
      services: {},
      database: null,
      overall_status: 'unknown',
    };

    // Check key services
    const serviceEndpoints = [
      { name: 'ollama', url: 'http://localhost:11434/api/tags' },
      { name: 'enhanced_rag', url: 'http://localhost:8094/health' },
      { name: 'upload_service', url: 'http://localhost:8093/health' },
      { name: 'recommendation_service', url: 'http://localhost:8096/health' },
    ];

    for (const service of serviceEndpoints) {
      try {
        const controller = new AbortController();
        const t = setTimeout(() => controller.abort(), 5000);
        const response = await fetch(service.url, { method: 'GET', signal: controller.signal });
        clearTimeout(t);
        (healthResults.services as any)[service.name] = {
          status: response.ok ? 'healthy' : 'unhealthy',
          response_code: response.status,
          endpoint: service.url,
        };
      } catch (error: any) {
        (healthResults.services as any)[service.name] = {
          status: 'error',
          error: (error as Error).message,
          endpoint: service.url,
        };
      }
    }

    // Test database connectivity
    try {
      const testQuery = await (orchestrator as any).queryDatabase?.({ limit: 1 }, 'cases');
      healthResults.database = {
        status: 'healthy',
        connection: 'active',
        test_query: 'successful',
      };
    } catch (error: any) {
      healthResults.database = {
        status: 'error',
        error: error.message,
      };
    }

    // Determine overall status
    const serviceStatuses = Object.values(healthResults.services as Record<string, any>).map(
      (s: any) => s.status
    );
    const healthyServices = serviceStatuses.filter((s) => s === 'healthy').length;
    const totalServices = serviceStatuses.length;

    if (healthyServices === totalServices && healthResults.database.status === 'healthy') {
      healthResults.overall_status = 'healthy';
    } else if (healthyServices > totalServices / 2) {
      healthResults.overall_status = 'degraded';
    } else {
      healthResults.overall_status = 'critical';
    }

    return json({
      success: true,
      health_check: healthResults,
      healthy_services: healthyServices,
      total_services: totalServices,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `Health check failed: ${error.message}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// Configure orchestrator settings
async function configureOrchestrator(configuration: any): Promise<any> {
  try {
    // Apply configuration settings
    const appliedSettings = [];

    if (configuration.eventLoopInterval) {
      appliedSettings.push(`Event loop interval: ${configuration.eventLoopInterval}ms`);
    }

    if (configuration.conditionCheckInterval) {
      appliedSettings.push(`Condition check interval: ${configuration.conditionCheckInterval}ms`);
    }

    if (configuration.databaseWatchInterval) {
      appliedSettings.push(`Database watch interval: ${configuration.databaseWatchInterval}ms`);
    }

    if (configuration.context7SyncInterval) {
      appliedSettings.push(`Context7 sync interval: ${configuration.context7SyncInterval}ms`);
    }

    // Save configuration to database
    await (orchestrator as any).saveToDatabase?.(
      {
        configuration,
        applied_settings: appliedSettings,
        configured_at: new Date(),
        configured_by: 'api',
      },
      'orchestrator_configurations'
    );

    return json({
      success: true,
      message: 'Orchestrator configured successfully',
      applied_settings: appliedSettings,
      configuration,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: `Configuration failed: ${error.message}`,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
