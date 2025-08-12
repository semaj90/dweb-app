/**
 * Automated Workflow Triggers API
 * 
 * RESTful API for managing and monitoring automated workflow triggers
 * Provides endpoints for:
 * - Trigger management (CRUD operations)
 * - Real-time metrics processing
 * - System monitoring and statistics
 * - Performance analytics
 * - Alert management
 */

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { automatedWorkflowEngine } from '$lib/orchestration/automated-workflow-triggers';
import type { 
    WorkflowTrigger,
    PerformanceMetrics,
    SystemAlert 
} from '$lib/ai/types';

// POST - Process metrics and evaluate triggers
export const POST: RequestHandler = async ({ request, url }) => {
    try {
        const action = url.searchParams.get('action');
        const body = await request.json();

        switch (action) {
            case 'process-metrics':
                return await processMetrics(body);
            
            case 'register-trigger':
                return await registerTrigger(body);
            
            case 'execute-trigger':
                return await executeTrigger(body);
            
            case 'send-alert':
                return await sendAlert(body);
            
            default:
                return error(400, 'Invalid action specified');
        }
    } catch (err) {
        console.error('❌ Automation triggers API error:', err);
        return error(500, `Server error: ${err.message}`);
    }
};

// GET - Retrieve trigger information and statistics
export const GET: RequestHandler = async ({ url }) => {
    try {
        const action = url.searchParams.get('action');
        
        switch (action) {
            case 'stats':
                return getTriggerStats();
            
            case 'metrics':
                return getSystemMetrics();
            
            case 'triggers':
                return getTriggers();
            
            case 'workflows':
                return getActiveWorkflows();
            
            case 'alerts':
                return getAlerts();
            
            case 'health':
                return getHealthStatus();
            
            default:
                return getDashboard();
        }
    } catch (err) {
        console.error('❌ Automation triggers API error:', err);
        return error(500, `Server error: ${err.message}`);
    }
};

// PUT - Update trigger configuration
export const PUT: RequestHandler = async ({ request }) => {
    try {
        const { triggerId, updates } = await request.json();
        
        if (!triggerId) {
            return error(400, 'Trigger ID is required');
        }

        // Update trigger configuration
        const result = await updateTrigger(triggerId, updates);
        
        return json({
            success: true,
            triggerId,
            updates,
            result,
            timestamp: new Date().toISOString()
        });
    } catch (err) {
        console.error('❌ Update trigger error:', err);
        return error(500, `Update error: ${err.message}`);
    }
};

// DELETE - Remove trigger
export const DELETE: RequestHandler = async ({ url }) => {
    try {
        const triggerId = url.searchParams.get('triggerId');
        
        if (!triggerId) {
            return error(400, 'Trigger ID is required');
        }

        const result = await removeTrigger(triggerId);
        
        return json({
            success: true,
            triggerId,
            removed: result,
            timestamp: new Date().toISOString()
        });
    } catch (err) {
        console.error('❌ Remove trigger error:', err);
        return error(500, `Remove error: ${err.message}`);
    }
};

/**
 * Handler Functions
 */

async function processMetrics(metricsData: any) {
    const startTime = performance.now();
    
    // Validate metrics data
    if (!metricsData || typeof metricsData !== 'object') {
        return error(400, 'Invalid metrics data');
    }

    // Enhanced metrics with system information
    const enhancedMetrics: PerformanceMetrics = {
        ...metricsData,
        timestamp: Date.now(),
        source: 'api',
        processedAt: new Date().toISOString()
    };

    // Process through automation engine
    await automatedWorkflowEngine.processMetrics(enhancedMetrics);
    
    const processingTime = performance.now() - startTime;
    
    return json({
        success: true,
        message: 'Metrics processed successfully',
        metrics: enhancedMetrics,
        processingTimeMs: processingTime.toFixed(2),
        triggersEvaluated: true,
        timestamp: new Date().toISOString()
    });
}

async function registerTrigger(triggerData: any) {
    // Validate trigger data
    if (!triggerData || !triggerData.id || !triggerData.conditions) {
        return error(400, 'Invalid trigger configuration');
    }

    const trigger: WorkflowTrigger = {
        ...triggerData,
        registeredAt: new Date(),
        registeredVia: 'api'
    };

    // Register with automation engine
    automatedWorkflowEngine.registerTrigger(trigger);
    
    return json({
        success: true,
        message: 'Trigger registered successfully',
        trigger: {
            id: trigger.id,
            name: trigger.name,
            type: trigger.type,
            enabled: trigger.enabled,
            conditions: trigger.conditions.length,
            actions: trigger.actions?.length || 0
        },
        timestamp: new Date().toISOString()
    });
}

async function executeTrigger(triggerData: any) {
    const { triggerId, forceExecute = false, testMode = false } = triggerData;
    
    if (!triggerId) {
        return error(400, 'Trigger ID is required');
    }

    // Manual trigger execution
    const result = await executeManualTrigger(triggerId, forceExecute, testMode);
    
    return json({
        success: true,
        message: 'Trigger executed successfully',
        triggerId,
        forceExecute,
        testMode,
        result,
        timestamp: new Date().toISOString()
    });
}

async function sendAlert(alertData: any) {
    const alert: SystemAlert = {
        id: `manual-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        ...alertData,
        source: 'manual-api',
        timestamp: new Date()
    };

    // Send through automation engine
    automatedWorkflowEngine.emit('alert', alert);
    
    return json({
        success: true,
        message: 'Alert sent successfully',
        alert: {
            id: alert.id,
            type: alert.type,
            severity: alert.severity,
            message: alert.message
        },
        timestamp: new Date().toISOString()
    });
}

function getTriggerStats() {
    const stats = automatedWorkflowEngine.getTriggerStats();
    
    return json({
        success: true,
        stats,
        timestamp: new Date().toISOString()
    });
}

function getSystemMetrics() {
    const metrics = automatedWorkflowEngine.getSystemMetrics();
    
    return json({
        success: true,
        metrics,
        timestamp: new Date().toISOString()
    });
}

function getTriggers() {
    const stats = automatedWorkflowEngine.getTriggerStats();
    
    return json({
        success: true,
        triggers: stats.triggerDetails,
        totalCount: stats.totalTriggers,
        activeCount: stats.activeTriggers,
        timestamp: new Date().toISOString()
    });
}

function getActiveWorkflows() {
    const metrics = automatedWorkflowEngine.getSystemMetrics();
    
    return json({
        success: true,
        activeWorkflows: metrics.activeWorkflows,
        count: metrics.activeWorkflows?.length || 0,
        timestamp: new Date().toISOString()
    });
}

function getAlerts() {
    const metrics = automatedWorkflowEngine.getSystemMetrics();
    
    return json({
        success: true,
        alerts: metrics.recentAlerts,
        count: metrics.recentAlerts?.length || 0,
        timestamp: new Date().toISOString()
    });
}

function getHealthStatus() {
    const stats = automatedWorkflowEngine.getTriggerStats();
    const metrics = automatedWorkflowEngine.getSystemMetrics();
    
    // Calculate health score
    const totalTriggers = stats.totalTriggers;
    const activeTriggers = stats.activeTriggers;
    const failedTriggers = stats.triggerDetails.filter(t => t.failureCount > t.successCount).length;
    const openCircuitBreakers = stats.triggerDetails.filter(t => t.circuitBreakerOpen).length;
    
    const healthScore = totalTriggers > 0 
        ? Math.max(0, Math.min(100, 
            ((activeTriggers - failedTriggers - openCircuitBreakers) / totalTriggers) * 100
        ))
        : 100;

    const status = healthScore > 80 ? 'healthy' : 
                  healthScore > 50 ? 'degraded' : 'unhealthy';
    
    return json({
        success: true,
        health: {
            status,
            score: Math.round(healthScore),
            triggers: {
                total: totalTriggers,
                active: activeTriggers,
                failed: failedTriggers,
                circuitBreakersOpen: openCircuitBreakers
            },
            activeWorkflows: stats.activeWorkflows,
            recentAlerts: metrics.recentAlerts?.length || 0,
            uptime: process.uptime(),
            lastCheck: new Date().toISOString()
        },
        timestamp: new Date().toISOString()
    });
}

function getDashboard() {
    const stats = automatedWorkflowEngine.getTriggerStats();
    const metrics = automatedWorkflowEngine.getSystemMetrics();
    
    // Enhanced dashboard with comprehensive information
    return json({
        success: true,
        dashboard: {
            overview: {
                totalTriggers: stats.totalTriggers,
                activeTriggers: stats.activeTriggers,
                activeWorkflows: stats.activeWorkflows,
                totalAlerts: stats.totalAlerts
            },
            recentActivity: {
                triggers: stats.triggerDetails.slice(-10),
                workflows: metrics.activeWorkflows?.slice(-5) || [],
                alerts: metrics.recentAlerts?.slice(-5) || []
            },
            performance: {
                currentMetrics: metrics.currentMetrics,
                historyLength: metrics.historyLength,
                windowSize: metrics.windowSizeMs
            },
            systemHealth: calculateSystemHealth(stats, metrics)
        },
        timestamp: new Date().toISOString()
    });
}

/**
 * Helper Functions
 */

async function executeManualTrigger(triggerId: string, forceExecute: boolean, testMode: boolean) {
    // This would integrate with the automation engine to manually execute a trigger
    // For now, return a mock result
    
    console.log(`🔧 Manual execution of trigger: ${triggerId}`);
    
    return {
        executed: true,
        triggerId,
        executionId: `manual-${Date.now()}`,
        forceExecute,
        testMode,
        result: testMode ? 'Test execution completed' : 'Trigger executed successfully',
        timestamp: new Date().toISOString()
    };
}

async function updateTrigger(triggerId: string, updates: any) {
    // This would update the trigger configuration in the automation engine
    console.log(`🔧 Updating trigger: ${triggerId}`, updates);
    
    return {
        updated: true,
        triggerId,
        updates,
        timestamp: new Date().toISOString()
    };
}

async function removeTrigger(triggerId: string) {
    // This would remove the trigger from the automation engine
    console.log(`🗑️ Removing trigger: ${triggerId}`);
    
    return {
        removed: true,
        triggerId,
        timestamp: new Date().toISOString()
    };
}

function calculateSystemHealth(stats: any, metrics: any) {
    const triggerHealth = stats.totalTriggers > 0 
        ? (stats.activeTriggers / stats.totalTriggers) * 100 
        : 100;
    
    const workflowHealth = stats.activeWorkflows < 10 ? 100 : 
                          stats.activeWorkflows < 20 ? 75 : 50;
    
    const alertHealth = stats.totalAlerts < 5 ? 100 :
                       stats.totalAlerts < 15 ? 75 : 50;
    
    const overallHealth = (triggerHealth + workflowHealth + alertHealth) / 3;
    
    return {
        overall: Math.round(overallHealth),
        triggers: Math.round(triggerHealth),
        workflows: Math.round(workflowHealth),
        alerts: Math.round(alertHealth),
        status: overallHealth > 80 ? 'excellent' :
               overallHealth > 60 ? 'good' :
               overallHealth > 40 ? 'fair' : 'poor'
    };
}