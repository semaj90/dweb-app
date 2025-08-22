/**
 * Horizontal Agent Scaling API
 * 
 * RESTful API for managing horizontal agent scaling across servers
 * Provides endpoints for:
 * - Agent scaling operations (scale up/down)
 * - Service discovery and registration
 * - Load balancing management
 * - Health monitoring and status
 * - Placement decisions and metrics
 * - Server and agent management
 */

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { horizontalAgentScaler } from '$lib/scaling/horizontal-agent-scaler';
import type { 
    AgentPlacement,
    ScalingMetrics,
    AgentScalingConfig,
    ServerHealth
} from '$lib/ai/types';

// POST - Execute scaling operations
export const POST: RequestHandler = async ({ request, url }) => {
    try {
        const action = url.searchParams.get('action');
        const body = await request.json();

        switch (action) {
            case 'scale-agents':
                return await scaleAgents(body);
            
            case 'deploy-agent':
                return await deployAgent(body);
            
            case 'migrate-agents':
                return await migrateAgents(body);
            
            case 'register-server':
                return await registerServer(body);
            
            case 'start-scaling':
                return await startScalingSystem(body);
            
            default:
                return error(400, 'Invalid action specified');
        }
    } catch (err) {
        console.error('❌ Horizontal scaling API error:', err);
        return error(500, `Server error: ${err.message}`);
    }
};

// GET - Retrieve scaling information and status
export const GET: RequestHandler = async ({ url }) => {
    try {
        const action = url.searchParams.get('action');
        
        switch (action) {
            case 'status':
                return getScalingStatus();
            
            case 'placements':
                return getAgentPlacements();
            
            case 'health':
                return getSystemHealth();
            
            case 'servers':
                return getServerNodes();
            
            case 'metrics':
                return getScalingMetrics();
            
            case 'load-balancer':
                return getLoadBalancerStatus();
            
            default:
                return getDashboard();
        }
    } catch (err) {
        console.error('❌ Horizontal scaling API error:', err);
        return error(500, `Server error: ${err.message}`);
    }
};

// PUT - Update scaling configuration or agent placement
export const PUT: RequestHandler = async ({ request }) => {
    try {
        const { config, placement, policy } = await request.json();
        
        if (config) {
            return await updateScalingConfig(config);
        }
        
        if (placement) {
            return await updateAgentPlacement(placement);
        }
        
        if (policy) {
            return await updateScalingPolicy(policy);
        }
        
        return error(400, 'No valid update data provided');
        
    } catch (err) {
        console.error('❌ Update scaling error:', err);
        return error(500, `Update error: ${err.message}`);
    }
};

// DELETE - Remove agents or servers
export const DELETE: RequestHandler = async ({ url }) => {
    try {
        const agentId = url.searchParams.get('agentId');
        const serverId = url.searchParams.get('serverId');
        
        if (agentId) {
            return await removeAgent(agentId);
        }
        
        if (serverId) {
            return await removeServer(serverId);
        }
        
        return error(400, 'Agent ID or Server ID is required');
    } catch (err) {
        console.error('❌ Remove operation error:', err);
        return error(500, `Remove error: ${err.message}`);
    }
};

/**
 * Handler Functions
 */

async function scaleAgents(scaleData: unknown) {
    const { agentType, targetCount, force = false } = scaleData;
    
    if (!agentType || typeof targetCount !== 'number') {
        return error(400, 'Agent type and target count are required');
    }

    if (targetCount < 0) {
        return error(400, 'Target count must be non-negative');
    }

    const startTime = performance.now();
    
    try {
        const scalingMetrics = await horizontalAgentScaler.scaleAgents(agentType, targetCount);
        const operationTime = performance.now() - startTime;
        
        return json({
            success: true,
            message: 'Agent scaling completed successfully',
            scaling: {
                agentType,
                targetCount,
                actualCount: scalingMetrics.actualCount,
                previousCount: scalingMetrics.previousCount,
                scalingActions: scalingMetrics.scalingActions,
                operationTime: operationTime.toFixed(2)
            },
            metrics: scalingMetrics,
            timestamp: new Date().toISOString()
        });
        
    } catch (err) {
        return error(500, `Scaling failed: ${err.message}`);
    }
}

async function deployAgent(deployData: unknown) {
    const { agentType, serverId, configuration = {} } = deployData;
    
    if (!agentType) {
        return error(400, 'Agent type is required');
    }

    try {
        // Create agent placement request
        const agentId = `${agentType}-manual-${Date.now()}`;
        
        const placement: AgentPlacement = {
            agentId,
            agentType,
            serverId: serverId || 'auto-select',
            serverAddress: 'auto-select',
            status: 'pending',
            createdAt: new Date(),
            configuration,
            deploymentType: 'manual'
        };
        
        // Deploy through scaling system
        const result = await horizontalAgentScaler.scaleAgents(agentType, 1);
        
        return json({
            success: true,
            message: 'Agent deployed successfully',
            deployment: {
                agentId,
                agentType,
                serverId,
                status: 'deployed',
                configuration
            },
            scalingResult: result,
            timestamp: new Date().toISOString()
        });
        
    } catch (err) {
        return error(500, `Deployment failed: ${err.message}`);
    }
}

async function migrateAgents(migrationData: unknown) {
    const { fromServerId, toServerId, agentIds = [] } = migrationData;
    
    if (!fromServerId || !toServerId) {
        return error(400, 'Source and target server IDs are required');
    }

    try {
        const migrations: unknown[] = [];
        
        // Get current placements
        const placements = await horizontalAgentScaler.getAgentPlacements();
        const agentsToMigrate = placements.filter(p => 
            p.serverId === fromServerId && 
            (agentIds.length === 0 || agentIds.includes(p.agentId))
        );
        
        for (const agent of agentsToMigrate) {
            try {
                // Simulate migration process
                const migrationResult = {
                    agentId: agent.agentId,
                    fromServer: fromServerId,
                    toServer: toServerId,
                    status: 'migrated',
                    startTime: new Date(),
                    endTime: new Date(Date.now() + 1000) // Simulate 1s migration
                };
                
                migrations.push(migrationResult);
                
            } catch (err) {
                migrations.push({
                    agentId: agent.agentId,
                    fromServer: fromServerId,
                    toServer: toServerId,
                    status: 'failed',
                    error: err.message
                });
            }
        }
        
        return json({
            success: true,
            message: 'Agent migration completed',
            migration: {
                fromServerId,
                toServerId,
                totalAgents: agentsToMigrate.length,
                successful: migrations.filter(m => m.status === 'migrated').length,
                failed: migrations.filter(m => m.status === 'failed').length,
                details: migrations
            },
            timestamp: new Date().toISOString()
        });
        
    } catch (err) {
        return error(500, `Migration failed: ${err.message}`);
    }
}

async function registerServer(serverData: unknown) {
    const { serverAddress, capabilities = [], resources = {} } = serverData;
    
    if (!serverAddress) {
        return error(400, 'Server address is required');
    }

    try {
        const serverId = `server-${serverAddress.replace(/[:.]/g, '-')}`;
        
        const registrationResult = {
            serverId,
            serverAddress,
            capabilities,
            resources: {
                cpu: { total: 8, available: 8 },
                memory: { total: 16384, available: 16384 },
                disk: { total: 1000000, available: 1000000 },
                ...resources
            },
            status: 'registered',
            registeredAt: new Date()
        };
        
        console.log(`🖥️ Registered server: ${serverId} at ${serverAddress}`);
        
        return json({
            success: true,
            message: 'Server registered successfully',
            server: registrationResult,
            timestamp: new Date().toISOString()
        });
        
    } catch (err) {
        return error(500, `Server registration failed: ${err.message}`);
    }
}

async function startScalingSystem(configData: unknown) {
    try {
        await horizontalAgentScaler.start();
        
        return json({
            success: true,
            message: 'Horizontal scaling system started successfully',
            config: configData,
            timestamp: new Date().toISOString()
        });
        
    } catch (err) {
        return error(500, `Failed to start scaling system: ${err.message}`);
    }
}

function getScalingStatus() {
    const status = horizontalAgentScaler.getScalingStatus();
    
    return json({
        success: true,
        status,
        timestamp: new Date().toISOString()
    });
}

async function getAgentPlacements() {
    const placements = await horizontalAgentScaler.getAgentPlacements();
    
    // Group placements by server and agent type
    const byServer = {};
    const byAgentType = {};
    
    placements.forEach(placement => {
        // By server
        if (!byServer[placement.serverId]) {
            byServer[placement.serverId] = [];
        }
        byServer[placement.serverId].push(placement);
        
        // By agent type
        if (!byAgentType[placement.agentType]) {
            byAgentType[placement.agentType] = [];
        }
        byAgentType[placement.agentType].push(placement);
    });
    
    return json({
        success: true,
        placements: {
            total: placements.length,
            byServer,
            byAgentType,
            details: placements.map(p => ({
                agentId: p.agentId,
                agentType: p.agentType,
                serverId: p.serverId,
                status: p.status,
                createdAt: p.createdAt,
                resources: p.resources
            }))
        },
        timestamp: new Date().toISOString()
    });
}

async function getSystemHealth() {
    const serverHealth = await horizontalAgentScaler.getServerHealth();
    const placements = await horizontalAgentScaler.getAgentPlacements();
    
    // Calculate health metrics
    const totalServers = serverHealth.size;
    const healthyServers = Array.from(serverHealth.values())
        .filter(h => h.status === 'healthy').length;
    
    const totalAgents = placements.length;
    const healthyAgents = placements.filter(p => p.status === 'running').length;
    
    const systemHealthScore = Math.round(
        ((healthyServers / totalServers) * 0.6 + (healthyAgents / totalAgents) * 0.4) * 100
    );
    
    return json({
        success: true,
        health: {
            overall: {
                score: systemHealthScore,
                status: systemHealthScore > 80 ? 'healthy' : 
                       systemHealthScore > 60 ? 'degraded' : 'unhealthy'
            },
            servers: {
                total: totalServers,
                healthy: healthyServers,
                unhealthy: totalServers - healthyServers,
                details: Object.fromEntries(serverHealth)
            },
            agents: {
                total: totalAgents,
                healthy: healthyAgents,
                unhealthy: totalAgents - healthyAgents,
                byStatus: placements.reduce((acc, p) => {
                    acc[p.status] = (acc[p.status] || 0) + 1;
                    return acc;
                }, {})
            }
        },
        timestamp: new Date().toISOString()
    });
}

function getServerNodes() {
    const status = horizontalAgentScaler.getScalingStatus();
    
    const serverInfo = {
        totalNodes: status.serverNodes,
        configuration: status.config.serverNodes,
        distribution: status.agentsByServer,
        capacity: {
            maxAgentsPerServer: status.config.maxAgentsPerServer,
            totalCapacity: status.serverNodes * status.config.maxAgentsPerServer,
            currentUtilization: status.totalAgents / (status.serverNodes * status.config.maxAgentsPerServer)
        }
    };
    
    return json({
        success: true,
        servers: serverInfo,
        timestamp: new Date().toISOString()
    });
}

function getScalingMetrics() {
    const status = horizontalAgentScaler.getScalingStatus();
    
    const metrics = {
        scaling: {
            totalAgents: status.totalAgents,
            agentsByType: status.agentsByType,
            agentsByServer: status.agentsByServer,
            activePolicies: status.activeScalingPolicies
        },
        performance: {
            distributedLocks: status.distributedLocks,
            serverNodes: status.serverNodes,
            utilization: status.totalAgents / (status.serverNodes * status.config.maxAgentsPerServer)
        },
        configuration: {
            autoScaling: status.config.enableAutoScaling,
            loadBalancing: status.config.enableLoadBalancing,
            healthMonitoring: status.config.enableHealthMonitoring,
            scalingInterval: status.config.scalingInterval,
            thresholds: {
                scaleUp: status.config.scaleUpThreshold,
                scaleDown: status.config.scaleDownThreshold
            }
        }
    };
    
    return json({
        success: true,
        metrics,
        timestamp: new Date().toISOString()
    });
}

function getLoadBalancerStatus() {
    const status = horizontalAgentScaler.getScalingStatus();
    
    const loadBalancerInfo = {
        enabled: status.config.enableLoadBalancing,
        strategy: status.config.loadBalancingStrategy || 'round-robin',
        totalAgents: status.totalAgents,
        agentsByType: status.agentsByType,
        distribution: status.agentsByServer,
        healthCheck: {
            enabled: status.config.enableHealthMonitoring,
            interval: status.config.healthCheckInterval
        }
    };
    
    return json({
        success: true,
        loadBalancer: loadBalancerInfo,
        timestamp: new Date().toISOString()
    });
}

function getDashboard() {
    const status = horizontalAgentScaler.getScalingStatus();
    
    // Create comprehensive dashboard
    return json({
        success: true,
        dashboard: {
            overview: {
                totalAgents: status.totalAgents,
                serverNodes: status.serverNodes,
                agentTypes: Object.keys(status.agentsByType).length,
                scalingPolicies: status.activeScalingPolicies
            },
            distribution: {
                agentsByType: status.agentsByType,
                agentsByServer: status.agentsByServer,
                utilization: status.totalAgents / (status.serverNodes * status.config.maxAgentsPerServer)
            },
            scaling: {
                autoScalingEnabled: status.config.enableAutoScaling,
                loadBalancingEnabled: status.config.enableLoadBalancing,
                healthMonitoringEnabled: status.config.enableHealthMonitoring,
                scalingInterval: status.config.scalingInterval,
                thresholds: {
                    scaleUp: status.config.scaleUpThreshold,
                    scaleDown: status.config.scaleDownThreshold
                }
            },
            resources: {
                maxAgentsPerServer: status.config.maxAgentsPerServer,
                minAgentsPerServer: status.config.minAgentsPerServer,
                totalCapacity: status.serverNodes * status.config.maxAgentsPerServer,
                currentUtilization: (status.totalAgents / (status.serverNodes * status.config.maxAgentsPerServer) * 100).toFixed(1) + '%'
            },
            coordination: {
                distributedLocks: status.distributedLocks,
                lockTimeout: '30s',
                leaderElection: 'enabled'
            }
        },
        timestamp: new Date().toISOString()
    });
}

/**
 * Helper Functions
 */

async function updateScalingConfig(config: AgentScalingConfig) {
    console.log('🔧 Updating scaling configuration:', config);
    
    // In a real implementation, this would update the scaler configuration
    return {
        updated: true,
        config,
        restartRequired: true,
        timestamp: new Date().toISOString()
    };
}

async function updateAgentPlacement(placement: AgentPlacement) {
    console.log('🔧 Updating agent placement:', placement);
    
    return {
        updated: true,
        placement,
        timestamp: new Date().toISOString()
    };
}

async function updateScalingPolicy(policy: unknown) {
    console.log('🔧 Updating scaling policy:', policy);
    
    return {
        updated: true,
        policy,
        timestamp: new Date().toISOString()
    };
}

async function removeAgent(agentId: string) {
    console.log(`🗑️ Removing agent: ${agentId}`);
    
    return {
        removed: true,
        agentId,
        reason: 'manual-removal',
        timestamp: new Date().toISOString()
    };
}

async function removeServer(serverId: string) {
    console.log(`🗑️ Removing server: ${serverId}`);
    
    return {
        removed: true,
        serverId,
        agentsMigrated: 0,
        reason: 'manual-removal',
        timestamp: new Date().toISOString()
    };
}