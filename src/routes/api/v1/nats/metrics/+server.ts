import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getNATSService } from '$lib/services/nats-messaging-service.js';

/**
 * GET /api/v1/nats/metrics
 * Get NATS messaging metrics and statistics
 */
export const GET: RequestHandler = async () => {
  try {
    const natsService = getNATSService();
    
    if (!natsService) {
      return json({
        error: 'NATS service not available',
        metrics: null,
        timestamp: Date.now()
      }, { status: 503 });
    }

    const status = natsService.connectionStatus;
    const messageStats = natsService.messageStats;
    const subscriptions = natsService.getSubscriptionStats();

    // Calculate derived metrics
    const currentTime = Date.now();
    const uptimeMs = status.lastConnected ? currentTime - status.lastConnected : 0;
    const uptimeHours = uptimeMs / (1000 * 60 * 60);

    // Message rate calculations (messages per hour)
    const publishRate = uptimeHours > 0 ? messageStats.published / uptimeHours : 0;
    const receiveRate = uptimeHours > 0 ? messageStats.received / uptimeHours : 0;

    // Bandwidth calculations (bytes per second)
    const uptimeSeconds = uptimeMs / 1000;
    const inboundBandwidth = uptimeSeconds > 0 ? status.bytesIn / uptimeSeconds : 0;
    const outboundBandwidth = uptimeSeconds > 0 ? status.bytesOut / uptimeSeconds : 0;

    // Health metrics
    const healthScore = calculateHealthScore(status, messageStats);
    const performanceGrade = getPerformanceGrade(publishRate, receiveRate, status.reconnectAttempts);

    return json({
      service: {
        name: 'Legal AI NATS Messaging',
        version: '1.0.0',
        uptime: {
          milliseconds: uptimeMs,
          seconds: Math.floor(uptimeSeconds),
          minutes: Math.floor(uptimeMs / (1000 * 60)),
          hours: Math.floor(uptimeHours),
          formatted: formatUptime(uptimeMs)
        }
      },
      connection: {
        status: status.connected ? 'connected' : 'disconnected',
        health: healthScore >= 80 ? 'excellent' : (healthScore >= 60 ? 'good' : (healthScore >= 40 ? 'fair' : 'poor')),
        healthScore,
        lastConnected: status.lastConnected,
        reconnectAttempts: status.reconnectAttempts,
        errorMessage: status.error
      },
      messaging: {
        published: {
          total: messageStats.published,
          rate: {
            perHour: Math.round(publishRate * 100) / 100,
            perMinute: Math.round((publishRate / 60) * 100) / 100,
            perSecond: Math.round((publishRate / 3600) * 100) / 100
          }
        },
        received: {
          total: messageStats.received,
          rate: {
            perHour: Math.round(receiveRate * 100) / 100,
            perMinute: Math.round((receiveRate / 60) * 100) / 100,
            perSecond: Math.round((receiveRate / 3600) * 100) / 100
          }
        },
        queued: messageStats.queued,
        totalThroughput: messageStats.published + messageStats.received
      },
      bandwidth: {
        inbound: {
          total: status.bytesIn,
          rate: {
            bytesPerSecond: Math.round(inboundBandwidth),
            kbPerSecond: Math.round((inboundBandwidth / 1024) * 100) / 100,
            mbPerSecond: Math.round((inboundBandwidth / (1024 * 1024)) * 100) / 100
          }
        },
        outbound: {
          total: status.bytesOut,
          rate: {
            bytesPerSecond: Math.round(outboundBandwidth),
            kbPerSecond: Math.round((outboundBandwidth / 1024) * 100) / 100,
            mbPerSecond: Math.round((outboundBandwidth / (1024 * 1024)) * 100) / 100
          }
        },
        totalBytes: status.bytesIn + status.bytesOut
      },
      subscriptions: {
        total: status.subscriptions,
        active: subscriptions.length,
        details: subscriptions.map(sub => ({
          subject: sub.subject,
          messageCount: sub.messageCount,
          uptime: currentTime - sub.createdAt,
          messagesPerHour: sub.messageCount > 0 ? 
            (sub.messageCount / ((currentTime - sub.createdAt) / (1000 * 60 * 60))) : 0
        })),
        subjectBreakdown: getSubjectBreakdown(subscriptions)
      },
      performance: {
        grade: performanceGrade,
        metrics: {
          averageLatency: 'N/A', // Would need actual latency measurements
          reliability: calculateReliability(status.reconnectAttempts, uptimeMs),
          efficiency: calculateEfficiency(messageStats, status.bytesIn + status.bytesOut)
        }
      },
      system: {
        timestamp: currentTime,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        environment: 'development' // Would be determined from env
      }
    });

  } catch (error) {
    console.error('NATS metrics API error:', error);
    return json({ 
      error: 'Failed to get NATS metrics',
      details: error.message,
      timestamp: Date.now()
    }, { status: 500 });
  }
};

// Helper functions
function calculateHealthScore(status: any, messageStats: any): number {
  let score = 0;
  
  // Connection status (40 points)
  if (status.connected) score += 40;
  else if (status.connecting) score += 20;
  
  // Message throughput (30 points)
  const totalMessages = messageStats.published + messageStats.received;
  if (totalMessages > 100) score += 30;
  else if (totalMessages > 50) score += 20;
  else if (totalMessages > 10) score += 10;
  
  // Reliability (30 points)
  if (status.reconnectAttempts === 0) score += 30;
  else if (status.reconnectAttempts < 3) score += 20;
  else if (status.reconnectAttempts < 10) score += 10;
  
  return Math.min(score, 100);
}

function getPerformanceGrade(publishRate: number, receiveRate: number, reconnects: number): string {
  const totalRate = publishRate + receiveRate;
  
  if (totalRate > 1000 && reconnects === 0) return 'A+';
  if (totalRate > 500 && reconnects < 2) return 'A';
  if (totalRate > 100 && reconnects < 5) return 'B';
  if (totalRate > 50 && reconnects < 10) return 'C';
  if (totalRate > 10) return 'D';
  return 'F';
}

function calculateReliability(reconnects: number, uptime: number): number {
  if (uptime === 0) return 0;
  
  const uptimeHours = uptime / (1000 * 60 * 60);
  const reconnectsPerHour = reconnects / Math.max(uptimeHours, 1);
  
  // Reliability decreases with reconnection frequency
  return Math.max(0, 100 - (reconnectsPerHour * 10));
}

function calculateEfficiency(messageStats: any, totalBytes: number): number {
  const totalMessages = messageStats.published + messageStats.received;
  if (totalMessages === 0) return 100;
  
  const bytesPerMessage = totalBytes / totalMessages;
  
  // Efficiency based on bytes per message (lower is better)
  if (bytesPerMessage < 1000) return 100;
  if (bytesPerMessage < 5000) return 80;
  if (bytesPerMessage < 10000) return 60;
  if (bytesPerMessage < 50000) return 40;
  return 20;
}

function getSubjectBreakdown(subscriptions: any[]): Record<string, number> {
  const breakdown: Record<string, number> = {};
  
  subscriptions.forEach(sub => {
    const category = sub.subject.split('.')[0] || 'unknown';
    breakdown[category] = (breakdown[category] || 0) + 1;
  });
  
  return breakdown;
}

function formatUptime(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (days > 0) return `${days}d ${hours % 24}h ${minutes % 60}m`;
  if (hours > 0) return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
}