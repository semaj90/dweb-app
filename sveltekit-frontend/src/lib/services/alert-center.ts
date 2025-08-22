/**
 * Alert Center - Legal AI Platform
 * 
 * Enterprise-grade alert routing, management and notification system
 * for legal AI platform with real-time monitoring, compliance tracking,
 * and automated remediation capabilities.
 * 
 * Features:
 * - Multi-tier alert classification and routing
 * - Legal compliance alert monitoring
 * - Performance anomaly detection
 * - Automated incident response
 * - Chain of custody alert tracking
 * - Client privilege breach detection
 * - Real-time dashboard notifications
 * - Integration with NATS messaging
 * 
 * @author Legal AI Platform Team
 * @version 3.2.0
 * @lastModified 2025-01-20
 */

import fs from 'fs';
import path from 'path';
import type { NATSMessagingService } from './nats-messaging-service';
import { 
  getQUICMetrics, 
  getAggregateAnomaliesLast5m, 
  getStageBaselineSnapshot, 
  resetBudgetCounters, 
  getBudgetCounters 
} from './pipeline-metrics';

// ===== ALERT INTERFACES & TYPES =====

/**
 * Runtime Alert Interface with Legal Compliance
 */
export interface RuntimeAlert {
  id: string;
  type: AlertType;
  severity: AlertSeverity;
  category: AlertCategory;
  message: string;
  description?: string;
  timestamp: number;
  caseId?: string;
  userId?: string;
  clientId?: string;
  legalContext?: LegalAlertContext;
  metadata?: Record<string, any>;
  acknowledgementRequired: boolean;
  escalationLevel: number;
  retentionPeriod: number; // days
  complianceFlags: string[];
}

/**
 * Legal Alert Context for Compliance Tracking
 */
export interface LegalAlertContext {
  confidentialityLevel: 'public' | 'confidential' | 'privileged' | 'attorney_client';
  chainOfCustodyId?: string;
  evidenceId?: string;
  documentId?: string;
  privilegeScope?: string;
  clientMatter?: string;
  jurisdictionCode?: string;
  regulatoryCompliance: string[];
  auditTrailRequired: boolean;
  legalHoldStatus?: 'active' | 'released' | 'pending';
}

/**
 * Alert Type Classifications
 */
export type AlertType = 
  // System Performance
  | 'p99_latency_exceeded'
  | 'error_spike'
  | 'pipeline_anomaly_spike'
  | 'service_unavailable'
  | 'memory_threshold_exceeded'
  | 'cpu_threshold_exceeded'
  | 'disk_space_low'
  // Legal Compliance
  | 'privilege_breach_detected'
  | 'unauthorized_access_attempt'
  | 'chain_of_custody_break'
  | 'retention_policy_violation'
  | 'confidentiality_breach'
  | 'document_tampering_detected'
  | 'illegal_discovery_access'
  // Security & Authentication
  | 'failed_authentication'
  | 'suspicious_activity'
  | 'data_exfiltration_attempt'
  | 'unauthorized_privilege_escalation'
  | 'concurrent_session_anomaly'
  // Client Management
  | 'client_data_anomaly'
  | 'billing_discrepancy'
  | 'deadline_approaching'
  | 'court_filing_deadline'
  | 'statute_of_limitations_warning'
  // AI/ML Specific
  | 'model_prediction_anomaly'
  | 'embedding_quality_degradation'
  | 'rag_accuracy_decline'
  | 'gpu_memory_overflow'
  | 'inference_timeout';

/**
 * Alert Severity Levels
 */
export type AlertSeverity = 'info' | 'warning' | 'critical' | 'emergency';

/**
 * Alert Categories for Organization
 */
export type AlertCategory = 
  | 'system_performance'
  | 'legal_compliance'
  | 'security_breach'
  | 'client_management'
  | 'ai_ml_operations'
  | 'data_integrity'
  | 'regulatory_compliance';

/**
 * Alert Configuration Settings
 */
export interface AlertConfig {
  ringBufferSize: number;
  retentionDays: number;
  escalationTimeouts: Record<AlertSeverity, number>;
  notificationChannels: NotificationChannel[];
  complianceSettings: ComplianceAlertSettings;
  autoRemediation: AutoRemediationSettings;
}

/**
 * Notification Channel Configuration
 */
export interface NotificationChannel {
  type: 'email' | 'sms' | 'slack' | 'teams' | 'webhook' | 'nats' | 'dashboard';
  endpoint: string;
  credentials?: Record<string, string>;
  filters: AlertFilterConfig;
  enabled: boolean;
  priority: number;
}

/**
 * Alert Filter Configuration
 */
export interface AlertFilterConfig {
  severityLevels: AlertSeverity[];
  categories: AlertCategory[];
  types: AlertType[];
  excludePatterns: string[];
  includePatterns: string[];
  timeWindows?: TimeWindowConfig[];
}

/**
 * Time Window Configuration
 */
export interface TimeWindowConfig {
  startHour: number;
  endHour: number;
  timezone: string;
  daysOfWeek: number[];
}

/**
 * Compliance Alert Settings
 */
export interface ComplianceAlertSettings {
  privilegeBreachThreshold: number;
  chainOfCustodyTimeout: number;
  retentionViolationGracePeriod: number;
  confidentialityLevelMapping: Record<string, AlertSeverity>;
  auditLogRequired: boolean;
  regulatoryNotificationRequired: boolean;
}

/**
 * Auto-Remediation Settings
 */
export interface AutoRemediationSettings {
  enabled: boolean;
  maxAttempts: number;
  cooldownPeriod: number;
  allowedActions: string[];
  escalationOnFailure: boolean;
  requireApproval: boolean;
}

/**
 * Alert Statistics
 */
export interface AlertStatistics {
  totalAlerts: number;
  alertsBySeverity: Record<AlertSeverity, number>;
  alertsByCategory: Record<AlertCategory, number>;
  alertsByType: Record<string, number>;
  averageResponseTime: number;
  escalationRate: number;
  autoRemediationSuccessRate: number;
  complianceViolations: number;
  mostFrequentAlerts: Array<{ type: AlertType; count: number }>;
  timeDistribution: Record<string, number>;
}

// ===== ALERT CENTER IMPLEMENTATION =====

/**
 * Enterprise Alert Center for Legal AI Platform
 */
export class AlertCenter {
  private alerts: RuntimeAlert[] = [];
  private config: AlertConfig;
  private natsService?: NATSMessagingService;
  private sustainedP99Breaches = 0;
  private lastP99Ok = Date.now();
  private autosolveInFlight = false;
  private lastAutosolveTs = 0;
  private persistenceState: PersistedState | null = null;
  
  // Constants
  private readonly RING_SIZE = 500;
  private readonly SUSTAINED_P99_THRESHOLD = 3;
  private readonly AUTOSOLVE_COOLDOWN_MS = 5 * 60 * 1000;
  private readonly SYSTEM_ALERTS_SUBJECT = 'system.alerts';
  private readonly LEGAL_ALERTS_SUBJECT = 'legal.compliance.alerts';
  private readonly RUNTIME_DIR = path.resolve(process.cwd(), '.runtime');
  private readonly STATE_FILE = path.join(this.RUNTIME_DIR, 'alert-center-state.json');
  
  constructor(config?: Partial<AlertConfig>, natsService?: NATSMessagingService) {
    this.config = this.buildDefaultConfig(config);
    this.natsService = natsService;
    this.initializePersistence();
    this.startBackgroundTasks();
  }
  
  /**
   * Build default configuration with overrides
   */
  private buildDefaultConfig(overrides?: Partial<AlertConfig>): AlertConfig {
    const defaultConfig: AlertConfig = {
      ringBufferSize: this.RING_SIZE,
      retentionDays: 90,
      escalationTimeouts: {
        info: 24 * 60 * 60 * 1000,      // 24 hours
        warning: 4 * 60 * 60 * 1000,    // 4 hours
        critical: 30 * 60 * 1000,       // 30 minutes
        emergency: 5 * 60 * 1000        // 5 minutes
      },
      notificationChannels: [
        {
          type: 'dashboard',
          endpoint: '/api/v1/notifications/dashboard',
          filters: {
            severityLevels: ['info', 'warning', 'critical', 'emergency'],
            categories: ['system_performance', 'legal_compliance', 'security_breach'],
            types: [],
            excludePatterns: [],
            includePatterns: []
          },
          enabled: true,
          priority: 1
        },
        {
          type: 'nats',
          endpoint: this.SYSTEM_ALERTS_SUBJECT,
          filters: {
            severityLevels: ['critical', 'emergency'],
            categories: ['legal_compliance', 'security_breach'],
            types: [],
            excludePatterns: [],
            includePatterns: []
          },
          enabled: true,
          priority: 2
        }
      ],
      complianceSettings: {
        privilegeBreachThreshold: 1,
        chainOfCustodyTimeout: 60 * 60 * 1000, // 1 hour
        retentionViolationGracePeriod: 7 * 24 * 60 * 60 * 1000, // 7 days
        confidentialityLevelMapping: {
          'public': 'info',
          'confidential': 'warning',
          'privileged': 'critical',
          'attorney_client': 'emergency'
        },
        auditLogRequired: true,
        regulatoryNotificationRequired: true
      },
      autoRemediation: {
        enabled: true,
        maxAttempts: 3,
        cooldownPeriod: this.AUTOSOLVE_COOLDOWN_MS,
        allowedActions: ['restart_service', 'clear_cache', 'rotate_logs'],
        escalationOnFailure: true,
        requireApproval: false
      }
    };
    
    return { ...defaultConfig, ...overrides };
  }
  
  /**
   * Initialize persistence layer
   */
  private initializePersistence(): void {
    try {
      if (fs.existsSync(this.STATE_FILE)) {
        const raw = fs.readFileSync(this.STATE_FILE, 'utf8');
        this.persistenceState = JSON.parse(raw) as PersistedState;
        
        // Restore runtime state
        this.sustainedP99Breaches = this.persistenceState.sustainedP99Breaches || 0;
        this.lastP99Ok = this.persistenceState.lastP99Ok || Date.now();
        
        console.log('Alert Center: Persistence state loaded successfully');
      }
    } catch (error) {
      console.warn('Alert Center: Failed to load persistence state:', (error as Error).message);
    }
  }
  
  /**
   * Start background tasks
   */
  private startBackgroundTasks(): void {
    // Periodic persistence
    setInterval(() => this.persistState(), 60_000).unref?.();
    
    // Daily reset scheduler
    this.scheduleDailyReset();
    
    // Alert cleanup
    setInterval(() => this.cleanupExpiredAlerts(), 60 * 60 * 1000).unref?.(); // Every hour
    
    console.log('Alert Center: Background tasks initialized');
  }
  
  /**
   * Create and route alerts
   */
  public async routeAlerts(
    alertCodes: string[], 
    context: Record<string, any> = {}
  ): Promise<RuntimeAlert[]> {
    if (!alertCodes.length) {
      this.sustainedP99Breaches = 0;
      return [];
    }
    
    const alerts: RuntimeAlert[] = [];
    const quicMetrics = getQUICMetrics();
    
    for (const code of alertCodes) {
      const alert = this.createAlert(code as AlertType, quicMetrics, context);
      await this.processAlert(alert);
      alerts.push(alert);
    }
    
    // Handle sustained P99 tracking
    this.updateP99Tracking(alertCodes);
    
    return alerts;
  }
  
  /**
   * Create a new alert
   */
  private createAlert(
    type: AlertType, 
    quicMetrics: any, 
    context: Record<string, any>
  ): RuntimeAlert {
    const severity = this.determineSeverity(type);
    const category = this.determineCategory(type);
    const legalContext = this.extractLegalContext(context);
    
    return {
      id: this.generateAlertId(),
      type,
      severity,
      category,
      message: this.humanizeAlertMessage(type, quicMetrics),
      description: this.generateAlertDescription(type, context),
      timestamp: Date.now(),
      caseId: context.caseId,
      userId: context.userId,
      clientId: context.clientId,
      legalContext,
      metadata: {
        quicP99: quicMetrics.p99,
        quicErrors1m: quicMetrics.error_rate_1m,
        anomalies5m: getAggregateAnomaliesLast5m(),
        systemLoad: context.systemLoad,
        ...context
      },
      acknowledgementRequired: severity === 'critical' || severity === 'emergency',
      escalationLevel: 0,
      retentionPeriod: this.calculateRetentionPeriod(type, legalContext),
      complianceFlags: this.generateComplianceFlags(type, legalContext)
    };
  }
  
  /**
   * Process alert through notification channels
   */
  private async processAlert(alert: RuntimeAlert): Promise<void> {
    // Add to ring buffer
    this.addToRingBuffer(alert);
    
    // Log alert
    this.logAlert(alert);
    
    // Send notifications
    await this.sendNotifications(alert);
    
    // Trigger auto-remediation if applicable
    if (this.shouldTriggerAutoRemediation(alert)) {
      await this.triggerAutoRemediation(alert);
    }
  }
  
  /**
   * Add alert to ring buffer
   */
  private addToRingBuffer(alert: RuntimeAlert): void {
    this.alerts.push(alert);
    if (this.alerts.length > this.config.ringBufferSize) {
      this.alerts.shift();
    }
  }
  
  /**
   * Log alert to console and audit trail
   */
  private logAlert(alert: RuntimeAlert): void {
    const logLevel = this.mapSeverityToLogLevel(alert.severity);
    console[logLevel](
      `[ALERT][${alert.severity.toUpperCase()}][${alert.category}] ${alert.type}: ${alert.message}`,
      {
        id: alert.id,
        timestamp: new Date(alert.timestamp).toISOString(),
        legalContext: alert.legalContext,
        metadata: alert.metadata
      }
    );
  }
  
  /**
   * Send notifications through configured channels
   */
  private async sendNotifications(alert: RuntimeAlert): Promise<void> {
    const channels = this.getMatchingChannels(alert);
    
    for (const channel of channels) {
      try {
        await this.sendNotificationToChannel(alert, channel);
      } catch (error) {
        console.error(`Failed to send notification via ${channel.type}:`, (error as Error).message);
      }
    }
  }
  
  /**
   * Send notification to specific channel
   */
  private async sendNotificationToChannel(
    alert: RuntimeAlert, 
    channel: NotificationChannel
  ): Promise<void> {
    switch (channel.type) {
      case 'nats':
        if (this.natsService) {
          await this.natsService.publish(channel.endpoint, {
            alert: {
              id: alert.id,
              type: alert.type,
              severity: alert.severity,
              category: alert.category,
              message: alert.message,
              timestamp: alert.timestamp,
              legalContext: alert.legalContext
            }
          });
        }
        break;
        
      case 'dashboard':
        // Dashboard notifications handled by WebSocket or SSE
        if (typeof window !== 'undefined' && (window as any).alertDashboard) {
          (window as any).alertDashboard.addAlert(alert);
        }
        break;
        
      case 'webhook':
        await fetch(channel.endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ alert, timestamp: Date.now() })
        });
        break;
        
      default:
        console.warn(`Unsupported notification channel type: ${channel.type}`);
    }
  }
  
  /**
   * Get notification channels that match alert criteria
   */
  private getMatchingChannels(alert: RuntimeAlert): NotificationChannel[] {
    return this.config.notificationChannels
      .filter(channel => {
        if (!channel.enabled) return false;
        
        const filters = channel.filters;
        
        // Check severity filter
        if (filters.severityLevels.length > 0 && !filters.severityLevels.includes(alert.severity)) {
          return false;
        }
        
        // Check category filter
        if (filters.categories.length > 0 && !filters.categories.includes(alert.category)) {
          return false;
        }
        
        // Check type filter
        if (filters.types.length > 0 && !filters.types.includes(alert.type)) {
          return false;
        }
        
        return true;
      })
      .sort((a, b) => a.priority - b.priority);
  }
  
  /**
   * Determine alert severity based on type
   */
  private determineSeverity(type: AlertType): AlertSeverity {
    const severityMap: Record<AlertType, AlertSeverity> = {
      // System Performance
      'p99_latency_exceeded': 'warning',
      'error_spike': 'critical',
      'pipeline_anomaly_spike': 'critical',
      'service_unavailable': 'critical',
      'memory_threshold_exceeded': 'warning',
      'cpu_threshold_exceeded': 'warning',
      'disk_space_low': 'warning',
      
      // Legal Compliance - High severity due to legal implications
      'privilege_breach_detected': 'emergency',
      'unauthorized_access_attempt': 'critical',
      'chain_of_custody_break': 'emergency',
      'retention_policy_violation': 'critical',
      'confidentiality_breach': 'emergency',
      'document_tampering_detected': 'emergency',
      'illegal_discovery_access': 'emergency',
      
      // Security & Authentication
      'failed_authentication': 'warning',
      'suspicious_activity': 'critical',
      'data_exfiltration_attempt': 'emergency',
      'unauthorized_privilege_escalation': 'emergency',
      'concurrent_session_anomaly': 'warning',
      
      // Client Management
      'client_data_anomaly': 'warning',
      'billing_discrepancy': 'warning',
      'deadline_approaching': 'warning',
      'court_filing_deadline': 'critical',
      'statute_of_limitations_warning': 'critical',
      
      // AI/ML Specific
      'model_prediction_anomaly': 'warning',
      'embedding_quality_degradation': 'warning',
      'rag_accuracy_decline': 'warning',
      'gpu_memory_overflow': 'critical',
      'inference_timeout': 'warning'
    };
    
    return severityMap[type] || 'info';
  }
  
  /**
   * Determine alert category based on type
   */
  private determineCategory(type: AlertType): AlertCategory {
    const categoryMap: Record<string, AlertCategory> = {
      'p99_latency_exceeded': 'system_performance',
      'error_spike': 'system_performance',
      'pipeline_anomaly_spike': 'system_performance',
      'service_unavailable': 'system_performance',
      'memory_threshold_exceeded': 'system_performance',
      'cpu_threshold_exceeded': 'system_performance',
      'disk_space_low': 'system_performance',
      
      'privilege_breach_detected': 'legal_compliance',
      'unauthorized_access_attempt': 'security_breach',
      'chain_of_custody_break': 'legal_compliance',
      'retention_policy_violation': 'legal_compliance',
      'confidentiality_breach': 'legal_compliance',
      'document_tampering_detected': 'security_breach',
      'illegal_discovery_access': 'legal_compliance',
      
      'failed_authentication': 'security_breach',
      'suspicious_activity': 'security_breach',
      'data_exfiltration_attempt': 'security_breach',
      'unauthorized_privilege_escalation': 'security_breach',
      'concurrent_session_anomaly': 'security_breach',
      
      'client_data_anomaly': 'client_management',
      'billing_discrepancy': 'client_management',
      'deadline_approaching': 'client_management',
      'court_filing_deadline': 'client_management',
      'statute_of_limitations_warning': 'client_management',
      
      'model_prediction_anomaly': 'ai_ml_operations',
      'embedding_quality_degradation': 'ai_ml_operations',
      'rag_accuracy_decline': 'ai_ml_operations',
      'gpu_memory_overflow': 'ai_ml_operations',
      'inference_timeout': 'ai_ml_operations'
    };
    
    return categoryMap[type] || 'system_performance';
  }
  
  /**
   * Extract legal context from alert context
   */
  private extractLegalContext(context: Record<string, any>): LegalAlertContext | undefined {
    if (!context.legal && !context.caseId && !context.evidenceId) {
      return undefined;
    }
    
    return {
      confidentialityLevel: context.confidentialityLevel || 'public',
      chainOfCustodyId: context.chainOfCustodyId,
      evidenceId: context.evidenceId,
      documentId: context.documentId,
      privilegeScope: context.privilegeScope,
      clientMatter: context.clientMatter,
      jurisdictionCode: context.jurisdictionCode,
      regulatoryCompliance: context.regulatoryCompliance || [],
      auditTrailRequired: context.auditTrailRequired !== false,
      legalHoldStatus: context.legalHoldStatus
    };
  }
  
  /**
   * Generate human-readable alert message
   */
  private humanizeAlertMessage(type: AlertType, quicMetrics: any): string {
    const messages: Record<AlertType, string> = {
      'p99_latency_exceeded': `QUIC p99 latency ${quicMetrics.p99}ms exceeded threshold`,
      'error_spike': `QUIC error spike detected (${quicMetrics.error_rate_1m} errors/min)`,
      'pipeline_anomaly_spike': `Pipeline anomaly spike (${getAggregateAnomaliesLast5m()} anomalies in 5 minutes)`,
      'service_unavailable': 'Critical service is unavailable',
      'memory_threshold_exceeded': 'System memory usage exceeded threshold',
      'cpu_threshold_exceeded': 'CPU usage exceeded threshold',
      'disk_space_low': 'Available disk space is critically low',
      
      'privilege_breach_detected': 'Attorney-client privilege breach detected',
      'unauthorized_access_attempt': 'Unauthorized access attempt to privileged content',
      'chain_of_custody_break': 'Evidence chain of custody integrity compromised',
      'retention_policy_violation': 'Document retention policy violation detected',
      'confidentiality_breach': 'Confidential document access breach',
      'document_tampering_detected': 'Evidence tampering or modification detected',
      'illegal_discovery_access': 'Illegal access to discovery materials',
      
      'failed_authentication': 'Authentication failure detected',
      'suspicious_activity': 'Suspicious user activity pattern detected',
      'data_exfiltration_attempt': 'Potential data exfiltration attempt',
      'unauthorized_privilege_escalation': 'Unauthorized privilege escalation attempt',
      'concurrent_session_anomaly': 'Unusual concurrent session activity',
      
      'client_data_anomaly': 'Client data anomaly detected',
      'billing_discrepancy': 'Billing discrepancy requires attention',
      'deadline_approaching': 'Important deadline approaching',
      'court_filing_deadline': 'Court filing deadline imminent',
      'statute_of_limitations_warning': 'Statute of limitations deadline warning',
      
      'model_prediction_anomaly': 'AI model prediction anomaly detected',
      'embedding_quality_degradation': 'Vector embedding quality degradation',
      'rag_accuracy_decline': 'RAG system accuracy declining',
      'gpu_memory_overflow': 'GPU memory overflow detected',
      'inference_timeout': 'AI inference timeout occurred'
    };
    
    return messages[type] || `Alert: ${type}`;
  }
  
  /**
   * Generate detailed alert description
   */
  private generateAlertDescription(type: AlertType, context: Record<string, any>): string {
    const baseDescription = `Alert triggered for ${type} at ${new Date().toISOString()}`;
    
    if (context.description) {
      return `${baseDescription}. ${context.description}`;
    }
    
    return baseDescription;
  }
  
  /**
   * Generate unique alert ID
   */
  private generateAlertId(): string {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 8);
    return `alert_${timestamp}_${random}`;
  }
  
  /**
   * Calculate retention period based on alert type and legal context
   */
  private calculateRetentionPeriod(type: AlertType, legalContext?: LegalAlertContext): number {
    // Legal compliance alerts require longer retention
    if (legalContext?.auditTrailRequired || type.includes('privilege') || type.includes('chain_of_custody')) {
      return 2555; // 7 years
    }
    
    // Security alerts require extended retention
    if (type.includes('breach') || type.includes('unauthorized') || type.includes('tampering')) {
      return 1095; // 3 years
    }
    
    // Default retention
    return this.config.retentionDays;
  }
  
  /**
   * Generate compliance flags for alert
   */
  private generateComplianceFlags(type: AlertType, legalContext?: LegalAlertContext): string[] {
    const flags: string[] = [];
    
    if (legalContext) {
      flags.push('legal_context_present');
      
      if (legalContext.confidentialityLevel === 'attorney_client') {
        flags.push('attorney_client_privilege');
      }
      
      if (legalContext.auditTrailRequired) {
        flags.push('audit_trail_required');
      }
      
      if (legalContext.legalHoldStatus === 'active') {
        flags.push('legal_hold_active');
      }
      
      if (legalContext.regulatoryCompliance.length > 0) {
        flags.push('regulatory_compliance');
      }
    }
    
    // Type-specific flags
    if (type.includes('privilege') || type.includes('confidentiality')) {
      flags.push('privilege_related');
    }
    
    if (type.includes('chain_of_custody')) {
      flags.push('evidence_integrity');
    }
    
    return flags;
  }
  
  /**
   * Map alert severity to console log level
   */
  private mapSeverityToLogLevel(severity: AlertSeverity): 'log' | 'info' | 'warn' | 'error' {
    const mapping: Record<AlertSeverity, 'log' | 'info' | 'warn' | 'error'> = {
      'info': 'info',
      'warning': 'warn',
      'critical': 'error',
      'emergency': 'error'
    };
    
    return mapping[severity];
  }
  
  /**
   * Update P99 latency tracking
   */
  private updateP99Tracking(alertCodes: string[]): void {
    if (alertCodes.includes('p99_latency_exceeded')) {
      this.sustainedP99Breaches++;
    } else {
      if (this.sustainedP99Breaches > 0) {
        this.lastP99Ok = Date.now();
      }
      this.sustainedP99Breaches = 0;
    }
  }
  
  /**
   * Determine if auto-remediation should be triggered
   */
  private shouldTriggerAutoRemediation(alert: RuntimeAlert): boolean {
    if (!this.config.autoRemediation.enabled) {
      return false;
    }
    
    // Don't auto-remediate legal compliance issues
    if (alert.category === 'legal_compliance') {
      return false;
    }
    
    // Check specific conditions
    return (
      alert.type === 'pipeline_anomaly_spike' ||
      (alert.type === 'p99_latency_exceeded' && this.sustainedP99Breaches >= this.SUSTAINED_P99_THRESHOLD)
    );
  }
  
  /**
   * Trigger auto-remediation for alert
   */
  private async triggerAutoRemediation(alert: RuntimeAlert): Promise<void> {
    if (this.autosolveInFlight) {
      return;
    }
    
    if (Date.now() - this.lastAutosolveTs < this.config.autoRemediation.cooldownPeriod) {
      return;
    }
    
    this.autosolveInFlight = true;
    
    try {
      const start = performance.now();
      
      // Call autosolve endpoint
      const response = await fetch('/api/context7-autosolve?action=trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          triggeredBy: alert.id,
          alertType: alert.type,
          severity: alert.severity,
          context: alert.metadata
        })
      });
      
      const duration = performance.now() - start;
      this.lastAutosolveTs = Date.now();
      
      console.log(
        `Auto-remediation triggered for alert ${alert.id}: ${response.status} (${duration.toFixed(2)}ms)`
      );
      
    } catch (error) {
      console.error(`Auto-remediation failed for alert ${alert.id}:`, (error as Error).message);
    } finally {
      this.autosolveInFlight = false;
    }
  }
  
  /**
   * Get alert history with filtering and pagination
   */
  public getAlertHistory(
    filters?: Partial<AlertFilterConfig>,
    limit = 100,
    offset = 0
  ): RuntimeAlert[] {
    let filteredAlerts = [...this.alerts];
    
    // Apply filters
    if (filters) {
      if (filters.severityLevels?.length) {
        filteredAlerts = filteredAlerts.filter(alert => 
          filters.severityLevels!.includes(alert.severity)
        );
      }
      
      if (filters.categories?.length) {
        filteredAlerts = filteredAlerts.filter(alert => 
          filters.categories!.includes(alert.category)
        );
      }
      
      if (filters.types?.length) {
        filteredAlerts = filteredAlerts.filter(alert => 
          filters.types!.includes(alert.type)
        );
      }
    }
    
    // Sort by timestamp (most recent first)
    filteredAlerts.sort((a, b) => b.timestamp - a.timestamp);
    
    // Apply pagination
    return filteredAlerts.slice(offset, offset + limit);
  }
  
  /**
   * Get alert statistics
   */
  public getAlertStatistics(): AlertStatistics {
    const total = this.alerts.length;
    
    const bySeverity = this.alerts.reduce((acc, alert) => {
      acc[alert.severity] = (acc[alert.severity] || 0) + 1;
      return acc;
    }, {} as Record<AlertSeverity, number>);
    
    const byCategory = this.alerts.reduce((acc, alert) => {
      acc[alert.category] = (acc[alert.category] || 0) + 1;
      return acc;
    }, {} as Record<AlertCategory, number>);
    
    const byType = this.alerts.reduce((acc, alert) => {
      acc[alert.type] = (acc[alert.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const mostFrequent = Object.entries(byType)
      .map(([type, count]) => ({ type: type as AlertType, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
    
    return {
      totalAlerts: total,
      alertsBySeverity: bySeverity,
      alertsByCategory: byCategory,
      alertsByType: byType,
      averageResponseTime: 0, // TODO: Implement response time tracking
      escalationRate: 0, // TODO: Implement escalation tracking
      autoRemediationSuccessRate: 0, // TODO: Implement success rate tracking
      complianceViolations: this.alerts.filter(a => a.category === 'legal_compliance').length,
      mostFrequentAlerts: mostFrequent,
      timeDistribution: {} // TODO: Implement time distribution
    };
  }
  
  /**
   * Get sustained P99 information
   */
  public getSustainedP99Info(): {
    sustainedP99Breaches: number;
    threshold: number;
    lastP99OkTs: number;
  } {
    return {
      sustainedP99Breaches: this.sustainedP99Breaches,
      threshold: this.SUSTAINED_P99_THRESHOLD,
      lastP99OkTs: this.lastP99Ok
    };
  }
  
  /**
   * Build baseline snapshot
   */
  public buildBaseline(): BaselineFile {
    const baseline: BaselineFile = {
      created: new Date().toISOString(),
      stages: getStageBaselineSnapshot(),
      quic: getQUICMetrics()
    };
    
    // Store in persistence
    if (this.persistenceState) {
      this.persistenceState.lastBaseline = baseline;
      this.persistState();
    }
    
    return baseline;
  }
  
  /**
   * Compare baselines
   */
  public diffBaselines(oldBaseline: BaselineFile, newBaseline: BaselineFile): unknown {
    const stageDiff = newBaseline.stages.map(stage => {
      const prevStage = oldBaseline.stages.find(p => p.stage === stage.stage);
      
      if (!prevStage) {
        return {
          stage: stage.stage,
          change: 'added',
          current: stage
        };
      }
      
      const deltas = {
        p50: stage.p50 - prevStage.p50,
        p90: stage.p90 - prevStage.p90,
        p99: stage.p99 - prevStage.p99,
        anomalies: stage.anomalies - prevStage.anomalies
      };
      
      return {
        stage: stage.stage,
        deltas
      };
    });
    
    return {
      stageDiff,
      quicP99Delta: newBaseline.quic.p99 - (oldBaseline.quic.p99 || 0)
    };
  }
  
  /**
   * Persist state to disk
   */
  private persistState(): void {
    try {
      if (!fs.existsSync(this.RUNTIME_DIR)) {
        fs.mkdirSync(this.RUNTIME_DIR, { recursive: true });
      }
      
      const state: PersistedState = {
        sustainedP99Breaches: this.sustainedP99Breaches,
        lastP99Ok: this.lastP99Ok,
        lastBaseline: this.persistenceState?.lastBaseline,
        budgets: getBudgetCounters(),
        savedAt: new Date().toISOString()
      };
      
      fs.writeFileSync(this.STATE_FILE, JSON.stringify(state, null, 2));
    } catch (error) {
      console.warn('Alert Center: Failed to persist state:', (error as Error).message);
    }
  }
  
  /**
   * Schedule daily reset
   */
  private scheduleDailyReset(): void {
    const resetHour = Number(process.env.OBS_DAILY_RESET_HOUR || 0);
    
    const msUntilNextReset = (): number => {
      const now = new Date();
      const next = new Date(now.getTime());
      next.setUTCHours(resetHour, 0, 0, 0);
      
      if (next <= now) {
        next.setUTCDate(next.getUTCDate() + 1);
      }
      
      return next.getTime() - now.getTime();
    };
    
    const scheduleNext = (): void => {
      setTimeout(() => {
        // Perform daily reset
        resetBudgetCounters();
        this.sustainedP99Breaches = 0;
        this.lastP99Ok = Date.now();
        this.persistState();
        
        console.log('Alert Center: Daily reset completed');
        
        // Schedule next reset
        scheduleNext();
      }, msUntilNextReset()).unref?.();
    };
    
    scheduleNext();
  }
  
  /**
   * Clean up expired alerts
   */
  private cleanupExpiredAlerts(): void {
    const now = Date.now();
    const cutoffMs = this.config.retentionDays * 24 * 60 * 60 * 1000;
    
    const initialCount = this.alerts.length;
    this.alerts = this.alerts.filter(alert => {
      const retentionMs = alert.retentionPeriod * 24 * 60 * 60 * 1000;
      return now - alert.timestamp < retentionMs;
    });
    
    const cleanedCount = initialCount - this.alerts.length;
    if (cleanedCount > 0) {
      console.log(`Alert Center: Cleaned up ${cleanedCount} expired alerts`);
    }
  }
}

// ===== PERSISTENCE INTERFACES =====

interface BaselineFile {
  created: string;
  stages: ReturnType<typeof getStageBaselineSnapshot>;
  quic: any;
}

interface PersistedState {
  sustainedP99Breaches: number;
  lastP99Ok: number;
  lastBaseline?: BaselineFile;
  budgets?: unknown;
  savedAt: string;
}

// ===== SINGLETON INSTANCE =====

/**
 * Global Alert Center Instance
 */
let alertCenterInstance: AlertCenter | null = null;

/**
 * Get or create Alert Center singleton
 */
export function getAlertCenter(config?: Partial<AlertConfig>, natsService?: NATSMessagingService): AlertCenter {
  if (!alertCenterInstance) {
    alertCenterInstance = new AlertCenter(config, natsService);
  }
  return alertCenterInstance;
}

// ===== LEGACY COMPATIBILITY EXPORTS =====

/**
 * Legacy route alerts function for backward compatibility
 */
export function routeAlerts(raw: string[], ctx: Record<string, any> = {}): Promise<RuntimeAlert[]> {
  return getAlertCenter().routeAlerts(raw, ctx);
}

/**
 * Legacy get alert history function
 */
export function getAlertHistory(): RuntimeAlert[] {
  return getAlertCenter().getAlertHistory();
}

/**
 * Legacy trigger autosolve function
 */
export async function maybeTriggerAutosolve(fetchFn: typeof fetch, rawCodes: string[]): Promise<any> {
  const alertCenter = getAlertCenter();
  const alerts = await alertCenter.routeAlerts(rawCodes, { fetchFn });
  
  // Check if any alerts triggered auto-remediation
  for (const alert of alerts) {
    if (alertCenter['shouldTriggerAutoRemediation'](alert)) {
      return { triggered: true };
    }
  }
  
  return { triggered: false };
}

/**
 * Legacy get sustained P99 info function
 */
export function getSustainedP99Info(): unknown {
  return getAlertCenter().getSustainedP99Info();
}

/**
 * Legacy build baseline function
 */
export function buildBaseline(): BaselineFile {
  return getAlertCenter().buildBaseline();
}

/**
 * Legacy diff baselines function
 */
export function diffBaselines(oldB: BaselineFile, newB: BaselineFile): unknown {
  return getAlertCenter().diffBaselines(oldB, newB);
}