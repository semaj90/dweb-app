#!/usr/bin/env node

/**
 * Legal AI Comprehensive Logging Integration
 * 
 * Provides unified logging across all orchestration components with:
 * - Structured logging with JSON format
 * - Log aggregation and routing to ELK stack
 * - Performance metrics and traces
 * - Error tracking and alerting
 * - Log level management and filtering
 * - Real-time log streaming
 */

const fs = require('fs');
const path = require('path');
const { createWriteStream } = require('fs');

class LegalAILoggingIntegration {
  constructor(orchestrator) {
    this.orchestrator = orchestrator;
    this.logStreams = new Map();
    this.logBuffers = new Map();
    this.logFilters = new Map();
    this.logMetrics = new Map();
    this.logLevel = process.env.LOG_LEVEL || 'info';
    this.environment = process.env.LEGAL_AI_ENV || 'development';
    
    // Log levels hierarchy
    this.logLevels = {
      error: 0,
      warn: 1,
      info: 2,
      debug: 3,
      trace: 4
    };
    
    this.initializeLoggingSystem();
  }

  initializeLoggingSystem() {
    console.log('[LOGGING] Initializing Legal AI Logging Integration...');
    console.log(`[LOGGING] Log Level: ${this.logLevel}`);
    console.log(`[LOGGING] Environment: ${this.environment}`);
    
    // Create log directories
    this.ensureLogDirectories();
    
    // Set up log streams
    this.setupLogStreams();
    
    // Initialize log aggregation
    this.initializeLogAggregation();
    
    // Set up ELK stack integration
    this.setupELKIntegration();
    
    // Initialize performance monitoring
    this.initializePerformanceLogging();
    
    // Set up real-time log streaming
    this.setupRealTimeStreaming();
    
    console.log('[LOGGING] Logging system initialized successfully');
  }

  ensureLogDirectories() {
    const logDirs = [
      './logs',
      './logs/services',
      './logs/orchestration',
      './logs/performance',
      './logs/errors',
      './logs/audit'
    ];
    
    logDirs.forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
        console.log(`[LOGGING] Created log directory: ${dir}`);
      }
    });
  }

  setupLogStreams() {
    console.log('[LOGGING] Setting up log streams...');
    
    // Service-specific log streams
    const services = this.orchestrator.config.orchestration.services;
    Object.keys(services).forEach(serviceId => {
      this.createServiceLogStream(serviceId);
    });
    
    // System log streams
    this.createSystemLogStreams();
    
    console.log(`[LOGGING] Created ${this.logStreams.size} log streams`);
  }

  createServiceLogStream(serviceId) {
    const logFile = path.join('./logs/services', `${serviceId}.log`);
    const errorLogFile = path.join('./logs/errors', `${serviceId}-errors.log`);
    
    const logStream = createWriteStream(logFile, { flags: 'a' });
    const errorStream = createWriteStream(errorLogFile, { flags: 'a' });
    
    this.logStreams.set(serviceId, {
      info: logStream,
      error: errorStream,
      buffer: [],
      lastRotation: Date.now()
    });
    
    // Set up log rotation
    this.scheduleLogRotation(serviceId);
  }

  createSystemLogStreams() {
    const systemStreams = [
      'orchestration',
      'health-monitoring',
      'message-routing',
      'dependency-management',
      'configuration',
      'performance',
      'audit'
    ];
    
    systemStreams.forEach(streamName => {
      const logFile = path.join('./logs/orchestration', `${streamName}.log`);
      const stream = createWriteStream(logFile, { flags: 'a' });
      
      this.logStreams.set(`system.${streamName}`, {
        info: stream,
        buffer: [],
        lastRotation: Date.now()
      });
    });
  }

  initializeLogAggregation() {
    console.log('[LOGGING] Initializing log aggregation...');
    
    // Set up log collection from all services
    this.setupLogCollection();
    
    // Initialize log parsing and enrichment
    this.setupLogEnrichment();
    
    // Set up log forwarding to ELK
    this.setupLogForwarding();
  }

  setupLogCollection() {
    // Collect logs from orchestrated services
    if (this.orchestrator.messageRouter) {
      // Subscribe to log messages via NATS
      this.orchestrator.messageRouter.natsConnection?.subscribe('legal.logs.*', {
        callback: (err, msg) => {
          if (!err) {
            this.processLogMessage(msg);
          }
        }
      });
    }
    
    // Set up file-based log collection for services that write to files
    this.setupFileLogCollection();
  }

  setupFileLogCollection() {
    const serviceLogDir = './logs/services';
    
    if (fs.existsSync(serviceLogDir)) {
      fs.watch(serviceLogDir, (eventType, filename) => {
        if (eventType === 'change' && filename.endsWith('.log')) {
          this.processFileLogUpdate(filename);
        }
      });
    }
  }

  processLogMessage(message) {
    try {
      const logData = JSON.parse(message.data);
      const enrichedLog = this.enrichLogData(logData);
      
      // Route to appropriate log stream
      this.routeLogMessage(enrichedLog);
      
      // Forward to ELK stack
      this.forwardToELK(enrichedLog);
      
      // Update metrics
      this.updateLogMetrics(enrichedLog);
      
    } catch (error) {
      this.logError('LOG_PROCESSING_ERROR', error, { message: message.subject });
    }
  }

  enrichLogData(logData) {
    const enriched = {
      ...logData,
      '@timestamp': new Date().toISOString(),
      environment: this.environment,
      orchestration: {
        version: this.orchestrator.config.orchestration.metadata.version,
        instance_id: process.pid
      }
    };
    
    // Add service context if available
    if (logData.service) {
      const serviceConfig = this.orchestrator.config.orchestration.services[logData.service];
      if (serviceConfig) {
        enriched.service_context = {
          type: serviceConfig.type,
          host: serviceConfig.host,
          port: serviceConfig.port,
          tags: serviceConfig.tags
        };
      }
    }
    
    // Add request tracing if available
    if (logData.trace_id) {
      enriched.trace = {
        trace_id: logData.trace_id,
        span_id: logData.span_id,
        parent_span_id: logData.parent_span_id
      };
    }
    
    return enriched;
  }

  routeLogMessage(logData) {
    const level = logData.level || 'info';
    const service = logData.service || 'unknown';
    
    // Check if we should process this log level
    if (!this.shouldLog(level)) {
      return;
    }
    
    // Get appropriate log stream
    const streamKey = logData.service || 'system.orchestration';
    const streams = this.logStreams.get(streamKey);
    
    if (streams) {
      const targetStream = level === 'error' ? streams.error || streams.info : streams.info;
      const logLine = JSON.stringify(logData) + '\n';
      
      targetStream.write(logLine);
      
      // Buffer for real-time streaming
      if (!streams.buffer) streams.buffer = [];
      streams.buffer.push(logData);
      
      // Keep buffer size manageable
      if (streams.buffer.length > 1000) {
        streams.buffer.shift();
      }
    }
  }

  shouldLog(level) {
    const levelValue = this.logLevels[level];
    const currentLevelValue = this.logLevels[this.logLevel];
    
    return levelValue !== undefined && levelValue <= currentLevelValue;
  }

  setupELKIntegration() {
    console.log('[LOGGING] Setting up ELK stack integration...');
    
    // Configure Logstash pipeline
    this.createLogstashPipeline();
    
    // Set up Elasticsearch index templates
    this.createElasticsearchTemplates();
    
    // Configure Kibana dashboards
    this.setupKibanaDashboards();
  }

  createLogstashPipeline() {
    const pipelineConfig = {
      input: {
        beats: {
          port: 5044
        },
        file: {
          path: './logs/**/*.log',
          start_position: 'beginning',
          sincedb_path: './logs/.sincedb'
        }
      },
      filter: [
        {
          if: '[message] =~ /^{.*}$/',
          json: {
            source: 'message'
          }
        },
        {
          mutate: {
            add_field: {
              '[@metadata][index]': 'legal-ai-%{environment}-%{+YYYY.MM.dd}'
            }
          }
        },
        {
          if: '[service]',
          mutate: {
            add_tag: ['service-%{service}']
          }
        },
        {
          if: '[level] == "error"',
          mutate: {
            add_tag: ['error']
          }
        }
      ],
      output: {
        elasticsearch: {
          hosts: ['localhost:9200'],
          index: '%{[@metadata][index]}'
        },
        if: this.environment === 'development',
        stdout: {
          codec: 'rubydebug'
        }
      }
    };
    
    // Write Logstash configuration
    const configPath = './elk-stack/logstash/pipeline/legal-ai.conf';
    this.writeLogstashConfig(configPath, pipelineConfig);
  }

  writeLogstashConfig(configPath, config) {
    const configContent = this.convertToLogstashFormat(config);
    
    try {
      fs.writeFileSync(configPath, configContent);
      console.log('[LOGGING] Created Logstash pipeline configuration');
    } catch (error) {
      console.error('[LOGGING] Failed to write Logstash configuration:', error.message);
    }
  }

  convertToLogstashFormat(config) {
    // Simple converter for basic Logstash configuration
    let content = '';
    
    if (config.input) {
      content += 'input {\n';
      Object.entries(config.input).forEach(([plugin, settings]) => {
        content += `  ${plugin} {\n`;
        Object.entries(settings).forEach(([key, value]) => {
          content += `    ${key} => "${value}"\n`;
        });
        content += '  }\n';
      });
      content += '}\n\n';
    }
    
    if (config.filter) {
      content += 'filter {\n';
      config.filter.forEach(filter => {
        Object.entries(filter).forEach(([plugin, settings]) => {
          if (plugin === 'if') {
            content += `  if ${settings} {\n`;
          } else {
            content += `  ${plugin} {\n`;
            if (typeof settings === 'object') {
              Object.entries(settings).forEach(([key, value]) => {
                if (typeof value === 'object') {
                  content += `    ${key} {\n`;
                  Object.entries(value).forEach(([subKey, subValue]) => {
                    content += `      ${subKey} => "${subValue}"\n`;
                  });
                  content += '    }\n';
                } else {
                  content += `    ${key} => "${value}"\n`;
                }
              });
            }
            content += '  }\n';
          }
        });
      });
      content += '}\n\n';
    }
    
    if (config.output) {
      content += 'output {\n';
      Object.entries(config.output).forEach(([plugin, settings]) => {
        if (plugin === 'if') {
          content += `  if ${settings} {\n`;
        } else {
          content += `  ${plugin} {\n`;
          if (typeof settings === 'object') {
            Object.entries(settings).forEach(([key, value]) => {
              if (Array.isArray(value)) {
                content += `    ${key} => [${value.map(v => `"${v}"`).join(', ')}]\n`;
              } else {
                content += `    ${key} => "${value}"\n`;
              }
            });
          }
          content += '  }\n';
        }
      });
      content += '}\n';
    }
    
    return content;
  }

  createElasticsearchTemplates() {
    const indexTemplate = {
      index_patterns: ['legal-ai-*'],
      template: {
        settings: {
          number_of_shards: 1,
          number_of_replicas: 0,
          'index.lifecycle.name': 'legal-ai-policy',
          'index.lifecycle.rollover_alias': 'legal-ai'
        },
        mappings: {
          properties: {
            '@timestamp': { type: 'date' },
            level: { type: 'keyword' },
            message: { type: 'text' },
            service: { type: 'keyword' },
            environment: { type: 'keyword' },
            trace_id: { type: 'keyword' },
            span_id: { type: 'keyword' },
            duration: { type: 'long' },
            response_time: { type: 'long' },
            error: {
              type: 'object',
              properties: {
                message: { type: 'text' },
                stack: { type: 'text' },
                code: { type: 'keyword' }
              }
            }
          }
        }
      }
    };
    
    const templatePath = './elk-stack/elasticsearch/templates/legal-ai-template.json';
    try {
      fs.writeFileSync(templatePath, JSON.stringify(indexTemplate, null, 2));
      console.log('[LOGGING] Created Elasticsearch index template');
    } catch (error) {
      console.error('[LOGGING] Failed to write Elasticsearch template:', error.message);
    }
  }

  setupKibanaDashboards() {
    const dashboardConfig = {
      legal_ai_overview: {
        title: 'Legal AI System Overview',
        visualizations: [
          'service_health_status',
          'request_volume_timeline',
          'error_rate_by_service',
          'response_time_distribution'
        ]
      },
      service_performance: {
        title: 'Service Performance Metrics',
        visualizations: [
          'service_response_times',
          'throughput_by_service',
          'memory_usage_trends',
          'cpu_utilization'
        ]
      },
      error_analysis: {
        title: 'Error Analysis Dashboard',
        visualizations: [
          'error_timeline',
          'error_breakdown_by_service',
          'error_frequency_heatmap',
          'critical_errors_table'
        ]
      }
    };
    
    const dashboardPath = './elk-stack/kibana/dashboards/legal-ai-dashboards.json';
    try {
      fs.writeFileSync(dashboardPath, JSON.stringify(dashboardConfig, null, 2));
      console.log('[LOGGING] Created Kibana dashboard configurations');
    } catch (error) {
      console.error('[LOGGING] Failed to write Kibana dashboards:', error.message);
    }
  }

  initializePerformanceLogging() {
    console.log('[LOGGING] Initializing performance logging...');
    
    // Set up performance metric collection
    this.setupPerformanceMetrics();
    
    // Initialize request tracing
    this.setupRequestTracing();
    
    // Set up SLA monitoring
    this.setupSLAMonitoring();
  }

  setupPerformanceMetrics() {
    const performanceLog = {
      service: 'orchestration.performance',
      level: 'info',
      message: 'Performance metrics',
      metrics: {
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        cpu: process.cpuUsage(),
        event_loop_lag: 0 // Would need perf_hooks for real measurement
      }
    };
    
    // Log performance metrics every 30 seconds
    setInterval(() => {
      this.logPerformanceMetrics();
    }, 30000);
  }

  logPerformanceMetrics() {
    const metrics = {
      '@timestamp': new Date().toISOString(),
      service: 'orchestration.performance',
      level: 'info',
      type: 'metrics',
      orchestration: {
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        cpu: process.cpuUsage(),
        services: {
          total: this.orchestrator.serviceRegistry.size(),
          running: this.getRunningServicesCount(),
          healthy: this.getHealthyServicesCount()
        }
      }
    };
    
    this.routeLogMessage(metrics);
  }

  getRunningServicesCount() {
    // Implementation would check actual service states
    return this.orchestrator.services.size;
  }

  getHealthyServicesCount() {
    // Implementation would check health status
    return this.orchestrator.healthMonitor.getOverallHealth().services.healthy;
  }

  setupRealTimeStreaming() {
    console.log('[LOGGING] Setting up real-time log streaming...');
    
    // This would integrate with the orchestration controller's WebSocket server
    // to provide real-time log streaming to clients
  }

  // Public API methods
  log(level, message, data = {}, service = 'orchestration') {
    if (!this.shouldLog(level)) {
      return;
    }
    
    const logEntry = {
      '@timestamp': new Date().toISOString(),
      level,
      message,
      service,
      ...data
    };
    
    const enrichedLog = this.enrichLogData(logEntry);
    this.routeLogMessage(enrichedLog);
  }

  logError(message, error, context = {}) {
    const errorLog = {
      level: 'error',
      message,
      error: {
        message: error.message,
        stack: error.stack,
        name: error.name
      },
      ...context
    };
    
    this.log('error', message, errorLog);
  }

  logPerformance(operation, duration, metadata = {}) {
    const perfLog = {
      level: 'info',
      type: 'performance',
      operation,
      duration,
      ...metadata
    };
    
    this.log('info', `Performance: ${operation}`, perfLog);
  }

  logAudit(action, user, resource, result) {
    const auditLog = {
      '@timestamp': new Date().toISOString(),
      level: 'info',
      type: 'audit',
      action,
      user,
      resource,
      result,
      service: 'orchestration.audit'
    };
    
    const enrichedLog = this.enrichLogData(auditLog);
    this.routeLogMessage(enrichedLog);
  }

  // Log rotation
  scheduleLogRotation(serviceId) {
    // Rotate logs daily at midnight
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setDate(tomorrow.getDate() + 1);
    tomorrow.setHours(0, 0, 0, 0);
    
    const timeUntilMidnight = tomorrow - now;
    
    setTimeout(() => {
      this.rotateServiceLogs(serviceId);
      
      // Schedule next rotation
      setInterval(() => {
        this.rotateServiceLogs(serviceId);
      }, 24 * 60 * 60 * 1000); // Daily
    }, timeUntilMidnight);
  }

  rotateServiceLogs(serviceId) {
    const streams = this.logStreams.get(serviceId);
    if (!streams) return;
    
    try {
      // Close current streams
      streams.info.end();
      if (streams.error) streams.error.end();
      
      // Rename current log files
      const timestamp = new Date().toISOString().split('T')[0];
      const logFile = path.join('./logs/services', `${serviceId}.log`);
      const rotatedFile = path.join('./logs/services', `${serviceId}.${timestamp}.log`);
      
      if (fs.existsSync(logFile)) {
        fs.renameSync(logFile, rotatedFile);
      }
      
      // Create new streams
      this.createServiceLogStream(serviceId);
      
      console.log(`[LOGGING] Rotated logs for ${serviceId}`);
      
    } catch (error) {
      console.error(`[LOGGING] Failed to rotate logs for ${serviceId}:`, error.message);
    }
  }

  updateLogMetrics(logData) {
    const service = logData.service || 'unknown';
    const level = logData.level || 'info';
    
    const key = `${service}.${level}`;
    const current = this.logMetrics.get(key) || 0;
    this.logMetrics.set(key, current + 1);
  }

  getLogMetrics() {
    return Object.fromEntries(this.logMetrics);
  }

  // Cleanup
  async shutdown() {
    console.log('[LOGGING] Shutting down logging system...');
    
    // Close all log streams
    for (const [serviceId, streams] of this.logStreams) {
      try {
        streams.info.end();
        if (streams.error) streams.error.end();
        console.log(`[LOGGING] Closed log streams for ${serviceId}`);
      } catch (error) {
        console.error(`[LOGGING] Error closing streams for ${serviceId}:`, error);
      }
    }
    
    console.log('[LOGGING] Logging system shutdown complete');
  }
}

module.exports = LegalAILoggingIntegration;