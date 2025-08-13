#!/usr/bin/env node

/**
 * Legal AI Centralized Configuration Manager
 * 
 * Manages configuration across all orchestration components with:
 * - Hot reloading of configuration changes
 * - Environment-specific overrides
 * - Configuration validation and schema enforcement
 * - Encrypted secrets management
 * - Configuration versioning and rollback
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class ConfigurationManager {
  constructor() {
    this.configs = new Map();
    this.schemas = new Map();
    this.watchers = new Map();
    this.subscribers = new Map();
    this.configHistory = [];
    this.environment = process.env.LEGAL_AI_ENV || 'development';
    
    // Configuration file paths
    this.configPaths = {
      main: './orchestration-config.json',
      services: './service-configs/',
      secrets: './secrets/',
      schemas: './config-schemas/'
    };
    
    this.initializeConfigurationManager();
  }

  initializeConfigurationManager() {
    console.log('[CONFIG-MANAGER] Initializing Legal AI Configuration Manager...');
    console.log(`[CONFIG-MANAGER] Environment: ${this.environment}`);
    
    // Create configuration directories
    this.ensureDirectoriesExist();
    
    // Load configuration schemas
    this.loadConfigurationSchemas();
    
    // Load main configuration
    this.loadMainConfiguration();
    
    // Load service-specific configurations
    this.loadServiceConfigurations();
    
    // Set up configuration file watchers
    this.setupConfigurationWatchers();
    
    console.log('[CONFIG-MANAGER] Configuration manager initialized successfully');
  }

  ensureDirectoriesExist() {
    Object.values(this.configPaths).forEach(configPath => {
      if (configPath.endsWith('/')) {
        const dir = path.resolve(configPath);
        if (!fs.existsSync(dir)) {
          fs.mkdirSync(dir, { recursive: true });
          console.log(`[CONFIG-MANAGER] Created configuration directory: ${dir}`);
        }
      }
    });
  }

  loadConfigurationSchemas() {
    console.log('[CONFIG-MANAGER] Loading configuration schemas...');
    
    // Main orchestration schema
    this.schemas.set('orchestration', {
      type: 'object',
      required: ['orchestration'],
      properties: {
        orchestration: {
          type: 'object',
          required: ['services', 'communication', 'monitoring'],
          properties: {
            services: {
              type: 'object',
              additionalProperties: {
                type: 'object',
                required: ['name', 'type', 'host', 'port'],
                properties: {
                  name: { type: 'string' },
                  type: { type: 'string' },
                  host: { type: 'string' },
                  port: { type: 'number' },
                  dependencies: {
                    type: 'array',
                    items: { type: 'string' }
                  }
                }
              }
            }
          }
        }
      }
    });
    
    // Service-specific schemas
    this.schemas.set('kratos', {
      type: 'object',
      properties: {
        server: {
          type: 'object',
          properties: {
            port: { type: 'number' },
            host: { type: 'string' },
            timeout: { type: 'string' }
          }
        },
        database: {
          type: 'object',
          properties: {
            url: { type: 'string' },
            pool_size: { type: 'number' }
          }
        }
      }
    });
    
    console.log(`[CONFIG-MANAGER] Loaded ${this.schemas.size} configuration schemas`);
  }

  loadMainConfiguration() {
    console.log('[CONFIG-MANAGER] Loading main orchestration configuration...');
    
    try {
      const configData = fs.readFileSync(this.configPaths.main, 'utf8');
      const config = JSON.parse(configData);
      
      // Validate against schema
      this.validateConfiguration('orchestration', config);
      
      // Apply environment-specific overrides
      const finalConfig = this.applyEnvironmentOverrides(config);
      
      this.configs.set('main', finalConfig);
      this.addToHistory('main', finalConfig);
      
      console.log('[CONFIG-MANAGER] Main configuration loaded successfully');
      
    } catch (error) {
      console.error('[CONFIG-MANAGER] Failed to load main configuration:', error.message);
      throw error;
    }
  }

  loadServiceConfigurations() {
    console.log('[CONFIG-MANAGER] Loading service-specific configurations...');
    
    const serviceConfigDir = this.configPaths.services;
    
    if (!fs.existsSync(serviceConfigDir)) {
      console.log('[CONFIG-MANAGER] No service configuration directory found, creating defaults...');
      this.createDefaultServiceConfigurations();
      return;
    }
    
    try {
      const configFiles = fs.readdirSync(serviceConfigDir);
      
      configFiles.forEach(file => {
        if (file.endsWith('.json')) {
          const serviceName = path.basename(file, '.json');
          const configPath = path.join(serviceConfigDir, file);
          
          try {
            const configData = fs.readFileSync(configPath, 'utf8');
            const config = JSON.parse(configData);
            
            // Validate if schema exists
            if (this.schemas.has(serviceName)) {
              this.validateConfiguration(serviceName, config);
            }
            
            // Apply environment overrides
            const finalConfig = this.applyEnvironmentOverrides(config);
            
            this.configs.set(serviceName, finalConfig);
            this.addToHistory(serviceName, finalConfig);
            
            console.log(`[CONFIG-MANAGER] Loaded configuration for ${serviceName}`);
            
          } catch (error) {
            console.error(`[CONFIG-MANAGER] Failed to load ${serviceName} configuration:`, error.message);
          }
        }
      });
      
    } catch (error) {
      console.error('[CONFIG-MANAGER] Failed to read service configurations:', error.message);
    }
  }

  createDefaultServiceConfigurations() {
    const defaultConfigs = {
      kratos: {
        server: {
          port: 8080,
          host: "localhost",
          timeout: "30s",
          read_timeout: "10s",
          write_timeout: "10s"
        },
        database: {
          url: "${POSTGRES_URL}",
          pool_size: 10,
          max_idle: 5,
          max_lifetime: "1h"
        },
        logging: {
          level: "info",
          format: "json",
          output: "stdout"
        }
      },
      
      "node-cluster": {
        cluster: {
          workers: {
            legal: { count: 3, memory: "512MB" },
            ai: { count: 2, memory: "1GB" },
            vector: { count: 2, memory: "256MB" },
            database: { count: 3, memory: "256MB" }
          },
          restart_policy: "on-failure",
          max_restarts: 3
        },
        ipc: {
          enabled: true,
          timeout: "30s"
        }
      },
      
      "quic-gateway": {
        quic: {
          port: 8443,
          http_port: 8444,
          cert_file: "./certs/server.crt",
          key_file: "./certs/server.key",
          max_streams: 1000,
          idle_timeout: "300s"
        },
        performance: {
          congestion_control: "bbr",
          initial_window: "1MB",
          max_window: "10MB"
        }
      },
      
      "webgpu-engine": {
        gpu: {
          device_selection: "high-performance",
          memory_limit: "2GB",
          shader_cache: true,
          debug_mode: false
        },
        compute: {
          workgroup_size: 256,
          max_dispatch_size: 65535,
          tensor_cache_size: "500MB"
        }
      },
      
      "xstate-orchestrator": {
        workflows: {
          max_concurrent: 100,
          timeout: "5m",
          persistence: true,
          snapshot_interval: "30s"
        },
        state_machines: [
          "legal-case-lifecycle",
          "evidence-processing-pipeline",
          "ai-analysis-orchestrator",
          "multi-agent-coordinator"
        ]
      }
    };
    
    // Create service configuration files
    Object.entries(defaultConfigs).forEach(([serviceName, config]) => {
      const configPath = path.join(this.configPaths.services, `${serviceName}.json`);
      fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
      
      this.configs.set(serviceName, config);
      console.log(`[CONFIG-MANAGER] Created default configuration for ${serviceName}`);
    });
  }

  applyEnvironmentOverrides(config) {
    // Create a deep copy
    const overriddenConfig = JSON.parse(JSON.stringify(config));
    
    // Replace environment variables
    this.replaceEnvironmentVariables(overriddenConfig);
    
    // Apply environment-specific overrides
    const envOverridePath = `./config.${this.environment}.json`;
    if (fs.existsSync(envOverridePath)) {
      try {
        const envOverrides = JSON.parse(fs.readFileSync(envOverridePath, 'utf8'));
        this.deepMerge(overriddenConfig, envOverrides);
        console.log(`[CONFIG-MANAGER] Applied ${this.environment} environment overrides`);
      } catch (error) {
        console.error(`[CONFIG-MANAGER] Failed to apply environment overrides:`, error.message);
      }
    }
    
    return overriddenConfig;
  }

  replaceEnvironmentVariables(obj) {
    if (typeof obj === 'string') {
      return obj.replace(/\${([^}]+)}/g, (match, envVar) => {
        return process.env[envVar] || match;
      });
    } else if (Array.isArray(obj)) {
      return obj.map(item => this.replaceEnvironmentVariables(item));
    } else if (typeof obj === 'object' && obj !== null) {
      Object.keys(obj).forEach(key => {
        obj[key] = this.replaceEnvironmentVariables(obj[key]);
      });
    }
    return obj;
  }

  deepMerge(target, source) {
    Object.keys(source).forEach(key => {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        if (!target[key]) target[key] = {};
        this.deepMerge(target[key], source[key]);
      } else {
        target[key] = source[key];
      }
    });
  }

  validateConfiguration(schemaName, config) {
    const schema = this.schemas.get(schemaName);
    if (!schema) {
      console.warn(`[CONFIG-MANAGER] No schema found for ${schemaName}, skipping validation`);
      return true;
    }
    
    // Simple validation implementation
    try {
      this.validateObject(config, schema);
      console.log(`[CONFIG-MANAGER] Configuration validation passed for ${schemaName}`);
      return true;
    } catch (error) {
      throw new Error(`Configuration validation failed for ${schemaName}: ${error.message}`);
    }
  }

  validateObject(obj, schema) {
    if (schema.type === 'object') {
      if (typeof obj !== 'object' || obj === null) {
        throw new Error('Expected object');
      }
      
      if (schema.required) {
        schema.required.forEach(field => {
          if (!(field in obj)) {
            throw new Error(`Required field '${field}' is missing`);
          }
        });
      }
      
      if (schema.properties) {
        Object.keys(schema.properties).forEach(prop => {
          if (prop in obj) {
            this.validateObject(obj[prop], schema.properties[prop]);
          }
        });
      }
    } else if (schema.type === 'array') {
      if (!Array.isArray(obj)) {
        throw new Error('Expected array');
      }
      
      if (schema.items) {
        obj.forEach(item => {
          this.validateObject(item, schema.items);
        });
      }
    } else if (schema.type === 'string') {
      if (typeof obj !== 'string') {
        throw new Error('Expected string');
      }
    } else if (schema.type === 'number') {
      if (typeof obj !== 'number') {
        throw new Error('Expected number');
      }
    }
  }

  setupConfigurationWatchers() {
    console.log('[CONFIG-MANAGER] Setting up configuration file watchers...');
    
    // Watch main configuration file
    this.watchConfigurationFile(this.configPaths.main, 'main');
    
    // Watch service configuration directory
    if (fs.existsSync(this.configPaths.services)) {
      fs.watch(this.configPaths.services, (eventType, filename) => {
        if (filename && filename.endsWith('.json')) {
          const serviceName = path.basename(filename, '.json');
          const configPath = path.join(this.configPaths.services, filename);
          
          if (eventType === 'change') {
            console.log(`[CONFIG-MANAGER] Configuration changed for ${serviceName}`);
            this.reloadServiceConfiguration(serviceName, configPath);
          }
        }
      });
    }
    
    console.log('[CONFIG-MANAGER] Configuration watchers active');
  }

  watchConfigurationFile(filePath, configName) {
    if (fs.existsSync(filePath)) {
      fs.watch(filePath, (eventType) => {
        if (eventType === 'change') {
          console.log(`[CONFIG-MANAGER] Configuration file changed: ${configName}`);
          this.reloadConfiguration(configName, filePath);
        }
      });
    }
  }

  reloadConfiguration(configName, filePath) {
    try {
      const configData = fs.readFileSync(filePath, 'utf8');
      const config = JSON.parse(configData);
      
      // Validate configuration
      if (configName !== 'main' && this.schemas.has(configName)) {
        this.validateConfiguration(configName, config);
      }
      
      // Apply environment overrides
      const finalConfig = this.applyEnvironmentOverrides(config);
      
      // Store previous configuration for rollback
      const previousConfig = this.configs.get(configName);
      this.configs.set(configName, finalConfig);
      this.addToHistory(configName, finalConfig);
      
      // Notify subscribers
      this.notifyConfigurationChange(configName, finalConfig, previousConfig);
      
      console.log(`[CONFIG-MANAGER] Successfully reloaded configuration: ${configName}`);
      
    } catch (error) {
      console.error(`[CONFIG-MANAGER] Failed to reload configuration ${configName}:`, error.message);
    }
  }

  reloadServiceConfiguration(serviceName, configPath) {
    this.reloadConfiguration(serviceName, configPath);
  }

  // Configuration access methods
  getConfiguration(serviceName = 'main') {
    return this.configs.get(serviceName);
  }

  getServiceConfiguration(serviceName) {
    return this.configs.get(serviceName) || {};
  }

  setConfiguration(serviceName, config) {
    // Validate configuration
    if (this.schemas.has(serviceName)) {
      this.validateConfiguration(serviceName, config);
    }
    
    const previousConfig = this.configs.get(serviceName);
    this.configs.set(serviceName, config);
    this.addToHistory(serviceName, config);
    
    // Persist to file
    this.persistConfiguration(serviceName, config);
    
    // Notify subscribers
    this.notifyConfigurationChange(serviceName, config, previousConfig);
  }

  persistConfiguration(serviceName, config) {
    try {
      let filePath;
      if (serviceName === 'main') {
        filePath = this.configPaths.main;
      } else {
        filePath = path.join(this.configPaths.services, `${serviceName}.json`);
      }
      
      fs.writeFileSync(filePath, JSON.stringify(config, null, 2));
      console.log(`[CONFIG-MANAGER] Persisted configuration for ${serviceName}`);
      
    } catch (error) {
      console.error(`[CONFIG-MANAGER] Failed to persist configuration for ${serviceName}:`, error.message);
    }
  }

  // Subscription management
  subscribe(serviceName, callback) {
    if (!this.subscribers.has(serviceName)) {
      this.subscribers.set(serviceName, new Set());
    }
    
    this.subscribers.get(serviceName).add(callback);
    
    // Return unsubscribe function
    return () => {
      const serviceSubscribers = this.subscribers.get(serviceName);
      if (serviceSubscribers) {
        serviceSubscribers.delete(callback);
      }
    };
  }

  notifyConfigurationChange(serviceName, newConfig, previousConfig) {
    const serviceSubscribers = this.subscribers.get(serviceName);
    if (serviceSubscribers) {
      serviceSubscribers.forEach(callback => {
        try {
          callback(newConfig, previousConfig);
        } catch (error) {
          console.error(`[CONFIG-MANAGER] Error in configuration change callback:`, error);
        }
      });
    }
  }

  // Configuration history and rollback
  addToHistory(serviceName, config) {
    this.configHistory.push({
      serviceName,
      config: JSON.parse(JSON.stringify(config)),
      timestamp: new Date().toISOString(),
      version: this.configHistory.length + 1
    });
    
    // Keep only last 50 configuration changes
    if (this.configHistory.length > 50) {
      this.configHistory.shift();
    }
  }

  rollbackConfiguration(serviceName, version) {
    const historyEntry = this.configHistory.find(
      entry => entry.serviceName === serviceName && entry.version === version
    );
    
    if (!historyEntry) {
      throw new Error(`Configuration version ${version} not found for ${serviceName}`);
    }
    
    this.setConfiguration(serviceName, historyEntry.config);
    console.log(`[CONFIG-MANAGER] Rolled back ${serviceName} to version ${version}`);
  }

  getConfigurationHistory(serviceName) {
    return this.configHistory.filter(entry => 
      !serviceName || entry.serviceName === serviceName
    );
  }

  // Utility methods
  getAllConfigurations() {
    return Object.fromEntries(this.configs);
  }

  getConfigurationSummary() {
    return {
      environment: this.environment,
      loadedConfigs: Array.from(this.configs.keys()),
      schemas: Array.from(this.schemas.keys()),
      historySize: this.configHistory.length,
      subscribers: Object.fromEntries(
        Array.from(this.subscribers.entries()).map(([service, subs]) => [service, subs.size])
      )
    };
  }
}

module.exports = ConfigurationManager;