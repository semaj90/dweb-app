const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const grpc = require('@grpc/grpc-js');
const nats = require('nats');

/**
 * AI Coordination Service Worker
 * Orchestrates multi-agent workflows and coordinates with Go-Kratos services
 */
class AICoordinatorWorker {
  constructor(data) {
    this.workerId = data.workerId;
    this.services = data.services;
    this.coordinatedTasks = 0;
    this.startTime = Date.now();
    
    // AI coordination configuration
    this.config = {
      maxConcurrentTasks: 5,
      taskTimeout: 300000, // 5 minutes
      retryAttempts: 3,
      retryDelay: 2000,
      healthCheckInterval: 30000
    };
    
    // Active task tracking
    this.activeTasks = new Map();
    this.taskQueue = [];
    this.agentRegistry = new Map();
    
    // Service connections
    this.natsConnection = null;
    this.kratosClient = null;
    
    this.init();
  }
  
  async init() {
    console.log(`[AI-COORDINATOR-${this.workerId}] AI coordinator worker starting`);
    
    // Setup message handling
    this.setupMessageHandling();
    
    // Initialize connections
    await this.initializeConnections();
    
    // Setup agent registry
    this.setupAgentRegistry();
    
    // Start task processor
    this.startTaskProcessor();
    
    // Setup periodic health checks
    setInterval(() => {
      this.performHealthChecks();
    }, this.config.healthCheckInterval);
    
    this.sendMessage({
      type: 'worker-ready',
      worker: 'ai-coordinator',
      pid: process.pid
    });
  }
  
  setupMessageHandling() {
    parentPort.on('message', async (message) => {
      try {
        await this.handleMessage(message);
      } catch (error) {
        console.error(`[AI-COORDINATOR-${this.workerId}] Message handling error:`, error);
        this.sendMessage({
          type: 'error',
          worker: 'ai-coordinator',
          error: error.message,
          timestamp: Date.now()
        });
      }
    });
  }
  
  async handleMessage(message) {
    switch (message.type) {
      case 'coordinate-analysis':
        await this.coordinateAnalysis(message.data);
        break;
        
      case 'multi-agent-task':
        await this.executeMultiAgentTask(message.data);
        break;
        
      case 'legal-reasoning':
        await this.performLegalReasoning(message.data);
        break;
        
      case 'document-workflow':
        await this.executeDocumentWorkflow(message.data);
        break;
        
      case 'cancel-task':
        await this.cancelTask(message.data.taskId);
        break;
        
      case 'get-task-status':
        this.getTaskStatus(message.data.taskId);
        break;
        
      case 'health-check':
        this.sendHealthReport();
        break;
        
      case 'memory-cleanup':
        await this.performCleanup();
        break;
        
      default:
        console.log(`[AI-COORDINATOR-${this.workerId}] Unknown message type: ${message.type}`);
    }
  }
  
  async initializeConnections() {
    try {
      // Connect to NATS for agent communication
      this.natsConnection = await nats.connect({
        servers: [this.services.natsServer],
        reconnectTimeWait: 2000,
        maxReconnectAttempts: 10
      });
      
      console.log(`[AI-COORDINATOR-${this.workerId}] NATS connected`);
      
      // Initialize gRPC client for Go-Kratos services
      const protoLoader = require('@grpc/proto-loader');
      
      // Load proto definitions (assuming they exist)
      const packageDefinition = protoLoader.loadSync('./protos/legal-ai.proto', {
        keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true
      });
      
      const legalProto = grpc.loadPackageDefinition(packageDefinition).legal.v1;
      
      this.kratosClient = new legalProto.LegalAnalysisService(
        this.services.kratosGrpc,
        grpc.credentials.createInsecure(),
        {
          'grpc.keepalive_time_ms': 30000,
          'grpc.keepalive_timeout_ms': 5000
        }
      );
      
      console.log(`[AI-COORDINATOR-${this.workerId}] gRPC client connected`);
      
    } catch (error) {
      console.error(`[AI-COORDINATOR-${this.workerId}] Connection initialization error:`, error);
    }
  }
  
  setupAgentRegistry() {
    // Register available AI agents and their capabilities
    this.agentRegistry.set('legal-analyst', {
      name: 'Legal Document Analyst',
      capabilities: ['contract-analysis', 'risk-assessment', 'compliance-check'],
      endpoint: 'legal.agents.analyst',
      maxConcurrent: 3,
      currentLoad: 0
    });
    
    this.agentRegistry.set('research-agent', {
      name: 'Legal Research Agent',
      capabilities: ['case-law-search', 'precedent-analysis', 'citation-validation'],
      endpoint: 'legal.agents.research',
      maxConcurrent: 2,
      currentLoad: 0
    });
    
    this.agentRegistry.set('summarizer', {
      name: 'Document Summarizer',
      capabilities: ['text-summarization', 'key-extraction', 'executive-summary'],
      endpoint: 'legal.agents.summarizer',
      maxConcurrent: 5,
      currentLoad: 0
    });
    
    this.agentRegistry.set('classifier', {
      name: 'Document Classifier',
      capabilities: ['document-type', 'priority-assessment', 'category-tagging'],
      endpoint: 'legal.agents.classifier',
      maxConcurrent: 4,
      currentLoad: 0
    });
    
    this.agentRegistry.set('kratos-service', {
      name: 'Go-Kratos High-Performance Service',
      capabilities: ['vector-operations', 'bulk-processing', 'performance-critical'],
      endpoint: 'grpc',
      maxConcurrent: 10,
      currentLoad: 0
    });
    
    console.log(`[AI-COORDINATOR-${this.workerId}] Agent registry setup with ${this.agentRegistry.size} agents`);
  }
  
  async coordinateAnalysis(analysisData) {
    const taskId = this.generateTaskId();
    const startTime = Date.now();
    
    try {
      console.log(`[AI-COORDINATOR-${this.workerId}] Coordinating analysis task: ${taskId}`);
      
      const task = {
        id: taskId,
        type: 'coordinate-analysis',
        data: analysisData,
        status: 'running',
        startTime: startTime,
        steps: [],
        results: {}
      };
      
      this.activeTasks.set(taskId, task);
      
      // Step 1: Document classification
      const classificationResult = await this.callAgent('classifier', {
        action: 'document-type',
        content: analysisData.content,
        metadata: analysisData.metadata
      });
      
      task.steps.push({ step: 'classification', completed: true, result: classificationResult });
      
      // Step 2: Parallel analysis based on document type
      const analysisPromises = [];
      
      if (classificationResult.documentType === 'contract') {
        analysisPromises.push(
          this.callAgent('legal-analyst', {
            action: 'contract-analysis',
            content: analysisData.content,
            metadata: analysisData.metadata
          })
        );
      }
      
      analysisPromises.push(
        this.callAgent('summarizer', {
          action: 'executive-summary',
          content: analysisData.content,
          maxLength: 500
        })
      );
      
      analysisPromises.push(
        this.callAgent('research-agent', {
          action: 'precedent-analysis',
          content: analysisData.content,
          jurisdiction: analysisData.jurisdiction || 'federal'
        })
      );
      
      // Wait for parallel analysis to complete
      const analysisResults = await Promise.allSettled(analysisPromises);
      
      task.steps.push({ step: 'parallel-analysis', completed: true, results: analysisResults });
      
      // Step 3: Risk assessment
      const riskAssessment = await this.callAgent('legal-analyst', {
        action: 'risk-assessment',
        content: analysisData.content,
        analysisResults: analysisResults
      });
      
      task.steps.push({ step: 'risk-assessment', completed: true, result: riskAssessment });
      
      // Step 4: Final synthesis
      const synthesis = await this.synthesizeResults(task.steps, analysisData);
      
      task.status = 'completed';
      task.results = synthesis;
      task.completionTime = Date.now();
      task.duration = task.completionTime - startTime;
      
      this.coordinatedTasks++;
      
      this.sendMessage({
        type: 'analysis-completed',
        data: {
          taskId: taskId,
          results: synthesis,
          duration: task.duration,
          steps: task.steps.length
        }
      });
      
      console.log(`[AI-COORDINATOR-${this.workerId}] Analysis task ${taskId} completed in ${task.duration}ms`);
      
    } catch (error) {
      console.error(`[AI-COORDINATOR-${this.workerId}] Analysis coordination error:`, error);
      
      const task = this.activeTasks.get(taskId);
      if (task) {
        task.status = 'failed';
        task.error = error.message;
      }
      
      this.sendMessage({
        type: 'analysis-error',
        data: {
          taskId: taskId,
          error: error.message,
          duration: Date.now() - startTime
        }
      });
    } finally {
      // Clean up completed task after delay
      setTimeout(() => {
        this.activeTasks.delete(taskId);
      }, 60000);
    }
  }
  
  async executeMultiAgentTask(taskData) {
    const taskId = this.generateTaskId();
    
    try {
      console.log(`[AI-COORDINATOR-${this.workerId}] Executing multi-agent task: ${taskId}`);
      
      const { workflow, data } = taskData;
      const results = {};
      
      // Execute workflow steps sequentially or in parallel based on dependencies
      for (const step of workflow.steps) {
        if (step.parallel) {
          // Execute parallel steps
          const parallelPromises = step.agents.map(agentConfig => 
            this.callAgent(agentConfig.agent, {
              ...agentConfig.params,
              ...data
            })
          );
          
          const parallelResults = await Promise.allSettled(parallelPromises);
          results[step.name] = parallelResults;
          
        } else {
          // Execute sequential step
          const result = await this.callAgent(step.agent, {
            ...step.params,
            ...data,
            previousResults: results
          });
          
          results[step.name] = result;
        }
      }
      
      this.sendMessage({
        type: 'multi-agent-completed',
        data: {
          taskId: taskId,
          workflow: workflow.name,
          results: results
        }
      });
      
    } catch (error) {
      console.error(`[AI-COORDINATOR-${this.workerId}] Multi-agent task error:`, error);
      
      this.sendMessage({
        type: 'multi-agent-error',
        data: {
          taskId: taskId,
          error: error.message
        }
      });
    }
  }
  
  async callAgent(agentName, params) {
    const agent = this.agentRegistry.get(agentName);
    
    if (!agent) {
      throw new Error(`Agent not found: ${agentName}`);
    }
    
    // Check agent availability
    if (agent.currentLoad >= agent.maxConcurrent) {
      throw new Error(`Agent ${agentName} at maximum capacity`);
    }
    
    agent.currentLoad++;
    
    try {
      let result;
      
      if (agent.endpoint === 'grpc') {
        // Call Go-Kratos service via gRPC
        result = await this.callKratosService(params);
      } else {
        // Call agent via NATS
        result = await this.callNATSAgent(agent.endpoint, params);
      }
      
      return result;
      
    } finally {
      agent.currentLoad = Math.max(0, agent.currentLoad - 1);
    }
  }
  
  async callKratosService(params) {
    return new Promise((resolve, reject) => {
      if (!this.kratosClient) {
        return reject(new Error('Kratos client not available'));
      }
      
      // Map params to appropriate gRPC method
      let method = 'ProcessDocument';
      if (params.action === 'vector-operations') {
        method = 'ProcessVectors';
      } else if (params.action === 'bulk-processing') {
        method = 'ProcessBulk';
      }
      
      this.kratosClient[method](params, (error, response) => {
        if (error) {
          reject(error);
        } else {
          resolve(response);
        }
      });
    });
  }
  
  async callNATSAgent(endpoint, params) {
    if (!this.natsConnection) {
      throw new Error('NATS connection not available');
    }
    
    const requestData = {
      id: this.generateTaskId(),
      params: params,
      timestamp: Date.now(),
      requestedBy: this.workerId
    };
    
    // Publish request and wait for response
    const response = await this.natsConnection.request(
      endpoint,
      JSON.stringify(requestData),
      { timeout: this.config.taskTimeout }
    );
    
    return JSON.parse(new TextDecoder().decode(response.data));
  }
  
  async synthesizeResults(steps, originalData) {
    const synthesis = {
      documentId: originalData.documentId,
      analysisType: 'comprehensive',
      confidence: 0,
      recommendations: [],
      riskLevel: 'unknown',
      summary: '',
      keyFindings: [],
      actionItems: [],
      metadata: {
        analysisSteps: steps.length,
        processingTime: steps.reduce((total, step) => total + (step.duration || 0), 0),
        analyzedBy: this.workerId,
        timestamp: Date.now()
      }
    };
    
    // Extract key information from each step
    steps.forEach(step => {
      if (step.step === 'classification' && step.result) {
        synthesis.documentType = step.result.documentType;
        synthesis.confidence += step.result.confidence * 0.2;
      }
      
      if (step.step === 'risk-assessment' && step.result) {
        synthesis.riskLevel = step.result.riskLevel;
        synthesis.riskFactors = step.result.factors;
        synthesis.confidence += step.result.confidence * 0.3;
      }
      
      if (step.step === 'parallel-analysis' && step.results) {
        step.results.forEach(result => {
          if (result.status === 'fulfilled' && result.value) {
            if (result.value.summary) {
              synthesis.summary = result.value.summary;
            }
            if (result.value.keyFindings) {
              synthesis.keyFindings.push(...result.value.keyFindings);
            }
            if (result.value.recommendations) {
              synthesis.recommendations.push(...result.value.recommendations);
            }
          }
        });
      }
    });
    
    // Generate final recommendations
    synthesis.recommendations = this.generateFinalRecommendations(synthesis);
    
    // Calculate overall confidence
    synthesis.confidence = Math.min(1.0, synthesis.confidence);
    
    return synthesis;
  }
  
  generateFinalRecommendations(synthesis) {
    const recommendations = [];
    
    // Risk-based recommendations
    if (synthesis.riskLevel === 'high') {
      recommendations.push({
        type: 'urgent',
        priority: 1,
        action: 'Immediate legal review required',
        reason: 'High risk factors identified'
      });
    }
    
    // Document type specific recommendations
    if (synthesis.documentType === 'contract') {
      recommendations.push({
        type: 'review',
        priority: 2,
        action: 'Verify contract terms and conditions',
        reason: 'Contract analysis completed'
      });
    }
    
    // Confidence-based recommendations
    if (synthesis.confidence < 0.7) {
      recommendations.push({
        type: 'verification',
        priority: 3,
        action: 'Manual verification recommended',
        reason: 'Low confidence in automated analysis'
      });
    }
    
    return recommendations.sort((a, b) => a.priority - b.priority);
  }
  
  startTaskProcessor() {
    // Process queued tasks
    setInterval(() => {
      this.processTaskQueue();
    }, 1000);
  }
  
  processTaskQueue() {
    while (this.taskQueue.length > 0 && this.activeTasks.size < this.config.maxConcurrentTasks) {
      const task = this.taskQueue.shift();
      this.executeTask(task);
    }
  }
  
  async executeTask(task) {
    this.activeTasks.set(task.id, task);
    
    try {
      switch (task.type) {
        case 'coordinate-analysis':
          await this.coordinateAnalysis(task.data);
          break;
        case 'multi-agent-task':
          await this.executeMultiAgentTask(task.data);
          break;
        default:
          throw new Error(`Unknown task type: ${task.type}`);
      }
    } catch (error) {
      console.error(`[AI-COORDINATOR-${this.workerId}] Task execution error:`, error);
    }
  }
  
  performHealthChecks() {
    // Check agent health
    this.agentRegistry.forEach((agent, name) => {
      if (agent.currentLoad > agent.maxConcurrent) {
        console.warn(`[AI-COORDINATOR-${this.workerId}] Agent ${name} overloaded: ${agent.currentLoad}/${agent.maxConcurrent}`);
      }
    });
    
    // Check task timeouts
    const now = Date.now();
    this.activeTasks.forEach((task, taskId) => {
      if (now - task.startTime > this.config.taskTimeout) {
        console.warn(`[AI-COORDINATOR-${this.workerId}] Task ${taskId} timeout, cancelling`);
        this.cancelTask(taskId);
      }
    });
  }
  
  async cancelTask(taskId) {
    const task = this.activeTasks.get(taskId);
    if (task) {
      task.status = 'cancelled';
      task.cancelledAt = Date.now();
      
      this.sendMessage({
        type: 'task-cancelled',
        data: { taskId, reason: 'User requested or timeout' }
      });
      
      // Clean up after delay
      setTimeout(() => {
        this.activeTasks.delete(taskId);
      }, 10000);
    }
  }
  
  getTaskStatus(taskId) {
    const task = this.activeTasks.get(taskId);
    
    this.sendMessage({
      type: 'task-status',
      data: {
        taskId: taskId,
        status: task ? task.status : 'not-found',
        task: task
      }
    });
  }
  
  generateTaskId() {
    return `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  async performCleanup() {
    // Force garbage collection
    if (global.gc) {
      global.gc();
    }
    
    // Clean up old completed tasks
    const now = Date.now();
    const oldTasks = [];
    
    this.activeTasks.forEach((task, taskId) => {
      if (task.status === 'completed' || task.status === 'failed') {
        if (now - (task.completionTime || task.startTime) > 300000) { // 5 minutes
          oldTasks.push(taskId);
        }
      }
    });
    
    oldTasks.forEach(taskId => {
      this.activeTasks.delete(taskId);
    });
    
    console.log(`[AI-COORDINATOR-${this.workerId}] Cleanup completed. Removed ${oldTasks.length} old tasks`);
  }
  
  sendHealthReport() {
    const health = {
      worker: 'ai-coordinator',
      workerId: this.workerId,
      pid: process.pid,
      uptime: Date.now() - this.startTime,
      coordinatedTasks: this.coordinatedTasks,
      activeTasks: this.activeTasks.size,
      queuedTasks: this.taskQueue.length,
      memoryUsage: process.memoryUsage(),
      agents: Array.from(this.agentRegistry.entries()).map(([name, agent]) => ({
        name: name,
        currentLoad: agent.currentLoad,
        maxConcurrent: agent.maxConcurrent,
        utilization: (agent.currentLoad / agent.maxConcurrent * 100).toFixed(1) + '%'
      })),
      connections: {
        nats: this.natsConnection ? 'connected' : 'disconnected',
        kratos: this.kratosClient ? 'connected' : 'disconnected'
      },
      timestamp: Date.now()
    };
    
    this.sendMessage({
      type: 'health-report',
      data: health
    });
  }
  
  sendMessage(message) {
    try {
      parentPort.postMessage(message);
    } catch (error) {
      console.error(`[AI-COORDINATOR-${this.workerId}] Failed to send message:`, error);
    }
  }
}

// Initialize worker if running in worker thread
if (!isMainThread) {
  new AICoordinatorWorker(workerData);
}

module.exports = AICoordinatorWorker;