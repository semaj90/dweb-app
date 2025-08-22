
/**
 * Stateless API Coordinator for Phase 13
 * Redis/NATS/ZeroMQ integration with task coordination and load balancing
 * Designed for SvelteKit 2 with inverse relationship philosophy
 */

import { writable, derived, type Writable } from "svelte/store";
// TODO: Fix import - // Orphaned content: import {  // Task coordination types
export interface TaskMessage {
  id: string;
  type: "LEGAL_ANALYSIS" | "DOCUMENT_PROCESSING" | "AI_INFERENCE" | "VECTOR_SEARCH" | "REPORT_GENERATION";
  payload: any;
  priority: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  timestamp: number;
  retryCount: number;
  maxRetries: number;
  timeout: number;
  nodeAffinity?: string;
  dependencies?: string[];
  metadata: {
    caseId?: string;
    userId?: string;
    sessionId?: string;
    estimatedDuration?: number;
  };
}

export interface TaskResult {
  taskId: string;
  status: "SUCCESS" | "FAILURE" | "TIMEOUT" | "RETRY";
  result?: unknown;
  error?: string;
  processingTime: number;
  nodeId: string;
  timestamp: number;
}

export interface APINode {
  id: string;
  endpoint: string;
  type: "REDIS" | "NATS" | "ZEROMQ" | "WEBSOCKET";
  status: "ACTIVE" | "INACTIVE" | "DEGRADED" | "MAINTENANCE";
  load: number;
  capacity: number;
  lastHeartbeat: number;
  capabilities: string[];
  region?: string;
  metadata: {
    version: string;
    uptime: number;
    processedTasks: number;
    errorRate: number;
  };
}

export interface CoordinationConfig {
  enableRedis: boolean;
  enableNATS: boolean;
  enableZeroMQ: boolean;
  enableWebSocket: boolean;
  taskTimeout: number;
  maxRetries: number;
  heartbeatInterval: number;
  loadBalancingStrategy: "ROUND_ROBIN" | "LEAST_CONNECTIONS" | "WEIGHTED" | "AFFINITY";
  failoverThreshold: number;
  batchSize: number;
}

// Stateless API Coordinator Class
export class StatelessAPICoordinator {
  private config: CoordinationConfig;
  private nodes: Map<string, APINode> = new Map();
  private tasks: Map<string, TaskMessage> = new Map();
  private results: Map<string, TaskResult> = new Map();
  private connectionPool: Map<string, any> = new Map();
  
  // Reactive stores for Svelte integration
  public activeNodes = writable<APINode[]>([]);
  public queuedTasks = writable<TaskMessage[]>([]);
  public completedTasks = writable<TaskResult[]>([]);
  public systemHealth = writable<{ overall: number; redis: number; nats: number; zeromq: number }>({
    overall: 0,
    redis: 0,
    nats: 0,
    zeromq: 0
  });
  public throughputMetrics = writable<{
    tasksPerSecond: number;
    averageLatency: number;
    errorRate: number;
    queueDepth: number;
  }>({
    tasksPerSecond: 0,
    averageLatency: 0,
    errorRate: 0,
    queueDepth: 0
  });

  // Performance monitoring
  private metricsInterval?: number;
  private startTime = Date.now();
  private processedTaskCount = 0;
  private errorCount = 0;
  private latencySum = 0;

  constructor(config: Partial<CoordinationConfig> = {}) {
    this.config = {
      enableRedis: true,
      enableNATS: true,
      enableZeroMQ: false,
      enableWebSocket: true,
      taskTimeout: 30000,
      maxRetries: 3,
      heartbeatInterval: 5000,
      loadBalancingStrategy: "LEAST_CONNECTIONS",
      failoverThreshold: 0.1,
      batchSize: 10,
      ...config
    };

    this.initializeNodes();
    this.startHeartbeatMonitoring();
    this.startMetricsCollection();
  }

  // Initialize connection nodes
  private initializeNodes(): void {
    if (!browser) return;

    // Redis nodes (simulated - in production, use real Redis connections)
    if (this.config.enableRedis) {
      this.addNode({
        id: "redis-primary",
        endpoint: "redis://localhost:6379",
        type: "REDIS",
        status: "ACTIVE",
        load: 0,
        capacity: 100,
        lastHeartbeat: Date.now(),
        capabilities: ["QUEUE", "PUBSUB", "CACHE", "STREAM"],
        metadata: {
          version: "7.0.0",
          uptime: 0,
          processedTasks: 0,
          errorRate: 0
        }
      });

      this.addNode({
        id: "redis-secondary",
        endpoint: "redis://localhost:6380",
        type: "REDIS",
        status: "ACTIVE",
        load: 0,
        capacity: 100,
        lastHeartbeat: Date.now(),
        capabilities: ["QUEUE", "PUBSUB", "CACHE"],
        metadata: {
          version: "7.0.0",
          uptime: 0,
          processedTasks: 0,
          errorRate: 0
        }
      });
    }

    // NATS nodes (simulated - in production, use real NATS connections)
    if (this.config.enableNATS) {
      this.addNode({
        id: "nats-cluster-1",
        endpoint: "nats://localhost:4222",
        type: "NATS",
        status: "ACTIVE",
        load: 0,
        capacity: 200,
        lastHeartbeat: Date.now(),
        capabilities: ["PUBSUB", "REQUEST_REPLY", "STREAMING"],
        metadata: {
          version: "2.9.0",
          uptime: 0,
          processedTasks: 0,
          errorRate: 0
        }
      });
    }

    // WebSocket nodes for real-time communication
    if (this.config.enableWebSocket) {
      this.addNode({
        id: "websocket-gateway",
        endpoint: "ws://localhost:5173/ws",
        type: "WEBSOCKET",
        status: "ACTIVE",
        load: 0,
        capacity: 50,
        lastHeartbeat: Date.now(),
        capabilities: ["REALTIME", "EVENTS", "NOTIFICATIONS"],
        metadata: {
          version: "1.0.0",
          uptime: 0,
          processedTasks: 0,
          errorRate: 0
        }
      });
    }

    this.updateActiveNodes();
  }

  // Add a new node to the coordination system
  public addNode(node: APINode): void {
    this.nodes.set(node.id, node);
    this.initializeConnection(node);
    this.updateActiveNodes();
  }

  // Initialize connection to a node
  private async initializeConnection(node: APINode): Promise<void> {
    try {
      switch (node.type) {
        case "REDIS":
          // In production: const redis = new Redis(node.endpoint);
          const redisConnection = {
            endpoint: node.endpoint,
            connected: true,
            lastPing: Date.now(),
            send: this.createMockSender(node.id),
            subscribe: this.createMockSubscriber(node.id)
          };
          this.connectionPool.set(node.id, redisConnection);
          break;

        case "NATS":
          // In production: const nc = await connect({ servers: [node.endpoint] });
          const natsConnection = {
            endpoint: node.endpoint,
            connected: true,
            lastPing: Date.now(),
            publish: this.createMockPublisher(node.id),
            subscribe: this.createMockSubscriber(node.id),
            request: this.createMockRequester(node.id)
          };
          this.connectionPool.set(node.id, natsConnection);
          break;

        case "WEBSOCKET":
          // Real WebSocket connection for browser
          if (browser) {
            const ws = new WebSocket(node.endpoint);
            ws.onopen = () => {
              console.log(`WebSocket connected: ${node.id}`);
            };
            ws.onmessage = (event) => {
              this.handleWebSocketMessage(node.id, JSON.parse(event.data));
            };
            ws.onerror = () => {
              this.markNodeDegraded(node.id);
            };
            this.connectionPool.set(node.id, ws);
          }
          break;
      }

      node.status = "ACTIVE";
      node.lastHeartbeat = Date.now();
    } catch (error) {
      console.error(`Failed to initialize connection to ${node.id}:`, error);
      node.status = "INACTIVE";
    }
  }

  // Submit a task for processing
  public async submitTask(task: Omit<TaskMessage, "id" | "timestamp" | "retryCount">): Promise<string> {
    const taskId = `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const fullTask: TaskMessage = {
      ...task,
      id: taskId,
      timestamp: Date.now(),
      retryCount: 0,
      maxRetries: this.config.maxRetries
    };

    this.tasks.set(taskId, fullTask);
    this.updateQueuedTasks();

    // Select optimal node for task
    const selectedNode = this.selectNodeForTask(fullTask);
    if (!selectedNode) {
      throw new Error("No available nodes for task processing");
    }

    // Route task to selected node
    await this.routeTaskToNode(fullTask, selectedNode);
    
    return taskId;
  }

  // Select optimal node based on load balancing strategy
  private selectNodeForTask(task: TaskMessage): APINode | null {
    const availableNodes = Array.from(this.nodes.values())
      .filter((node: any) => node.status === "ACTIVE" && 
        node.load < node.capacity &&
        this.nodeSupportsTask(node, task)
      );

    if (availableNodes.length === 0) return null;

    switch (this.config.loadBalancingStrategy) {
      case "ROUND_ROBIN":
        return availableNodes[this.processedTaskCount % availableNodes.length];
        
      case "LEAST_CONNECTIONS":
        return availableNodes.reduce((prev, current) => 
          prev.load < current.load ? prev : current
        );
        
      case "WEIGHTED":
        const weightedNode = availableNodes.reduce((prev, current) => {
          const prevScore = (current.capacity - current.load) / current.capacity;
          const currentScore = (prev.capacity - prev.load) / prev.capacity;
          return prevScore > currentScore ? current : prev;
        });
        return weightedNode;
        
      case "AFFINITY":
        if (task.nodeAffinity) {
          const affinityNode = availableNodes.find((node: any) => node.id === task.nodeAffinity);
          if (affinityNode) return affinityNode;
        }
        return availableNodes[0];
        
      default:
        return availableNodes[0];
    }
  }

  // Check if node supports task type
  private nodeSupportsTask(node: APINode, task: TaskMessage): boolean {
    const requiredCapabilities = {
      "LEGAL_ANALYSIS": ["QUEUE", "PUBSUB"],
      "DOCUMENT_PROCESSING": ["QUEUE", "STREAM"],
      "AI_INFERENCE": ["QUEUE", "REQUEST_REPLY"],
      "VECTOR_SEARCH": ["QUEUE", "CACHE"],
      "REPORT_GENERATION": ["QUEUE", "PUBSUB"]
    };

    const required = requiredCapabilities[task.type] || ["QUEUE"];
    return required.some((capability: any) => node.capabilities.includes(capability));
  }

  // Route task to specific node
  private async routeTaskToNode(task: TaskMessage, node: APINode): Promise<void> {
    const connection = this.connectionPool.get(node.id);
    if (!connection) {
      throw new Error(`No connection available for node ${node.id}`);
    }

    try {
      const taskData = {
        task,
        routingKey: `legal.${task.type.toLowerCase()}`,
        timestamp: Date.now()
      };

      switch (node.type) {
        case "REDIS":
          await connection.send("task_queue", JSON.stringify(taskData));
          break;
          
        case "NATS":
          await connection.publish(`legal.tasks.${task.type.toLowerCase()}`, JSON.stringify(taskData));
          break;
          
        case "WEBSOCKET":
          if (connection.readyState === WebSocket.OPEN) {
            connection.send(JSON.stringify({ type: "TASK", data: taskData }));
          }
          break;
      }

      // Update node load
      node.load += 1;
      node.metadata.processedTasks += 1;
      this.updateActiveNodes();

    } catch (error) {
      console.error(`Failed to route task to ${node.id}:`, error);
      this.markNodeDegraded(node.id);
      
      // Retry with different node
      if (task.retryCount < task.maxRetries) {
        task.retryCount += 1;
        const alternateNode = this.selectNodeForTask(task);
        if (alternateNode) {
          await this.routeTaskToNode(task, alternateNode);
        }
      }
    }
  }

  // Handle task completion
  public handleTaskResult(result: TaskResult): void {
    const task = this.tasks.get(result.taskId);
    if (!task) return;

    this.results.set(result.taskId, result);
    this.tasks.delete(result.taskId);

    // Update metrics
    this.processedTaskCount += 1;
    this.latencySum += result.processingTime;
    
    if (result.status === "FAILURE") {
      this.errorCount += 1;
    }

    // Update node load
    const nodeId = result.nodeId;
    const node = this.nodes.get(nodeId);
    if (node) {
      node.load = Math.max(0, node.load - 1);
      if (result.status === "FAILURE") {
        node.metadata.errorRate = (node.metadata.errorRate + 1) / node.metadata.processedTasks;
      }
    }

    this.updateQueuedTasks();
    this.updateCompletedTasks();
    this.updateActiveNodes();
  }

  // WebSocket message handler
  private handleWebSocketMessage(nodeId: string, message: any): void {
    switch (message.type) {
      case "TASK_RESULT":
        this.handleTaskResult({
          ...message.data,
          nodeId
        });
        break;
        
      case "HEARTBEAT":
        const node = this.nodes.get(nodeId);
        if (node) {
          node.lastHeartbeat = Date.now();
          node.status = "ACTIVE";
        }
        break;
        
      case "LOAD_UPDATE":
        const loadNode = this.nodes.get(nodeId);
        if (loadNode) {
          loadNode.load = message.data.load;
        }
        break;
    }
  }

  // Heartbeat monitoring
  private startHeartbeatMonitoring(): void {
    if (!browser) return;

    setInterval(() => {
      const now = Date.now();
      const staleThreshold = this.config.heartbeatInterval * 2;

      for (const [nodeId, node] of this.nodes.entries()) {
        if (now - node.lastHeartbeat > staleThreshold) {
          this.markNodeDegraded(nodeId);
        }
      }

      this.updateSystemHealth();
    }, this.config.heartbeatInterval);
  }

  // Mark node as degraded
  private markNodeDegraded(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (node) {
      node.status = "DEGRADED";
      node.metadata.errorRate = Math.min(1.0, node.metadata.errorRate + 0.1);
      this.updateActiveNodes();
    }
  }

  // Metrics collection
  private startMetricsCollection(): void {
    if (!browser) return;

    this.metricsInterval = window.setInterval(() => {
      const timeElapsed = (Date.now() - this.startTime) / 1000;
      const tasksPerSecond = this.processedTaskCount / timeElapsed;
      const averageLatency = this.processedTaskCount > 0 ? this.latencySum / this.processedTaskCount : 0;
      const errorRate = this.processedTaskCount > 0 ? this.errorCount / this.processedTaskCount : 0;
      const queueDepth = this.tasks.size;

      this.throughputMetrics.set({
        tasksPerSecond,
        averageLatency,
        errorRate,
        queueDepth
      });
    }, 1000);
  }

  // Update reactive stores
  private updateActiveNodes(): void {
    this.activeNodes.set(Array.from(this.nodes.values()));
  }

  private updateQueuedTasks(): void {
    this.queuedTasks.set(Array.from(this.tasks.values()));
  }

  private updateCompletedTasks(): void {
    const recent = Array.from(this.results.values())
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 100);
    this.completedTasks.set(recent);
  }

  private updateSystemHealth(): void {
    const totalNodes = this.nodes.size;
    if (totalNodes === 0) {
      this.systemHealth.set({ overall: 0, redis: 0, nats: 0, zeromq: 0 });
      return;
    }

    const activeNodes = Array.from(this.nodes.values()).filter((n: any) => n.status === "ACTIVE");
    const redisNodes = Array.from(this.nodes.values()).filter((n: any) => n.type === "REDIS");
    const natsNodes = Array.from(this.nodes.values()).filter((n: any) => n.type === "NATS");
    const zeromqNodes = Array.from(this.nodes.values()).filter((n: any) => n.type === "ZEROMQ");

    const overall = (activeNodes.length / totalNodes) * 100;
    const redis = redisNodes.length > 0 ? (redisNodes.filter((n: any) => n.status === "ACTIVE").length / redisNodes.length) * 100 : 0;
    const nats = natsNodes.length > 0 ? (natsNodes.filter((n: any) => n.status === "ACTIVE").length / natsNodes.length) * 100 : 0;
    const zeromq = zeromqNodes.length > 0 ? (zeromqNodes.filter((n: any) => n.status === "ACTIVE").length / zeromqNodes.length) * 100 : 0;

    this.systemHealth.set({ overall, redis, nats, zeromq });
  }

  // Mock implementations for development
  private createMockSender(nodeId: string) {
    return async (queue: string, data: string) => {
      setTimeout(() => {
        // Simulate task processing
        const task = JSON.parse(data).task;
        const result: TaskResult = {
          taskId: task.id,
          status: Math.random() > 0.1 ? "SUCCESS" : "FAILURE",
          result: { processed: true, nodeId },
          processingTime: Math.floor(Math.random() * 1000) + 100,
          nodeId,
          timestamp: Date.now()
        };
        this.handleTaskResult(result);
      }, Math.floor(Math.random() * 2000) + 500);
    };
  }

  private createMockPublisher(nodeId: string) {
    return async (subject: string, data: string) => {
      setTimeout(() => {
        const task = JSON.parse(data).task;
        const result: TaskResult = {
          taskId: task.id,
          status: Math.random() > 0.05 ? "SUCCESS" : "FAILURE",
          result: { processed: true, nodeId },
          processingTime: Math.floor(Math.random() * 800) + 100,
          nodeId,
          timestamp: Date.now()
        };
        this.handleTaskResult(result);
      }, Math.floor(Math.random() * 1500) + 300);
    };
  }

  private createMockSubscriber(nodeId: string) {
    return (subject: string, handler: (data: any) => void) => {
      // Mock subscription
      return () => {}; // Unsubscribe function
    };
  }

  private createMockRequester(nodeId: string) {
    return async (subject: string, data: string) => {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({ success: true, nodeId, processed: Date.now() });
        }, Math.floor(Math.random() * 500) + 100);
      });
    };
  }

  // Cleanup
  public destroy(): void {
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
    }

    // Close all connections
    for (const [nodeId, connection] of this.connectionPool.entries()) {
      const node = this.nodes.get(nodeId);
      if (node?.type === "WEBSOCKET" && connection instanceof WebSocket) {
        connection.close();
      }
    }

    this.connectionPool.clear();
    this.nodes.clear();
    this.tasks.clear();
    this.results.clear();
  }
}

// Factory function for Svelte integration
export function createStatelessAPICoordinator(config?: Partial<CoordinationConfig>) {
  const coordinator = new StatelessAPICoordinator(config);
  
  return {
    coordinator,
    stores: {
      activeNodes: coordinator.activeNodes,
      queuedTasks: coordinator.queuedTasks,
      completedTasks: coordinator.completedTasks,
      systemHealth: coordinator.systemHealth,
      throughputMetrics: coordinator.throughputMetrics
    },
    
    // Derived stores
    derived: {
      isHealthy: derived(coordinator.systemHealth, ($health) => $health.overall > 70),
      totalTasks: derived(
        [coordinator.queuedTasks, coordinator.completedTasks],
        ([$queued, $completed]) => $queued.length + $completed.length
      ),
      averageProcessingTime: derived(
        coordinator.completedTasks,
        ($completed) => {
          if ($completed.length === 0) return 0;
          const sum = $completed.reduce((acc, task) => acc + task.processingTime, 0);
          return sum / $completed.length;
        }
      )
    },
    
    // API methods
    submitTask: coordinator.submitTask.bind(coordinator),
    addNode: coordinator.addNode.bind(coordinator),
    destroy: coordinator.destroy.bind(coordinator)
  };
}

// Helper functions for common task types
export const TaskTemplates = {
  legalAnalysis: (caseId: string, documentIds: string[]): Omit<TaskMessage, "id" | "timestamp" | "retryCount"> => ({
    type: "LEGAL_ANALYSIS",
    payload: { caseId, documentIds },
    priority: "HIGH",
    maxRetries: 2,
    timeout: 60000,
    metadata: { caseId, estimatedDuration: 45000 }
  }),

  documentProcessing: (documentId: string, extractText: boolean = true): Omit<TaskMessage, "id" | "timestamp" | "retryCount"> => ({
    type: "DOCUMENT_PROCESSING",
    payload: { documentId, extractText },
    priority: "MEDIUM",
    maxRetries: 3,
    timeout: 30000,
    metadata: { estimatedDuration: 20000 }
  }),

  aiInference: (prompt: string, model: string = "gemma3-legal"): Omit<TaskMessage, "id" | "timestamp" | "retryCount"> => ({
    type: "AI_INFERENCE",
    payload: { prompt, model },
    priority: "HIGH",
    maxRetries: 2,
    timeout: 45000,
    nodeAffinity: "nats-cluster-1",
    metadata: { estimatedDuration: 30000 }
  }),

  vectorSearch: (query: string, topK: number = 10): Omit<TaskMessage, "id" | "timestamp" | "retryCount"> => ({
    type: "VECTOR_SEARCH",
    payload: { query, topK },
    priority: "MEDIUM",
    maxRetries: 2,
    timeout: 15000,
    metadata: { estimatedDuration: 5000 }
  }),

  reportGeneration: (caseId: string, template: string): Omit<TaskMessage, "id" | "timestamp" | "retryCount"> => ({
    type: "REPORT_GENERATION",
    payload: { caseId, template },
    priority: "LOW",
    maxRetries: 1,
    timeout: 120000,
    metadata: { caseId, estimatedDuration: 90000 }
  })
};

export default StatelessAPICoordinator;