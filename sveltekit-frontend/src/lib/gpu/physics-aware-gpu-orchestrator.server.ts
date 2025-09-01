/**
 * @file Manages GPU resources for physics-based simulations or ML models.
 * This is a server-only module with '.server.ts' extension.
 */

interface GpuDevice {
  id: number;
  name: string;
  memoryUsage: number; // in GB
  memoryTotal: number; // in GB
  load: number; // 0-1 score
  temperature: number; // in Celsius
  powerUsage: number; // in watts
  computeUnits: number;
  clockSpeed: number; // in MHz
  lastUpdate: number;
}

interface GpuTask {
  id: string;
  type: 'inference' | 'training' | 'ocr' | 'vector_computation' | 'physics_simulation';
  requiredMemory: number;
  estimatedDuration: number; // in ms
  priority: 'low' | 'medium' | 'high' | 'critical';
  deviceId?: number;
  status: 'queued' | 'running' | 'completed' | 'failed';
  startTime?: number;
  endTime?: number;
}

class PhysicsAwareGpuOrchestrator {
  private availableDevices: GpuDevice[] = [];
  private taskQueue: GpuTask[] = [];
  private runningTasks: Map<string, GpuTask> = new Map();
  private completedTasks: GpuTask[] = [];
  private isInitialized = false;
  private totalGpuUtilization = 0.65; // Mock GPU utilization

  constructor() {
    console.log("PhysicsAwareGpuOrchestrator (server-only) instance created.");
  }

  /**
   * Initializes the orchestrator, detecting available GPU devices.
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;
    console.log("Initializing Physics-Aware GPU Orchestrator (server-only)...");
    
    // In a real implementation, this would use libraries like `node-nvidia-smi`
    // to query actual GPU hardware on the server.
    await this.detectHardwareGpus();
    this.startPerformanceMonitoring();
    
    this.isInitialized = true;
    console.log("Physics-Aware GPU Orchestrator (server-only) initialized.");
  }

  private async detectHardwareGpus(): Promise<void> {
    // Simulate hardware detection delay
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // Mock detection of RTX 3060 Ti as specified in system requirements
    this.availableDevices = [
      {
        id: 0,
        name: 'NVIDIA GeForce RTX 3060 Ti',
        memoryUsage: 4.2,
        memoryTotal: 8.0, // 8GB VRAM as specified
        load: 0.25,
        temperature: 65,
        powerUsage: 180,
        computeUnits: 4864, // CUDA cores
        clockSpeed: 1665, // Base clock in MHz
        lastUpdate: Date.now()
      },
      // Simulate additional GPU if available (for scaling)
      {
        id: 1,
        name: 'NVIDIA GeForce RTX 3060 Ti',
        memoryUsage: 2.8,
        memoryTotal: 8.0,
        load: 0.15,
        temperature: 58,
        powerUsage: 165,
        computeUnits: 4864,
        clockSpeed: 1665,
        lastUpdate: Date.now()
      }
    ];

    console.log(`Detected ${this.availableDevices.length} GPU devices:`);
    this.availableDevices.forEach(gpu => {
      console.log(`  - ${gpu.name} (${gpu.memoryTotal}GB VRAM, ${gpu.computeUnits} cores)`);
    });
  }

  private startPerformanceMonitoring(): void {
    // Update GPU metrics every 5 seconds
    setInterval(() => {
      this.updateDeviceMetrics();
      this.processTaskQueue();
      this.cleanupCompletedTasks();
    }, 5000);
  }

  private updateDeviceMetrics(): void {
    this.availableDevices.forEach(device => {
      // Simulate realistic GPU metric fluctuations
      const baseLoad = 0.1 + (this.runningTasks.size * 0.2);
      device.load = Math.min(0.95, baseLoad + (Math.random() * 0.1 - 0.05));
      device.temperature = 45 + (device.load * 30) + (Math.random() * 5 - 2.5);
      device.powerUsage = 120 + (device.load * 80) + (Math.random() * 10 - 5);
      device.memoryUsage = Math.max(1.0, device.memoryUsage + (Math.random() * 0.2 - 0.1));
      device.lastUpdate = Date.now();
    });

    // Update total utilization
    this.totalGpuUtilization = this.availableDevices.reduce((sum, device) => 
      sum + device.load, 0) / this.availableDevices.length;
  }

  private processTaskQueue(): void {
    // Process queued tasks if GPU resources are available
    const queuedTasks = this.taskQueue.filter(task => task.status === 'queued');
    
    for (const task of queuedTasks) {
      const suitableDevice = this.findOptimalDevice(task);
      if (suitableDevice) {
        this.executeTask(task, suitableDevice);
      }
    }
  }

  private cleanupCompletedTasks(): void {
    // Keep only recent completed tasks (last 100)
    if (this.completedTasks.length > 100) {
      this.completedTasks = this.completedTasks.slice(-100);
    }
  }

  /**
   * Schedules a task on the optimal GPU based on physics-aware algorithms.
   */
  scheduleTask(taskRequest: Omit<GpuTask, 'id' | 'status' | 'deviceId'>): { 
    taskId: string; 
    deviceId?: number; 
    estimatedStartTime: number;
  } | { error: string } {
    const task: GpuTask = {
      ...taskRequest,
      id: `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      status: 'queued'
    };

    // Immediate scheduling attempt
    const optimalDevice = this.findOptimalDevice(task);
    
    if (optimalDevice && this.canExecuteImmediately(task, optimalDevice)) {
      // Execute immediately
      this.executeTask(task, optimalDevice);
      return {
        taskId: task.id,
        deviceId: optimalDevice.id,
        estimatedStartTime: Date.now()
      };
    } else {
      // Queue for later execution
      this.taskQueue.push(task);
      const estimatedWaitTime = this.calculateEstimatedWaitTime(task);
      
      return {
        taskId: task.id,
        estimatedStartTime: Date.now() + estimatedWaitTime
      };
    }
  }

  private findOptimalDevice(task: GpuTask): GpuDevice | null {
    // Physics-aware device selection algorithm
    const candidateDevices = this.availableDevices.filter(device => {
      const availableMemory = device.memoryTotal - device.memoryUsage;
      const loadOk = device.load < 0.85; // Don't overload GPU
      const temperatureOk = device.temperature < 85; // Thermal protection
      
      return availableMemory >= task.requiredMemory && loadOk && temperatureOk;
    });

    if (candidateDevices.length === 0) return null;

    // Score devices based on multiple factors
    const deviceScores = candidateDevices.map(device => {
      const memoryScore = (device.memoryTotal - device.memoryUsage - task.requiredMemory) / device.memoryTotal;
      const loadScore = 1 - device.load;
      const temperatureScore = 1 - (device.temperature / 90);
      const powerEfficiencyScore = 1 - (device.powerUsage / 250);
      
      // Task-specific scoring
      let taskTypeScore = 1.0;
      if (task.type === 'physics_simulation' && device.computeUnits > 4000) {
        taskTypeScore = 1.2; // Prefer high-core-count GPUs for physics
      } else if (task.type === 'inference' && device.memoryTotal >= 8) {
        taskTypeScore = 1.1; // Prefer high-memory GPUs for inference
      }

      const totalScore = (
        memoryScore * 0.3 +
        loadScore * 0.25 +
        temperatureScore * 0.2 +
        powerEfficiencyScore * 0.15 +
        taskTypeScore * 0.1
      );

      return { device, score: totalScore };
    });

    // Sort by score and return the best device
    deviceScores.sort((a, b) => b.score - a.score);
    return deviceScores[0].device;
  }

  private canExecuteImmediately(task: GpuTask, device: GpuDevice): boolean {
    // Check if device can handle the task immediately
    const runningTasksOnDevice = Array.from(this.runningTasks.values())
      .filter(t => t.deviceId === device.id);
    
    // For high-priority tasks, allow some overlap
    if (task.priority === 'critical') {
      return runningTasksOnDevice.length < 2;
    } else if (task.priority === 'high') {
      return runningTasksOnDevice.length < 1;
    } else {
      return runningTasksOnDevice.length === 0;
    }
  }

  private executeTask(task: GpuTask, device: GpuDevice): void {
    task.status = 'running';
    task.deviceId = device.id;
    task.startTime = Date.now();
    
    this.runningTasks.set(task.id, task);
    
    // Update device memory usage
    device.memoryUsage += task.requiredMemory;
    
    console.log(`Executing task ${task.id} on GPU ${device.id} (${task.type})`);
    
    // Simulate task execution
    setTimeout(() => {
      this.completeTask(task.id);
    }, task.estimatedDuration);
  }

  private completeTask(taskId: string): void {
    const task = this.runningTasks.get(taskId);
    if (!task) return;

    task.status = 'completed';
    task.endTime = Date.now();
    
    // Free device memory
    const device = this.availableDevices.find(d => d.id === task.deviceId);
    if (device) {
      device.memoryUsage = Math.max(0, device.memoryUsage - task.requiredMemory);
    }
    
    this.runningTasks.delete(taskId);
    this.completedTasks.push(task);
    
    console.log(`Task ${taskId} completed in ${(task.endTime! - task.startTime!) / 1000}s`);
  }

  private calculateEstimatedWaitTime(task: GpuTask): number {
    // Estimate wait time based on queue and running tasks
    const queuePosition = this.taskQueue.filter(t => 
      t.status === 'queued' && t.priority >= task.priority
    ).length;
    
    const averageTaskDuration = 5000; // 5 seconds average
    return queuePosition * averageTaskDuration;
  }

  /**
   * Get current GPU utilization across all devices
   */
  getGPUUtilization(): number {
    return Math.round(this.totalGpuUtilization * 100) / 100;
  }

  /**
   * Get detailed GPU statistics
   */
  getGPUStats(): {
    devices: GpuDevice[];
    queuedTasks: number;
    runningTasks: number;
    completedTasks: number;
    totalUtilization: number;
    averageTemperature: number;
    totalPowerUsage: number;
  } {
    const averageTemperature = this.availableDevices.reduce((sum, device) => 
      sum + device.temperature, 0) / this.availableDevices.length;
    
    const totalPowerUsage = this.availableDevices.reduce((sum, device) => 
      sum + device.powerUsage, 0);

    return {
      devices: [...this.availableDevices],
      queuedTasks: this.taskQueue.filter(t => t.status === 'queued').length,
      runningTasks: this.runningTasks.size,
      completedTasks: this.completedTasks.length,
      totalUtilization: this.totalGpuUtilization,
      averageTemperature: Math.round(averageTemperature * 10) / 10,
      totalPowerUsage: Math.round(totalPowerUsage)
    };
  }

  /**
   * Get task information
   */
  getTaskInfo(taskId: string): GpuTask | null {
    // Check running tasks first
    const runningTask = this.runningTasks.get(taskId);
    if (runningTask) return runningTask;
    
    // Check queued tasks
    const queuedTask = this.taskQueue.find(t => t.id === taskId);
    if (queuedTask) return queuedTask;
    
    // Check completed tasks
    const completedTask = this.completedTasks.find(t => t.id === taskId);
    return completedTask || null;
  }

  /**
   * Cancel a queued task
   */
  cancelTask(taskId: string): boolean {
    const taskIndex = this.taskQueue.findIndex(t => t.id === taskId && t.status === 'queued');
    if (taskIndex >= 0) {
      this.taskQueue.splice(taskIndex, 1);
      return true;
    }
    return false;
  }

  /**
   * Health check for GPU orchestrator
   */
  async healthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    availableDevices: number;
    averageLoad: number;
    averageTemperature: number;
    issues: string[];
  }> {
    const availableDevices = this.availableDevices.filter(device => 
      device.temperature < 85 && device.load < 0.95
    ).length;
    
    const averageLoad = this.availableDevices.reduce((sum, device) => 
      sum + device.load, 0) / this.availableDevices.length;
    
    const averageTemperature = this.availableDevices.reduce((sum, device) => 
      sum + device.temperature, 0) / this.availableDevices.length;
    
    const issues: string[] = [];
    
    // Check for overheating
    this.availableDevices.forEach((device, index) => {
      if (device.temperature > 80) {
        issues.push(`GPU ${index}: High temperature (${device.temperature}°C)`);
      }
      if (device.load > 0.90) {
        issues.push(`GPU ${index}: High utilization (${Math.round(device.load * 100)}%)`);
      }
    });
    
    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    if (availableDevices === 0) {
      status = 'unhealthy';
    } else if (availableDevices < this.availableDevices.length || averageTemperature > 75) {
      status = 'degraded';
    }
    
    return {
      status,
      availableDevices,
      averageLoad: Math.round(averageLoad * 100) / 100,
      averageTemperature: Math.round(averageTemperature * 10) / 10,
      issues
    };
  }
}

// Export a singleton instance for use across the server
export const physicsAwareGPUOrchestrator = new PhysicsAwareGpuOrchestrator();

// Also export the class type if needed for dependency injection
export { PhysicsAwareGpuOrchestrator };