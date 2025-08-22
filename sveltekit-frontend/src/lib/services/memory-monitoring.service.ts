
import { browser } from '$app/environment';

export class MemoryMonitoringService {
  private intervalId: NodeJS.Timeout | null = null;
  private callbacks: Array<(data: any) => void> = [];

  start(intervalMs: number = 10000) {
    if (!browser || this.intervalId) return;

    this.intervalId = setInterval(async () => {
      try {
        const response = await fetch('/api/memory/neural?action=status');
        const result = await response.json();
        
        if (result.success) {
          this.notifyCallbacks(result.data);
        }
      } catch (error) {
        console.error('Memory monitoring error:', error);
      }
    }, intervalMs);

    console.log('✅ Memory monitoring started');
  }

  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
      console.log('⏹️ Memory monitoring stopped');
    }
  }

  onUpdate(callback: (data: any) => void) {
    this.callbacks.push(callback);
  }

  private notifyCallbacks(data: any) {
    this.callbacks.forEach((callback: any) => {
      try {
        callback(data);
      } catch (error) {
        console.error('Memory monitoring callback error:', error);
      }
    });
  }

  async triggerOptimization() {
    try {
      const response = await fetch('/api/memory/neural?action=optimize');
      const result = await response.json();
      return result.success;
    } catch (error) {
      console.error('Failed to trigger optimization:', error);
      return false;
    }
  }

  async getPrediction(horizonMinutes: number = 30) {
    try {
      const response = await fetch(`/api/memory/neural?action=predict&horizon=${horizonMinutes}`);
      const result = await response.json();
      return result.success ? result.data : null;
    } catch (error) {
      console.error('Failed to get memory prediction:', error);
      return null;
    }
  }
}

export const memoryMonitoring = new MemoryMonitoringService();