/**
 * Upload Telemetry Service - Structured Event Emission
 * Provides observability for upload operations with JSON event logging
 */

export interface TelemetryEvent {
  timestamp: number;
  sessionId: string;
  eventType: string;
  data: Record<string, any>;
}

export interface UploadStartEvent {
  batchId: string;
  fileCount: number;
  totalSizeBytes: number;
  maxConcurrency: number;
  enableGPU: boolean;
  caseId?: string;
}

export interface RetryScheduledEvent {
  batchId: string;
  fileId: string;
  fileName: string;
  attempt: number;
  maxRetries: number;
  delayMs: number;
  reason: string;
}

export interface RetryExecutedEvent {
  batchId: string;
  fileId: string;
  fileName: string;
  attempt: number;
  previousFailureReason: string;
}

export interface UploadCompleteEvent {
  batchId: string;
  fileId: string;
  fileName: string;
  fileSizeBytes: number;
  durationMs: number;
  attempts: number;
  gpuTaskIds: string[];
  status: 'completed' | 'failed' | 'canceled';
  errorMessage?: string;
}

export interface BatchSummaryEvent {
  batchId: string;
  totalFiles: number;
  completedFiles: number;
  failedFiles: number;
  canceledFiles: number;
  totalDurationMs: number;
  averageFileTime: number;
  totalGpuTasks: number;
  concurrencyUtilization: number;
}

export interface CancelAllEvent {
  batchId: string;
  activeUploads: number;
  queuedUploads: number;
  reason: 'user_initiated' | 'system_error';
}

export interface SessionRestoredEvent {
  restoredFiles: number;
  placeholderFiles: number;
  sessionAge: number;
}

class UploadTelemetryService {
  private sessionId: string;
  private events: TelemetryEvent[] = [];
  private maxEvents = 1000;
  private endpoint = '/api/v1/telemetry/upload';
  private batchSize = 10;
  private flushInterval = 30000; // 30 seconds
  private flushTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    this.sessionId = `session-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`;
    this.startAutoFlush();
  }

  private startAutoFlush() {
    this.flushTimer = setInterval(() => {
      this.flush();
    }, this.flushInterval);
  }

  private emit(eventType: string, data: Record<string, any>) {
    const event: TelemetryEvent = {
      timestamp: Date.now(),
      sessionId: this.sessionId,
      eventType,
      data
    };

    this.events.push(event);
    console.log(`📊 Telemetry [${eventType}]:`, data);

    // Maintain max events limit
    if (this.events.length > this.maxEvents) {
      this.events = this.events.slice(-this.maxEvents);
    }

    // Auto-flush on critical events
    if (['upload_start', 'batch_summary', 'canceled_all'].includes(eventType)) {
      setTimeout(() => this.flush(), 100);
    }
  }

  /**
   * Upload batch started
   */
  uploadStart(data: UploadStartEvent) {
    this.emit('upload_start', data);
  }

  /**
   * Retry scheduled for a file
   */
  retryScheduled(data: RetryScheduledEvent) {
    this.emit('retry_scheduled', data);
  }

  /**
   * Retry being executed
   */
  retryExecuted(data: RetryExecutedEvent) {
    this.emit('retry_executed', data);
  }

  /**
   * Single file upload completed
   */
  uploadComplete(data: UploadCompleteEvent) {
    this.emit('upload_complete', data);
  }

  /**
   * Batch summary when all uploads finish
   */
  batchSummary(data: BatchSummaryEvent) {
    this.emit('batch_summary', data);
  }

  /**
   * Cancel all uploads
   */
  canceledAll(data: CancelAllEvent) {
    this.emit('canceled_all', data);
  }

  /**
   * Session restored from storage
   */
  sessionRestored(data: SessionRestoredEvent) {
    this.emit('session_restored', data);
  }

  /**
   * Custom event emission
   */
  customEvent(eventType: string, data: Record<string, any>) {
    this.emit(eventType, data);
  }

  /**
   * Flush events to endpoint
   */
  async flush() {
    if (this.events.length === 0) return;

    const eventsToSend = this.events.splice(0, this.batchSize);
    
    try {
      const response = await fetch(this.endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessionId: this.sessionId,
          events: eventsToSend
        })
      });

      if (!response.ok) {
        console.warn('Telemetry flush failed:', response.status);
        // Re-add events back to queue on failure
        this.events.unshift(...eventsToSend);
      }
    } catch (error) {
      console.warn('Telemetry flush error:', error);
      // Re-add events back to queue on error
      this.events.unshift(...eventsToSend);
    }
  }

  /**
   * Get current session statistics
   */
  getStats() {
    const eventCounts = this.events.reduce((acc, event) => {
      acc[event.eventType] = (acc[event.eventType] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      sessionId: this.sessionId,
      totalEvents: this.events.length,
      eventCounts,
      queuedForFlush: this.events.length
    };
  }

  /**
   * Get recent events for debugging
   */
  getRecentEvents(count = 10): TelemetryEvent[] {
    return this.events.slice(-count);
  }

  /**
   * Clear all events
   */
  clear() {
    this.events = [];
  }

  /**
   * Cleanup resources
   */
  destroy() {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
    this.flush(); // Final flush
  }
}

// Export singleton instance
export const uploadTelemetry = new UploadTelemetryService();

// Cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    uploadTelemetry.destroy();
  });
}