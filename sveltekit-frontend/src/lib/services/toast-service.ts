/**
 * Toast Notification Service with NES.css Integration
 * Provides user feedback for upload progress, errors, and system status
 */

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info' | 'upload';
  title: string;
  message: string;
  duration?: number; // ms, 0 for persistent
  progress?: number; // 0-100 for upload toasts
  actions?: Array<{
    label: string;
    action: () => void;
    style?: 'primary' | 'secondary' | 'danger';
  }>;
  createdAt: Date;
  dismissible?: boolean;
}

export interface ToastOptions {
  type?: Toast['type'];
  duration?: number;
  progress?: number;
  actions?: Toast['actions'];
  dismissible?: boolean;
}

class ToastService {
  private toasts: Toast[] = [];
  private subscribers: Array<(toasts: Toast[]) => void> = [];
  private maxToasts = 5;
  private defaultDurations = {
    success: 4000,
    error: 8000,
    warning: 6000,
    info: 4000,
    upload: 0 // Persistent until manually dismissed
  };

  /**
   * Subscribe to toast updates
   */
  subscribe(callback: (toasts: Toast[]) => void): () => void {
    this.subscribers.push(callback);
    callback([...this.toasts]); // Initial state
    
    return () => {
      const index = this.subscribers.indexOf(callback);
      if (index > -1) {
        this.subscribers.splice(index, 1);
      }
    };
  }

  /**
   * Notify all subscribers of toast changes
   */
  private notify() {
    this.subscribers.forEach(callback => callback([...this.toasts]));
  }

  /**
   * Show a toast notification
   */
  show(title: string, message: string, options: ToastOptions = {}): string {
    const toast: Toast = {
      id: `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: options.type || 'info',
      title,
      message,
      duration: options.duration ?? this.defaultDurations[options.type || 'info'],
      progress: options.progress,
      actions: options.actions,
      createdAt: new Date(),
      dismissible: options.dismissible ?? true
    };

    this.toasts.unshift(toast);

    // Limit max toasts
    if (this.toasts.length > this.maxToasts) {
      this.toasts = this.toasts.slice(0, this.maxToasts);
    }

    this.notify();

    // Auto-dismiss if duration is set
    if (toast.duration && toast.duration > 0) {
      setTimeout(() => {
        this.dismiss(toast.id);
      }, toast.duration);
    }

    return toast.id;
  }

  /**
   * Update an existing toast (useful for upload progress)
   */
  update(id: string, updates: Partial<Pick<Toast, 'title' | 'message' | 'progress' | 'type'>>): void {
    const toast = this.toasts.find(t => t.id === id);
    if (toast) {
      Object.assign(toast, updates);
      this.notify();
    }
  }

  /**
   * Dismiss a toast by ID
   */
  dismiss(id: string): void {
    const index = this.toasts.findIndex(t => t.id === id);
    if (index > -1) {
      this.toasts.splice(index, 1);
      this.notify();
    }
  }

  /**
   * Dismiss all toasts
   */
  dismissAll(): void {
    this.toasts = [];
    this.notify();
  }

  /**
   * Convenience methods for common toast types
   */
  success(title: string, message: string, options: Omit<ToastOptions, 'type'> = {}): string {
    return this.show(title, message, { ...options, type: 'success' });
  }

  error(title: string, message: string, options: Omit<ToastOptions, 'type'> = {}): string {
    return this.show(title, message, { ...options, type: 'error' });
  }

  warning(title: string, message: string, options: Omit<ToastOptions, 'type'> = {}): string {
    return this.show(title, message, { ...options, type: 'warning' });
  }

  info(title: string, message: string, options: Omit<ToastOptions, 'type'> = {}): string {
    return this.show(title, message, { ...options, type: 'info' });
  }

  /**
   * Specialized upload toast with progress tracking
   */
  upload(title: string, message: string, options: Omit<ToastOptions, 'type'> = {}): string {
    return this.show(title, message, { 
      ...options, 
      type: 'upload',
      dismissible: false,
      actions: [
        {
          label: 'Cancel',
          action: () => {
            // This would be overridden by the caller with actual cancel logic
            console.log('Cancel upload requested');
          },
          style: 'danger'
        }
      ]
    });
  }

  /**
   * Upload progress update helper
   */
  updateUploadProgress(id: string, progress: number, message?: string): void {
    this.update(id, { 
      progress, 
      message: message || `${Math.round(progress)}% complete`
    });
  }

  /**
   * Complete an upload toast
   */
  completeUpload(id: string, successMessage?: string): void {
    this.update(id, {
      type: 'success',
      message: successMessage || 'Upload completed successfully',
      progress: 100
    });

    // Auto-dismiss after 3 seconds
    setTimeout(() => {
      this.dismiss(id);
    }, 3000);
  }

  /**
   * Fail an upload toast
   */
  failUpload(id: string, errorMessage: string, retryAction?: () => void): void {
    const actions: Toast['actions'] = [
      {
        label: 'Dismiss',
        action: () => this.dismiss(id),
        style: 'secondary'
      }
    ];

    if (retryAction) {
      actions.unshift({
        label: 'Retry',
        action: retryAction,
        style: 'primary'
      });
    }

    this.update(id, {
      type: 'error',
      message: errorMessage,
      progress: undefined
    });

    // Update actions
    const toast = this.toasts.find(t => t.id === id);
    if (toast) {
      toast.actions = actions;
      toast.dismissible = true;
      this.notify();
    }
  }

  /**
   * System status notifications
   */
  systemStatus(service: string, status: 'connected' | 'disconnected' | 'degraded', details?: string): string {
    const statusMap = {
      connected: { type: 'success' as const, icon: '✅' },
      disconnected: { type: 'error' as const, icon: '❌' },
      degraded: { type: 'warning' as const, icon: '⚠️' }
    };

    const config = statusMap[status];
    return this.show(
      `${config.icon} ${service}`,
      details || `Service is ${status}`,
      { 
        type: config.type,
        duration: status === 'connected' ? 3000 : 0 // Keep errors visible
      }
    );
  }

  /**
   * GPU processing notifications
   */
  gpuTask(taskType: string, status: 'queued' | 'processing' | 'completed' | 'failed', details?: string): string {
    const statusMap = {
      queued: { type: 'info' as const, icon: '⏳', message: 'Task queued for GPU processing' },
      processing: { type: 'upload' as const, icon: '🚀', message: 'Processing on GPU...' },
      completed: { type: 'success' as const, icon: '✅', message: 'GPU processing completed' },
      failed: { type: 'error' as const, icon: '❌', message: 'GPU processing failed' }
    };

    const config = statusMap[status];
    return this.show(
      `${config.icon} GPU ${taskType}`,
      details || config.message,
      { 
        type: config.type,
        duration: status === 'completed' ? 3000 : (status === 'processing' ? 0 : 5000)
      }
    );
  }

  /**
   * Vector similarity notifications
   */
  vectorSimilarity(count: number, similarity: number, processingTime: number): string {
    return this.success(
      `🎯 Found ${count} similar items`,
      `Average similarity: ${(similarity * 100).toFixed(1)}% (processed in ${processingTime}ms)`,
      { duration: 4000 }
    );
  }

  /**
   * RabbitMQ message notifications
   */
  rabbitMQEvent(eventType: string, message: string, isError = false): string {
    return this.show(
      `🐰 ${eventType}`,
      message,
      { 
        type: isError ? 'error' : 'info',
        duration: isError ? 6000 : 3000
      }
    );
  }

  /**
   * Get current toasts (for external access)
   */
  getToasts(): Toast[] {
    return [...this.toasts];
  }

  /**
   * Clear expired toasts (maintenance)
   */
  clearExpired(): void {
    const now = Date.now();
    const before = this.toasts.length;
    
    this.toasts = this.toasts.filter(toast => {
      if (!toast.duration || toast.duration === 0) return true;
      return (now - toast.createdAt.getTime()) < toast.duration;
    });

    if (this.toasts.length !== before) {
      this.notify();
    }
  }
}

// Export singleton instance
export const toastService = new ToastService();

// Auto-cleanup expired toasts every 10 seconds
setInterval(() => {
  toastService.clearExpired();
}, 10000);