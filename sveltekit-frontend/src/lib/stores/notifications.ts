import { writable } from 'svelte/store';

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'system';
  title?: string;
  message: string;
  duration?: number;
  persistent?: boolean;
  closable?: boolean;
  icon?: string;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'center';
  showProgress?: boolean;
}

export interface NotificationOptions {
  type?: Notification['type'];
  title?: string;
  duration?: number;
  persistent?: boolean;
  closable?: boolean;
  icon?: string;
  position?: Notification['position'];
  showProgress?: boolean;
}

function createNotificationStore() {
  const { subscribe, update } = writable<Notification[]>([]);

  function generateId(): string {
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
  }

  function add(message: string, options: NotificationOptions = {}): string {
    const id = generateId();
    const notification: Notification = {
      id,
      message,
      type: options.type || 'info',
      title: options.title,
      duration: options.duration ?? 5000,
      persistent: options.persistent ?? false,
      closable: options.closable ?? true,
      icon: options.icon,
      position: options.position || 'top-right',
      showProgress: options.showProgress ?? true
    };

    update(notifications => [...notifications, notification]);
    return id;
  }

  function remove(id: string) {
    update(notifications => notifications.filter(n => n.id !== id));
  }

  function clear() {
    update(() => []);
  }

  // Convenience methods for different notification types
  function info(message: string, options: Omit<NotificationOptions, 'type'> = {}): string {
    return add(message, { ...options, type: 'info' });
  }

  function success(message: string, options: Omit<NotificationOptions, 'type'> = {}): string {
    return add(message, { ...options, type: 'success' });
  }

  function warning(message: string, options: Omit<NotificationOptions, 'type'> = {}): string {
    return add(message, { ...options, type: 'warning' });
  }

  function error(message: string, options: Omit<NotificationOptions, 'type'> = {}): string {
    return add(message, { ...options, type: 'error' });
  }

  function system(message: string, options: Omit<NotificationOptions, 'type'> = {}): string {
    return add(message, { 
      ...options, 
      type: 'system',
      persistent: options.persistent ?? true,
      position: options.position || 'center'
    });
  }

  // Legal AI specific notifications
  function caseUpdate(message: string, caseId?: string): string {
    const title = caseId ? `Case ${caseId}` : 'Case Update';
    return add(message, {
      type: 'info',
      title,
      icon: '📋',
      duration: 7000
    });
  }

  function evidenceProcessed(message: string, evidenceId?: string): string {
    const title = evidenceId ? `Evidence ${evidenceId}` : 'Evidence Processed';
    return add(message, {
      type: 'success',
      title,
      icon: '🔍',
      duration: 5000
    });
  }

  function aiAnalysisComplete(message: string, confidence?: number): string {
    const confidenceText = confidence ? ` (${Math.round(confidence)}% confidence)` : '';
    return add(message + confidenceText, {
      type: 'success',
      title: 'AI Analysis Complete',
      icon: '🤖',
      duration: 8000
    });
  }

  function securityAlert(message: string): string {
    return add(message, {
      type: 'error',
      title: 'Security Alert',
      icon: '🚨',
      persistent: true,
      position: 'center'
    });
  }

  function systemStatus(message: string, isOnline: boolean = true): string {
    return add(message, {
      type: 'system',
      title: 'System Status',
      icon: isOnline ? '🟢' : '🔴',
      persistent: !isOnline,
      position: 'top-left'
    });
  }

  return {
    subscribe,
    add,
    remove,
    clear,
    info,
    success,
    warning,
    error,
    system,
    // Legal AI specific methods
    caseUpdate,
    evidenceProcessed,
    aiAnalysisComplete,
    securityAlert,
    systemStatus
  };
}

export const notificationStore = createNotificationStore();
;
// Export convenience functions for use throughout the app
export const notify = notificationStore;