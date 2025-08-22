
// Real-time Document Update Notifications
// WebSocket-based notifications for document re-embedding and re-ranking progress

import { writable } from "svelte/store";
// Orphaned content: import {

// ============================================================================
// TYPES
// ============================================================================

export interface UpdateNotification {
  id: string;
  type: 'document_changed' | 'reembedding_started' | 'reembedding_complete' | 'reranking_complete' | 'error';
  documentId: string;
  timestamp: string;
  data: {
    title?: string;
    progress?: number;
    chunksProcessed?: number;
    totalChunks?: number;
    queriesReranked?: number;
    similarityImprovement?: number;
    error?: string;
    priority?: 'low' | 'medium' | 'high' | 'critical';
  };
}

export interface NotificationState {
  connected: boolean;
  notifications: UpdateNotification[];
  activeUpdates: Map<string, UpdateNotification>;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
}

// ============================================================================
// NOTIFICATION STORE
// ============================================================================

const initialState: NotificationState = {
  connected: false,
  notifications: [],
  activeUpdates: new Map(),
  connectionStatus: 'disconnected'
};

export const documentUpdateNotifications = writable<NotificationState>(initialState);

// ============================================================================
// WEBSOCKET MANAGER
// ============================================================================

class DocumentUpdateNotificationManager {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private pingInterval: NodeJS.Timeout | null = null;

  constructor() {
    if (browser) {
      this.connect();
    }
  }

  private connect() {
    if (!browser) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/ws/document-updates`;

    console.log('🔗 Connecting to document update notifications:', wsUrl);

    documentUpdateNotifications.update((state: any) => ({
      ...state,
      connectionStatus: 'connecting'
    }));

    try {
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('✅ Document update notifications connected');
        this.reconnectAttempts = 0;
        
        documentUpdateNotifications.update((state: any) => ({
          ...state,
          connected: true,
          connectionStatus: 'connected'
        }));

        // Send periodic pings to keep connection alive
        this.pingInterval = setInterval(() => {
          if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000);
      };

      this.ws.onmessage = (event) => {
        try {
          const notification: UpdateNotification = JSON.parse(event.data);
          this.handleNotification(notification);
        } catch (error) {
          console.warn('Failed to parse notification:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('🔌 Document update notifications disconnected');
        
        documentUpdateNotifications.update((state: any) => ({
          ...state,
          connected: false,
          connectionStatus: 'disconnected'
        }));

        if (this.pingInterval) {
          clearInterval(this.pingInterval);
          this.pingInterval = null;
        }

        // Attempt to reconnect
        this.attemptReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('❌ Document update notification error:', error);
        
        documentUpdateNotifications.update((state: any) => ({
          ...state,
          connected: false,
          connectionStatus: 'error'
        }));
      };

    } catch (error) {
      console.error('❌ Failed to create WebSocket connection:', error);
      
      documentUpdateNotifications.update((state: any) => ({
        ...state,
        connectionStatus: 'error'
      }));
    }
  }

  private handleNotification(notification: UpdateNotification) {
    console.log('📬 Document update notification:', notification);

    documentUpdateNotifications.update((state: any) => {
      const newNotifications = [...state.notifications, notification].slice(-50); // Keep last 50
      const newActiveUpdates = new Map(state.activeUpdates);

      // Update active updates tracking
      if (notification.type === 'reembedding_started') {
        newActiveUpdates.set(notification.documentId, notification);
      } else if (notification.type === 'reembedding_complete' || notification.type === 'error') {
        newActiveUpdates.delete(notification.documentId);
      }

      return {
        ...state,
        notifications: newNotifications,
        activeUpdates: newActiveUpdates
      };
    });

    // Show browser notification if permission granted
    this.showBrowserNotification(notification);
  }

  private showBrowserNotification(notification: UpdateNotification) {
    if (!browser || Notification.permission !== 'granted') return;

    let title = 'Document Update';
    let body = '';
    let icon = '/favicon.svg';

    switch (notification.type) {
      case 'document_changed':
        title = 'Document Changed';
        body = `"${notification.data.title || 'Document'}" has been modified and is being re-processed`;
        break;
        
      case 'reembedding_started':
        title = 'Re-embedding Started';
        body = `Processing "${notification.data.title || 'document'}" for improved search accuracy`;
        break;
        
      case 'reembedding_complete':
        title = 'Re-embedding Complete';
        body = `"${notification.data.title || 'Document'}" updated with ${notification.data.chunksProcessed} chunks`;
        break;
        
      case 'reranking_complete':
        title = 'Search Results Updated';
        body = `${notification.data.queriesReranked} search queries re-ranked for improved accuracy`;
        break;
        
      case 'error':
        title = 'Update Error';
        body = `Failed to update "${notification.data.title || 'document'}": ${notification.data.error}`;
        icon = '/error-icon.svg';
        break;
    }

    const browserNotification = new Notification(title, {
      body,
      icon,
      tag: `document-update-${notification.documentId}`, // Replace previous notifications for same document
      requireInteraction: notification.type === 'error'
    });

    // Auto-close success notifications after 5 seconds
    if (notification.type !== 'error') {
      setTimeout(() => browserNotification.close(), 5000);
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('❌ Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1); // Exponential backoff

    console.log(`🔄 Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  // ============================================================================
  // PUBLIC METHODS
  // ============================================================================

  public disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  public clearNotifications() {
    documentUpdateNotifications.update((state: any) => ({
      ...state,
      notifications: []
    }));
  }

  public async requestNotificationPermission() {
    if (!browser || !('Notification' in window)) {
      return false;
    }

    if (Notification.permission === 'default') {
      const permission = await Notification.requestPermission();
      return permission === 'granted';
    }

    return Notification.permission === 'granted';
  }

  public getConnectionStatus() {
    return {
      connected: this.ws?.readyState === WebSocket.OPEN,
      readyState: this.ws?.readyState,
      reconnectAttempts: this.reconnectAttempts
    };
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

export const notificationManager = browser ? new DocumentUpdateNotificationManager() : null;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

export function formatNotificationTime(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  if (diff < 60000) { // Less than 1 minute
    return 'Just now';
  } else if (diff < 3600000) { // Less than 1 hour
    const minutes = Math.floor(diff / 60000);
    return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
  } else if (diff < 86400000) { // Less than 1 day
    const hours = Math.floor(diff / 3600000);
    return `${hours} hour${hours > 1 ? 's' : ''} ago`;
  } else {
    return date.toLocaleDateString();
  }
}

export function getNotificationIcon(type: UpdateNotification['type']): string {
  switch (type) {
    case 'document_changed': return '📝';
    case 'reembedding_started': return '🔄';
    case 'reembedding_complete': return '✅';
    case 'reranking_complete': return '🏆';
    case 'error': return '❌';
    default: return '📬';
  }
}

export function getPriorityColor(priority?: string): string {
  switch (priority) {
    case 'critical': return 'text-red-600 bg-red-50';
    case 'high': return 'text-orange-600 bg-orange-50';
    case 'medium': return 'text-yellow-600 bg-yellow-50';
    case 'low': return 'text-blue-600 bg-blue-50';
    default: return 'text-gray-600 bg-gray-50';
  }
}