
/**
 * WebSocket Store for Real-time AI System Communication
 * Handles real-time updates for system health, metrics, and recommendations
 */

import { writable } from 'svelte/store';
import type { Writable } from 'svelte/store';
import { browser } from '$app/environment';

interface WebSocketState {
  connected: boolean;
  connecting: boolean;
  error: string | null;
  lastMessage: unknown;
  reconnectAttempts: number;
  maxReconnectAttempts: number;
  reconnectDelay: number;
}

interface WebSocketMessage {
  type: string;
  data: unknown;
  timestamp: number;
  id?: string;
}

class WebSocketStore {
  private socket: WebSocket | null = null;
  private connectionState: Writable<WebSocketState>;
  private eventListeners: Map<string, ((data: unknown) => void)[]> = new Map();
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private url: string = '';

  constructor() {
    this.connectionState = writable<WebSocketState>({
      connected: false,
      connecting: false,
      error: null,
      lastMessage: null,
      reconnectAttempts: 0,
      maxReconnectAttempts: 5,
      reconnectDelay: 3000
    });
  }

  async connect(url: string) {
    if (!browser) return;

    this.url = url;
    
    this.connectionState.update(state => ({
      ...state,
      connecting: true,
      error: null
    }));

    try {
      this.socket = new WebSocket(url);
      this.setupEventHandlers();
      
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('WebSocket connection timeout'));
        }, 10000);

        this.socket!.addEventListener('open', () => {
          clearTimeout(timeout);
          resolve(this.socket);
        });

        this.socket!.addEventListener('error', (error) => {
          clearTimeout(timeout);
          reject(error);
        });
      });

    } catch (error) {
      this.connectionState.update(state => ({
        ...state,
        connecting: false,
        error: error.message
      }));
      throw error;
    }
  }

  private setupEventHandlers() {
    if (!this.socket) return;

    this.socket.addEventListener('open', () => {
      console.log('🌐 WebSocket connected');
      
      this.connectionState.update(state => ({
        ...state,
        connected: true,
        connecting: false,
        error: null,
        reconnectAttempts: 0
      }));

      this.startHeartbeat();
      this.emit('connected', { timestamp: Date.now() });
    });

    this.socket.addEventListener('close', (event) => {
      console.log('🔌 WebSocket disconnected:', event.code, event.reason);
      
      this.connectionState.update(state => ({
        ...state,
        connected: false,
        connecting: false
      }));

      this.stopHeartbeat();
      this.emit('disconnected', { code: event.code, reason: event.reason });

      // Attempt reconnection unless it's a normal closure
      if (event.code !== 1000 && event.code !== 1001) {
        this.scheduleReconnect();
      }
    });

    this.socket.addEventListener('error', (error) => {
      console.error('❌ WebSocket error:', error);
      
      this.connectionState.update(state => ({
        ...state,
        error: 'WebSocket connection error',
        connected: false,
        connecting: false
      }));

      this.emit('error', { error: error.toString() });
    });

    this.socket.addEventListener('message', (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        
        this.connectionState.update(state => ({
          ...state,
          lastMessage: message
        }));

        this.handleMessage(message);
        
      } catch (error) {
        console.warn('Failed to parse WebSocket message:', error);
      }
    });
  }

  private handleMessage(message: WebSocketMessage) {
    const { type, data } = message;
    
    switch (type) {
      case 'system-health':
        this.emit('system-health', data);
        break;
        
      case 'performance-metrics':
        this.emit('performance-metrics', data);
        break;
        
      case 'ai-recommendations':
        this.emit('ai-recommendations', data);
        break;
        
      case 'processor-complete':
        this.emit('processor-complete', data);
        break;
        
      case 'synthesis-started':
        this.emit('synthesis-started', data);
        break;
        
      case 'synthesis-complete':
        this.emit('synthesis-complete', data);
        break;
        
      case 'processing-error':
        this.emit('processing-error', data);
        break;
        
      case 'recommendations-generated':
        this.emit('recommendations-generated', data);
        break;
        
      case 'heartbeat':
        this.handleHeartbeat(data);
        break;
        
      case 'pong':
        // Response to ping
        break;
        
      default:
        console.log(`📨 Unknown message type: ${type}`, data);
        this.emit('message', { type, data });
    }
  }

  private startHeartbeat() {
    this.heartbeatTimer = setInterval(() => {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.send('ping', { timestamp: Date.now() });
      }
    }, 30000); // Every 30 seconds
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private handleHeartbeat(data: unknown) {
    // Respond to heartbeat from server
    this.send('heartbeat-response', { 
      timestamp: Date.now(),
      serverTimestamp: data.timestamp 
    });
  }

  private scheduleReconnect() {
    this.connectionState.update(state => {
      if (state.reconnectAttempts >= state.maxReconnectAttempts) {
        console.error('🔄 Max reconnection attempts reached');
        return {
          ...state,
          error: 'Maximum reconnection attempts exceeded'
        };
      }

      const delay = state.reconnectDelay * Math.pow(2, state.reconnectAttempts); // Exponential backoff
      
      console.log(`🔄 Reconnecting in ${delay}ms (attempt ${state.reconnectAttempts + 1}/${state.maxReconnectAttempts})`);
      
      this.reconnectTimer = setTimeout(() => {
        this.connect(this.url).catch(error => {
          console.error('Reconnection failed:', error);
        });
      }, delay);

      return {
        ...state,
        reconnectAttempts: state.reconnectAttempts + 1
      };
    });
  }

  // Public API methods
  send(type: string, data: unknown = {}) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn('Cannot send message: WebSocket not connected');
      return false;
    }

    const message: WebSocketMessage = {
      type,
      data,
      timestamp: Date.now(),
      id: this.generateMessageId()
    };

    try {
      this.socket.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Failed to send WebSocket message:', error);
      return false;
    }
  }

  // Event system
  on(eventType: string, callback: (data: unknown) => void) {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, []);
    }
    
    this.eventListeners.get(eventType)!.push(callback);
    
    // Return unsubscribe function
    return () => {
      const listeners = this.eventListeners.get(eventType);
      if (listeners) {
        const index = listeners.indexOf(callback);
        if (index > -1) {
          listeners.splice(index, 1);
        }
      }
    };
  }

  private emit(eventType: string, data: unknown) {
    const listeners = this.eventListeners.get(eventType);
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in WebSocket event listener for ${eventType}:`, error);
        }
      });
    }
  }

  // Specialized message senders for AI system
  requestSystemHealth() {
    return this.send('request-system-health');
  }

  requestPerformanceMetrics() {
    return this.send('request-performance-metrics');
  }

  requestRecommendations(userId?: string) {
    return this.send('request-recommendations', { userId });
  }

  startDocumentProcessing(documentId: string, options: unknown = {}) {
    return this.send('start-document-processing', { documentId, options });
  }

  startAnalysisSession(sessionId: string, documents: unknown[], options: unknown = {}) {
    return this.send('start-analysis-session', { sessionId, documents, options });
  }

  updateUserPreferences(preferences: unknown) {
    return this.send('update-user-preferences', preferences);
  }

  abortProcessing(sessionId?: string) {
    return this.send('abort-processing', { sessionId });
  }

  // Utility methods
  private generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  get socket() {
    return this.socket;
  }

  get isConnected() {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  get connectionState() {
    return this.connectionState;
  }

  // Store subscription
  subscribe(callback: (value: WebSocketState) => void) {
    return this.connectionState.subscribe(callback);
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    this.stopHeartbeat();

    if (this.socket) {
      this.socket.close(1000, 'Client disconnect');
      this.socket = null;
    }

    this.connectionState.update(state => ({
      ...state,
      connected: false,
      connecting: false,
      reconnectAttempts: 0
    }));

    this.eventListeners.clear();
    
    console.log('🔌 WebSocket disconnected by client');
  }
}

// Export factory function
export function createWebSocketStore() {
  return new WebSocketStore();
}

// Export types
export type { WebSocketState, WebSocketMessage };