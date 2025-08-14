/**
 * Real-Time Async Communication Layer for Legal AI Platform
 * Provides WebSocket, SSE, and WebRTC connections with intelligent fallback
 */

import { writable, type Writable } from 'svelte/store';

export interface ConnectionStatus {
  websocket: 'connected' | 'connecting' | 'disconnected' | 'error';
  sse: 'connected' | 'connecting' | 'disconnected' | 'error';
  webrtc: 'connected' | 'connecting' | 'disconnected' | 'error';
  primaryChannel: 'websocket' | 'sse' | 'webrtc' | null;
}

export interface RealtimeMessage {
  id: string;
  type:
    | 'ai_response'
    | 'document_analysis'
    | 'system_notification'
    | 'user_activity'
    | 'rag_result'
    | 'gpu_compute'
    | 'semantic_update';
  channel: 'websocket' | 'sse' | 'webrtc';
  data: any;
  timestamp: Date;
  userId?: string;
  sessionId?: string;
  priority: 'low' | 'normal' | 'high' | 'critical';
}

export interface StreamingResponse {
  id: string;
  type: 'ai_chat' | 'document_processing' | 'rag_query' | 'semantic_analysis';
  status: 'started' | 'streaming' | 'completed' | 'error';
  chunks: string[];
  metadata?: {
    totalTokens?: number;
    processingTime?: number;
    confidence?: number;
    model?: string;
  };
}

export interface WebRTCDataChannel {
  channel: RTCDataChannel;
  connection: RTCPeerConnection;
  status: 'connecting' | 'connected' | 'closed';
}

class RealtimeCommunicationLayer {
  private connections: {
    websocket: WebSocket | null;
    sse: EventSource | null;
    webrtc: WebRTCDataChannel | null;
  } = {
    websocket: null,
    sse: null,
    webrtc: null,
  };

  private reconnectAttempts = {
    websocket: 0,
    sse: 0,
    webrtc: 0,
  };

  private readonly maxReconnectAttempts = 5;
  private readonly reconnectDelay = 1000;

  private messageQueue: RealtimeMessage[] = [];
  private streamingResponses: Map<string, StreamingResponse> = new Map();

  // Service worker for background communication
  private serviceWorkerRegistration: ServiceWorkerRegistration | null = null;

  // Message handlers
  private messageHandlers: Map<string, (message: RealtimeMessage) => void> = new Map();
  private streamHandlers: Map<string, (response: StreamingResponse) => void> = new Map();

  // Connection URLs
  private readonly urls = {
    websocket: 'ws://localhost:8094/ws',
    sse: 'http://localhost:8094/api/events',
    webrtc: 'http://localhost:8094/api/webrtc/signaling',
  };

  /**
   * Initialize all communication channels
   */
  async initialize(userId: string, sessionId: string): Promise<void> {
    console.log('Initializing real-time communication layer...');

    // Initialize service worker for background communication
    await this.initializeServiceWorker();

    // Initialize connections in priority order
    await Promise.allSettled([
      this.initializeWebSocket(userId, sessionId),
      this.initializeSSE(userId, sessionId),
      this.initializeWebRTC(userId, sessionId),
    ]);

    // Set up heartbeat mechanism
    this.startHeartbeat();

    // Set up message queue processing
    this.startMessageQueueProcessor();

    console.log('Real-time communication layer initialized');
  }

  /**
   * Initialize Service Worker for background communication
   */
  private async initializeServiceWorker(): Promise<void> {
    if ('serviceWorker' in navigator) {
      try {
        this.serviceWorkerRegistration =
          await navigator.serviceWorker.register('/service-worker.js');
        console.log('Service Worker registered for background communication');

        // Listen for messages from service worker
        navigator.serviceWorker.addEventListener('message', (event) => {
          this.handleServiceWorkerMessage(event.data);
        });
      } catch (error) {
        console.warn('Service Worker registration failed:', error);
      }
    }
  }

  /**
   * Initialize WebSocket connection
   */
  private async initializeWebSocket(userId: string, sessionId: string): Promise<void> {
    try {
      connectionStatus.update((status) => ({ ...status, websocket: 'connecting' }));

      const wsUrl = `${this.urls.websocket}/${userId}?session=${sessionId}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('WebSocket connected');
        this.connections.websocket = ws;
        this.reconnectAttempts.websocket = 0;
        connectionStatus.update((status) => ({
          ...status,
          websocket: 'connected',
          primaryChannel: status.primaryChannel || 'websocket',
        }));

        // Send queued messages
        this.processQueuedMessages('websocket');
      };

      ws.onmessage = (event) => {
        try {
          const message: RealtimeMessage = JSON.parse(event.data);
          message.channel = 'websocket';
          message.timestamp = new Date(message.timestamp);
          this.handleMessage(message);
        } catch (error) {
          console.error('WebSocket message parsing failed:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        this.connections.websocket = null;
        connectionStatus.update((status) => ({
          ...status,
          websocket: 'disconnected',
          primaryChannel: status.primaryChannel === 'websocket' ? null : status.primaryChannel,
        }));

        // Attempt reconnection
        this.scheduleReconnect('websocket', userId, sessionId);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        connectionStatus.update((status) => ({ ...status, websocket: 'error' }));
      };
    } catch (error) {
      console.error('WebSocket initialization failed:', error);
      connectionStatus.update((status) => ({ ...status, websocket: 'error' }));
    }
  }

  /**
   * Initialize Server-Sent Events connection
   */
  private async initializeSSE(userId: string, sessionId: string): Promise<void> {
    try {
      connectionStatus.update((status) => ({ ...status, sse: 'connecting' }));

      const sseUrl = `${this.urls.sse}?userId=${userId}&sessionId=${sessionId}`;
      const eventSource = new EventSource(sseUrl);

      eventSource.onopen = () => {
        console.log('SSE connected');
        this.connections.sse = eventSource;
        this.reconnectAttempts.sse = 0;
        connectionStatus.update((status) => ({
          ...status,
          sse: 'connected',
          primaryChannel: status.primaryChannel || 'sse',
        }));
      };

      eventSource.onmessage = (event) => {
        try {
          const message: RealtimeMessage = JSON.parse(event.data);
          message.channel = 'sse';
          message.timestamp = new Date(message.timestamp);
          this.handleMessage(message);
        } catch (error) {
          console.error('SSE message parsing failed:', error);
        }
      };

      // Handle streaming AI responses
      eventSource.addEventListener('ai_stream', (event) => {
        this.handleStreamingResponse(JSON.parse(event.data));
      });

      // Handle document processing updates
      eventSource.addEventListener('document_progress', (event) => {
        this.handleStreamingResponse(JSON.parse(event.data));
      });

      eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        this.connections.sse = null;
        connectionStatus.update((status) => ({
          ...status,
          sse: 'error',
          primaryChannel: status.primaryChannel === 'sse' ? null : status.primaryChannel,
        }));

        // Attempt reconnection
        this.scheduleReconnect('sse', userId, sessionId);
      };
    } catch (error) {
      console.error('SSE initialization failed:', error);
      connectionStatus.update((status) => ({ ...status, sse: 'error' }));
    }
  }

  /**
   * Initialize WebRTC data channel for low-latency communication
   */
  private async initializeWebRTC(userId: string, sessionId: string): Promise<void> {
    try {
      connectionStatus.update((status) => ({ ...status, webrtc: 'connecting' }));

      // Create peer connection
      const config: RTCConfiguration = {
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      };

      const peerConnection = new RTCPeerConnection(config);

      // Create data channel for legal AI communication
      const dataChannel = peerConnection.createDataChannel('legal-ai-channel', {
        ordered: false, // Allow out-of-order delivery for speed
        maxRetransmits: 3,
      });

      dataChannel.onopen = () => {
        console.log('WebRTC data channel opened');
        this.connections.webrtc = {
          channel: dataChannel,
          connection: peerConnection,
          status: 'connected',
        };
        connectionStatus.update((status) => ({
          ...status,
          webrtc: 'connected',
          primaryChannel: status.primaryChannel || 'webrtc',
        }));
      };

      dataChannel.onmessage = (event) => {
        try {
          const message: RealtimeMessage = JSON.parse(event.data);
          message.channel = 'webrtc';
          message.timestamp = new Date(message.timestamp);
          this.handleMessage(message);
        } catch (error) {
          console.error('WebRTC message parsing failed:', error);
        }
      };

      dataChannel.onclose = () => {
        console.log('WebRTC data channel closed');
        this.connections.webrtc = null;
        connectionStatus.update((status) => ({
          ...status,
          webrtc: 'disconnected',
          primaryChannel: status.primaryChannel === 'webrtc' ? null : status.primaryChannel,
        }));
      };

      dataChannel.onerror = (error) => {
        console.error('WebRTC data channel error:', error);
        connectionStatus.update((status) => ({ ...status, webrtc: 'error' }));
      };

      // Handle ICE candidates and signaling
      await this.handleWebRTCSignaling(peerConnection, userId, sessionId);
    } catch (error) {
      console.error('WebRTC initialization failed:', error);
      connectionStatus.update((status) => ({ ...status, webrtc: 'error' }));
    }
  }

  /**
   * Handle WebRTC signaling through HTTP API
   */
  private async handleWebRTCSignaling(
    peerConnection: RTCPeerConnection,
    userId: string,
    sessionId: string
  ): Promise<void> {
    // Create offer
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);

    // Send offer to signaling server
    const signalResponse = await fetch(this.urls.webrtc, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type: 'offer',
        offer: offer,
        userId,
        sessionId,
      }),
    });

    if (signalResponse.ok) {
      const { answer } = await signalResponse.json();
      await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
    }

    // Handle ICE candidates
    peerConnection.onicecandidate = (event) => {
      if (event.candidate) {
        fetch(this.urls.webrtc, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: 'ice-candidate',
            candidate: event.candidate,
            userId,
            sessionId,
          }),
        });
      }
    };
  }

  /**
   * Send message through the best available channel
   */
  async sendMessage(
    type: RealtimeMessage['type'],
    data: any,
    priority: RealtimeMessage['priority'] = 'normal'
  ): Promise<void> {
    const message: RealtimeMessage = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type,
      channel: 'websocket', // Will be updated based on actual channel used
      data,
      timestamp: new Date(),
      priority,
    };

    // Choose best available channel based on priority and connection status
    const channel = this.selectBestChannel(priority);

    if (!channel) {
      // Queue message for later delivery
      this.messageQueue.push(message);
      console.warn('No active connections, message queued:', message);
      return;
    }

    try {
      message.channel = channel;
      await this.sendThroughChannel(message, channel);
    } catch (error) {
      console.error(`Failed to send message through ${channel}:`, error);
      // Try alternative channels or queue
      this.messageQueue.push(message);
    }
  }

  /**
   * Send streaming request (for AI responses, document processing, etc.)
   */
  async sendStreamingRequest(type: StreamingResponse['type'], data: any): Promise<string> {
    const requestId = `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const streamingResponse: StreamingResponse = {
      id: requestId,
      type,
      status: 'started',
      chunks: [],
    };

    this.streamingResponses.set(requestId, streamingResponse);
    streamingResponses.update((responses) => new Map(responses.set(requestId, streamingResponse)));

    // Send initial request
    await this.sendMessage(
      'ai_response',
      {
        requestId,
        type,
        data,
        streaming: true,
      },
      'high'
    );

    return requestId;
  }

  /**
   * Select best communication channel based on priority and availability
   */
  private selectBestChannel(
    priority: RealtimeMessage['priority']
  ): 'websocket' | 'sse' | 'webrtc' | null {
    const status = this.getCurrentConnectionStatus();

    // For critical messages, prefer WebRTC for lowest latency
    if (priority === 'critical' && status.webrtc === 'connected') {
      return 'webrtc';
    }

    // For high priority, prefer WebSocket
    if (priority === 'high' && status.websocket === 'connected') {
      return 'websocket';
    }

    // Fallback priority: WebSocket > WebRTC > SSE
    if (status.websocket === 'connected') return 'websocket';
    if (status.webrtc === 'connected') return 'webrtc';
    if (status.sse === 'connected') return 'sse';

    return null;
  }

  /**
   * Send message through specific channel
   */
  private async sendThroughChannel(
    message: RealtimeMessage,
    channel: 'websocket' | 'sse' | 'webrtc'
  ): Promise<void> {
    const messageData = JSON.stringify(message);

    switch (channel) {
      case 'websocket':
        if (this.connections.websocket?.readyState === WebSocket.OPEN) {
          this.connections.websocket.send(messageData);
        } else {
          throw new Error('WebSocket not connected');
        }
        break;

      case 'webrtc':
        if (this.connections.webrtc?.channel.readyState === 'open') {
          this.connections.webrtc.channel.send(messageData);
        } else {
          throw new Error('WebRTC data channel not open');
        }
        break;

      case 'sse':
        // SSE is read-only, fallback to HTTP POST
        await fetch('http://localhost:8094/api/messages', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: messageData,
        });
        break;

      default:
        throw new Error(`Unknown channel: ${channel}`);
    }
  }

  /**
   * Handle incoming messages
   */
  private handleMessage(message: RealtimeMessage): void {
    // Update message store
    messages.update((msgs) => [...msgs.slice(-99), message]); // Keep last 100 messages

    // Handle streaming responses
    if (message.type === 'ai_response' && message.data.streaming) {
      this.handleStreamingResponse(message.data);
    }

    // Call registered handlers
    const handler = this.messageHandlers.get(message.type);
    if (handler) {
      handler(message);
    }

    // Emit to custom event handlers
    document.dispatchEvent(new CustomEvent('realtime-message', { detail: message }));
  }

  /**
   * Handle streaming responses
   */
  private handleStreamingResponse(data: any): void {
    const { requestId, chunk, status, metadata } = data;

    if (!this.streamingResponses.has(requestId)) {
      return;
    }

    const response = this.streamingResponses.get(requestId)!;

    if (chunk) {
      response.chunks.push(chunk);
    }

    if (status) {
      response.status = status;
    }

    if (metadata) {
      response.metadata = { ...response.metadata, ...metadata };
    }

    // Update store
    streamingResponses.update((responses) => new Map(responses.set(requestId, response)));

    // Call stream handlers
    const handler = this.streamHandlers.get(requestId);
    if (handler) {
      handler(response);
    }

    // Clean up completed streams
    if (status === 'completed' || status === 'error') {
      setTimeout(() => {
        this.streamingResponses.delete(requestId);
        streamingResponses.update((responses) => {
          responses.delete(requestId);
          return new Map(responses);
        });
      }, 30000); // Keep for 30 seconds after completion
    }
  }

  /**
   * Handle service worker messages
   */
  private handleServiceWorkerMessage(data: any): void {
    if (data.type === 'background-sync') {
      // Handle background synchronization
      this.processQueuedMessages();
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(
    channel: 'websocket' | 'sse' | 'webrtc',
    userId: string,
    sessionId: string
  ): void {
    const attempts = this.reconnectAttempts[channel];

    if (attempts < this.maxReconnectAttempts) {
      const delay = this.reconnectDelay * Math.pow(2, attempts); // Exponential backoff

      setTimeout(async () => {
        console.log(
          `Attempting ${channel} reconnection (${attempts + 1}/${this.maxReconnectAttempts})`
        );
        this.reconnectAttempts[channel]++;

        switch (channel) {
          case 'websocket':
            await this.initializeWebSocket(userId, sessionId);
            break;
          case 'sse':
            await this.initializeSSE(userId, sessionId);
            break;
          case 'webrtc':
            await this.initializeWebRTC(userId, sessionId);
            break;
        }
      }, delay);
    }
  }

  /**
   * Process queued messages
   */
  private processQueuedMessages(preferredChannel?: string): void {
    const queue = [...this.messageQueue];
    this.messageQueue = [];

    for (const message of queue) {
      const candidate = preferredChannel || this.selectBestChannel(message.priority);
      const channel: 'websocket' | 'sse' | 'webrtc' | null =
        candidate === 'websocket' || candidate === 'sse' || candidate === 'webrtc'
          ? candidate
          : null;
      if (channel) {
        this.sendThroughChannel(message, channel).catch((error) => {
          console.error('Failed to send queued message:', error);
          this.messageQueue.push(message); // Re-queue
        });
      } else {
        this.messageQueue.push(message); // Re-queue
      }
    }
  }

  /**
   * Start message queue processor
   */
  private startMessageQueueProcessor(): void {
    setInterval(() => {
      if (this.messageQueue.length > 0) {
        this.processQueuedMessages();
      }
    }, 5000); // Process every 5 seconds
  }

  /**
   * Start heartbeat mechanism
   */
  private startHeartbeat(): void {
    setInterval(() => {
      this.sendMessage('system_notification', { type: 'heartbeat' }, 'low').catch(() => {
        // Heartbeat failed, connections may be down
      });
    }, 30000); // Every 30 seconds
  }

  /**
   * Register message handler
   */
  onMessage(type: RealtimeMessage['type'], handler: (message: RealtimeMessage) => void): void {
    this.messageHandlers.set(type, handler);
  }

  /**
   * Register stream handler
   */
  onStream(requestId: string, handler: (response: StreamingResponse) => void): void {
    this.streamHandlers.set(requestId, handler);
  }

  /**
   * Get current connection status
   */
  private getCurrentConnectionStatus(): ConnectionStatus {
    return {
      websocket:
        this.connections.websocket?.readyState === WebSocket.OPEN ? 'connected' : 'disconnected',
      sse: this.connections.sse?.readyState === EventSource.OPEN ? 'connected' : 'disconnected',
      // Map internal 'closed' status to union-compatible 'disconnected'
      webrtc:
        this.connections.webrtc?.status === 'connected'
          ? 'connected'
          : this.connections.webrtc?.status === 'connecting'
            ? 'connecting'
            : 'disconnected',
      primaryChannel: null, // Will be set by store
    };
  }

  /**
   * Disconnect all channels
   */
  disconnect(): void {
    if (this.connections.websocket) {
      this.connections.websocket.close();
    }

    if (this.connections.sse) {
      this.connections.sse.close();
    }

    if (this.connections.webrtc) {
      this.connections.webrtc.channel.close();
      this.connections.webrtc.connection.close();
    }

    this.messageHandlers.clear();
    this.streamHandlers.clear();
    this.messageQueue = [];
    this.streamingResponses.clear();
  }
}

// Svelte stores for reactive state management
export const connectionStatus: Writable<ConnectionStatus> = writable({
  websocket: 'disconnected',
  sse: 'disconnected',
  webrtc: 'disconnected',
  primaryChannel: null,
});

export const messages: Writable<RealtimeMessage[]> = writable([]);
export const streamingResponses: Writable<Map<string, StreamingResponse>> = writable(new Map());

// Export singleton instance
export const realtimeComm = new RealtimeCommunicationLayer();
