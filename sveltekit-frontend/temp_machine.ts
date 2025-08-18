// src/lib/machines/uploadMachine.ts
import { createMachine, assign, fromCallback } from 'xstate';
import type { ProgressMsg } from '$lib/types/progress';

interface UploadContext {
  files: Record<string, {
    uploadProgress?: number;
    step?: string;
    stepProgress?: number;
    fragment?: any;
    result?: any;
    status?: 'uploading' | 'processing' | 'done' | 'error';
    error?: string;
  }>;
  sessionId?: string;
  wsConnected: boolean;
  retryCount: number;
  lastError?: string;
}

type UploadEvent =
  | { type: 'START_PROCESS'; sessionId: string; fileId: string }
  | { type: 'UPLOAD_PROGRESS'; fileId: string; progress: number }
  | { type: 'PROCESSING_STEP'; fileId: string; step: string; progress?: number; fragment?: any }
  | { type: 'PROCESSING_COMPLETE'; fileId: string; result?: any }
  | { type: 'ERROR'; fileId?: string; error: any }
  | { type: 'WS_CLOSED' }
  | { type: 'WS_OPENED' }
  | { type: 'RETRY' }
  | { type: 'CANCEL'; fileId?: string }
  | { type: 'RESET' };

// REMOVED - Old actor factory moved to machine actors config
    let ws: WebSocket | null = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    let reconnectTimer: NodeJS.Timeout | null = null;
    
    function connect() {
      try {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${location.host}/api/evidence/stream/${sessionId}`;
        
        console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);
        
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
          console.log('✅ WebSocket connected');
          reconnectAttempts = 0;
          callback({ type: 'WS_OPENED' });
        };
        
        ws.onmessage = (event) => {
          try {
            const msg: ProgressMsg = JSON.parse(event.data);
            
            console.log('📨 WebSocket message:', msg);
            
            // Transform WebSocket messages to machine events
            switch (msg.type) {
              case 'upload-progress':
                callback({
                  type: 'UPLOAD_PROGRESS',
                  fileId: msg.fileId,
                  progress: msg.progress
                });
                break;
                
              case 'processing-step':
                callback({
                  type: 'PROCESSING_STEP',
                  fileId: msg.fileId,
                  step: msg.step,
                  progress: msg.stepProgress,
                  fragment: msg.fragment
                });
                break;
                
              case 'processing-complete':
                callback({
                  type: 'PROCESSING_COMPLETE',
                  fileId: msg.fileId,
                  result: msg.finalResult
                });
                break;
                
              case 'error':
                callback({
                  type: 'ERROR',
                  fileId: msg.fileId,
                  error: msg.error
                });
                break;
            }
            
          } catch (parseError) {
            console.error('❌ Failed to parse WebSocket message:', parseError);
            callback({
              type: 'ERROR',
              error: { message: 'Failed to parse WebSocket message' }
            });
          }
        };
        
        ws.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          callback({
            type: 'ERROR',
            error: { message: 'WebSocket connection error' }
          });
        };
        
        ws.onclose = (event) => {
          console.log(`🔌 WebSocket closed: ${event.code} - ${event.reason}`);
          
          callback({ type: 'WS_CLOSED' });
          
          // Attempt to reconnect if not manually closed
          if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
            
            console.log(`🔄 Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
            
            reconnectTimer = setTimeout(() => {
              connect();
            }, delay);
          }
        };
        
      } catch (error) {
        console.error('❌ Failed to create WebSocket connection:', error);
        callback({
          type: 'ERROR',
          error: { message: 'Failed to create WebSocket connection' }
        });
      }
    }
    
    // Initial connection
    connect();
    
    // Cleanup function
    return () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
      
      if (ws) {
        ws.close(1000, 'Component unmounted');
        ws = null;
      }
    };
  });
}

// Server-Sent Events fallback actor
function createSseActor(sessionId: string) {
  return fromCallback<UploadEvent>((callback) => {
    let eventSource: EventSource | null = null;
    
    try {
      const sseUrl = `/api/evidence/stream/${sessionId}`;
      eventSource = new EventSource(sseUrl);
      
      eventSource.onopen = () => {
        console.log('✅ SSE connected');
        callback({ type: 'WS_OPENED' });
      };
      
      eventSource.onmessage = (event) => {
        try {
          const msg: ProgressMsg = JSON.parse(event.data);
          
          // Transform SSE messages to machine events (same as WebSocket)
          switch (msg.type) {
            case 'upload-progress':
              callback({
                type: 'UPLOAD_PROGRESS',
                fileId: msg.fileId,
                progress: msg.progress
              });
              break;
              
            case 'processing-step':
              callback({
                type: 'PROCESSING_STEP',
                fileId: msg.fileId,
                step: msg.step,
                progress: msg.stepProgress,
                fragment: msg.fragment
              });
              break;
              
            case 'processing-complete':
              callback({
                type: 'PROCESSING_COMPLETE',
                fileId: msg.fileId,
                result: msg.finalResult
              });
              break;
              
            case 'error':
              callback({
                type: 'ERROR',
                fileId: msg.fileId,
                error: msg.error
              });
              break;
          }
          
        } catch (parseError) {
          console.error('❌ Failed to parse SSE message:', parseError);
        }
      };
      
      eventSource.onerror = (error) => {
        console.error('❌ SSE error:', error);
        callback({
          type: 'ERROR',
          error: { message: 'Server-Sent Events connection error' }
        });
      };
      
    } catch (error) {
      console.error('❌ Failed to create SSE connection:', error);
      callback({
        type: 'ERROR',
        error: { message: 'Failed to create SSE connection' }
      });
    }
    
    // Cleanup function
    return () => {
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
    };
  });
}

export const uploadMachine = createMachine({
  id: 'upload',
  initial: 'idle',
  types: {
    context: {} as UploadContext,
    events: {} as UploadEvent
  },
