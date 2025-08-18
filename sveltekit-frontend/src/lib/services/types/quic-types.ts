// QUIC protocol types for high-performance data transfer
export interface QUICConnection {
  id: string;
  endpoint: string;
  protocol: 'quic' | 'http3';
  status: 'connecting' | 'connected' | 'disconnected' | 'error';
  latency?: number;
  bandwidth?: number;
}

export interface TensorOperation {
  id: string;
  type: 'embedding' | 'inference' | 'training';
  input: ArrayBuffer | Float32Array;
  output?: ArrayBuffer | Float32Array;
  metadata: {
    shape: number[];
    dtype: 'float32' | 'float64' | 'int32';
    device: 'cpu' | 'gpu' | 'webgpu';
  };
  timing: {
    start: number;
    end?: number;
    duration?: number;
  };
}

export interface StreamingResponse<T = any> {
  id: string;
  sequence: number;
  type: 'data' | 'error' | 'complete';
  payload: T;
  metadata?: {
    total?: number;
    progress?: number;
    estimated_remaining?: number;
  };
}

export interface QUICMetrics {
  connection: {
    latency: number;
    bandwidth: number;
    packetsLost: number;
    packetsReceived: number;
    bytesSent: number;
    bytesReceived: number;
  };
  performance: {
    avgResponseTime: number;
    throughput: number;
    errorRate: number;
    uptime: number;
  };
  endpoints: {
    [endpoint: string]: {
      requestCount: number;
      avgLatency: number;
      errorCount: number;
    };
  };
}

export interface QUICClientConfig {
  endpoint: string;
  maxRetries?: number;
  timeout?: number;
  keepAlive?: boolean;
  compression?: boolean;
  encryption?: 'tls1.3' | 'aes256';
}

export interface QUICStreamOptions {
  priority?: 'low' | 'normal' | 'high' | 'critical';
  buffer?: boolean;
  compression?: boolean;
  reliable?: boolean;
}

export type QUICEventType =
  | 'connection:established'
  | 'connection:lost'
  | 'stream:opened'
  | 'stream:closed'
  | 'data:received'
  | 'error:network'
  | 'error:protocol';

export interface QUICEvent {
  type: QUICEventType;
  timestamp: number;
  connectionId: string;
  data?: any;
  error?: Error;
}
