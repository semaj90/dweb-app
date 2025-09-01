/**
 * Advanced Binary Encoding Middleware for Legal AI Platform
 * Supports CBOR, MessagePack, and JSON with intelligent format selection
 * Optimized for shader caching, legal document processing, and high-performance data transfer
 */

import * as CBOR from 'cbor';
import { encode as msgpackEncode, decode as msgpackDecode } from '@msgpack/msgpack';
import type { RequestEvent } from '@sveltejs/kit';

export type EncodingFormat = 'cbor' | 'msgpack' | 'json';

export interface BinaryEncodingOptions {
  format: EncodingFormat;
  compression: boolean;
  validation: boolean;
  fallback: boolean;
  performance: boolean;
  caching: boolean;
  streaming: boolean;
}

export interface EncodingMetrics {
  format: EncodingFormat;
  originalSize: number;
  encodedSize: number;
  compressionRatio: number;
  encodeTime: number;
  decodeTime: number;
  bandwidth: number;
  efficiency: 'excellent' | 'good' | 'moderate' | 'poor';
  cacheHit?: boolean;
}

export interface LegalWorkflowContext {
  type: 'document_upload' | 'evidence_review' | 'case_analysis' | 'contract_review' | 'litigation_prep';
  complexity: 'low' | 'medium' | 'high' | 'expert';
  dataSize: number;
  binaryContent: boolean;
  realTime: boolean;
  gpuAccelerated: boolean;
}

export interface BinaryStreamConfig {
  chunkSize: number;
  compression: boolean;
  priority: 'low' | 'normal' | 'high' | 'critical';
  caching: boolean;
  encryption?: boolean;
}

export class AdvancedBinaryEncodingService {
  private metrics: Map<string, EncodingMetrics> = new Map();
  private cache: Map<string, { data: ArrayBuffer | string; format: EncodingFormat; timestamp: number }> = new Map();
  private defaultOptions: BinaryEncodingOptions = {
    format: 'json',
    compression: true,
    validation: true,
    fallback: true,
    performance: true,
    caching: true,
    streaming: false
  };

  constructor(private options: Partial<BinaryEncodingOptions> = {}) {
    this.options = { ...this.defaultOptions, ...options };
    
    // Initialize cache cleanup interval
    setInterval(() => this.cleanupCache(), 300000); // 5 minutes
  }

  /**
   * Intelligent format detection based on data characteristics and legal workflow context
   */
  detectOptimalFormat(data: unknown, context?: LegalWorkflowContext): EncodingFormat {
    const jsonStr = JSON.stringify(data);
    const size = new TextEncoder().encode(jsonStr).length;
    
    // Context-aware format selection for legal workflows
    if (context) {
      switch (context.type) {
        case 'document_upload':
          // Large document uploads with binary content - prefer CBOR
          if (size > 50000 || context.binaryContent || this.hasBinaryData(data)) {
            return 'cbor';
          }
          break;
          
        case 'evidence_review':
          // Structured evidence data - prefer MessagePack for efficiency
          if (size > 5000 && this.isStructuredData(data)) {
            return 'msgpack';
          }
          break;
          
        case 'case_analysis':
          // Complex analysis data with mixed content - prefer CBOR
          if (size > 10000 || context.complexity === 'expert') {
            return 'cbor';
          }
          break;
          
        case 'contract_review':
          // Moderate structured data - prefer MessagePack
          if (size > 2000) {
            return 'msgpack';
          }
          break;
          
        case 'litigation_prep':
          // Critical performance - prefer CBOR for large datasets
          if (size > 15000 || context.realTime) {
            return 'cbor';
          }
          break;
      }
    }
    
    // General format detection rules
    if (size > 100000 || this.hasBinaryData(data)) {
      return 'cbor'; // Best for very large or binary data
    }
    
    if (size > 5000 && this.isStructuredData(data)) {
      return 'msgpack'; // Best for structured data
    }
    
    return 'json'; // Default for small or simple data
  }

  /**
   * Advanced encoding with caching and performance optimization
   */
  async encode(data: unknown, format?: EncodingFormat, context?: LegalWorkflowContext): Promise<{
    encoded: ArrayBuffer | string;
    format: EncodingFormat;
    metrics: EncodingMetrics;
    cacheKey?: string;
  }> {
    const startTime = performance.now();
    const targetFormat = format || this.detectOptimalFormat(data, context);
    const originalSize = new TextEncoder().encode(JSON.stringify(data)).length;
    
    // Generate cache key for reusable data
    const cacheKey = this.generateCacheKey(data, targetFormat);
    
    // Check cache first
    if (this.options.caching && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey)!;
      const metrics: EncodingMetrics = {
        format: targetFormat,
        originalSize,
        encodedSize: cached.data instanceof ArrayBuffer ? cached.data.byteLength : new TextEncoder().encode(cached.data).length,
        compressionRatio: originalSize / (cached.data instanceof ArrayBuffer ? cached.data.byteLength : new TextEncoder().encode(cached.data).length),
        encodeTime: 0.1, // Cache hit time
        decodeTime: 0,
        bandwidth: 0,
        efficiency: 'excellent',
        cacheHit: true
      };
      
      return {
        encoded: cached.data,
        format: cached.format,
        metrics,
        cacheKey
      };
    }

    let encoded: ArrayBuffer | string;
    let encodedSize: number;

    try {
      switch (targetFormat) {
        case 'cbor': {
          const cborBuffer = CBOR.encode(data);
          encoded = new ArrayBuffer(cborBuffer.byteLength);
          new Uint8Array(encoded).set(new Uint8Array(cborBuffer));
          encodedSize = encoded.byteLength;
          break;
        }
          
        case 'msgpack': {
          const msgpackData = msgpackEncode(data);
          encoded = new ArrayBuffer(msgpackData.byteLength);
          new Uint8Array(encoded).set(new Uint8Array(msgpackData));
          encodedSize = encoded.byteLength;
          break;
        }
          
        case 'json':
        default:
          encoded = JSON.stringify(data, null, this.options.compression ? 0 : 2);
          encodedSize = new TextEncoder().encode(encoded).length;
          break;
      }

      const encodeTime = performance.now() - startTime;
      const compressionRatio = originalSize / encodedSize;
      
      const metrics: EncodingMetrics = {
        format: targetFormat,
        originalSize,
        encodedSize,
        compressionRatio,
        encodeTime,
        decodeTime: 0,
        bandwidth: encodedSize / (encodeTime / 1000), // bytes per second
        efficiency: this.calculateEfficiency(compressionRatio, encodeTime),
        cacheHit: false
      };

      // Store in cache for future use
      if (this.options.caching) {
        this.cache.set(cacheKey, {
          data: encoded,
          format: targetFormat,
          timestamp: Date.now()
        });
      }

      // Store performance metrics
      if (this.options.performance) {
        this.metrics.set(`encode_${targetFormat}_${Date.now()}`, metrics);
      }

      return { encoded, format: targetFormat, metrics, cacheKey };

    } catch (error: any) {
      if (this.options.fallback && targetFormat !== 'json') {
        console.warn(`Encoding failed for ${targetFormat}, falling back to JSON:`, error);
        return this.encode(data, 'json', context);
      }
      throw new Error(`Encoding failed: ${error}`);
    }
  }

  /**
   * Advanced decoding with validation and error recovery
   */
  async decode(data: ArrayBuffer | string, format: EncodingFormat): Promise<{
    decoded: unknown;
    metrics: EncodingMetrics;
  }> {
    const startTime = performance.now();
    let decoded: unknown;

    try {
      switch (format) {
        case 'cbor':
          decoded = CBOR.decode(new Uint8Array(data as ArrayBuffer));
          break;
          
        case 'msgpack':
          decoded = msgpackDecode(new Uint8Array(data as ArrayBuffer));
          break;
          
        case 'json':
        default:
          decoded = JSON.parse(data as string);
          break;
      }

      const decodeTime = performance.now() - startTime;
      const dataSize = data instanceof ArrayBuffer ? data.byteLength : new TextEncoder().encode(data).length;
      
      const metrics: EncodingMetrics = {
        format,
        originalSize: 0,
        encodedSize: dataSize,
        compressionRatio: 1,
        encodeTime: 0,
        decodeTime,
        bandwidth: dataSize / (decodeTime / 1000),
        efficiency: this.calculateEfficiency(1, decodeTime)
      };

      if (this.options.performance) {
        this.metrics.set(`decode_${format}_${Date.now()}`, metrics);
      }

      return { decoded, metrics };

    } catch (error: any) {
      if (this.options.fallback && format !== 'json') {
        console.warn(`Decoding failed for ${format}, attempting JSON fallback:`, error);
        return this.decode(data, 'json');
      }
      throw new Error(`Decoding failed: ${error}`);
    }
  }

  /**
   * Streaming encoder for large datasets
   */
  async *encodeStream(data: AsyncIterable<any>, config: BinaryStreamConfig): AsyncGenerator<{
    chunk: ArrayBuffer | string;
    format: EncodingFormat;
    chunkIndex: number;
    totalChunks?: number;
    metrics: EncodingMetrics;
  }> {
    let chunkIndex = 0;
    const format = this.detectOptimalFormat(data);
    
    for await (const chunk of data) {
      const result = await this.encode(chunk, format);
      
      yield {
        chunk: result.encoded,
        format: result.format,
        chunkIndex: chunkIndex++,
        metrics: result.metrics
      };
      
      // Respect chunk size and priority
      if (config.priority === 'low') {
        await new Promise(resolve => setTimeout(resolve, 10)); // Throttle low priority
      }
    }
  }

  /**
   * Legal workflow optimization analyzer
   */
  analyzeWorkflowOptimization(context: LegalWorkflowContext): {
    recommendedFormat: EncodingFormat;
    expectedCompressionRatio: number;
    expectedPerformanceGain: number;
    memoryImpact: 'low' | 'medium' | 'high';
    recommendations: string[];
  } {
    const recommendations: string[] = [];
    let recommendedFormat: EncodingFormat = 'json';
    let expectedCompressionRatio = 1.0;
    let expectedPerformanceGain = 0.0;
    let memoryImpact: 'low' | 'medium' | 'high' = 'low';

    switch (context.type) {
      case 'document_upload':
        if (context.binaryContent || context.dataSize > 100000) {
          recommendedFormat = 'cbor';
          expectedCompressionRatio = 2.5;
          expectedPerformanceGain = 0.65;
          memoryImpact = 'medium';
          recommendations.push('Use CBOR for binary document content');
          recommendations.push('Enable compression for large files');
        }
        break;
        
      case 'evidence_review':
        if (context.complexity === 'high' || context.realTime) {
          recommendedFormat = 'msgpack';
          expectedCompressionRatio = 1.8;
          expectedPerformanceGain = 0.45;
          memoryImpact = 'low';
          recommendations.push('MessagePack optimal for structured evidence data');
          recommendations.push('Consider caching for real-time workflows');
        }
        break;
        
      case 'case_analysis':
        if (context.gpuAccelerated || context.dataSize > 50000) {
          recommendedFormat = 'cbor';
          expectedCompressionRatio = 2.2;
          expectedPerformanceGain = 0.55;
          memoryImpact = 'high';
          recommendations.push('CBOR recommended for GPU-accelerated analysis');
          recommendations.push('Enable streaming for large datasets');
        }
        break;
        
      case 'contract_review':
        recommendedFormat = 'msgpack';
        expectedCompressionRatio = 1.6;
        expectedPerformanceGain = 0.35;
        memoryImpact = 'low';
        recommendations.push('MessagePack efficient for contract metadata');
        break;
        
      case 'litigation_prep':
        recommendedFormat = 'cbor';
        expectedCompressionRatio = 2.0;
        expectedPerformanceGain = 0.50;
        memoryImpact = 'medium';
        recommendations.push('CBOR for high-performance litigation data');
        recommendations.push('Enable caching for frequently accessed data');
        break;
    }

    return {
      recommendedFormat,
      expectedCompressionRatio,
      expectedPerformanceGain,
      memoryImpact,
      recommendations
    };
  }

  /**
   * SvelteKit middleware with legal workflow awareness
   */
  createMiddleware(workflowContext?: LegalWorkflowContext) {
    return async (event: RequestEvent, resolve: Function) => {
      const { request } = event;
      
      // Detect preferred encoding from Accept header
      const acceptHeader = request.headers.get('accept') || '';
      let preferredFormat: EncodingFormat = 'json';
      
      if (acceptHeader.includes('application/cbor')) {
        preferredFormat = 'cbor';
      } else if (acceptHeader.includes('application/msgpack')) {
        preferredFormat = 'msgpack';
      }

      // Override based on workflow context
      if (workflowContext) {
        const optimization = this.analyzeWorkflowOptimization(workflowContext);
        if (optimization.expectedPerformanceGain > 0.4) {
          preferredFormat = optimization.recommendedFormat;
        }
      }

      // Handle request body decoding
      if (request.body && request.method !== 'GET') {
        const contentType = request.headers.get('content-type') || '';
        let format: EncodingFormat = 'json';
        
        if (contentType.includes('application/cbor')) {
          format = 'cbor';
        } else if (contentType.includes('application/msgpack')) {
          format = 'msgpack';
        }

        if (format !== 'json') {
          const arrayBuffer = await request.arrayBuffer();
          const { decoded } = await this.decode(arrayBuffer, format);
          
          // Replace request body with decoded data
          const newRequest = new Request(request.url, {
            method: request.method,
            headers: {
              ...Object.fromEntries(request.headers.entries()),
              'content-type': 'application/json'
            },
            body: JSON.stringify(decoded)
          });
          
          event.request = newRequest;
        }
      }

      // Process response
      const response = await resolve(event);

      // Handle response encoding
      if (response.headers.get('content-type')?.includes('application/json') && preferredFormat !== 'json') {
        const text = await response.text();
        const data = JSON.parse(text);
        const { encoded, format, metrics } = await this.encode(data, preferredFormat, workflowContext);

        const contentType = format === 'cbor' ? 'application/cbor' : 
                           format === 'msgpack' ? 'application/msgpack' : 
                           'application/json';

        return new Response(encoded, {
          status: response.status,
          statusText: response.statusText,
          headers: {
            ...Object.fromEntries(response.headers.entries()),
            'content-type': contentType,
            'x-encoding-format': format,
            'x-compression-ratio': metrics.compressionRatio.toFixed(2),
            'x-encode-time': `${metrics.encodeTime.toFixed(2)}ms`,
            'x-bandwidth': `${(metrics.bandwidth / 1024).toFixed(1)}KB/s`,
            'x-efficiency': metrics.efficiency
          }
        });
      }

      return response;
    };
  }

  /**
   * Performance analytics and reporting
   */
  getPerformanceReport(): {
    totalEncodings: number;
    totalDecodings: number;
    averageCompressionRatio: number;
    averageEncodeTime: number;
    averageDecodeTime: number;
    formatDistribution: Record<EncodingFormat, number>;
    efficiencyDistribution: Record<string, number>;
    cacheHitRate: number;
    totalBandwidthSaved: number;
  } {
    const metrics = Array.from(this.metrics.values());
    const encodings = metrics.filter(m => m.encodeTime > 0);
    const decodings = metrics.filter(m => m.decodeTime > 0);
    
    const formatDistribution: Record<EncodingFormat, number> = {
      cbor: metrics.filter(m => m.format === 'cbor').length,
      msgpack: metrics.filter(m => m.format === 'msgpack').length,
      json: metrics.filter(m => m.format === 'json').length
    };
    
    const efficiencyDistribution = {
      excellent: metrics.filter(m => m.efficiency === 'excellent').length,
      good: metrics.filter(m => m.efficiency === 'good').length,
      moderate: metrics.filter(m => m.efficiency === 'moderate').length,
      poor: metrics.filter(m => m.efficiency === 'poor').length
    };
    
    const cacheHits = metrics.filter(m => m.cacheHit).length;
    const totalBandwidthSaved = metrics.reduce((sum, m) => 
      sum + (m.originalSize - m.encodedSize), 0
    );

    return {
      totalEncodings: encodings.length,
      totalDecodings: decodings.length,
      averageCompressionRatio: encodings.reduce((sum, m) => sum + m.compressionRatio, 0) / encodings.length || 1,
      averageEncodeTime: encodings.reduce((sum, m) => sum + m.encodeTime, 0) / encodings.length || 0,
      averageDecodeTime: decodings.reduce((sum, m) => sum + m.decodeTime, 0) / decodings.length || 0,
      formatDistribution,
      efficiencyDistribution,
      cacheHitRate: metrics.length > 0 ? cacheHits / metrics.length : 0,
      totalBandwidthSaved
    };
  }

  /**
   * Utility methods
   */
  getMetrics(): EncodingMetrics[] {
    return Array.from(this.metrics.values());
  }

  clearMetrics(): void {
    this.metrics.clear();
  }

  clearCache(): void {
    this.cache.clear();
  }

  private generateCacheKey(data: unknown, format: EncodingFormat): string {
    const jsonStr = JSON.stringify(data);
    const hash = this.hashString(jsonStr);
    return `${format}_${hash}`;
  }

  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return hash.toString(36);
  }

  private calculateEfficiency(compressionRatio: number, processingTime: number): 'excellent' | 'good' | 'moderate' | 'poor' {
    const score = compressionRatio * (1000 / (processingTime + 1));
    
    if (score > 50) return 'excellent';
    if (score > 20) return 'good';
    if (score > 10) return 'moderate';
    return 'poor';
  }

  private cleanupCache(): void {
    const now = Date.now();
    const maxAge = 600000; // 10 minutes
    
    for (const [key, value] of this.cache.entries()) {
      if (now - value.timestamp > maxAge) {
        this.cache.delete(key);
      }
    }
  }

  private hasBinaryData(data: unknown): boolean {
    return this.traverseObject(data, (value) => {
      return value instanceof ArrayBuffer || 
             value instanceof Uint8Array || 
             (typeof value === 'string' && (
               value.startsWith('data:') ||
               value.includes('base64') ||
               /^[A-Za-z0-9+/]*={0,2}$/.test(value.slice(-20)) // Base64 pattern
             ));
    });
  }

  private isStructuredData(data: unknown): boolean {
    if (typeof data !== 'object' || data === null) return false;
    
    if (Array.isArray(data)) {
      return data.length > 5 && data.some(item => typeof item === 'object');
    }
    
    const keys = Object.keys(data);
    return keys.length > 3 && keys.some(key => 
      typeof (data as any)[key] === 'object'
    );
  }

  private traverseObject(obj: unknown, condition: (value: unknown) => boolean): boolean {
    if (condition(obj)) return true;
    
    if (typeof obj === 'object' && obj !== null) {
      for (const value of Object.values(obj)) {
        if (this.traverseObject(value, condition)) return true;
      }
    }
    
    return false;
  }
}

// Global instance with legal workflow optimization
export const binaryEncoder = new AdvancedBinaryEncodingService({
  performance: true,
  caching: true,
  compression: true,
  fallback: true
});

// Specialized instances for different legal workflows
export const documentUploadEncoder = new AdvancedBinaryEncodingService({
  format: 'cbor',
  compression: true,
  caching: true,
  streaming: true
});

export const evidenceReviewEncoder = new AdvancedBinaryEncodingService({
  format: 'msgpack',
  compression: true,
  caching: true,
  performance: true
});

export const caseAnalysisEncoder = new AdvancedBinaryEncodingService({
  format: 'cbor',
  compression: true,
  caching: true,
  streaming: true,
  performance: true
});

// Helper functions for direct use
export async function encodeCBOR(data: unknown): Promise<ArrayBuffer> {
  const { encoded } = await binaryEncoder.encode(data, 'cbor');
  return encoded as ArrayBuffer;
}

export async function encodeMessagePack(data: unknown): Promise<ArrayBuffer> {
  const { encoded } = await binaryEncoder.encode(data, 'msgpack');
  return encoded as ArrayBuffer;
}

export async function decodeCBOR(data: ArrayBuffer): Promise<any> {
  const { decoded } = await binaryEncoder.decode(data, 'cbor');
  return decoded;
}

export async function decodeMessagePack(data: ArrayBuffer): Promise<any> {
  const { decoded } = await binaryEncoder.decode(data, 'msgpack');
  return decoded;
}

// Legal workflow-specific encoding helpers
export async function encodeLegalDocument(data: unknown, context: LegalWorkflowContext): Promise<{
  encoded: ArrayBuffer | string;
  format: EncodingFormat;
  metrics: EncodingMetrics;
  optimization: any;
}> {
  const optimization = binaryEncoder.analyzeWorkflowOptimization(context);
  const result = await binaryEncoder.encode(data, optimization.recommendedFormat, context);
  
  return {
    ...result,
    optimization
  };
}

export async function createWorkflowMiddleware(workflowType: LegalWorkflowContext['type']): Promise<any> {
  const context: LegalWorkflowContext = {
    type: workflowType,
    complexity: 'medium',
    dataSize: 0,
    binaryContent: false,
    realTime: false,
    gpuAccelerated: false
  };
  
  return binaryEncoder.createMiddleware(context);
}