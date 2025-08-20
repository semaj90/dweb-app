/**
 * Binary Encoding Middleware for Legal AI Platform
 * Supports CBOR, MessagePack, and JSON encoding with performance optimization
 */

import CBOR from 'cbor';
import { pack as msgpackPack, unpack as msgpackUnpack } from '@msgpack/msgpack';
import type { RequestEvent } from '@sveltejs/kit';

export type EncodingFormat = 'cbor' | 'msgpack' | 'json';

export interface BinaryEncodingOptions {
  format: EncodingFormat;
  compression: boolean;
  validation: boolean;
  fallback: boolean;
  performance: boolean;
}

export interface EncodingMetrics {
  format: EncodingFormat;
  originalSize: number;
  encodedSize: number;
  compressionRatio: number;
  encodeTime: number;
  decodeTime: number;
}

export class BinaryEncodingService {
  private metrics: Map<string, EncodingMetrics> = new Map();
  private defaultOptions: BinaryEncodingOptions = {
    format: 'json',
    compression: true,
    validation: true,
    fallback: true,
    performance: true
  };

  constructor(private options: Partial<BinaryEncodingOptions> = {}) {
    this.options = { ...this.defaultOptions, ...options };
  }

  /**
   * Detect optimal encoding format based on data characteristics
   */
  detectOptimalFormat(data: any): EncodingFormat {
    const jsonStr = JSON.stringify(data);
    const size = new TextEncoder().encode(jsonStr).length;
    
    // Large binary data - use CBOR
    if (size > 10000 || this.hasBinaryData(data)) {
      return 'cbor';
    }
    
    // Medium structured data - use MessagePack
    if (size > 1000 && this.isStructuredData(data)) {
      return 'msgpack';
    }
    
    // Small or simple data - use JSON
    return 'json';
  }

  /**
   * Encode data using specified format
   */
  async encode(data: any, format?: EncodingFormat): Promise<{
    encoded: ArrayBuffer | string;
    format: EncodingFormat;
    metrics: EncodingMetrics;
  }> {
    const startTime = performance.now();
    const targetFormat = format || this.detectOptimalFormat(data);
    const originalSize = new TextEncoder().encode(JSON.stringify(data)).length;

    let encoded: ArrayBuffer | string;
    let encodedSize: number;

    try {
      switch (targetFormat) {
        case 'cbor':
          encoded = CBOR.encode(data);
          encodedSize = encoded.byteLength;
          break;
          
        case 'msgpack':
          encoded = msgpackPack(data);
          encodedSize = encoded.byteLength;
          break;
          
        case 'json':
        default:
          encoded = JSON.stringify(data);
          encodedSize = new TextEncoder().encode(encoded).length;
          break;
      }

      const encodeTime = performance.now() - startTime;
      const metrics: EncodingMetrics = {
        format: targetFormat,
        originalSize,
        encodedSize,
        compressionRatio: originalSize / encodedSize,
        encodeTime,
        decodeTime: 0
      };

      if (this.options.performance) {
        this.metrics.set(`encode_${targetFormat}_${Date.now()}`, metrics);
      }

      return { encoded, format: targetFormat, metrics };

    } catch (error) {
      if (this.options.fallback && targetFormat !== 'json') {
        console.warn(`Encoding failed for ${targetFormat}, falling back to JSON:`, error);
        return this.encode(data, 'json');
      }
      throw new Error(`Encoding failed: ${error}`);
    }
  }

  /**
   * Decode data using specified format
   */
  async decode(data: ArrayBuffer | string, format: EncodingFormat): Promise<{
    decoded: any;
    metrics: EncodingMetrics;
  }> {
    const startTime = performance.now();
    let decoded: any;

    try {
      switch (format) {
        case 'cbor':
          decoded = CBOR.decode(data as ArrayBuffer);
          break;
          
        case 'msgpack':
          decoded = msgpackUnpack(data as ArrayBuffer);
          break;
          
        case 'json':
        default:
          decoded = JSON.parse(data as string);
          break;
      }

      const decodeTime = performance.now() - startTime;
      const metrics: EncodingMetrics = {
        format,
        originalSize: 0,
        encodedSize: data instanceof ArrayBuffer ? data.byteLength : new TextEncoder().encode(data).length,
        compressionRatio: 1,
        encodeTime: 0,
        decodeTime
      };

      if (this.options.performance) {
        this.metrics.set(`decode_${format}_${Date.now()}`, metrics);
      }

      return { decoded, metrics };

    } catch (error) {
      if (this.options.fallback && format !== 'json') {
        console.warn(`Decoding failed for ${format}, attempting JSON fallback:`, error);
        return this.decode(data, 'json');
      }
      throw new Error(`Decoding failed: ${error}`);
    }
  }

  /**
   * SvelteKit middleware for automatic encoding/decoding
   */
  createMiddleware() {
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
            headers: request.headers,
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
        const { encoded, format } = await this.encode(data, preferredFormat);

        const contentType = format === 'cbor' ? 'application/cbor' : 
                           format === 'msgpack' ? 'application/msgpack' : 
                           'application/json';

        return new Response(encoded, {
          status: response.status,
          statusText: response.statusText,
          headers: {
            ...Object.fromEntries(response.headers.entries()),
            'content-type': contentType
          }
        });
      }

      return response;
    };
  }

  /**
   * Get encoding performance metrics
   */
  getMetrics(): EncodingMetrics[] {
    return Array.from(this.metrics.values());
  }

  /**
   * Clear performance metrics
   */
  clearMetrics(): void {
    this.metrics.clear();
  }

  /**
   * Check if data contains binary content
   */
  private hasBinaryData(data: any): boolean {
    return this.traverseObject(data, (value) => {
      return value instanceof ArrayBuffer || 
             value instanceof Uint8Array || 
             (typeof value === 'string' && value.startsWith('data:'));
    });
  }

  /**
   * Check if data is structured (objects/arrays)
   */
  private isStructuredData(data: any): boolean {
    return typeof data === 'object' && data !== null && 
           (Array.isArray(data) || Object.keys(data).length > 3);
  }

  /**
   * Traverse object and test condition
   */
  private traverseObject(obj: any, condition: (value: any) => boolean): boolean {
    if (condition(obj)) return true;
    
    if (typeof obj === 'object' && obj !== null) {
      for (const value of Object.values(obj)) {
        if (this.traverseObject(value, condition)) return true;
      }
    }
    
    return false;
  }
}

// Global instance
export const binaryEncoder = new BinaryEncodingService();

// Helper functions for direct use
export async function encodeCBOR(data: any): Promise<ArrayBuffer> {
  const { encoded } = await binaryEncoder.encode(data, 'cbor');
  return encoded as ArrayBuffer;
}

export async function encodeMessagePack(data: any): Promise<ArrayBuffer> {
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