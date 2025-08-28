/**
 * Test API endpoint for Binary Encoding Middleware
 * Demonstrates CBOR, MessagePack, and JSON encoding for legal AI data
 */
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { binaryEncoder } from '$lib/middleware/binary-encoding';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const data = await request.json();
    
    // Test all three formats
    const results = {
      json: await binaryEncoder.encode(data, 'json'),
      msgpack: await binaryEncoder.encode(data, 'msgpack'),  
      cbor: await binaryEncoder.encode(data, 'cbor'),
      auto: await binaryEncoder.encode(data) // Auto-detection
    };

    // Calculate compression savings
    const jsonSize = results.json.metrics.encodedSize;
    const msgpackSavings = ((jsonSize - results.msgpack.metrics.encodedSize) / jsonSize * 100);
    const cborSavings = ((jsonSize - results.cbor.metrics.encodedSize) / jsonSize * 100);

    const response = {
      success: true,
      originalData: data,
      encodingResults: {
        json: {
          size: results.json.metrics.encodedSize,
          format: results.json.format,
          encodeTime: results.json.metrics.encodeTime
        },
        msgpack: {
          size: results.msgpack.metrics.encodedSize,
          format: results.msgpack.format,
          encodeTime: results.msgpack.metrics.encodeTime,
          compressionSavings: `${msgpackSavings.toFixed(1)}%`
        },
        cbor: {
          size: results.cbor.metrics.encodedSize,
          format: results.cbor.format,
          encodeTime: results.cbor.metrics.encodeTime,
          compressionSavings: `${cborSavings.toFixed(1)}%`
        },
        autoDetected: results.auto.format
      },
      performanceMetrics: binaryEncoder.getMetrics()
    };

    return json(response);
    
  } catch (error) {
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      hint: 'Send a JSON payload to test binary encoding formats'
    }, { status: 400 });
  }
};

export const GET: RequestHandler = async () => {
  // Provide usage example and stats
  const metrics = binaryEncoder.getMetrics();
  
  return json({
    message: 'Binary Encoding Middleware Test Endpoint',
    usage: {
      endpoint: '/api/test-binary-encoding',
      method: 'POST',
      contentType: 'application/json',
      examplePayload: {
        legalCase: {
          id: 'case-123',
          title: 'Contract Dispute',
          documents: [
            { id: 'doc-1', type: 'contract', size: 1024 }
          ]
        }
      }
    },
    supportedFormats: ['json', 'msgpack', 'cbor'],
    currentMetrics: {
      totalEncodings: metrics.length,
      recentOperations: metrics.slice(-5)
    },
    features: [
      'Automatic format detection based on data characteristics',
      'Performance metrics tracking',
      'Fallback to JSON on encoding errors',
      'Support for legal document structures',
      'Compression ratio calculation'
    ]
  });
};