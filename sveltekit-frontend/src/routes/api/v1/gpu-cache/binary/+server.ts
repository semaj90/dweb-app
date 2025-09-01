import type { RequestHandler } from './$types.js';

/**
 * Binary-Optimized GPU Shader Cache API
 * Combines GPU shader caching with binary encoding middleware for maximum performance
 */

import { binaryGPUShaderCache } from '../../../../../lib/services/gpu-shader-cache-binary-extension.js';
import { binaryEncoder } from '../../../../../lib/middleware/binary-encoding.js';
import { URL } from "url";

// GET /api/v1/gpu-cache/binary/shader?key=<cacheKey>
export const GET: RequestHandler = async ({ url, request }) => {
  try {
    const cacheKey = url.searchParams.get('key');
    if (!cacheKey) {
      return json({ error: 'Missing cache key' }, { status: 400 });
    }

    // Retrieve shader with binary optimization
    const shader = await binaryGPUShaderCache.retrieveShader(cacheKey);
    if (!shader) {
      return json({ error: 'Shader not found' }, { status: 404 });
    }

    // Detect client's preferred encoding format
    const acceptHeader = request.headers.get('accept') || '';
    let preferredFormat: 'cbor' | 'msgpack' | 'json' = 'json';
    
    if (acceptHeader.includes('application/cbor')) {
      preferredFormat = 'cbor';
    } else if (acceptHeader.includes('application/msgpack')) {
      preferredFormat = 'msgpack';
    }

    // Encode response with optimal format
    const responseData = {
      shader: {
        sourceCode: shader.sourceCode,
        metadata: shader.metadata,
        metrics: shader.metrics
      },
      cacheKey,
      timestamp: Date.now(),
      compressionSavings: `${((1 - 1/shader.metrics.compressionRatio) * 100).toFixed(1)}%`,
      decodingTime: `${shader.metrics.decodingTime.toFixed(2)}ms`
    };

    if (preferredFormat === 'json') {
      return json(responseData);
    }

    // Binary encoding for better performance
    const { encoded, format, metrics } = await binaryEncoder.encode(responseData, preferredFormat);
    
    const contentType = format === 'cbor' ? 'application/cbor' : 'application/msgpack';
    
    return new Response(encoded, {
      status: 200,
      headers: {
        'content-type': contentType,
        'x-encoding-format': format,
        'x-compression-ratio': metrics.compressionRatio.toString(),
        'x-encode-time': `${metrics.encodeTime.toFixed(2)}ms`
      }
    });

  } catch (error: any) {
    console.error('Binary shader cache GET error:', error);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
};

// POST /api/v1/gpu-cache/binary/shader
export const POST: RequestHandler = async ({ request }) => {
  try {
    // Auto-detect request encoding
    const contentType = request.headers.get('content-type') || '';
    let requestData: any;

    if (contentType.includes('application/cbor')) {
      const buffer = await request.arrayBuffer();
      const { decoded } = await binaryEncoder.decode(buffer, 'cbor');
      requestData = decoded;
    } else if (contentType.includes('application/msgpack')) {
      const buffer = await request.arrayBuffer();
      const { decoded } = await binaryEncoder.decode(buffer, 'msgpack');
      requestData = decoded;
    } else {
      requestData = await request.json();
    }

    const { sourceCode, compiledBinary, metadata, workflowType } = requestData;

    if (!sourceCode || !compiledBinary) {
      return json({ error: 'Missing required fields: sourceCode, compiledBinary' }, { status: 400 });
    }

    // Convert base64 binary data if needed
    let binaryData: ArrayBuffer;
    if (typeof compiledBinary === 'string') {
      const base64 = compiledBinary.split(',')[1] || compiledBinary;
      binaryData = Uint8Array.from(atob(base64), c => c.charCodeAt(0)).buffer as ArrayBuffer;
    } else {
      binaryData = compiledBinary;
    }

    // Store shader with binary optimization
    const entry = await binaryGPUShaderCache.storeShader({
      sourceCode,
      compiledBinary: binaryData,
      metadata: metadata || {}
    });

    // Get workflow optimization recommendations
    let optimizationRecommendations = null;
    if (workflowType) {
      optimizationRecommendations = await binaryGPUShaderCache.optimizeForLegalWorkflow(workflowType);
    }

    const response = {
      success: true,
      cacheKey: entry.cacheKey,
      entry: {
        id: entry.id,
        shaderType: entry.shaderType,
        encodingFormat: entry.encodingFormat,
        compressionRatio: entry.compressionRatio,
        memoryFootprint: entry.memoryFootprint
      },
      optimizationRecommendations,
      metrics: {
        compressionSavings: `${((1 - 1/entry.compressionRatio) * 100).toFixed(1)}%`,
        memoryReduction: `${(entry.memoryFootprint / 1024).toFixed(1)}KB`,
        storageEfficiency: entry.compressionRatio > 1.5 ? 'excellent' : entry.compressionRatio > 1.2 ? 'good' : 'moderate'
      }
    };

    return json(response);

  } catch (error: any) {
    console.error('Binary shader cache POST error:', error);
    return json({ error: 'Failed to store shader' }, { status: 500 });
  }
};

// PUT /api/v1/gpu-cache/binary/batch
export const PUT: RequestHandler = async ({ request }) => {
  try {
    const { shaders, workflowType } = await request.json();

    if (!Array.isArray(shaders) || shaders.length === 0) {
      return json({ error: 'Invalid or empty shaders array' }, { status: 400 });
    }

    // Process shaders in batch for better performance
    const startTime = performance.now();
    const results = await binaryGPUShaderCache.batchEncodeShaders(shaders);
    const processingTime = performance.now() - startTime;

    // Get workflow optimization for the batch
    let workflowOptimization = null;
    if (workflowType) {
      workflowOptimization = await binaryGPUShaderCache.optimizeForLegalWorkflow(workflowType);
    }

    const response = {
      success: true,
      processed: results.encodedShaders.length,
      totalCompressionRatio: results.totalCompressionRatio,
      totalEncodingTime: results.totalEncodingTime,
      processingTime: processingTime,
      workflowOptimization,
      shaders: results.encodedShaders.map(shader => ({
        cacheKey: shader.cacheKey,
        shaderType: shader.shaderType,
        encodingFormat: shader.encodingFormat,
        compressionRatio: shader.compressionRatio
      })),
      batchMetrics: {
        averageCompressionRatio: results.totalCompressionRatio / results.encodedShaders.length,
        averageEncodingTime: results.totalEncodingTime / results.encodedShaders.length,
        totalMemorySaved: results.encodedShaders.reduce((total, shader) => {
          return total + (shader.memoryFootprint * (1 - 1/shader.compressionRatio));
        }, 0),
        recommendedFormat: workflowOptimization?.recommendedEncodingFormat || 'cbor'
      }
    };

    return json(response);

  } catch (error: any) {
    console.error('Binary shader cache batch error:', error);
    return json({ error: 'Batch processing failed' }, { status: 500 });
  }
};

// GET /api/v1/gpu-cache/binary/webgpu?key=<cacheKey>
export const PATCH: RequestHandler = async ({ url }) => {
  try {
    const cacheKey = url.searchParams.get('key');
    if (!cacheKey) {
      return json({ error: 'Missing cache key' }, { status: 400 });
    }

    // Retrieve shader optimized for WebGPU
    const webgpuShader = await binaryGPUShaderCache.retrieveForWebGPU(cacheKey);
    if (!webgpuShader) {
      return json({ error: 'Shader not found' }, { status: 404 });
    }

    return json({
      shaderModule: webgpuShader.shaderModule,
      binaryAssets: Array.from(webgpuShader.binaryAssets).map(buffer => 
        Array.from(new Uint8Array(buffer))
      ),
      compressionSavings: webgpuShader.compressionSavings,
      webgpuReady: true,
      loadingInstructions: {
        createShaderModule: true,
        binaryData: webgpuShader.binaryAssets.length,
        estimatedLoadTime: `${(webgpuShader.compressionSavings / 1024 / 100).toFixed(1)}ms` // rough estimate
      }
    });

  } catch (error: any) {
    console.error('WebGPU shader cache error:', error);
    return json({ error: 'WebGPU shader retrieval failed' }, { status: 500 });
  }
};

// DELETE /api/v1/gpu-cache/binary/metrics
export const DELETE: RequestHandler = async () => {
  try {
    // Clear encoding performance metrics
    binaryEncoder.clearMetrics();

    return json({ 
      success: true, 
      message: 'Binary encoding metrics cleared',
      timestamp: Date.now()
    });

  } catch (error: any) {
    console.error('Metrics clear error:', error);
    return json({ error: 'Failed to clear metrics' }, { status: 500 });
  }
};

// OPTIONS for CORS support
export const OPTIONS: RequestHandler = async () => {
  return new Response(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, PATCH, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Accept, X-Encoding-Format',
      'Access-Control-Expose-Headers': 'X-Encoding-Format, X-Compression-Ratio, X-Encode-Time'
    }
  });
};