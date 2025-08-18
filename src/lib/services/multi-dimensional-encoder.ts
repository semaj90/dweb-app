/**
 * Multi-Dimensional Data Encoder
 * Advanced data encoding for efficient transfer and GPU processing
 * Implements: Image texture encoding, QUIC/gRPC/REST protocol selection, vertex buffer optimization
 */

interface EncodingConfig {
  dimensions: 1 | 2 | 3;
  protocol: 'rest' | 'grpc' | 'quic';
  compression: 'none' | 'gzip' | 'brotli' | 'custom';
  precision: 'fp32' | 'fp16' | 'int8' | 'int4';
  chunkSize: number;
  enableGPUOptimization: boolean;
}

interface EncodedData {
  format: 'raw' | 'texture' | 'vertex_buffer' | 'compressed';
  dimensions: number;
  shape: number[];
  data: ArrayBuffer | ImageData | GPUBuffer;
  metadata: {
    originalSize: number;
    compressedSize: number;
    compressionRatio: number;
    encoding: string;
    checksum: string;
  };
}

interface StreamConfig {
  protocol: 'rest' | 'grpc' | 'quic';
  chunkSize: number;
  enableCompression: boolean;
  priority: number;
  timeout: number;
}

class MultiDimensionalEncoder {
  private webGPUDevice: GPUDevice | null = null;
  private compressionWorker: Worker | null = null;
  private protocolHandlers = new Map<string, Function>();

  constructor() {
    this.initializeProtocolHandlers();
    this.initializeCompressionWorker();
  }

  async initialize() {
    // Initialize WebGPU for advanced encoding
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          this.webGPUDevice = await adapter.requestDevice();
          console.log('🎮 WebGPU encoder initialized');
        }
      } catch (error) {
        console.warn('WebGPU not available for encoding:', error);
      }
    }
  }

  private initializeProtocolHandlers() {
    // REST handler - simple HTTP requests
    this.protocolHandlers.set('rest', async (data: ArrayBuffer, endpoint: string) => {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: data
      });
      return response.arrayBuffer();
    });

    // gRPC handler - binary protocol buffers
    this.protocolHandlers.set('grpc', async (data: ArrayBuffer, endpoint: string) => {
      // Mock gRPC implementation - in practice would use @grpc/grpc-js
      const response = await fetch(endpoint.replace('grpc://', 'http://'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/grpc+proto' },
        body: data
      });
      return response.arrayBuffer();
    });

    // QUIC handler - UDP-based protocol for 3D data
    this.protocolHandlers.set('quic', async (data: ArrayBuffer, endpoint: string) => {
      // Mock QUIC implementation - would use WebTransport in real implementation
      if ('WebTransport' in window) {
        try {
          const transport = new (window as any).WebTransport(endpoint);
          await transport.ready;
          
          const writer = transport.datagrams.writable.getWriter();
          await writer.write(new Uint8Array(data));
          await writer.close();
          
          return data; // Echo for demo
        } catch (error) {
          console.warn('WebTransport not available, falling back to WebSocket');
          return this.fallbackToWebSocket(data, endpoint);
        }
      } else {
        return this.fallbackToWebSocket(data, endpoint);
      }
    });
  }

  private async fallbackToWebSocket(data: ArrayBuffer, endpoint: string): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(endpoint.replace('quic://', 'ws://'));
      
      ws.onopen = () => {
        ws.send(data);
      };
      
      ws.onmessage = (event) => {
        if (event.data instanceof Blob) {
          event.data.arrayBuffer().then(resolve);
        } else {
          resolve(new TextEncoder().encode(event.data).buffer);
        }
        ws.close();
      };
      
      ws.onerror = reject;
      
      setTimeout(() => reject(new Error('WebSocket timeout')), 10000);
    });
  }

  private initializeCompressionWorker() {
    // Create compression worker for background processing
    const workerCode = `
      self.onmessage = function(e) {
        const { data, compression, precision } = e.data;
        
        let result;
        switch (compression) {
          case 'gzip':
            // Mock gzip compression
            result = data; // In practice would use actual compression
            break;
          case 'brotli':
            // Mock brotli compression
            result = data;
            break;
          case 'custom':
            // Custom compression for float arrays
            result = customCompress(data, precision);
            break;
          default:
            result = data;
        }
        
        self.postMessage(result);
      };
      
      function customCompress(data, precision) {
        // Custom compression algorithm for float data
        const view = new Float32Array(data);
        let compressed;
        
        switch (precision) {
          case 'fp16':
            compressed = compressToFP16(view);
            break;
          case 'int8':
            compressed = compressToInt8(view);
            break;
          case 'int4':
            compressed = compressToInt4(view);
            break;
          default:
            compressed = data;
        }
        
        return compressed;
      }
      
      function compressToFP16(data) {
        // Convert float32 to float16
        const result = new Uint16Array(data.length);
        for (let i = 0; i < data.length; i++) {
          result[i] = floatToHalf(data[i]);
        }
        return result.buffer;
      }
      
      function compressToInt8(data) {
        // Quantize to int8 [-128, 127]
        const result = new Int8Array(data.length);
        for (let i = 0; i < data.length; i++) {
          result[i] = Math.round(Math.max(-128, Math.min(127, data[i] * 127)));
        }
        return result.buffer;
      }
      
      function compressToInt4(data) {
        // Pack two 4-bit values into each byte
        const result = new Uint8Array(Math.ceil(data.length / 2));
        for (let i = 0; i < data.length; i += 2) {
          const val1 = Math.round(Math.max(0, Math.min(15, (data[i] + 1) * 7.5))) & 0xF;
          const val2 = i + 1 < data.length ? 
            Math.round(Math.max(0, Math.min(15, (data[i + 1] + 1) * 7.5))) & 0xF : 0;
          result[Math.floor(i / 2)] = (val1 << 4) | val2;
        }
        return result.buffer;
      }
      
      function floatToHalf(val) {
        // IEEE 754 float32 to float16 conversion
        const floatView = new Float32Array([val]);
        const intView = new Uint32Array(floatView.buffer);
        const bits = intView[0];
        
        const sign = (bits >>> 31) << 15;
        const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
        const mantissa = (bits >>> 13) & 0x3ff;
        
        if (exponent <= 0) return sign;
        if (exponent >= 31) return sign | 0x7c00 | (mantissa ? 0x200 : 0);
        
        return sign | (exponent << 10) | mantissa;
      }
    `;

    const blob = new Blob([workerCode], { type: 'application/javascript' });
    this.compressionWorker = new Worker(URL.createObjectURL(blob));
  }

  /**
   * Encode 1D data (simple arrays) - Use REST protocol
   */
  async encode1D(data: Float32Array, config: Partial<EncodingConfig> = {}): Promise<EncodedData> {
    const fullConfig: EncodingConfig = {
      dimensions: 1,
      protocol: 'rest',
      compression: 'gzip',
      precision: 'fp32',
      chunkSize: 1024,
      enableGPUOptimization: false,
      ...config
    };

    let processedData: ArrayBuffer = data.buffer.slice(0);

    // Apply compression if requested
    if (fullConfig.compression !== 'none') {
      processedData = await this.compressData(processedData, fullConfig.compression, fullConfig.precision);
    }

    const originalSize = data.buffer.byteLength;
    const compressedSize = processedData.byteLength;

    return {
      format: fullConfig.compression === 'none' ? 'raw' : 'compressed',
      dimensions: 1,
      shape: [data.length],
      data: processedData,
      metadata: {
        originalSize,
        compressedSize,
        compressionRatio: originalSize / compressedSize,
        encoding: `${fullConfig.precision}_${fullConfig.compression}`,
        checksum: await this.calculateChecksum(processedData)
      }
    };
  }

  /**
   * Encode 2D data (matrices, images) - Use gRPC protocol
   */
  async encode2D(data: Float32Array, width: number, height: number, config: Partial<EncodingConfig> = {}): Promise<EncodedData> {
    const fullConfig: EncodingConfig = {
      dimensions: 2,
      protocol: 'grpc',
      compression: 'brotli',
      precision: 'fp16',
      chunkSize: 4096,
      enableGPUOptimization: true,
      ...config
    };

    // Convert to image texture for efficient transfer
    const imageData = this.encodeAsTexture(data, width, height, fullConfig.precision);
    
    let processedData: ArrayBuffer;

    if (fullConfig.enableGPUOptimization && this.webGPUDevice) {
      // Use GPU for texture compression
      processedData = await this.compressTextureOnGPU(imageData);
    } else {
      // CPU fallback
      processedData = await this.compressImageData(imageData, fullConfig.compression);
    }

    const originalSize = data.buffer.byteLength;
    const compressedSize = processedData.byteLength;

    return {
      format: 'texture',
      dimensions: 2,
      shape: [width, height],
      data: processedData,
      metadata: {
        originalSize,
        compressedSize,
        compressionRatio: originalSize / compressedSize,
        encoding: `texture_${fullConfig.precision}_${fullConfig.compression}`,
        checksum: await this.calculateChecksum(processedData)
      }
    };
  }

  /**
   * Encode 3D data (volumes, vertex buffers) - Use QUIC protocol
   */
  async encode3D(data: Float32Array, width: number, height: number, depth: number, config: Partial<EncodingConfig> = {}): Promise<EncodedData> {
    const fullConfig: EncodingConfig = {
      dimensions: 3,
      protocol: 'quic',
      compression: 'custom',
      precision: 'fp16',
      chunkSize: 8192,
      enableGPUOptimization: true,
      ...config
    };

    let processedData: ArrayBuffer | GPUBuffer;

    if (fullConfig.enableGPUOptimization && this.webGPUDevice) {
      // Create GPU vertex buffer for maximum performance
      processedData = await this.createVertexBuffer(data, fullConfig);
    } else {
      // CPU-based 3D encoding
      processedData = await this.encode3DCPU(data, width, height, depth, fullConfig);
    }

    const originalSize = data.buffer.byteLength;
    const compressedSize = processedData instanceof GPUBuffer ? 
      originalSize * 0.6 : // Estimate for GPU buffer
      processedData.byteLength;

    return {
      format: processedData instanceof GPUBuffer ? 'vertex_buffer' : 'compressed',
      dimensions: 3,
      shape: [width, height, depth],
      data: processedData,
      metadata: {
        originalSize,
        compressedSize,
        compressionRatio: originalSize / compressedSize,
        encoding: `3d_${fullConfig.precision}_${fullConfig.compression}`,
        checksum: processedData instanceof GPUBuffer ? 
          'gpu_buffer_checksum' : 
          await this.calculateChecksum(processedData)
      }
    };
  }

  private encodeAsTexture(data: Float32Array, width: number, height: number, precision: string): ImageData {
    const imageData = new ImageData(width, height);
    const pixels = imageData.data;

    for (let i = 0; i < data.length && i * 4 < pixels.length; i++) {
      const value = data[i];
      
      if (precision === 'fp32') {
        // Encode float32 into RGBA channels
        const floatView = new Float32Array([value]);
        const intView = new Uint32Array(floatView.buffer);
        const bits = intView[0];

        pixels[i * 4] = (bits >>> 24) & 0xFF; // R
        pixels[i * 4 + 1] = (bits >>> 16) & 0xFF; // G
        pixels[i * 4 + 2] = (bits >>> 8) & 0xFF; // B
        pixels[i * 4 + 3] = bits & 0xFF; // A
      } else if (precision === 'fp16') {
        // Encode as 16-bit values in RG channels
        const halfValue = this.floatToHalf(value);
        pixels[i * 4] = (halfValue >>> 8) & 0xFF; // R
        pixels[i * 4 + 1] = halfValue & 0xFF; // G
        pixels[i * 4 + 2] = 0; // B
        pixels[i * 4 + 3] = 255; // A
      } else {
        // int8 encoding
        const intValue = Math.round(Math.max(0, Math.min(255, (value + 1) * 127.5)));
        pixels[i * 4] = intValue;
        pixels[i * 4 + 1] = intValue;
        pixels[i * 4 + 2] = intValue;
        pixels[i * 4 + 3] = 255;
      }
    }

    return imageData;
  }

  private async compressTextureOnGPU(imageData: ImageData): Promise<ArrayBuffer> {
    if (!this.webGPUDevice) throw new Error('WebGPU not available');

    // Create texture from image data
    const texture = this.webGPUDevice.createTexture({
      size: { width: imageData.width, height: imageData.height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC
    });

    // Upload image data to texture
    this.webGPUDevice.queue.writeTexture(
      { texture },
      imageData.data,
      { bytesPerRow: imageData.width * 4 },
      { width: imageData.width, height: imageData.height }
    );

    // Create compute shader for compression
    const compressionShader = this.webGPUDevice.createShaderModule({
      code: `
        @group(0) @binding(0) var inputTexture: texture_2d<f32>;
        @group(0) @binding(1) var<storage, read_write> outputBuffer: array<f32>;
        
        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let coords = vec2<i32>(i32(global_id.x), i32(global_id.y));
          let pixel = textureLoad(inputTexture, coords, 0);
          
          // Custom compression: pack RGBA into fewer bits
          let compressed = pack4x8unorm(pixel);
          let index = global_id.y * 1024u + global_id.x; // Assume max 1024 width
          
          if (index < arrayLength(&outputBuffer)) {
            outputBuffer[index] = bitcast<f32>(compressed);
          }
        }
      `
    });

    // Execute compression on GPU
    const outputBuffer = this.webGPUDevice.createBuffer({
      size: imageData.width * imageData.height * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const computePipeline = this.webGPUDevice.createComputePipeline({
      layout: 'auto',
      compute: { module: compressionShader, entryPoint: 'main' }
    });

    const bindGroup = this.webGPUDevice.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: { buffer: outputBuffer } }
      ]
    });

    const commandEncoder = this.webGPUDevice.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(
      Math.ceil(imageData.width / 8),
      Math.ceil(imageData.height / 8)
    );
    
    computePass.end();

    // Read back results
    const readBuffer = this.webGPUDevice.createBuffer({
      size: outputBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBuffer.size);
    this.webGPUDevice.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = readBuffer.getMappedRange().slice(0);
    readBuffer.unmap();

    // Cleanup
    texture.destroy();
    outputBuffer.destroy();
    readBuffer.destroy();

    return result;
  }

  private async createVertexBuffer(data: Float32Array, config: EncodingConfig): Promise<GPUBuffer> {
    if (!this.webGPUDevice) throw new Error('WebGPU not available');

    // Create optimized vertex buffer for 3D data
    const buffer = this.webGPUDevice.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });

    // Apply precision reduction if requested
    let bufferData: ArrayBufferView;
    
    switch (config.precision) {
      case 'fp16':
        bufferData = this.convertToFP16(data);
        break;
      case 'int8':
        bufferData = this.convertToInt8(data);
        break;
      case 'int4':
        bufferData = this.convertToInt4(data);
        break;
      default:
        bufferData = data;
    }

    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(bufferData.buffer));
    buffer.unmap();

    return buffer;
  }

  private async encode3DCPU(data: Float32Array, width: number, height: number, depth: number, config: EncodingConfig): Promise<ArrayBuffer> {
    // Spatial encoding for 3D data
    const encoded = new Float32Array(data.length);
    
    // Apply 3D wavelet compression or similar
    for (let z = 0; z < depth; z++) {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const index = z * width * height + y * width + x;
          
          // Simple 3D encoding: apply spatial filtering
          const neighbors = this.get3DNeighbors(data, x, y, z, width, height, depth);
          const average = neighbors.reduce((sum, val) => sum + val, 0) / neighbors.length;
          
          // Store difference from local average for better compression
          encoded[index] = data[index] - average;
        }
      }
    }

    // Apply final compression
    return this.compressData(encoded.buffer, config.compression, config.precision);
  }

  private get3DNeighbors(data: Float32Array, x: number, y: number, z: number, width: number, height: number, depth: number): number[] {
    const neighbors: number[] = [];
    
    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          const nz = z + dz;
          
          if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
            const index = nz * width * height + ny * width + nx;
            neighbors.push(data[index]);
          }
        }
      }
    }
    
    return neighbors;
  }

  /**
   * Stream encoded data using optimal protocol
   */
  async streamData(encodedData: EncodedData, endpoint: string, config: Partial<StreamConfig> = {}): Promise<void> {
    const streamConfig: StreamConfig = {
      protocol: 'rest',
      chunkSize: 8192,
      enableCompression: true,
      priority: 1,
      timeout: 30000,
      ...config
    };

    // Select optimal protocol based on data dimensions
    const protocol = encodedData.dimensions === 3 ? 'quic' : 
                    encodedData.dimensions === 2 ? 'grpc' : 'rest';

    const handler = this.protocolHandlers.get(protocol);
    if (!handler) {
      throw new Error(`Protocol ${protocol} not supported`);
    }

    console.log(`🚀 Streaming ${encodedData.format} data via ${protocol.toUpperCase()}`);

    if (encodedData.data instanceof ArrayBuffer) {
      // Stream in chunks for large data
      const chunks = this.chunkData(encodedData.data, streamConfig.chunkSize);
      
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const chunkEndpoint = `${endpoint}?chunk=${i}&total=${chunks.length}`;
        
        try {
          await handler(chunk, chunkEndpoint);
          console.log(`📦 Streamed chunk ${i + 1}/${chunks.length}`);
        } catch (error) {
          console.error(`❌ Failed to stream chunk ${i}:`, error);
          throw error;
        }
      }
    } else if (encodedData.data instanceof GPUBuffer) {
      // Handle GPU buffer streaming
      console.log('🎮 GPU buffer streaming not yet implemented');
    }
  }

  private chunkData(data: ArrayBuffer, chunkSize: number): ArrayBuffer[] {
    const chunks: ArrayBuffer[] = [];
    const uint8Array = new Uint8Array(data);
    
    for (let i = 0; i < uint8Array.length; i += chunkSize) {
      const chunk = uint8Array.slice(i, i + chunkSize);
      chunks.push(chunk.buffer);
    }
    
    return chunks;
  }

  // Utility methods
  private async compressData(data: ArrayBuffer, compression: string, precision: string): Promise<ArrayBuffer> {
    if (!this.compressionWorker) return data;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Compression timeout')), 10000);
      
      this.compressionWorker!.onmessage = (e) => {
        clearTimeout(timeout);
        resolve(e.data);
      };
      
      this.compressionWorker!.onerror = (error) => {
        clearTimeout(timeout);
        reject(error);
      };
      
      this.compressionWorker!.postMessage({ data, compression, precision });
    });
  }

  private async compressImageData(imageData: ImageData, compression: string): Promise<ArrayBuffer> {
    // Convert ImageData to ArrayBuffer
    const canvas = new OffscreenCanvas(imageData.width, imageData.height);
    const ctx = canvas.getContext('2d')!;
    ctx.putImageData(imageData, 0, 0);
    
    // Convert to compressed format (WebP, AVIF, etc.)
    const blob = await canvas.convertToBlob({ 
      type: compression === 'brotli' ? 'image/webp' : 'image/png',
      quality: 0.8 
    });
    
    return blob.arrayBuffer();
  }

  private convertToFP16(data: Float32Array): Uint16Array {
    const result = new Uint16Array(data.length);
    for (let i = 0; i < data.length; i++) {
      result[i] = this.floatToHalf(data[i]);
    }
    return result;
  }

  private convertToInt8(data: Float32Array): Int8Array {
    const result = new Int8Array(data.length);
    for (let i = 0; i < data.length; i++) {
      result[i] = Math.round(Math.max(-128, Math.min(127, data[i] * 127)));
    }
    return result;
  }

  private convertToInt4(data: Float32Array): Uint8Array {
    const result = new Uint8Array(Math.ceil(data.length / 2));
    for (let i = 0; i < data.length; i += 2) {
      const val1 = Math.round(Math.max(0, Math.min(15, (data[i] + 1) * 7.5))) & 0xF;
      const val2 = i + 1 < data.length ? 
        Math.round(Math.max(0, Math.min(15, (data[i + 1] + 1) * 7.5))) & 0xF : 0;
      result[Math.floor(i / 2)] = (val1 << 4) | val2;
    }
    return result;
  }

  private floatToHalf(val: number): number {
    const floatView = new Float32Array([val]);
    const intView = new Uint32Array(floatView.buffer);
    const bits = intView[0];
    
    const sign = (bits >>> 31) << 15;
    const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
    const mantissa = (bits >>> 13) & 0x3ff;
    
    if (exponent <= 0) return sign;
    if (exponent >= 31) return sign | 0x7c00 | (mantissa ? 0x200 : 0);
    
    return sign | (exponent << 10) | mantissa;
  }

  private async calculateChecksum(data: ArrayBuffer): Promise<string> {
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  /**
   * Decode data back to original format
   */
  async decode(encodedData: EncodedData): Promise<Float32Array> {
    switch (encodedData.format) {
      case 'raw':
        return new Float32Array(encodedData.data as ArrayBuffer);
        
      case 'texture':
        return this.decodeTexture(encodedData);
        
      case 'vertex_buffer':
        return this.decodeVertexBuffer(encodedData);
        
      case 'compressed':
        return this.decodeCompressed(encodedData);
        
      default:
        throw new Error(`Unsupported format: ${encodedData.format}`);
    }
  }

  private decodeTexture(encodedData: EncodedData): Float32Array {
    // Decode texture data back to float array
    const imageData = encodedData.data as ImageData;
    const result = new Float32Array(imageData.width * imageData.height);
    const pixels = imageData.data;

    for (let i = 0; i < result.length; i++) {
      const r = pixels[i * 4];
      const g = pixels[i * 4 + 1];
      const b = pixels[i * 4 + 2];
      const a = pixels[i * 4 + 3];

      // Reconstruct float from RGBA channels
      const bits = (r << 24) | (g << 16) | (b << 8) | a;
      const intView = new Uint32Array([bits]);
      const floatView = new Float32Array(intView.buffer);
      
      result[i] = floatView[0];
    }

    return result;
  }

  private async decodeVertexBuffer(encodedData: EncodedData): Promise<Float32Array> {
    const buffer = encodedData.data as GPUBuffer;
    
    if (!this.webGPUDevice) {
      throw new Error('WebGPU required for vertex buffer decoding');
    }

    // Read back from GPU buffer
    const readBuffer = this.webGPUDevice.createBuffer({
      size: buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const commandEncoder = this.webGPUDevice.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, buffer.size);
    this.webGPUDevice.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readBuffer.destroy();

    return data;
  }

  private async decodeCompressed(encodedData: EncodedData): Promise<Float32Array> {
    // Decompress using appropriate algorithm
    const data = encodedData.data as ArrayBuffer;
    
    // For now, assume data is already decompressed
    // In practice, would apply inverse of compression algorithm
    return new Float32Array(data);
  }

  destroy() {
    if (this.compressionWorker) {
      this.compressionWorker.terminate();
      this.compressionWorker = null;
    }
  }
}

// Singleton instance
export const multiDimensionalEncoder = new MultiDimensionalEncoder();