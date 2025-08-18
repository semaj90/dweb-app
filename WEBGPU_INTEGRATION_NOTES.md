# WebGPU Integration Notes
## GPU Orchestrator Scaffold - Client-Side GPU Processing

### Overview
This document explains how to integrate WebGPU for client-side GPU processing that complements the native CUDA worker system. The CUDA workers handle server-side heavy lifting, while WebGPU provides real-time client-side visualization and lightweight GPU compute.

## Architecture Flow

```
[CUDA Worker (Server)] → [Node.js Orchestrator] → [WebSocket] → [Browser WebGPU]
                                                               ↓
[Vertex Buffers] ← [GPU Compute Shaders] ← [Float32Array Data]
```

## WebGPU Integration Points

### 1. Receiving Vector Data from Server

```javascript
// orchestrator/webgpu_client.js
class WebGPUVectorProcessor {
    constructor() {
        this.device = null;
        this.context = null;
        this.pipeline = null;
        this.buffers = new Map();
    }

    async initialize() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        this.device = await adapter.requestDevice();
        
        const canvas = document.getElementById('gpu-canvas');
        this.context = canvas.getContext('webgpu');
        
        this.context.configure({
            device: this.device,
            format: 'bgra8unorm'
        });

        await this.createComputePipeline();
        console.log('✅ WebGPU initialized');
    }

    async createComputePipeline() {
        const shaderCode = `
            @group(0) @binding(0) var<storage, read> inputVectors: array<f32>;
            @group(0) @binding(1) var<storage, read_write> outputBuffer: array<f32>;
            @group(0) @binding(2) var<uniform> params: array<f32, 4>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index >= arrayLength(&inputVectors)) {
                    return;
                }
                
                // Example: SOM-style neighbor calculation
                let vector_size = u32(params[0]);
                let som_width = u32(params[1]);
                let som_height = u32(params[2]);
                
                // Compute distance to SOM nodes (simplified)
                outputBuffer[index] = inputVectors[index] * 0.95 + 0.1 * sin(f32(index) * 0.1);
            }
        `;

        const shaderModule = this.device.createShaderModule({ code: shaderCode });
        
        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
    }

    async processVectorStream(vectorData) {
        const inputBuffer = this.device.createBuffer({
            size: vectorData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        const outputBuffer = this.device.createBuffer({
            size: vectorData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        const paramBuffer = this.device.createBuffer({
            size: 16, // 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Upload data
        this.device.queue.writeBuffer(inputBuffer, 0, vectorData);
        this.device.queue.writeBuffer(paramBuffer, 0, new Float32Array([
            vectorData.length / 4, // vector_size
            8, // som_width
            8, // som_height
            0  // unused
        ]));

        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } },
                { binding: 2, resource: { buffer: paramBuffer } }
            ]
        });

        // Dispatch compute
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(vectorData.length / 64));
        passEncoder.end();

        this.device.queue.submit([commandEncoder.finish()]);

        return outputBuffer;
    }
}
```

### 2. WebSocket Integration with Node.js Orchestrator

```javascript
// client/gpu_websocket_client.js
class GPUWebSocketClient {
    constructor(webgpuProcessor) {
        this.ws = null;
        this.webgpu = webgpuProcessor;
        this.vectorQueue = [];
    }

    connect(url = 'ws://localhost:8095') {
        this.ws = new WebSocket(url);
        
        this.ws.onopen = () => {
            console.log('🔌 Connected to GPU orchestrator');
            this.ws.send(JSON.stringify({
                type: 'subscribe',
                channels: ['vector_updates', 'job_results']
            }));
        };

        this.ws.onmessage = async (event) => {
            try {
                const data = JSON.parse(event.data);
                await this.handleMessage(data);
            } catch (error) {
                // Handle binary data (vector streams)
                const vectorData = new Float32Array(event.data);
                await this.handleVectorStream(vectorData);
            }
        };
    }

    async handleMessage(message) {
        switch (message.type) {
            case 'job_completed':
                if (message.result && message.result.vector) {
                    const vectorData = new Float32Array(message.result.vector);
                    await this.handleVectorStream(vectorData);
                }
                break;
                
            case 'vector_batch':
                // Handle batch of vectors from auto-indexing
                for (const vector of message.vectors) {
                    const vectorData = new Float32Array(vector);
                    this.vectorQueue.push(vectorData);
                }
                await this.processVectorQueue();
                break;
        }
    }

    async handleVectorStream(vectorData) {
        // Process with WebGPU
        const processedBuffer = await this.webgpu.processVectorStream(vectorData);
        
        // Read back results for visualization
        const readBuffer = this.webgpu.device.createBuffer({
            size: vectorData.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const commandEncoder = this.webgpu.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(processedBuffer, 0, readBuffer, 0, vectorData.byteLength);
        this.webgpu.device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const resultData = new Float32Array(readBuffer.getMappedRange());
        
        // Update visualization
        this.updateVisualization(resultData);
        
        readBuffer.unmap();
    }

    updateVisualization(vectorData) {
        // Example: Update SOM grid visualization
        const canvas = document.getElementById('som-visualization');
        if (canvas) {
            this.renderSOMGrid(canvas, vectorData);
        }
    }

    renderSOMGrid(canvas, data) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // Simple grid visualization
        const gridSize = Math.sqrt(data.length);
        const cellWidth = width / gridSize;
        const cellHeight = height / gridSize;
        
        for (let i = 0; i < data.length; i++) {
            const x = (i % gridSize) * cellWidth;
            const y = Math.floor(i / gridSize) * cellHeight;
            const intensity = Math.abs(data[i]) * 255;
            
            ctx.fillStyle = `rgb(${intensity}, ${intensity/2}, ${255-intensity})`;
            ctx.fillRect(x, y, cellWidth, cellHeight);
        }
    }
}
```

### 3. Vertex Buffer Management for 3D Visualization

```javascript
// client/vertex_buffer_manager.js
class VertexBufferManager {
    constructor(device) {
        this.device = device;
        this.vertexBuffers = new Map();
        this.indexBuffers = new Map();
    }

    createVertexBuffer(name, vertices) {
        const buffer = this.device.createBuffer({
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        
        this.device.queue.writeBuffer(buffer, 0, vertices);
        this.vertexBuffers.set(name, buffer);
        
        return buffer;
    }

    updateVertexBuffer(name, newVertices) {
        const buffer = this.vertexBuffers.get(name);
        if (buffer) {
            this.device.queue.writeBuffer(buffer, 0, newVertices);
        }
    }

    createVectorPointCloud(vectors) {
        // Convert vectors to 3D points for visualization
        const vertices = new Float32Array(vectors.length * 6); // position + color
        
        for (let i = 0; i < vectors.length; i += 3) {
            const baseIdx = (i / 3) * 6;
            
            // Position (use first 3 components, or derive from vector)
            vertices[baseIdx + 0] = vectors[i] || 0;
            vertices[baseIdx + 1] = vectors[i + 1] || 0;
            vertices[baseIdx + 2] = vectors[i + 2] || 0;
            
            // Color (based on vector magnitude)
            const magnitude = Math.sqrt(
                vertices[baseIdx + 0] ** 2 + 
                vertices[baseIdx + 1] ** 2 + 
                vertices[baseIdx + 2] ** 2
            );
            
            vertices[baseIdx + 3] = magnitude; // R
            vertices[baseIdx + 4] = magnitude * 0.7; // G
            vertices[baseIdx + 5] = magnitude * 0.3; // B
        }
        
        return this.createVertexBuffer('point_cloud', vertices);
    }
}
```

### 4. Integration with Node.js WebSocket Server

```javascript
// orchestrator/websocket_server.js
const WebSocket = require('ws');

class GPUWebSocketServer {
    constructor(port = 8095) {
        this.wss = new WebSocket.Server({ port });
        this.clients = new Set();
        this.setupEventHandlers();
        console.log(`🌐 WebSocket server listening on port ${port}`);
    }

    setupEventHandlers() {
        this.wss.on('connection', (ws) => {
            this.clients.add(ws);
            console.log(`📱 Client connected (${this.clients.size} total)`);

            ws.on('message', (data) => {
                try {
                    const message = JSON.parse(data);
                    this.handleClientMessage(ws, message);
                } catch (error) {
                    console.error('Invalid message:', error.message);
                }
            });

            ws.on('close', () => {
                this.clients.delete(ws);
                console.log(`📱 Client disconnected (${this.clients.size} total)`);
            });
        });
    }

    handleClientMessage(ws, message) {
        switch (message.type) {
            case 'subscribe':
                ws.channels = message.channels || [];
                break;
                
            case 'request_vectors':
                this.sendRecentVectors(ws);
                break;
        }
    }

    broadcastJobResult(jobResult) {
        const message = JSON.stringify({
            type: 'job_completed',
            result: jobResult,
            timestamp: new Date().toISOString()
        });

        this.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN && 
                client.channels?.includes('job_results')) {
                client.send(message);
            }
        });
    }

    broadcastVectorBatch(vectors) {
        const message = JSON.stringify({
            type: 'vector_batch',
            vectors: vectors,
            count: vectors.length,
            timestamp: new Date().toISOString()
        });

        this.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN && 
                client.channels?.includes('vector_updates')) {
                client.send(message);
            }
        });
    }

    sendBinaryVector(vectorData) {
        const buffer = Buffer.from(vectorData.buffer);
        
        this.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(buffer);
            }
        });
    }
}

module.exports = GPUWebSocketServer;
```

### 5. HTML Integration Example

```html
<!-- client/gpu_visualization.html -->
<!DOCTYPE html>
<html>
<head>
    <title>GPU Orchestrator - WebGPU Visualization</title>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; background: #1a1a1a; color: white; }
        .container { display: flex; height: 100vh; }
        .sidebar { width: 300px; padding: 20px; background: #2a2a2a; }
        .main { flex: 1; position: relative; }
        canvas { width: 100%; height: 100%; display: block; }
        .status { margin-bottom: 10px; padding: 10px; background: #3a3a3a; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>GPU Orchestrator</h2>
            <div class="status" id="connection-status">Disconnected</div>
            <div class="status" id="vector-count">Vectors: 0</div>
            <div class="status" id="gpu-status">WebGPU: Not initialized</div>
            
            <button onclick="initializeWebGPU()">Initialize WebGPU</button>
            <button onclick="connectWebSocket()">Connect WebSocket</button>
            <button onclick="requestVectors()">Request Vectors</button>
        </div>
        
        <div class="main">
            <canvas id="gpu-canvas" width="800" height="600"></canvas>
            <canvas id="som-visualization" width="400" height="400" 
                    style="position: absolute; top: 10px; right: 10px; border: 2px solid #555;"></canvas>
        </div>
    </div>

    <script type="module">
        let webgpuProcessor = null;
        let wsClient = null;

        window.initializeWebGPU = async () => {
            try {
                webgpuProcessor = new WebGPUVectorProcessor();
                await webgpuProcessor.initialize();
                document.getElementById('gpu-status').textContent = 'WebGPU: Ready';
            } catch (error) {
                document.getElementById('gpu-status').textContent = `WebGPU: Error - ${error.message}`;
            }
        };

        window.connectWebSocket = () => {
            if (!webgpuProcessor) {
                alert('Initialize WebGPU first');
                return;
            }
            
            wsClient = new GPUWebSocketClient(webgpuProcessor);
            wsClient.connect();
            
            // Update connection status
            setTimeout(() => {
                document.getElementById('connection-status').textContent = 'Connected';
            }, 1000);
        };

        window.requestVectors = () => {
            if (wsClient && wsClient.ws) {
                wsClient.ws.send(JSON.stringify({ type: 'request_vectors' }));
            }
        };

        // Auto-initialize on page load
        document.addEventListener('DOMContentLoaded', async () => {
            if (navigator.gpu) {
                await initializeWebGPU();
                connectWebSocket();
            } else {
                document.getElementById('gpu-status').textContent = 'WebGPU: Not supported';
            }
        });
    </script>
</body>
</html>
```

## Performance Considerations

### Memory Management
- Use `GPUBuffer.destroy()` when buffers are no longer needed
- Implement buffer pooling for frequently updated data
- Monitor GPU memory usage via `adapter.limits`

### Workgroup Optimization
- Use appropriate workgroup sizes (multiples of 32/64)
- Balance between occupancy and memory usage
- Profile different workgroup sizes for your use case

### Data Transfer Optimization
- Batch multiple vector updates into single transfers
- Use persistent buffers for frequently updated data
- Consider using mapped buffers for streaming data

## Integration with Existing System

1. **CUDA Worker Output**: Modify `worker_process.js` to send results to WebSocket server
2. **Redis Integration**: Store visualization state in Redis for persistence
3. **XState Integration**: Add WebGPU states to the idle/processing state machine
4. **Auto-indexing**: Trigger WebGPU visualization updates during auto-index operations

## Next Steps for Extension

1. **NATS Integration**: Stream vectors through NATS for lower latency
2. **Kratos Security**: Add authentication to WebSocket connections
3. **ELK Monitoring**: Log WebGPU performance metrics
4. **Multi-GPU**: Detect and utilize multiple GPUs in the system
5. **WebGL Fallback**: Provide WebGL implementation for older browsers

This WebGPU integration provides real-time visualization of your CUDA worker results while maintaining the native Windows performance advantage for heavy computation.