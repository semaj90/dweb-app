// WebGPU type declarations for TypeScript compatibility

declare global {
  interface Navigator {
    gpu?: GPU;
  }

  interface GPU {
    requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
  }

  interface GPUAdapter {
    features: Set<string>;
    limits: GPULimits;
    requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
  }

  interface GPUDevice {
    features: Set<string>;
    limits: GPULimits;
    lost: Promise<GPUDeviceLostInfo>;
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
    createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
    createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline;
    createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
    createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
    createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
    queue: GPUQueue;
  }

  interface GPUBuffer {
    size: number;
    usage: number;
    mapState: string;
    mapAsync(mode: number, offset?: number, size?: number): Promise<any>;
    getMappedRange(offset?: number, size?: number): ArrayBuffer;
    unmap(): void;
    destroy(): void;
  }

  interface GPUQueue {
    submit(commandBuffers: GPUCommandBuffer[]): void;
    writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: BufferSource, dataOffset?: number, size?: number): void;
  }

  interface GPUCommandEncoder {
    beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
    copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
    finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer;
  }

  interface GPUComputePassEncoder {
    setPipeline(pipeline: GPUComputePipeline): void;
    setBindGroup(index: number, bindGroup: GPUBindGroup): void;
    dispatchWorkgroups(workgroupCountX: number, workgroupCountY?: number, workgroupCountZ?: number): void;
    end(): void;
  }

  // Basic interface definitions
  interface GPURequestAdapterOptions {}
  interface GPUDeviceDescriptor {}
  interface GPULimits {
    maxComputeWorkgroupsPerDimension?: number;
    maxComputeInvocationsPerWorkgroup?: number;
    maxStorageBufferBindingSize?: number;
  }
  interface GPUDeviceLostInfo {
    reason: string;
    message: string;
  }
  interface GPUBufferDescriptor {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }
  interface GPUShaderModuleDescriptor {
    code: string;
  }
  interface GPUComputePipelineDescriptor {
    compute: GPUProgrammableStage;
    layout: GPUPipelineLayout | string;
  }
  interface GPUBindGroupLayoutDescriptor {
    entries: GPUBindGroupLayoutEntry[];
  }
  interface GPUBindGroupDescriptor {
    layout: GPUBindGroupLayout;
    entries: GPUBindGroupEntry[];
  }
  interface GPUCommandEncoderDescriptor {}
  interface GPUComputePassDescriptor {}
  interface GPUCommandBufferDescriptor {}
  interface GPUProgrammableStage {
    module: GPUShaderModule;
    entryPoint: string;
  }
  interface GPUPipelineLayout {}
  interface GPUBindGroupLayoutEntry {}
  interface GPUBindGroupEntry {}
  interface GPUShaderModule {}
  interface GPUComputePipeline {
    getBindGroupLayout(index: number): GPUBindGroupLayout;
  }
  interface GPUBindGroupLayout {}
  interface GPUBindGroup {}
  interface GPUCommandBuffer {}
  
  // Canvas and Render Pipeline interfaces
  interface GPUCanvasContext {
    canvas: HTMLCanvasElement;
    configure(descriptor: GPUCanvasConfiguration): void;
    unconfigure(): void;
    getCurrentTexture(): GPUTexture;
  }
  
  interface GPUCanvasConfiguration {
    device: GPUDevice;
    format: GPUTextureFormat;
    usage?: GPUTextureUsageFlags;
    alphaMode?: GPUCanvasAlphaMode;
  }
  
  interface GPURenderPipeline {
    label: string | null;
    getBindGroupLayout(index: number): GPUBindGroupLayout;
  }
  
  interface GPURenderPipelineDescriptor {
    vertex: GPUVertexState;
    primitive?: GPUPrimitiveState;
    depthStencil?: GPUDepthStencilState;
    multisample?: GPUMultisampleState;
    fragment?: GPUFragmentState;
    layout: GPUPipelineLayout | string;
  }
  
  interface GPUTexture {
    createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView;
    destroy(): void;
  }
  
  interface GPUTextureView {}
  interface GPUTextureViewDescriptor {}
  interface GPUVertexState {}
  interface GPUPrimitiveState {}
  interface GPUDepthStencilState {}
  interface GPUMultisampleState {}
  interface GPUFragmentState {}
  
  type GPUTextureFormat = string;
  type GPUTextureUsageFlags = number;
  type GPUCanvasAlphaMode = "opaque" | "premultiplied";

  // Constants
  declare const GPUBufferUsage: {
    MAP_READ: number;
    MAP_WRITE: number;
    COPY_SRC: number;
    COPY_DST: number;
    INDEX: number;
    VERTEX: number;
    UNIFORM: number;
    STORAGE: number;
    INDIRECT: number;
    QUERY_RESOLVE: number;
  };

  declare const GPUMapMode: {
    READ: number;
    WRITE: number;
  };

  type GPUFeatureName = string;
}

export {};