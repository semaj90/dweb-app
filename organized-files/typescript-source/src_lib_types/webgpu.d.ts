// WebGPU Type Definitions
// Provides type definitions for WebGPU API

declare global {
  const GPUBufferUsage: {
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

  const GPUShaderStage: {
    VERTEX: number;
    FRAGMENT: number;
    COMPUTE: number;
  };

  const GPUMapMode: {
    READ: number;
    WRITE: number;
  };

  const GPUTextureUsage: {
    COPY_SRC: number;
    COPY_DST: number;
    TEXTURE_BINDING: number;
    STORAGE_BINDING: number;
    RENDER_ATTACHMENT: number;
  };

  interface GPUDevice {
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
    createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
    createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
    createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
    createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
    createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
    queue: GPUQueue;
  }

  interface GPUBuffer {
    size: number;
    usage: number;
    mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
    getMappedRange(offset?: number, size?: number): ArrayBuffer;
    unmap(): void;
    destroy(): void;
  }

  interface GPUBindGroup {
    label?: string;
  }

  interface GPUBindGroupLayout {
    label?: string;
  }

  interface GPUPipelineLayout {
    label?: string;
  }

  interface GPUComputePipeline {
    label?: string;
    getBindGroupLayout(index: number): GPUBindGroupLayout;
  }

  interface GPUShaderModule {
    label?: string;
  }

  interface GPUCommandEncoder {
    beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
    copyBufferToBuffer(
      source: GPUBuffer,
      sourceOffset: number,
      destination: GPUBuffer,
      destinationOffset: number,
      size: number
    ): void;
    finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer;
  }

  interface GPUComputePassEncoder {
    setPipeline(pipeline: GPUComputePipeline): void;
    setBindGroup(index: number, bindGroup: GPUBindGroup): void;
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
    end(): void;
  }

  interface GPUQueue {
    submit(commandBuffers: GPUCommandBuffer[]): void;
    writeBuffer(
      buffer: GPUBuffer,
      bufferOffset: number,
      data: BufferSource,
      dataOffset?: number,
      size?: number
    ): void;
  }

  interface GPUCommandBuffer {
    label?: string;
  }

  interface GPUBufferDescriptor {
    label?: string;
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }

  interface GPUBindGroupDescriptor {
    label?: string;
    layout: GPUBindGroupLayout;
    entries: GPUBindGroupEntry[];
  }

  interface GPUBindGroupEntry {
    binding: number;
    resource: GPUBindingResource;
  }

  type GPUBindingResource = 
    | { buffer: GPUBuffer; offset?: number; size?: number }
    | GPUSampler
    | GPUTextureView
    | GPUExternalTexture;

  interface GPUSampler {
    label?: string;
  }

  interface GPUTextureView {
    label?: string;
  }

  interface GPUExternalTexture {
    label?: string;
  }

  interface GPUBindGroupLayoutDescriptor {
    label?: string;
    entries: GPUBindGroupLayoutEntry[];
  }

  interface GPUBindGroupLayoutEntry {
    binding: number;
    visibility: number;
    buffer?: GPUBufferBindingLayout;
    sampler?: GPUSamplerBindingLayout;
    texture?: GPUTextureBindingLayout;
    storageTexture?: GPUStorageTextureBindingLayout;
    externalTexture?: GPUExternalTextureBindingLayout;
  }

  interface GPUBufferBindingLayout {
    type?: "uniform" | "storage" | "read-only-storage";
    hasDynamicOffset?: boolean;
    minBindingSize?: number;
  }

  interface GPUSamplerBindingLayout {
    type?: "filtering" | "non-filtering" | "comparison";
  }

  interface GPUTextureBindingLayout {
    sampleType?: "float" | "unfilterable-float" | "depth" | "sint" | "uint";
    viewDimension?: GPUTextureViewDimension;
    multisampled?: boolean;
  }

  interface GPUStorageTextureBindingLayout {
    access?: "write-only" | "read-only" | "read-write";
    format: GPUTextureFormat;
    viewDimension?: GPUTextureViewDimension;
  }

  interface GPUExternalTextureBindingLayout {}

  type GPUTextureViewDimension = "1d" | "2d" | "2d-array" | "cube" | "cube-array" | "3d";
  type GPUTextureFormat = string;

  interface GPUPipelineLayoutDescriptor {
    label?: string;
    bindGroupLayouts: GPUBindGroupLayout[];
  }

  interface GPUComputePipelineDescriptor {
    label?: string;
    layout: GPUPipelineLayout | "auto";
    compute: GPUProgrammableStage;
  }

  interface GPUProgrammableStage {
    module: GPUShaderModule;
    entryPoint: string;
    constants?: Record<string, number>;
  }

  interface GPUShaderModuleDescriptor {
    label?: string;
    code: string;
    sourceMap?: object;
    hints?: Record<string, GPUShaderModuleCompilationHint>;
  }

  interface GPUShaderModuleCompilationHint {
    layout?: GPUPipelineLayout | "auto";
  }

  interface GPUCommandEncoderDescriptor {
    label?: string;
  }

  interface GPUComputePassDescriptor {
    label?: string;
    timestampWrites?: GPUComputePassTimestampWrites;
  }

  interface GPUComputePassTimestampWrites {
    querySet: GPUQuerySet;
    beginningOfPassWriteIndex?: number;
    endOfPassWriteIndex?: number;
  }

  interface GPUQuerySet {
    label?: string;
    type: "occlusion" | "timestamp";
    count: number;
  }

  interface GPUCommandBufferDescriptor {
    label?: string;
  }

  interface Navigator {
    gpu?: GPU;
  }

  interface GPU {
    requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
  }

  interface GPURequestAdapterOptions {
    powerPreference?: "low-power" | "high-performance";
    forceFallbackAdapter?: boolean;
  }

  interface GPUAdapter {
    requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
    features: GPUSupportedFeatures;
    limits: GPUSupportedLimits;
  }

  interface GPUDeviceDescriptor {
    label?: string;
    requiredFeatures?: GPUFeatureName[];
    requiredLimits?: Record<string, number>;
  }

  type GPUFeatureName = string;

  interface GPUSupportedFeatures {
    has(feature: GPUFeatureName): boolean;
    size: number;
    entries(): IterableIterator<GPUFeatureName>;
    keys(): IterableIterator<GPUFeatureName>;
    values(): IterableIterator<GPUFeatureName>;
    forEach(callbackfn: (value: GPUFeatureName, value2: GPUFeatureName, set: GPUSupportedFeatures) => void): void;
  }

  interface GPUSupportedLimits {
    maxTextureDimension1D?: number;
    maxTextureDimension2D?: number;
    maxTextureDimension3D?: number;
    maxTextureArrayLayers?: number;
    maxBindGroups?: number;
    maxBindingsPerBindGroup?: number;
    maxDynamicUniformBuffersPerPipelineLayout?: number;
    maxDynamicStorageBuffersPerPipelineLayout?: number;
    maxSampledTexturesPerShaderStage?: number;
    maxSamplersPerShaderStage?: number;
    maxStorageBuffersPerShaderStage?: number;
    maxStorageTexturesPerShaderStage?: number;
    maxUniformBuffersPerShaderStage?: number;
    maxUniformBufferBindingSize?: number;
    maxStorageBufferBindingSize?: number;
    minUniformBufferOffsetAlignment?: number;
    minStorageBufferOffsetAlignment?: number;
    maxVertexBuffers?: number;
    maxBufferSize?: number;
    maxVertexAttributes?: number;
    maxVertexBufferArrayStride?: number;
    maxInterStageShaderComponents?: number;
    maxInterStageShaderVariables?: number;
    maxColorAttachments?: number;
    maxColorAttachmentBytesPerSample?: number;
    maxComputeWorkgroupStorageSize?: number;
    maxComputeInvocationsPerWorkgroup?: number;
    maxComputeWorkgroupSizeX?: number;
    maxComputeWorkgroupSizeY?: number;
    maxComputeWorkgroupSizeZ?: number;
    maxComputeWorkgroupsPerDimension?: number;
  }
}

// Export empty object to make this a module
export {};