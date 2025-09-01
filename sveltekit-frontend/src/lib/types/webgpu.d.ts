// WebGPU type definitions for Legal AI Platform
// These are placeholders for WebGPU types when not available

declare global {
  interface GPUAdapter {
    requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
    features: GPUSupportedFeatures;
    limits: GPUSupportedLimits;
    isFallbackAdapter: boolean;
  }

  interface GPUDevice {
    features: GPUSupportedFeatures;
    limits: GPUSupportedLimits;
    queue: GPUQueue;
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
    createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
    createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
    createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
    createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
    destroy(): void;
  }

  interface GPUQueue {
    submit(commandBuffers: GPUCommandBuffer[]): void;
    writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: BufferSource): void;
  }

  interface GPUBuffer {
    size: number;
    usage: GPUBufferUsageFlags;
    mapState: GPUBufferMapState;
    mapAsync(mode: GPUMapModeFlags, offset?: number, size?: number): Promise<void>;
    getMappedRange(offset?: number, size?: number): ArrayBuffer;
    unmap(): void;
    destroy(): void;
  }

  interface GPUShaderModule {
    getCompilationInfo(): Promise<GPUCompilationInfo>;
  }

  interface GPUComputePipeline {
    getBindGroupLayout(index: number): GPUBindGroupLayout;
  }

  interface GPUBindGroupLayout {}

  interface GPUBindGroup {}

  interface GPUCommandEncoder {
    beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
    finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer;
  }

  interface GPUCommandBuffer {}

  interface GPUComputePassEncoder {
    setPipeline(pipeline: GPUComputePipeline): void;
    setBindGroup(index: number, bindGroup: GPUBindGroup): void;
    dispatchWorkgroups(workgroupCountX: number, workgroupCountY?: number, workgroupCountZ?: number): void;
    end(): void;
  }

  // Descriptor types
  interface GPUDeviceDescriptor {
    requiredFeatures?: GPUFeatureName[];
    requiredLimits?: Record<string, number>;
  }

  interface GPUBufferDescriptor {
    size: number;
    usage: GPUBufferUsageFlags;
    mappedAtCreation?: boolean;
  }

  interface GPUShaderModuleDescriptor {
    code: string;
  }

  interface GPUComputePipelineDescriptor {
    compute: GPUProgrammableStage;
    layout?: GPUPipelineLayout | 'auto';
  }

  interface GPUProgrammableStage {
    module: GPUShaderModule;
    entryPoint: string;
  }

  interface GPUBindGroupLayoutDescriptor {
    entries: GPUBindGroupLayoutEntry[];
  }

  interface GPUBindGroupLayoutEntry {
    binding: number;
    visibility: GPUShaderStageFlags;
    buffer?: GPUBufferBindingLayout;
  }

  interface GPUBufferBindingLayout {
    type?: GPUBufferBindingType;
    hasDynamicOffset?: boolean;
  }

  interface GPUBindGroupDescriptor {
    layout: GPUBindGroupLayout;
    entries: GPUBindGroupEntry[];
  }

  interface GPUBindGroupEntry {
    binding: number;
    resource: GPUBindingResource;
  }

  interface GPUComputePassDescriptor {
    label?: string;
  }

  interface GPUCommandEncoderDescriptor {
    label?: string;
  }

  interface GPUCommandBufferDescriptor {
    label?: string;
  }

  interface GPUCompilationInfo {
    messages: GPUCompilationMessage[];
  }

  interface GPUCompilationMessage {
    message: string;
    type: GPUCompilationMessageType;
    lineNum: number;
    linePos: number;
  }

  // Type aliases
  type GPUSupportedFeatures = Set<GPUFeatureName>;
  type GPUSupportedLimits = Record<string, number>;
  type GPUBufferUsageFlags = number;
  type GPUMapModeFlags = number;
  type GPUShaderStageFlags = number;
  type GPUBufferMapState = 'unmapped' | 'pending' | 'mapped';
  type GPUFeatureName = string;
  type GPUBufferBindingType = 'uniform' | 'storage' | 'read-only-storage';
  type GPUBindingResource = GPUBuffer;
  type GPUCompilationMessageType = 'error' | 'warning' | 'info';
  type GPUPipelineLayout = any;

  // Global navigator extension
  interface Navigator {
    gpu?: {
      requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
    };
  }

  interface GPURequestAdapterOptions {
    powerPreference?: GPUPowerPreference;
    forceFallbackAdapter?: boolean;
  }

  type GPUPowerPreference = 'low-power' | 'high-performance';

  // WebGPU Constants
  const GPUBufferUsage: {
    readonly MAP_READ: 0x0001;
    readonly MAP_WRITE: 0x0002;
    readonly COPY_SRC: 0x0004;
    readonly COPY_DST: 0x0008;
    readonly INDEX: 0x0010;
    readonly VERTEX: 0x0020;
    readonly UNIFORM: 0x0040;
    readonly STORAGE: 0x0080;
    readonly INDIRECT: 0x0100;
    readonly QUERY_RESOLVE: 0x0200;
  };

  const GPUShaderStage: {
    readonly VERTEX: 0x1;
    readonly FRAGMENT: 0x2;
    readonly COMPUTE: 0x4;
  };

  const GPUMapMode: {
    readonly READ: 0x0001;
    readonly WRITE: 0x0002;
  };
}

