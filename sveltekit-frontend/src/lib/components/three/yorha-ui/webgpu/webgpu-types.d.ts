/**
 * Basic WebGPU type declarations for YoRHa WebGPU Math
 * These are stub definitions to avoid TypeScript errors when WebGPU is not available
 */

declare global {
  interface Navigator {
    gpu?: GPU;
  }

  interface GPU {
    requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
  }

  interface GPUAdapter {
    requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
  }

  interface GPUDevice {
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
    createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
    createCommandEncoder(): GPUCommandEncoder;
    queue: GPUQueue;
  }

  interface GPUBuffer {
    mapAsync(mode: GPUMapModeFlags, offset?: number, size?: number): Promise<void>;
    getMappedRange(offset?: number, size?: number): ArrayBuffer;
    unmap(): void;
    destroy(): void;
  }

  interface GPUComputePipeline {}
  interface GPUShaderModule {}
  interface GPUCommandEncoder {}
  interface GPUQueue {}

  interface GPURequestAdapterOptions {}
  interface GPUDeviceDescriptor {}
  interface GPUBufferDescriptor {
    size: number;
    usage: GPUBufferUsageFlags;
  }
  interface GPUShaderModuleDescriptor {}
  interface GPUComputePipelineDescriptor {}

  type GPUMapModeFlags = number;
  type GPUBufferUsageFlags = number;

  const GPUBufferUsage: {
    STORAGE: number;
    COPY_SRC: number;
    COPY_DST: number;
  };

  const GPUMapMode: {
    READ: number;
  };
}

