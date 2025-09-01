// Project-level ambient declarations for YoRHa Legal AI Platform
// Svelte 5 Runes and TypeScript compatibility

// Svelte 5 Runes - Global declarations
declare global {
  // Rune functions
  function $state<T>(initial: T): T;
  function $state<T>(): T | undefined;
  function $derived<T>(fn: () => T): T;
  function $derived<T>(value: T): T;
  function $props<T extends Record<string, any> = Record<string, any>>(): T;
  function $bindable<T>(value?: T): T;
  function $inspect(...values: any[]): void;
  function $effect(fn: () => void | (() => void)): void;
  
  // Rune variables (alternative syntax)
  const $state: any;
  const $derived: any;
  const $props: any;
  const $bindable: any;
  const $inspect: any;
  const $effect: any;
}

// Import Meta
interface ImportMeta {
  url: string;
  env: {
    MODE: string;
    BASE_URL: string;
    PROD: boolean;
    DEV: boolean;
    SSR: boolean;
    [key: string]: any;
  };
  hot?: {
    accept: (cb?: (mod: any) => void) => void;
    dispose: (cb: () => void) => void;
    decline: () => void;
    invalidate: () => void;
    on: (event: string, cb: (...args: any[]) => void) => void;
  };
  glob: (pattern: string) => Record<string, () => Promise<any>>;
  globEager: (pattern: string) => Record<string, any>;
}

// WebGPU API Types
interface Navigator {
  gpu?: GPU;
}

interface GPU {
  requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
}

interface GPUAdapter {
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
  limits: GPUSupportedLimits;
  features: GPUSupportedFeatures;
}

interface GPUDevice {
  queue: GPUQueue;
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createTexture(descriptor: GPUTextureDescriptor): GPUTexture;
  createSampler(descriptor?: GPUSamplerDescriptor): GPUSampler;
  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
  createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline;
  createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
  destroy(): void;
}

interface GPUQueue {
  submit(commandBuffers: GPUCommandBuffer[]): void;
  writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: BufferSource, dataOffset?: number, size?: number): void;
  writeTexture(destination: GPUImageCopyTexture, data: BufferSource, dataLayout: GPUImageDataLayout, size: GPUExtent3D): void;
}

interface GPUBuffer {
  size: number;
  usage: number;
  mapState: string;
  label?: string;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

interface GPUTexture {
  createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView;
  destroy(): void;
  width: number;
  height: number;
  depthOrArrayLayers: number;
  mipLevelCount: number;
  sampleCount: number;
  dimension: string;
  format: string;
  usage: number;
}

interface GPUSampler {}
interface GPUTextureView {}
interface GPUBindGroupLayout {}
interface GPUPipelineLayout {}
interface GPUBindGroup {}
interface GPUShaderModule {}
interface GPUComputePipeline {}
interface GPURenderPipeline {}
interface GPUCommandEncoder {
  beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
  beginRenderPass(descriptor: GPURenderPassDescriptor): GPURenderPassEncoder;
  copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
  copyBufferToTexture(source: GPUImageCopyBuffer, destination: GPUImageCopyTexture, copySize: GPUExtent3D): void;
  copyTextureToBuffer(source: GPUImageCopyTexture, destination: GPUImageCopyBuffer, copySize: GPUExtent3D): void;
  copyTextureToTexture(source: GPUImageCopyTexture, destination: GPUImageCopyTexture, copySize: GPUExtent3D): void;
  finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer;
}

interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

interface GPURenderPassEncoder {
  setPipeline(pipeline: GPURenderPipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
  setVertexBuffer(slot: number, buffer: GPUBuffer, offset?: number, size?: number): void;
  setIndexBuffer(buffer: GPUBuffer, indexFormat: string, offset?: number, size?: number): void;
  draw(vertexCount: number, instanceCount?: number, firstVertex?: number, firstInstance?: number): void;
  drawIndexed(indexCount: number, instanceCount?: number, firstIndex?: number, baseVertex?: number, firstInstance?: number): void;
  end(): void;
}

interface GPUCommandBuffer {}

// Type stubs for GPU descriptor interfaces
interface GPURequestAdapterOptions {
  powerPreference?: 'low-power' | 'high-performance';
  forceFallbackAdapter?: boolean;
}

interface GPUDeviceDescriptor {
  requiredFeatures?: string[];
  requiredLimits?: Record<string, number>;
  label?: string;
}

interface GPUBufferDescriptor {
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
  label?: string;
}

interface GPUTextureDescriptor {
  size: GPUExtent3D;
  mipLevelCount?: number;
  sampleCount?: number;
  dimension?: string;
  format: string;
  usage: number;
  label?: string;
}

interface GPUSamplerDescriptor {
  addressModeU?: string;
  addressModeV?: string;
  addressModeW?: string;
  magFilter?: string;
  minFilter?: string;
  mipmapFilter?: string;
  lodMinClamp?: number;
  lodMaxClamp?: number;
  compare?: string;
  maxAnisotropy?: number;
  label?: string;
}

interface GPUBindGroupLayoutDescriptor {
  entries: GPUBindGroupLayoutEntry[];
  label?: string;
}

interface GPUBindGroupLayoutEntry {
  binding: number;
  visibility: number;
  buffer?: GPUBufferBindingLayout;
  sampler?: GPUSamplerBindingLayout;
  texture?: GPUTextureBindingLayout;
  storageTexture?: GPUStorageTextureBindingLayout;
}

interface GPUBufferBindingLayout {
  type?: string;
  hasDynamicOffset?: boolean;
  minBindingSize?: number;
}

interface GPUSamplerBindingLayout {
  type?: string;
}

interface GPUTextureBindingLayout {
  sampleType?: string;
  viewDimension?: string;
  multisampled?: boolean;
}

interface GPUStorageTextureBindingLayout {
  access?: string;
  format: string;
  viewDimension?: string;
}

interface GPUPipelineLayoutDescriptor {
  bindGroupLayouts: GPUBindGroupLayout[];
  label?: string;
}

interface GPUBindGroupDescriptor {
  layout: GPUBindGroupLayout;
  entries: GPUBindGroupEntry[];
  label?: string;
}

interface GPUBindGroupEntry {
  binding: number;
  resource: GPUBindingResource;
}

type GPUBindingResource = GPUSampler | GPUTextureView | GPUBufferBinding;

interface GPUBufferBinding {
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
}

interface GPUShaderModuleDescriptor {
  code: string;
  label?: string;
}

interface GPUComputePipelineDescriptor {
  layout: GPUPipelineLayout | 'auto';
  compute: GPUProgrammableStage;
  label?: string;
}

interface GPURenderPipelineDescriptor {
  layout: GPUPipelineLayout | 'auto';
  vertex: GPUVertexState;
  primitive?: GPUPrimitiveState;
  depthStencil?: GPUDepthStencilState;
  multisample?: GPUMultisampleState;
  fragment?: GPUFragmentState;
  label?: string;
}

interface GPUProgrammableStage {
  module: GPUShaderModule;
  entryPoint: string;
}

interface GPUVertexState extends GPUProgrammableStage {
  buffers?: GPUVertexBufferLayout[];
}

interface GPUVertexBufferLayout {
  arrayStride: number;
  stepMode?: string;
  attributes: GPUVertexAttribute[];
}

interface GPUVertexAttribute {
  format: string;
  offset: number;
  shaderLocation: number;
}

interface GPUPrimitiveState {
  topology?: string;
  stripIndexFormat?: string;
  frontFace?: string;
  cullMode?: string;
}

interface GPUDepthStencilState {
  format: string;
  depthWriteEnabled?: boolean;
  depthCompare?: string;
  stencilFront?: GPUStencilFaceState;
  stencilBack?: GPUStencilFaceState;
  stencilReadMask?: number;
  stencilWriteMask?: number;
  depthBias?: number;
  depthBiasSlopeScale?: number;
  depthBiasClamp?: number;
}

interface GPUStencilFaceState {
  compare?: string;
  failOp?: string;
  depthFailOp?: string;
  passOp?: string;
}

interface GPUMultisampleState {
  count?: number;
  mask?: number;
  alphaToCoverageEnabled?: boolean;
}

interface GPUFragmentState extends GPUProgrammableStage {
  targets: GPUColorTargetState[];
}

interface GPUColorTargetState {
  format: string;
  blend?: GPUBlendState;
  writeMask?: number;
}

interface GPUBlendState {
  color: GPUBlendComponent;
  alpha: GPUBlendComponent;
}

interface GPUBlendComponent {
  srcFactor?: string;
  dstFactor?: string;
  operation?: string;
}

interface GPUCommandEncoderDescriptor {
  label?: string;
}

interface GPUComputePassDescriptor {
  label?: string;
}

interface GPURenderPassDescriptor {
  colorAttachments: (GPURenderPassColorAttachment | null)[];
  depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
  occlusionQuerySet?: GPUQuerySet;
  label?: string;
}

interface GPURenderPassColorAttachment {
  view: GPUTextureView;
  resolveTarget?: GPUTextureView;
  loadOp: string;
  storeOp: string;
  clearValue?: GPUColor;
}

interface GPURenderPassDepthStencilAttachment {
  view: GPUTextureView;
  depthClearValue?: number;
  depthLoadOp?: string;
  depthStoreOp?: string;
  depthReadOnly?: boolean;
  stencilClearValue?: number;
  stencilLoadOp?: string;
  stencilStoreOp?: string;
  stencilReadOnly?: boolean;
}

interface GPUQuerySet {}

interface GPUColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

interface GPUCommandBufferDescriptor {
  label?: string;
}

interface GPUImageCopyBuffer {
  buffer: GPUBuffer;
  offset?: number;
  bytesPerRow?: number;
  rowsPerImage?: number;
}

interface GPUImageCopyTexture {
  texture: GPUTexture;
  mipLevel?: number;
  origin?: GPUOrigin3D;
  aspect?: string;
}

interface GPUImageDataLayout {
  offset?: number;
  bytesPerRow?: number;
  rowsPerImage?: number;
}

type GPUExtent3D = [number, number, number] | GPUExtent3DDict;

interface GPUExtent3DDict {
  width: number;
  height?: number;
  depthOrArrayLayers?: number;
}

type GPUOrigin3D = [number, number, number] | GPUOrigin3DDict;

interface GPUOrigin3DDict {
  x?: number;
  y?: number;
  z?: number;
}

interface GPUTextureViewDescriptor {
  format?: string;
  dimension?: string;
  aspect?: string;
  baseMipLevel?: number;
  mipLevelCount?: number;
  baseArrayLayer?: number;
  arrayLayerCount?: number;
  label?: string;
}

interface GPUSupportedLimits {
  maxTextureDimension1D?: number;
  maxTextureDimension2D?: number;
  maxTextureDimension3D?: number;
  maxTextureArrayLayers?: number;
  maxBindGroups?: number;
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
  maxVertexAttributes?: number;
  maxVertexBufferArrayStride?: number;
  maxInterStageShaderComponents?: number;
  maxComputeWorkgroupStorageSize?: number;
  maxComputeInvocationsPerWorkgroup?: number;
  maxComputeWorkgroupSizeX?: number;
  maxComputeWorkgroupSizeY?: number;
  maxComputeWorkgroupSizeZ?: number;
  maxComputeWorkgroupsPerDimension?: number;
}

interface GPUSupportedFeatures {
  has(feature: string): boolean;
  size: number;
  forEach(callbackfn: (value: string, value2: string, set: Set<string>) => void, thisArg?: any): void;
}

declare const GPUBufferUsage: {
  MAP_READ: 0x0001;
  MAP_WRITE: 0x0002;
  COPY_SRC: 0x0004;
  COPY_DST: 0x0008;
  INDEX: 0x0010;
  VERTEX: 0x0020;
  UNIFORM: 0x0040;
  STORAGE: 0x0080;
  INDIRECT: 0x0100;
  QUERY_RESOLVE: 0x0200;
};

declare const GPUTextureUsage: {
  COPY_SRC: 0x01;
  COPY_DST: 0x02;
  TEXTURE_BINDING: 0x04;
  STORAGE_BINDING: 0x08;
  RENDER_ATTACHMENT: 0x10;
};

declare const GPUShaderStage: {
  VERTEX: 0x1;
  FRAGMENT: 0x2;
  COMPUTE: 0x4;
};

// Compression Streams API
declare class CompressionStream {
  constructor(format: 'gzip' | 'deflate' | 'deflate-raw');
  readonly readable: ReadableStream;
  readonly writable: WritableStream;
}

declare class DecompressionStream {
  constructor(format: 'gzip' | 'deflate' | 'deflate-raw');
  readonly readable: ReadableStream;
  readonly writable: WritableStream;
}

// Performance Memory API
interface Performance {
  memory?: {
    jsHeapSizeLimit: number;
    totalJSHeapSize: number;
    usedJSHeapSize: number;
  };
}

// Window extensions
interface Window {
  __gpuCacheBuffers?: Record<string, GPUBuffer>;
  CompressionStream?: typeof CompressionStream;
  DecompressionStream?: typeof DecompressionStream;
  fs?: {
    readFile(path: string, options?: { encoding?: string }): Promise<string | Uint8Array>;
  };
}

// Node.js compatibility shims for browser
declare global {
  var process: {
    env: Record<string, string>;
    browser: boolean;
    version: string;
    versions: { node: string };
    cwd: () => string;
  };
  var global: typeof globalThis;
  var Buffer: {
    from(str: string, encoding?: string): Uint8Array;
    isBuffer(obj: any): boolean;
    alloc(size: number): Uint8Array;
  };
}


