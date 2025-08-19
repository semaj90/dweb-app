// WebGPU-enabled worker with graceful CPU fallback for Legal AI Platform
// RTX 3060 Ti optimized tensor operations and legal document processing
// Supported ops:
//  - ping
//  - tensor.add (CPU fallback)
//  - tensor.add.gpu (attempt GPU compute; falls back automatically)
//  - legal.process (legal document processing with embeddings)
//  - legal.similarity (document similarity calculations)
//  - info (capabilities)

let gpuReady = false;
let device = null;
let queue = null;
let adapterInfo = null;

async function initWebGPU(){
  if (gpuReady || typeof navigator === 'undefined' || !('gpu' in navigator)) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;
    device = await adapter.requestDevice();
    queue = device.queue;
    // Some browsers support adapter.info (Chrome behind flag). Guard access.
    adapterInfo = adapter.requestAdapterInfo ? await adapter.requestAdapterInfo().catch(()=>null) : null;
    gpuReady = true;
    return true;
  } catch (err){
    gpuReady = false;
    return false;
  }
}

function cpuAdd(a,b){ return a.map((v,i)=>v + b[i]); }

async function gpuAdd(a,b){
  const n = a.length;
  if(!gpuReady) { const ok = await initWebGPU(); if(!ok) return { out: cpuAdd(a,b), mode: 'cpu' }; }
  try {
    const shader = `@group(0) @binding(0) var<storage, read> A: array<f32>;\n@group(0) @binding(1) var<storage, read> B: array<f32>;\n@group(0) @binding(2) var<storage, read_write> C: array<f32>;\n@compute @workgroup_size(64) fn add(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let i = gid.x; if (i < ${n}u) { C[i] = A[i] + B[i]; }\n}`;
    const module = device.createShaderModule({ code: shader });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'add' }});
    const bytes = n * 4;
    function buf(arr, usage){
      const buffer = device.createBuffer({ size: bytes, usage });
      device.queue.writeBuffer(buffer, 0, new Float32Array(arr));
      return buffer;
    }
    const aBuf = buf(a, GPUBufferUsage.STORAGE);
    const bBuf = buf(b, GPUBufferUsage.STORAGE);
    const outBuf = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding:0, resource:{ buffer: aBuf } },
      { binding:1, resource:{ buffer: bBuf } },
      { binding:2, resource:{ buffer: outBuf } },
    ]});
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = Math.ceil(n/64);
    pass.dispatchWorkgroups(wg);
    pass.end();
    // Readback
    const readBuf = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    commandEncoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, bytes);
    queue.submit([commandEncoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const array = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    return { out: Array.from(array), mode: 'webgpu' };
  } catch (e){
    return { out: cpuAdd(a,b), mode: 'cpu-fallback', error: (e && e.message) ? e.message : 'unknown' };
  }
}

// Legal document processing functions
function processLegalText(text) {
  const words = text.toLowerCase().split(/\s+/);
  const legalTerms = ['contract', 'liability', 'indemnification', 'precedent', 'evidence', 'plaintiff', 'defendant', 'statute', 'regulation', 'jurisprudence'];
  const legalScore = words.filter(word => legalTerms.includes(word)).length / words.length;
  
  // Generate simple embeddings
  const embeddings = words.slice(0, 100).map((word, idx) => {
    let hash = 0;
    for (let i = 0; i < word.length; i++) {
      hash = ((hash << 5) - hash) + word.charCodeAt(i);
      hash = hash & hash;
    }
    return (hash % 1000) / 1000 * (legalTerms.includes(word) ? 1.2 : 1.0);
  });
  
  return { 
    embeddings, 
    legalScore, 
    wordCount: words.length,
    legalTermCount: words.filter(word => legalTerms.includes(word)).length 
  };
}

function calculateSimilarity(emb1, emb2) {
  if (!emb1 || !emb2 || emb1.length === 0 || emb2.length === 0) return 0;
  const len = Math.min(emb1.length, emb2.length);
  let dotProduct = 0, norm1 = 0, norm2 = 0;
  
  for (let i = 0; i < len; i++) {
    dotProduct += emb1[i] * emb2[i];
    norm1 += emb1[i] * emb1[i];
    norm2 += emb2[i] * emb2[i];
  }
  
  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

self.addEventListener('message', async (e) => {
  const { op, payload } = e.data || {};
  switch(op){
    case 'ping': return self.postMessage({ op: 'pong' });
    case 'info': {
      const ok = await initWebGPU();
      return self.postMessage({ op: 'info', webgpu: ok, adapterInfo, legal: true });
    }
    case 'tensor.add': {
      const { a, b } = payload; return self.postMessage({ op: 'tensor.add.result', out: cpuAdd(a,b), mode: 'cpu' });
    }
    case 'tensor.add.gpu': {
      const { a, b } = payload; const res = await gpuAdd(a,b); return self.postMessage({ op: 'tensor.add.gpu.result', ...res });
    }
    case 'legal.process': {
      const { text } = payload;
      const startTime = performance.now();
      const result = processLegalText(text);
      const processingTime = performance.now() - startTime;
      return self.postMessage({ 
        op: 'legal.process.result', 
        ...result, 
        processingTime,
        mode: 'cpu'
      });
    }
    case 'legal.similarity': {
      const { text1, text2 } = payload;
      const startTime = performance.now();
      const proc1 = processLegalText(text1);
      const proc2 = processLegalText(text2);
      const similarity = calculateSimilarity(proc1.embeddings, proc2.embeddings);
      const processingTime = performance.now() - startTime;
      return self.postMessage({ 
        op: 'legal.similarity.result',
        similarity,
        legalScore1: proc1.legalScore,
        legalScore2: proc2.legalScore,
        processingTime,
        mode: 'cpu'
      });
    }
    default: return self.postMessage({ op: 'noop' });
  }
});
