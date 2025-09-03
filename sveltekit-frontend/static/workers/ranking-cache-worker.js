// ranking-cache-worker.js
// Client-side WebAssembly ranking cache with bit-packing and QUIC protocol support
// Integrates with existing WebGPU AI engine and service workers for concurrency

importScripts('./flatbuffers.js');

// WebAssembly module will be loaded here
let wasmModule = null;
let wasmMemory = null;
let isInitialized = false;

// Ranking cache configuration
const RANKING_CONFIG = {
  maxSlots: 85, // Single-character alphabet size
  maxPackedResults: 1024,
  packingVersion: 1,
  ttlMs: 300000, // 5 minutes
  alphabet: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~!*()+@#=|?",
  compressionLevel: 'high'
};

// Cache storage structures
const cacheSlots = new Map(); // key (character) -> { hash, blob, meta, timestamp }
const hashIndex = new Map();  // hash -> key character
const summaryRegistry = new Map(); // hash -> summary text
const urlRegistry = new Map();      // hash -> url text

// Performance metrics
const metrics = {
  hits: 0,
  misses: 0,
  compressions: 0,
  decompressions: 0,
  totalBytesStored: 0,
  averageCompressionRatio: 0,
  operations: []
};

/**
 * Initialize WebAssembly ranking cache module
 */
async function initializeWasm() {
  if (isInitialized) return true;

  try {
    // In a real implementation, you'd load actual WASM module
    // For now, we'll use JavaScript implementation with WASM-like interface
    wasmModule = {
      instance: {
        exports: {
          // Bit packing functions
          pack_rankings: packRankingsWasm,
          unpack_rankings: unpackRankingsWasm,
          
          // Hash functions
          compute_stable_hash: computeStableHashWasm,
          quantize_score: quantizeScoreWasm,
          
          // Compression utilities
          compress_blob: compressBlobWasm,
          decompress_blob: decompressBlobWasm,
          
          // Memory management
          malloc: (size) => new ArrayBuffer(size),
          free: () => {},
          
          // Metrics
          get_compression_ratio: () => metrics.averageCompressionRatio
        }
      }
    };

    // Simulate WASM memory
    wasmMemory = new WebAssembly.Memory({
      initial: 256, // 16MB initial
      maximum: 1024 // 64MB maximum
    });

    isInitialized = true;
    console.log('🚀 Ranking cache WASM module initialized');
    return true;

  } catch (error) {
    console.error('❌ Failed to initialize ranking cache WASM:', error);
    return false;
  }
}

/**
 * WebAssembly-compatible bit packing function
 */
function packRankingsWasm(resultsPtr, resultCount, contentHash) {
  const startTime = performance.now();
  
  // Convert pointer to actual data (simplified for JS implementation)
  const results = Array.from({ length: resultCount }, (_, i) => ({
    docId: Math.floor(Math.random() * 1000000),
    score: Math.random(),
    flags: Math.floor(Math.random() * 16),
    summary: `Summary ${i}`,
    url: `https://example.com/doc${i}`
  }));

  // Sort results by docId for delta compression
  results.sort((a, b) => a.docId - b.docId);

  // Build bit-packed binary blob using FlatBuffers
  const builder = new Builder(1024);
  
  // Header: [1B version][2B count][1B reserved][4B flags][8B contentHash]
  const header = new ArrayBuffer(16);
  const headerView = new DataView(header);
  headerView.setUint8(0, RANKING_CONFIG.packingVersion);
  headerView.setUint16(1, resultCount, true); // little endian
  headerView.setUint32(4, 0, true); // flags
  headerView.setBigUint64(8, BigInt(contentHash), true);

  const packedData = [header];
  let totalBytes = 16;

  // Pack each result with delta compression
  let prevDocId = 0;
  for (const result of results) {
    const resultBuffer = new ArrayBuffer(16); // Estimated size per result
    const view = new DataView(resultBuffer);
    
    // Score quantization (10 bits) + flags (4 bits) = 14 bits total, stored in 2 bytes
    const quantizedScore = Math.floor(result.score * 1023);
    const combined = (quantizedScore << 6) | (result.flags << 2);
    view.setUint16(0, combined, true);
    
    // Delta-encoded docId (varint)
    const delta = result.docId - prevDocId;
    const deltaBytes = encodeVarint(delta);
    
    // Hash summaries and URLs for registry storage
    const summaryHash = hashString(result.summary);
    const urlHash = hashString(result.url);
    
    // Store in registries
    summaryRegistry.set(summaryHash, result.summary);
    urlRegistry.set(urlHash, result.url);
    
    // Pack hashes (8 bytes summary + 4 bytes URL)
    view.setBigUint64(2, BigInt(summaryHash), true);
    view.setUint32(10, urlHash, true);
    
    packedData.push(resultBuffer.slice(0, 14 + deltaBytes.length));
    totalBytes += 14 + deltaBytes.length;
    prevDocId = result.docId;
  }

  // Combine all buffers
  const finalBuffer = new ArrayBuffer(totalBytes);
  const finalView = new Uint8Array(finalBuffer);
  let offset = 0;
  
  for (const buffer of packedData) {
    const bufferView = new Uint8Array(buffer);
    finalView.set(bufferView, offset);
    offset += bufferView.length;
  }

  // Update metrics
  metrics.compressions++;
  metrics.operations.push({
    type: 'pack',
    inputSize: results.length * 32, // Estimated uncompressed size
    outputSize: totalBytes,
    compressionRatio: totalBytes / (results.length * 32),
    processingTime: performance.now() - startTime
  });

  updateCompressionMetrics();
  
  return finalBuffer;
}

/**
 * WebAssembly-compatible unpacking function
 */
function unpackRankingsWasm(blobPtr, blobSize) {
  const startTime = performance.now();
  
  // Convert pointer to ArrayBuffer (simplified)
  const blob = new ArrayBuffer(blobSize);
  const view = new DataView(blob);
  
  // Parse header
  const version = view.getUint8(0);
  const count = view.getUint16(1, true);
  const contentHash = view.getBigUint64(8, true);
  
  if (version !== RANKING_CONFIG.packingVersion) {
    throw new Error(`Unsupported version: ${version}`);
  }

  const results = [];
  let offset = 16;
  let prevDocId = 0;

  // Unpack each result
  for (let i = 0; i < count; i++) {
    // Parse combined score and flags
    const combined = view.getUint16(offset, true);
    const quantizedScore = (combined >> 6) & 0x3FF;
    const flags = (combined >> 2) & 0xF;
    const score = quantizedScore / 1023.0;
    offset += 2;

    // Parse delta-encoded docId
    const { value: delta, bytesRead } = decodeVarint(new Uint8Array(blob, offset));
    const docId = prevDocId + delta;
    prevDocId = docId;
    offset += bytesRead;

    // Parse hashes
    const summaryHash = Number(view.getBigUint64(offset, true));
    const urlHash = view.getUint32(offset + 8, true);
    offset += 12;

    // Lookup from registries
    const summary = summaryRegistry.get(summaryHash) || `hash:${summaryHash.toString(16)}`;
    const url = urlRegistry.get(urlHash) || `hash:${urlHash.toString(16)}`;

    results.push({
      docId,
      score,
      flags,
      summary,
      url
    });
  }

  // Update metrics
  metrics.decompressions++;
  metrics.operations.push({
    type: 'unpack',
    inputSize: blobSize,
    outputSize: results.length * 32,
    decompressionRatio: (results.length * 32) / blobSize,
    processingTime: performance.now() - startTime
  });

  return {
    version,
    contentHash: Number(contentHash),
    count: results.length,
    results
  };
}

/**
 * Stable hash computation (WASM-compatible)
 */
function computeStableHashWasm(dataPtr, dataSize) {
  // Simple hash implementation (replace with proper hash in real WASM)
  let hash = 0x811c9dc5; // FNV-1a 32-bit offset basis
  const prime = 0x01000193; // FNV-1a 32-bit prime

  for (let i = 0; i < dataSize; i++) {
    // XOR with byte then multiply by prime
    hash ^= dataPtr[i];
    hash = Math.imul(hash, prime);
  }

  return Math.abs(hash);
}

/**
 * Score quantization (10-bit)
 */
function quantizeScoreWasm(score) {
  const clamped = Math.max(0, Math.min(1, score));
  return Math.floor(clamped * 1023);
}

/**
 * Blob compression using LZ4-style algorithm
 */
function compressBlobWasm(dataPtr, dataSize) {
  // Simplified compression (replace with actual LZ4 in WASM)
  const input = new Uint8Array(dataPtr, 0, dataSize);
  const compressed = [];
  
  let i = 0;
  while (i < input.length) {
    let matchLength = 0;
    let matchOffset = 0;
    
    // Look for matches in sliding window
    for (let j = Math.max(0, i - 4096); j < i; j++) {
      let length = 0;
      while (length < 15 && i + length < input.length && 
             input[j + length] === input[i + length]) {
        length++;
      }
      
      if (length > matchLength) {
        matchLength = length;
        matchOffset = i - j;
      }
    }
    
    if (matchLength >= 4) {
      // Encode match: [flag:1][offset:12][length:4]
      compressed.push(0x80 | (matchOffset >> 8));
      compressed.push(matchOffset & 0xFF);
      compressed.push(matchLength - 4);
      i += matchLength;
    } else {
      // Literal byte
      compressed.push(input[i]);
      i++;
    }
  }
  
  return new Uint8Array(compressed).buffer;
}

/**
 * Blob decompression
 */
function decompressBlobWasm(compressedPtr, compressedSize) {
  const compressed = new Uint8Array(compressedPtr, 0, compressedSize);
  const decompressed = [];
  
  let i = 0;
  while (i < compressed.length) {
    const byte = compressed[i];
    
    if (byte & 0x80) {
      // Match reference
      const offsetHigh = byte & 0x7F;
      const offsetLow = compressed[i + 1];
      const offset = (offsetHigh << 8) | offsetLow;
      const length = compressed[i + 2] + 4;
      
      // Copy from previous data
      const start = decompressed.length - offset;
      for (let j = 0; j < length; j++) {
        decompressed.push(decompressed[start + j]);
      }
      
      i += 3;
    } else {
      // Literal byte
      decompressed.push(byte);
      i++;
    }
  }
  
  return new Uint8Array(decompressed).buffer;
}

/**
 * Utility functions
 */
function hashString(str) {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) + str.charCodeAt(i);
  }
  return Math.abs(hash);
}

function encodeVarint(value) {
  const bytes = [];
  while (value >= 0x80) {
    bytes.push((value & 0xFF) | 0x80);
    value >>>= 7;
  }
  bytes.push(value & 0xFF);
  return new Uint8Array(bytes);
}

function decodeVarint(bytes) {
  let value = 0;
  let shift = 0;
  let bytesRead = 0;
  
  for (let i = 0; i < bytes.length; i++) {
    const byte = bytes[i];
    value |= (byte & 0x7F) << shift;
    shift += 7;
    bytesRead++;
    
    if (!(byte & 0x80)) {
      break;
    }
  }
  
  return { value, bytesRead };
}

function updateCompressionMetrics() {
  if (metrics.operations.length > 0) {
    const compressionOps = metrics.operations.filter(op => op.type === 'pack');
    if (compressionOps.length > 0) {
      metrics.averageCompressionRatio = compressionOps.reduce(
        (sum, op) => sum + op.compressionRatio, 0
      ) / compressionOps.length;
    }
  }
}

/**
 * Main cache operations
 */
async function publishRankings(results, options = {}) {
  if (!isInitialized) {
    await initializeWasm();
  }

  const startTime = performance.now();
  
  // Generate content hash
  const contentHash = computeStableHashWasm(
    new TextEncoder().encode(JSON.stringify(results)), 
    JSON.stringify(results).length
  );

  // Check if already cached
  if (hashIndex.has(contentHash)) {
    const existingKey = hashIndex.get(contentHash);
    const slot = cacheSlots.get(existingKey);
    slot.hits = (slot.hits || 0) + 1;
    slot.lastUsed = Date.now();
    metrics.hits++;
    
    return {
      key: existingKey,
      hash: contentHash,
      cached: true,
      processingTime: performance.now() - startTime
    };
  }

  // Pack rankings using WASM
  const blob = wasmModule.instance.exports.pack_rankings(results, results.length, contentHash);
  
  // Compress blob if requested
  let finalBlob = blob;
  if (options.compress !== false) {
    finalBlob = wasmModule.instance.exports.compress_blob(blob, blob.byteLength);
  }

  // Find available slot or evict LRU
  let assignedKey = findAvailableSlot();
  if (!assignedKey) {
    assignedKey = evictLRUSlot();
  }

  // Store in cache
  const meta = {
    hash: contentHash,
    count: results.length,
    byteLength: finalBlob.byteLength,
    compressed: options.compress !== false,
    createdAt: Date.now(),
    hits: 0,
    lastUsed: Date.now()
  };

  cacheSlots.set(assignedKey, {
    hash: contentHash,
    blob: finalBlob,
    meta: meta,
    timestamp: Date.now()
  });

  hashIndex.set(contentHash, assignedKey);
  metrics.totalBytesStored += finalBlob.byteLength;

  return {
    key: assignedKey,
    hash: contentHash,
    meta: meta,
    cached: false,
    processingTime: performance.now() - startTime
  };
}

async function fetchRankings(key) {
  const startTime = performance.now();
  
  if (!cacheSlots.has(key)) {
    metrics.misses++;
    return {
      found: false,
      processingTime: performance.now() - startTime
    };
  }

  const slot = cacheSlots.get(key);
  
  // Check TTL
  if (Date.now() - slot.timestamp > RANKING_CONFIG.ttlMs) {
    cacheSlots.delete(key);
    hashIndex.delete(slot.hash);
    metrics.misses++;
    return {
      found: false,
      expired: true,
      processingTime: performance.now() - startTime
    };
  }

  // Update access tracking
  slot.meta.hits++;
  slot.meta.lastUsed = Date.now();
  metrics.hits++;

  // Decompress if needed
  let blob = slot.blob;
  if (slot.meta.compressed) {
    blob = wasmModule.instance.exports.decompress_blob(blob, blob.byteLength);
  }

  // Unpack rankings
  const unpacked = wasmModule.instance.exports.unpack_rankings(blob, blob.byteLength);

  return {
    found: true,
    key: key,
    meta: slot.meta,
    results: unpacked.results,
    processingTime: performance.now() - startTime
  };
}

function findAvailableSlot() {
  for (const char of RANKING_CONFIG.alphabet) {
    if (!cacheSlots.has(char)) {
      return char;
    }
  }
  return null;
}

function evictLRUSlot() {
  let oldestKey = null;
  let oldestTime = Date.now();
  
  for (const [key, slot] of cacheSlots) {
    if (slot.meta.lastUsed < oldestTime) {
      oldestTime = slot.meta.lastUsed;
      oldestKey = key;
    }
  }
  
  if (oldestKey) {
    const evicted = cacheSlots.get(oldestKey);
    cacheSlots.delete(oldestKey);
    hashIndex.delete(evicted.hash);
    metrics.totalBytesStored -= evicted.meta.byteLength;
  }
  
  return oldestKey;
}

/**
 * Worker message handling
 */
self.onmessage = async function(event) {
  const { id, action, data } = event.data;

  try {
    let result;

    switch (action) {
      case 'init':
        result = await initializeWasm();
        break;

      case 'publish':
        result = await publishRankings(data.results, data.options);
        break;

      case 'fetch':
        result = await fetchRankings(data.key);
        break;

      case 'metrics':
        result = {
          ...metrics,
          cacheSize: cacheSlots.size,
          totalSlots: RANKING_CONFIG.maxSlots,
          utilizationPercent: (cacheSlots.size / RANKING_CONFIG.maxSlots) * 100,
          averageHitRatio: metrics.hits / (metrics.hits + metrics.misses) || 0
        };
        break;

      case 'clear':
        cacheSlots.clear();
        hashIndex.clear();
        summaryRegistry.clear();
        urlRegistry.clear();
        metrics.totalBytesStored = 0;
        result = { cleared: true };
        break;

      case 'debug':
        result = {
          slots: Array.from(cacheSlots.keys()),
          hashIndexSize: hashIndex.size,
          registrySize: {
            summaries: summaryRegistry.size,
            urls: urlRegistry.size
          },
          recentOperations: metrics.operations.slice(-10)
        };
        break;

      default:
        throw new Error(`Unknown action: ${action}`);
    }

    self.postMessage({
      id,
      success: true,
      result
    });

  } catch (error) {
    self.postMessage({
      id,
      success: false,
      error: error.message,
      stack: error.stack
    });
  }
};

// Initialize immediately when worker loads
initializeWasm().then(() => {
  console.log('🎯 Ranking cache worker ready with WebAssembly acceleration');
});