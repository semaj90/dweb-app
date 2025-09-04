// ======================================================================
// CANONICAL RESULT CACHE - Bit-Packed Single-Character Key System
// Ultra-low latency ranking cache with single-character indirection
// ======================================================================

import { browser } from '$app/environment';

// Canonical result record structure
export interface CanonicalResult {
  docId: string;
  score: number;          // 0.0 - 1.0 (will be quantized to 10-bit 0-1023)
  flags: number;          // 4-bit flags (type, priority, etc.)
  summaryHash: string;    // 22-bit hash → 4M summary entries
  targetUrlId?: string;   // Optional URL reference
  metadata?: Record<string, any>;
}

export interface RankingSet {
  results: CanonicalResult[];
  query: string;
  totalResults: number;
  timestamp: number;
  version: number;
}

// Header structure (64 bits)
export interface PackedHeader {
  version: number;        // 6 bits (0-63)
  resultCount: number;    // 10 bits (0-1023) 
  summaryStrategy: number; // 8 bits enum
  flags: number;          // 8 bits (personalized, filtered, etc.)
  crc32: number;          // 32 bits payload integrity
}

// Alphabet for single-character keys (Base85 compatible)
const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&()*+,-./:;<=>?@[]^_`{|}~';
const ALPHABET_SIZE = ALPHABET.length; // 85 characters

// Slot table entry
export interface SlotEntry {
  hash: bigint;           // 64-bit Blake3/Murmur3 hash
  blobPtr: Uint8Array;    // Packed blob data
  ttl: number;           // Time-to-live timestamp
  accessCount: number;   // LRU tracking
  createdAt: number;     // Creation timestamp
}

export interface CacheMetrics {
  hitRate: number;
  totalLookups: number;
  totalHits: number;
  totalMisses: number;
  sizeBytes: number;
  slotUtilization: number;
}

export class CanonicalResultCache {
  private slotTable: Map<string, SlotEntry> = new Map();
  private hashToSlot: Map<string, string> = new Map(); // Quick hash lookup
  private nextSlotIndex = 0;
  private metrics: CacheMetrics = {
    hitRate: 0,
    totalLookups: 0,
    totalHits: 0,
    totalMisses: 0,
    sizeBytes: 0,
    slotUtilization: 0
  };

  private readonly TTL_MS = 30 * 1000; // 30 seconds default TTL
  private readonly MAX_RESULTS = 1024;

  constructor(
    private redisClient?: any, // Optional Redis client
    private enableCompression = true
  ) {
    if (browser && 'performance' in window) {
      // Browser environment optimizations
      this.setupPerformanceMonitoring();
    }
  }

  // Main caching API
  async storeRankingSet(rankingSet: RankingSet): Promise<string> {
    try {
      // Validate input
      if (!rankingSet.results.length || rankingSet.results.length > this.MAX_RESULTS) {
        throw new Error(`Invalid result count: ${rankingSet.results.length}`);
      }

      // Generate content hash
      const contentHash = this.computeContentHash(rankingSet);
      const hashStr = contentHash.toString(16);

      // Check if we already have this hash cached
      if (this.hashToSlot.has(hashStr)) {
        const existingSlot = this.hashToSlot.get(hashStr)!;
        this.updateAccessMetrics(existingSlot, true);
        return existingSlot;
      }

      // Pack the ranking set into binary format
      const packedBlob = this.packRankingSet(rankingSet);
      
      // Find or allocate a slot
      const slotChar = this.allocateSlot(contentHash, packedBlob);

      // Update metrics
      this.updateMetrics();

      // Store in Redis if available
      if (this.redisClient) {
        await this.storeInRedis(hashStr, packedBlob, rankingSet);
      }

      return slotChar;

    } catch (error) {
      console.error('Failed to store ranking set:', error);
      throw error;
    }
  }

  async retrieveRankingSet(slotChar: string): Promise<RankingSet | null> {
    this.metrics.totalLookups++;

    try {
      // Validate slot character
      if (!ALPHABET.includes(slotChar)) {
        this.metrics.totalMisses++;
        return null;
      }

      // Check in-memory slot table
      const slotEntry = this.slotTable.get(slotChar);
      if (slotEntry) {
        // Check TTL
        if (Date.now() > slotEntry.ttl) {
          this.evictSlot(slotChar);
          this.metrics.totalMisses++;
          return null;
        }

        // Unpack and return
        const rankingSet = this.unpackRankingSet(slotEntry.blobPtr);
        this.updateAccessMetrics(slotChar, true);
        this.metrics.totalHits++;
        return rankingSet;
      }

      // Try Redis fallback
      if (this.redisClient) {
        const cachedData = await this.retrieveFromRedis(slotChar);
        if (cachedData) {
          this.metrics.totalHits++;
          return cachedData;
        }
      }

      this.metrics.totalMisses++;
      return null;

    } catch (error) {
      console.error('Failed to retrieve ranking set:', error);
      this.metrics.totalMisses++;
      return null;
    }
  }

  // Bit-packing implementation
  private packRankingSet(rankingSet: RankingSet): Uint8Array {
    const buffer = new ArrayBuffer(8192); // Start with 8KB, will resize if needed
    const view = new DataView(buffer);
    let offset = 0;

    // Pack header (64 bits)
    const header: PackedHeader = {
      version: 1,
      resultCount: Math.min(rankingSet.results.length, 1023),
      summaryStrategy: 0, // Default strategy
      flags: this.computeFlags(rankingSet),
      crc32: 0 // Will compute later
    };

    // Write header
    view.setUint8(offset++, (header.version << 2) | ((header.resultCount >> 8) & 0x03));
    view.setUint8(offset++, header.resultCount & 0xFF);
    view.setUint8(offset++, header.summaryStrategy);
    view.setUint8(offset++, header.flags);

    // Reserve space for CRC32
    const crcOffset = offset;
    offset += 4;

    // Pack results with delta encoding
    let prevDocId = '';
    for (let i = 0; i < header.resultCount; i++) {
      const result = rankingSet.results[i];
      
      // Quantize score to 10 bits (0-1023)
      const scoreQ = Math.min(1023, Math.max(0, Math.round(result.score * 1023)));
      
      // Pack score (10 bits) + flags (4 bits) + reserved (2 bits)
      const packed16 = (scoreQ << 6) | ((result.flags & 0xF) << 2);
      view.setUint16(offset, packed16);
      offset += 2;

      // Store docId delta (varint encoding)
      const deltaId = this.computeDocIdDelta(result.docId, prevDocId);
      offset = this.writeVarint(view, offset, deltaId);
      prevDocId = result.docId;

      // Store summary hash (22 bits compressed)
      const summaryHash = this.computeSummaryHash(result.summaryHash || '');
      offset = this.write22Bits(view, offset, summaryHash);

      // Store optional targetUrlId
      if (result.targetUrlId) {
        view.setUint8(offset++, 1); // Has URL flag
        offset = this.writeString(view, offset, result.targetUrlId);
      } else {
        view.setUint8(offset++, 0); // No URL flag
      }
    }

    // Compute and store CRC32
    const payloadSize = offset - 8;
    const payload = new Uint8Array(buffer, 8, payloadSize);
    const crc = this.computeCRC32(payload);
    view.setUint32(crcOffset, crc);

    // Return correctly sized buffer
    return new Uint8Array(buffer, 0, offset);
  }

  private unpackRankingSet(packedBlob: Uint8Array): RankingSet {
    const view = new DataView(packedBlob.buffer, packedBlob.byteOffset, packedBlob.byteLength);
    let offset = 0;

    // Unpack header
    const byte0 = view.getUint8(offset++);
    const byte1 = view.getUint8(offset++);
    const version = (byte0 >> 2) & 0x3F;
    const resultCount = ((byte0 & 0x03) << 8) | byte1;
    const summaryStrategy = view.getUint8(offset++);
    const flags = view.getUint8(offset++);
    const crc32 = view.getUint32(offset);
    offset += 4;

    // Verify CRC32
    const payload = packedBlob.slice(8);
    const computedCRC = this.computeCRC32(payload);
    if (crc32 !== computedCRC) {
      throw new Error('CRC32 validation failed - corrupted data');
    }

    // Unpack results
    const results: CanonicalResult[] = [];
    let prevDocId = '';

    for (let i = 0; i < resultCount; i++) {
      // Unpack score and flags
      const packed16 = view.getUint16(offset);
      offset += 2;
      
      const scoreQ = (packed16 >> 6) & 0x3FF; // Extract 10-bit score
      const flags = (packed16 >> 2) & 0xF;    // Extract 4-bit flags
      const score = scoreQ / 1023.0;

      // Read docId delta
      const deltaResult = this.readVarint(view, offset);
      const deltaId = deltaResult.value;
      offset = deltaResult.newOffset;
      const docId = this.applyDocIdDelta(prevDocId, deltaId);
      prevDocId = docId;

      // Read summary hash
      const summaryHashResult = this.read22Bits(view, offset);
      const summaryHash = summaryHashResult.value.toString(16);
      offset = summaryHashResult.newOffset;

      // Read optional targetUrlId
      const hasUrl = view.getUint8(offset++);
      let targetUrlId: string | undefined;
      if (hasUrl) {
        const urlResult = this.readString(view, offset);
        targetUrlId = urlResult.value;
        offset = urlResult.newOffset;
      }

      results.push({
        docId,
        score,
        flags,
        summaryHash,
        targetUrlId
      });
    }

    return {
      results,
      query: '', // Will be restored from metadata if needed
      totalResults: results.length,
      timestamp: Date.now(),
      version
    };
  }

  // Single-character key allocation
  private allocateSlot(contentHash: bigint, packedBlob: Uint8Array): string {
    const hashStr = contentHash.toString(16);

    // Check if we can reuse existing slot for same hash
    if (this.hashToSlot.has(hashStr)) {
      return this.hashToSlot.get(hashStr)!;
    }

    // Find next available slot (ring buffer)
    let attempts = 0;
    while (attempts < ALPHABET_SIZE) {
      const slotChar = ALPHABET[this.nextSlotIndex];
      this.nextSlotIndex = (this.nextSlotIndex + 1) % ALPHABET_SIZE;

      const existing = this.slotTable.get(slotChar);
      if (!existing || Date.now() > existing.ttl || this.shouldEvict(existing)) {
        // Evict old mapping if exists
        if (existing) {
          const oldHashStr = existing.hash.toString(16);
          this.hashToSlot.delete(oldHashStr);
        }

        // Create new slot entry
        const slotEntry: SlotEntry = {
          hash: contentHash,
          blobPtr: packedBlob,
          ttl: Date.now() + this.TTL_MS,
          accessCount: 1,
          createdAt: Date.now()
        };

        this.slotTable.set(slotChar, slotEntry);
        this.hashToSlot.set(hashStr, slotChar);
        return slotChar;
      }

      attempts++;
    }

    throw new Error('No available slots - cache full');
  }

  // Utility methods
  private computeContentHash(rankingSet: RankingSet): bigint {
    // Simple hash for demo - in production use Blake3
    let hash = 0n;
    const data = JSON.stringify({
      results: rankingSet.results.map(r => ({ 
        docId: r.docId, 
        score: Math.round(r.score * 1023), // Quantized for consistency
        flags: r.flags 
      })),
      query: rankingSet.query
    });

    for (let i = 0; i < data.length; i++) {
      hash = (hash * 31n + BigInt(data.charCodeAt(i))) & 0xFFFFFFFFFFFFFFFFn;
    }
    return hash;
  }

  private computeFlags(rankingSet: RankingSet): number {
    let flags = 0;
    // Example flag encoding
    if (rankingSet.results.some(r => r.metadata?.personalized)) flags |= 0x01;
    if (rankingSet.results.some(r => r.metadata?.filtered)) flags |= 0x02;
    if (rankingSet.totalResults > rankingSet.results.length) flags |= 0x04; // More results available
    return flags;
  }

  private computeSummaryHash(summaryText: string): number {
    // Simple 22-bit hash (4M buckets)
    let hash = 0;
    for (let i = 0; i < summaryText.length; i++) {
      hash = ((hash << 5) - hash + summaryText.charCodeAt(i)) & 0x3FFFFF; // 22 bits
    }
    return hash;
  }

  private computeDocIdDelta(currentId: string, prevId: string): number {
    // Simplified delta encoding - in production use proper string diff
    if (!prevId) return parseInt(currentId) || 0;
    return (parseInt(currentId) || 0) - (parseInt(prevId) || 0);
  }

  private applyDocIdDelta(prevId: string, delta: number): string {
    if (!prevId) return delta.toString();
    return ((parseInt(prevId) || 0) + delta).toString();
  }

  private computeCRC32(data: Uint8Array): number {
    // Simplified CRC32 - use proper implementation in production
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < data.length; i++) {
      crc ^= data[i];
      for (let j = 0; j < 8; j++) {
        crc = (crc >>> 1) ^ (crc & 1 ? 0xEDB88320 : 0);
      }
    }
    return (~crc) >>> 0;
  }

  // Varint encoding/decoding
  private writeVarint(view: DataView, offset: number, value: number): number {
    while (value >= 0x80) {
      view.setUint8(offset++, (value & 0xFF) | 0x80);
      value >>>= 7;
    }
    view.setUint8(offset++, value & 0xFF);
    return offset;
  }

  private readVarint(view: DataView, offset: number): { value: number; newOffset: number } {
    let value = 0;
    let shift = 0;
    while (true) {
      const byte = view.getUint8(offset++);
      value |= (byte & 0x7F) << shift;
      if ((byte & 0x80) === 0) break;
      shift += 7;
    }
    return { value, newOffset: offset };
  }

  // 22-bit encoding/decoding
  private write22Bits(view: DataView, offset: number, value: number): number {
    value &= 0x3FFFFF; // Ensure 22 bits
    view.setUint8(offset++, (value >> 16) & 0xFF);
    view.setUint8(offset++, (value >> 8) & 0xFF);
    view.setUint8(offset++, value & 0xFF);
    return offset;
  }

  private read22Bits(view: DataView, offset: number): { value: number; newOffset: number } {
    const byte0 = view.getUint8(offset++);
    const byte1 = view.getUint8(offset++);
    const byte2 = view.getUint8(offset++);
    const value = (byte0 << 16) | (byte1 << 8) | byte2;
    return { value, newOffset: offset };
  }

  // String encoding/decoding (simplified)
  private writeString(view: DataView, offset: number, str: string): number {
    const bytes = new TextEncoder().encode(str);
    offset = this.writeVarint(view, offset, bytes.length);
    for (let i = 0; i < bytes.length; i++) {
      view.setUint8(offset++, bytes[i]);
    }
    return offset;
  }

  private readString(view: DataView, offset: number): { value: string; newOffset: number } {
    const lengthResult = this.readVarint(view, offset);
    const length = lengthResult.value;
    offset = lengthResult.newOffset;
    
    const bytes = new Uint8Array(length);
    for (let i = 0; i < length; i++) {
      bytes[i] = view.getUint8(offset++);
    }
    
    const value = new TextDecoder().decode(bytes);
    return { value, newOffset: offset };
  }

  // Redis integration
  private async storeInRedis(hashStr: string, packedBlob: Uint8Array, rankingSet: RankingSet): Promise<void> {
    if (!this.redisClient) return;

    try {
      const key = `rank:${hashStr}`;
      await this.redisClient.setex(key, 300, Buffer.from(packedBlob)); // 5 min TTL
      
      // Store metadata separately
      const metaKey = `meta:${hashStr}`;
      await this.redisClient.setex(metaKey, 300, JSON.stringify({
        query: rankingSet.query,
        timestamp: rankingSet.timestamp,
        totalResults: rankingSet.totalResults
      }));
    } catch (error) {
      console.warn('Redis storage failed:', error);
    }
  }

  private async retrieveFromRedis(slotChar: string): Promise<RankingSet | null> {
    if (!this.redisClient) return null;

    try {
      // Get slot entry to find hash
      const slotEntry = this.slotTable.get(slotChar);
      if (!slotEntry) return null;

      const hashStr = slotEntry.hash.toString(16);
      const key = `rank:${hashStr}`;
      const metaKey = `meta:${hashStr}`;

      const [blobBuffer, metaJson] = await Promise.all([
        this.redisClient.get(key),
        this.redisClient.get(metaKey)
      ]);

      if (!blobBuffer) return null;

      const packedBlob = new Uint8Array(blobBuffer);
      const rankingSet = this.unpackRankingSet(packedBlob);

      if (metaJson) {
        const metadata = JSON.parse(metaJson);
        rankingSet.query = metadata.query || '';
        rankingSet.timestamp = metadata.timestamp || Date.now();
        rankingSet.totalResults = metadata.totalResults || rankingSet.results.length;
      }

      return rankingSet;
    } catch (error) {
      console.warn('Redis retrieval failed:', error);
      return null;
    }
  }

  // Cache management
  private shouldEvict(entry: SlotEntry): boolean {
    const age = Date.now() - entry.createdAt;
    const maxAge = this.TTL_MS * 2; // Double TTL for eviction threshold
    return age > maxAge || entry.accessCount < 2; // LRU-like eviction
  }

  private evictSlot(slotChar: string): void {
    const entry = this.slotTable.get(slotChar);
    if (entry) {
      const hashStr = entry.hash.toString(16);
      this.hashToSlot.delete(hashStr);
      this.slotTable.delete(slotChar);
    }
  }

  private updateAccessMetrics(slotChar: string, hit: boolean): void {
    const entry = this.slotTable.get(slotChar);
    if (entry) {
      entry.accessCount++;
    }
  }

  private updateMetrics(): void {
    this.metrics.hitRate = this.metrics.totalLookups > 0 
      ? this.metrics.totalHits / this.metrics.totalLookups 
      : 0;
    
    this.metrics.slotUtilization = this.slotTable.size / ALPHABET_SIZE;
    
    // Estimate memory usage
    let sizeBytes = 0;
    for (const entry of this.slotTable.values()) {
      sizeBytes += entry.blobPtr.byteLength + 64; // Overhead estimate
    }
    this.metrics.sizeBytes = sizeBytes;
  }

  private setupPerformanceMonitoring(): void {
    if (browser && 'PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.name.includes('canonical-cache')) {
              console.debug(`Cache operation ${entry.name}: ${entry.duration}ms`);
            }
          }
        });
        observer.observe({ entryTypes: ['measure'] });
      } catch (error) {
        console.debug('Performance monitoring not available');
      }
    }
  }

  // Public API
  async clear(): Promise<void> {
    this.slotTable.clear();
    this.hashToSlot.clear();
    this.nextSlotIndex = 0;
    this.metrics = {
      hitRate: 0,
      totalLookups: 0,
      totalHits: 0,
      totalMisses: 0,
      sizeBytes: 0,
      slotUtilization: 0
    };
  }

  getMetrics(): CacheMetrics {
    this.updateMetrics();
    return { ...this.metrics };
  }

  getSlotTableStatus(): { total: number; used: number; utilization: number } {
    return {
      total: ALPHABET_SIZE,
      used: this.slotTable.size,
      utilization: this.slotTable.size / ALPHABET_SIZE
    };
  }
}

// Export singleton instance
export const canonicalResultCache = new CanonicalResultCache();