// Updated OCR service with WebAssembly llama.cpp integration
// Supports both client-side and server-side AI processing

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { createWorker } from 'tesseract.js';
import sharp from 'sharp';
import pdfParse from 'pdf-parse';
import { Redis } from 'ioredis';
import { webLlamaService } from '$lib/ai/webasm-llamacpp.js';
import { llamaCppService } from '$lib/ai/llamacpp-service.js';

// LFU Cache implementation without constructor issues
class LFUCache<K, V> {
  private capacity: number;
  private cache = new Map<K, { value: V; frequency: number; lastUsed: number }>();
  private frequencies = new Map<number, Set<K>>();
  private minFreq = 0;
  private currentTime = 0;

  static create<K, V>(capacity: number): LFUCache<K, V> {
    const instance = Object.create(LFUCache.prototype);
    instance.capacity = capacity;
    instance.cache = new Map();
    instance.frequencies = new Map();
    instance.minFreq = 0;
    instance.currentTime = 0;
    return instance;
  }

  get(key: K): V | undefined {
    const item = this.cache.get(key);
    if (!item) return undefined;

    this.updateFrequency(key, item.frequency);
    item.lastUsed = ++this.currentTime;
    return item.value;
  }

  set(key: K, value: V): void {
    if (this.capacity <= 0) return;

    if (this.cache.has(key)) {
      const item = this.cache.get(key)!;
      item.value = value;
      this.updateFrequency(key, item.frequency);
      item.lastUsed = ++this.currentTime;
      return;
    }

    if (this.cache.size >= this.capacity) {
      this.evictLFU();
    }

    this.cache.set(key, { value, frequency: 1, lastUsed: ++this.currentTime });
    this.addToFrequency(key, 1);
    this.minFreq = 1;
  }

  private updateFrequency(key: K, oldFreq: number): void {
    const newFreq = oldFreq + 1;
    
    // Remove from old frequency
    const oldSet = this.frequencies.get(oldFreq);
    if (oldSet) {
      oldSet.delete(key);
      if (oldSet.size === 0 && oldFreq === this.minFreq) {
        this.minFreq++;
      }
    }

    // Add to new frequency
    this.addToFrequency(key, newFreq);
    
    // Update cache
    const item = this.cache.get(key)!;
    item.frequency = newFreq;
  }

  private addToFrequency(key: K, freq: number): void {
    if (!this.frequencies.has(freq)) {
      this.frequencies.set(freq, new Set());
    }
    this.frequencies.get(freq)!.add(key);
  }

  private evictLFU(): void {
    const minFreqSet = this.frequencies.get(this.minFreq);
    if (!minFreqSet || minFreqSet.size === 0) return;

    // Among items with minimum frequency, evict the least recently used
    let lruKey: K | undefined;
    let lruTime = Infinity;

    for (const key of minFreqSet) {
      const item = this.cache.get(key);
      if (item && item.lastUsed < lruTime) {
        lruTime = item.lastUsed;
        lruKey = key;
      }
    }

    if (lruKey !== undefined) {
      minFreqSet.delete(lruKey);
      this.cache.delete(lruKey);
    }
  }

  clear(): void {
    this.cache.clear();
    this.frequencies.clear();
    this.minFreq = 0;
    this.currentTime = 0;
  }
}

// Dynamic port allocation for microservices
class DynamicPortManager {
  private usedPorts = new Set<number>();
  private basePort: number;
  private maxRetries: number;

  static create(basePort = 3000, maxRetries = 100): DynamicPortManager {
    const instance = Object.create(DynamicPortManager.prototype);
    instance.basePort = basePort;
    instance.maxRetries = maxRetries;
    instance.usedPorts = new Set();
    return instance;
  }

  async getAvailablePort(): Promise<number> {
    for (let i = 0; i < this.maxRetries; i++) {
      const port = this.basePort + i;
      
      if (!this.usedPorts.has(port) && await this.isPortAvailable(port)) {
        this.usedPorts.add(port);
        return port;
      }
    }
    
    throw new Error(`No available ports found in range ${this.basePort}-${this.basePort + this.maxRetries}`);
  }

  releasePort(port: number): void {
    this.usedPorts.delete(port);
  }

  private async isPortAvailable(port: number): Promise<boolean> {
    return new Promise((resolve) => {
      const { createServer } = require('net');
      const server = createServer();
      
      server.listen(port, () => {
        server.close(() => resolve(true));
      });
      
      server.on('error', () => resolve(false));
    });
  }
}

// Redis connection factory
function createRedisConnection(): Redis {
  const redisConfig = {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379', 10),
    retryDelayOnFailover: 100,
    maxRetriesPerRequest: 3,
    lazyConnect: true,
    keepAlive: 30000,
    family: 4,
    // Connection pool settings
    enableReadyCheck: true,
    maxLoadingTimeout: 5000
  };

  const redis = new Redis(redisConfig);
  
  redis.on('error', (err) => {
    console.error('[Redis Client Error]', err);
  });

  redis.on('connect', () => {
    console.log('[Redis] Connected successfully');
  });

  redis.on('ready', () => {
    console.log('[Redis] Ready to accept commands');
  });

  return redis;
}

// Initialize services
const redis = createRedisConnection();
const ocrCache = LFUCache.create<string, any>(1000);
const portManager = DynamicPortManager.create(3100);

/**
 * Enhanced legal analysis using WebAssembly or server-side llama.cpp
 */
async function performAdvancedLegalAnalysis(
  text: string, 
  useClientSide = false,
  analysisType: 'comprehensive' | 'quick' | 'risk-focused' = 'comprehensive'
): Promise<any> {
  console.log(`[Legal Analysis] Using ${useClientSide ? 'client-side WebAssembly' : 'server-side'} processing...`);

  try {
    let analysis;

    if (useClientSide && typeof window !== 'undefined') {
      // Client-side WebAssembly processing
      analysis = await webLlamaService.analyzeLegalDocument(
        'OCR Document',
        text,
        analysisType
      );
    } else {
      // Server-side llama.cpp processing
      analysis = await llamaCppService.analyzeLegalDocument(
        'OCR Document',
        text,
        analysisType
      );
    }

    return {
      ...analysis,
      analysisMethod: useClientSide ? 'Client-side WebAssembly (Gemma 3 Legal)' : 'Server-side llama.cpp (Gemma 3 Legal)',
      processingLocation: useClientSide ? 'client' : 'server',
      modelType: 'gemma-3-legal-8b',
      quantization: 'Q4_K_M'
    };

  } catch (error: any) {
    console.error('[Legal Analysis] Failed:', error);
    
    // Fallback analysis
    return {
      summary: text.substring(0, 200) + '...',
      keyTerms: extractBasicKeyTerms(text),
      entities: extractBasicEntities(text),
      risks: [],
      recommendations: ['Consider manual review of this document'],
      confidence: 0.3,
      error: 'Advanced analysis failed, using fallback',
      analysisMethod: 'Fallback pattern matching',
      processingLocation: 'server'
    };
  }
}

/**
 * Fallback entity extraction
 */
function extractBasicEntities(text: string): Array<{ type: string; value: string; confidence: number }> {
  const entities: Array<{ type: string; value: string; confidence: number }> = [];
  
  // Basic regex patterns for common legal entities
  const patterns = {
    person: /(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+/g,
    organization: /(?:LLC|Inc\.|Corp\.|Ltd\.|Company|Corporation)\b/gi,
    date: /\b\d{1,2}\/\d{1,2}\/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b/g,
    currency: /\$[\d,]+(?:\.\d{2})?/g,
    legal_term: /\b(?:contract|agreement|liability|damages|breach|plaintiff|defendant)\b/gi
  };

  for (const [type, pattern] of Object.entries(patterns)) {
    const matches = text.match(pattern);
    if (matches) {
      matches.forEach(match => {
        entities.push({
          type,
          value: match.trim(),
          confidence: 0.7
        });
      });
    }
  }

  return entities.slice(0, 20); // Limit to top 20
}

/**
 * Extract basic key terms
 */
function extractBasicKeyTerms(text: string): string[] {
  const legalTerms = [
    'contract', 'agreement', 'liability', 'damages', 'breach', 'plaintiff', 
    'defendant', 'jurisdiction', 'clause', 'provision', 'penalty', 'indemnity',
    'warranty', 'representation', 'covenant', 'consideration', 'force majeure'
  ];

  const foundTerms = legalTerms.filter(term => 
    text.toLowerCase().includes(term.toLowerCase())
  );

  return foundTerms.slice(0, 10);
}

/**
 * Main POST handler for OCR with enhanced AI analysis
 */
export const POST: RequestHandler = async ({ request }) => {
  try {
    console.log('[OCR] Processing request with enhanced AI analysis...');
    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      throw error(400, 'No file provided');
    }

    // Configuration from headers
    const useClientSideAI = request.headers.get('X-Use-Client-AI') === 'true';
    const enableLegalAnalysis = request.headers.get('X-Enable-Legal-Analysis') !== 'false';
    const analysisType = (request.headers.get('X-Analysis-Type') as any) || 'comprehensive';
    const useCache = request.headers.get('X-Use-Cache') !== 'false';

    // Generate cache key
    const cacheKey = `ocr:${file.name}:${file.size}:${file.lastModified}:${analysisType}`;

    // Check LFU cache first
    if (useCache) {
      const cachedResult = ocrCache.get(cacheKey);
      if (cachedResult) {
        console.log(`[OCR] LFU Cache hit for ${file.name}`);
        return json({ ...cachedResult, fromCache: true });
      }

      // Check Redis cache
      try {
        const redisCached = await redis.get(cacheKey);
        if (redisCached) {
          const result = JSON.parse(redisCached);
          ocrCache.set(cacheKey, result);
          console.log(`[OCR] Redis cache hit for ${file.name}`);
          return json({ ...result, fromCache: true });
        }
      } catch (err) {
        console.warn('[OCR] Redis cache unavailable:', err);
      }
    }

    console.log(`[OCR] Processing ${file.name}, size: ${file.size}, type: ${file.type}`);
    const buffer = Buffer.from(await file.arrayBuffer());

    let extractedText = '';
    let confidence = 0;
    let processingMethod = '';
    const processingStart = Date.now();

    // Text extraction based on file type
    if (file.type === 'application/pdf') {
      processingMethod = 'PDF Text Extraction';
      try {
        const pdfData = await pdfParse(buffer);
        extractedText = pdfData.text;
        confidence = extractedText.length > 0 ? 0.95 : 0.1;
      } catch (err) {
        console.error('[OCR] PDF parsing failed:', err);
        throw new Error('Failed to extract text from PDF');
      }
    } else if (file.type.startsWith('image/')) {
      processingMethod = 'Enhanced Tesseract.js OCR';
      try {
        // Enhanced image preprocessing
        const processedBuffer = await sharp(buffer)
          .greyscale()
          .normalize()
          .sharpen({ sigma: 1.5 })
          .threshold(128)
          .toBuffer();

        const worker = await createWorker({
          logger: m => console.log(`[Tesseract] ${m.status}: ${m.progress * 100}%`)
        });

        await worker.loadLanguage('eng');
        await worker.initialize('eng');
        
        // Optimize for legal documents
        await worker.setParameters({
          tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}"\'-/$%& ',
          tessedit_pageseg_mode: '1', // Automatic page segmentation with OSD
          preserve_interword_spaces: '1'
        });

        const { data } = await worker.recognize(processedBuffer);
        extractedText = data.text;
        confidence = data.confidence / 100;
        
        await worker.terminate();
      } catch (err) {
        console.error('[OCR] Tesseract processing failed:', err);
        throw new Error('Failed to extract text from image');
      }
    } else if (file.type === 'text/plain') {
      processingMethod = 'Plain Text';
      extractedText = buffer.toString('utf-8');
      confidence = 1.0;
    } else {
      throw new Error(`Unsupported file type: ${file.type}. Supported types: PDF, images (PNG, JPG, TIFF), plain text`);
    }

    // Clean extracted text
    extractedText = extractedText
      .replace(/\s+/g, ' ')
      .replace(/[^\x20-\x7E\n\r\t]/g, '') // Remove non-printable chars
      .trim();

    if (!extractedText) {
      throw new Error('No text could be extracted from the file');
    }

    console.log(`[OCR] Extracted ${extractedText.length} characters with ${confidence * 100}% confidence`);

    // Enhanced legal analysis
    let legalAnalysis = null;
    if (enableLegalAnalysis && extractedText.length > 50) {
      try {
        legalAnalysis = await performAdvancedLegalAnalysis(
          extractedText,
          useClientSideAI,
          analysisType
        );
      } catch (err) {
        console.error('[OCR] Legal analysis failed:', err);
        legalAnalysis = {
          error: 'Legal analysis failed',
          message: err instanceof Error ? err.message : 'Unknown error'
        };
      }
    }

    const processingTime = Date.now() - processingStart;

    const result = {
      success: true,
      text: extractedText,
      confidence: confidence,
      processingMethod,
      processingTime,
      metadata: {
        filename: file.name,
        filesize: file.size,
        mimetype: file.type,
        characterCount: extractedText.length,
        wordCount: extractedText.split(/\s+/).length,
        timestamp: new Date().toISOString()
      },
      legal: legalAnalysis,
      fromCache: false,
      clientSideProcessing: useClientSideAI,
      analysisType
    };

    // Cache successful results
    if (useCache && confidence > 0.5) {
      try {
        // LFU cache (in-memory)
        ocrCache.set(cacheKey, result);
        
        // Redis cache (persistent, 1 hour TTL)
        await redis.setex(cacheKey, 3600, JSON.stringify(result));
        console.log('[OCR] Result cached successfully');
      } catch (err) {
        console.warn('[OCR] Failed to cache result:', err);
      }
    }

    return json(result);

  } catch (err: any) {
    console.error('[OCR] Error:', err);
    return json({
      success: false,
      error: err.message,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

/**
 * Enhanced health check endpoint
 */
export const GET: RequestHandler = async () => {
  try {
    const healthChecks = await Promise.allSettled([
      // Redis health
      redis.ping().then(() => ({ redis: 'healthy' })).catch(() => ({ redis: 'unhealthy' })),
      
      // WebAssembly service health
      Promise.resolve(webLlamaService.getHealthStatus()).then(status => ({ webasm: status })),
      
      // Server-side llama.cpp health (if available)
      llamaCppService.getCurrentModel ? 
        Promise.resolve({ serverLlama: llamaCppService.getCurrentModel() ? 'loaded' : 'not-loaded' }) :
        Promise.resolve({ serverLlama: 'not-available' }),
      
      // Port manager status
      Promise.resolve({ 
        portManager: {
          basePort: 3100,
          available: true
        }
      })
    ]);

    const results = healthChecks.reduce((acc, result) => {
      if (result.status === 'fulfilled') {
        Object.assign(acc, result.value);
      }
      return acc;
    }, {} as any);

    // Check if Tesseract.js is working
    let tesseractStatus = 'unknown';
    try {
      const worker = await createWorker();
      await worker.loadLanguage('eng');
      await worker.initialize('eng');
      await worker.terminate();
      tesseractStatus = 'healthy';
    } catch (err) {
      tesseractStatus = 'unhealthy';
    }

    const overallStatus = Object.values(results).every(status => 
      status === 'healthy' || (typeof status === 'object' && status !== null)
    ) && tesseractStatus === 'healthy' ? 'healthy' : 'degraded';

    return json({
      status: overallStatus,
      timestamp: new Date().toISOString(),
      services: {
        tesseract: tesseractStatus,
        cache: {
          lfu: { size: ocrCache['cache'].size, capacity: 1000 },
          ...results
        },
        ai: {
          webAssembly: results.webasm || 'not-available',
          serverSide: results.serverLlama || 'not-available'
        },
        infrastructure: {
          portManager: results.portManager || 'not-available'
        }
      },
      version: '2.0.0',
      features: [
        'Enhanced OCR with preprocessing',
        'WebAssembly AI processing',
        'LFU caching with Redis fallback',
        'Dynamic port management',
        'Legal document analysis',
        'Multi-format support (PDF, images, text)'
      ]
    });

  } catch (err: any) {
    console.error('[OCR Health Check] Error:', err);
    return json({
      status: 'unhealthy',
      error: err.message,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

// Cleanup on module unload
if (typeof process !== 'undefined') {
  process.on('SIGTERM', () => {
    console.log('[OCR] Cleaning up...');
    redis.disconnect();
    ocrCache.clear();
  });

  process.on('SIGINT', () => {
    console.log('[OCR] Cleaning up...');
    redis.disconnect();
    ocrCache.clear();
    process.exit(0);
  });
}
