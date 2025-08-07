/**
 * SIMD JSON + Redis Client Service
 * Integrates with Go microservice for ultra-high performance JSON processing
 */

export interface SIMDParseResult {
  parse_time: string;
  size: number;
  fields: number;
}

export interface SIMDCacheResult {
  cached: boolean;
  key?: string;
}

export interface SIMDHealthCheck {
  status: string;
  simd: boolean;
  redis?: boolean;
}

export class SIMDRedisClient {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl = 'http://localhost:8080', timeout = 5000) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
  }

  /**
   * Check SIMD + Redis service health
   */
  async healthCheck(): Promise<SIMDHealthCheck> {
    const response = await fetch(`${this.baseUrl}/health`, {
      signal: AbortSignal.timeout(this.timeout)
    });
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }
    
    return response.json();
  }

  /**
   * Parse JSON with SIMD optimization
   */
  async parseJSON(data: any): Promise<SIMDParseResult> {
    const jsonData = typeof data === 'string' ? data : JSON.stringify(data);
    
    const response = await fetch(`${this.baseUrl}/simd-parse`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: jsonData,
      signal: AbortSignal.timeout(this.timeout)
    });

    if (!response.ok) {
      throw new Error(`SIMD parse failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Cache JSON data in Redis
   */
  async cacheJSON(key: string, data: any): Promise<SIMDCacheResult> {
    const jsonData = typeof data === 'string' ? data : JSON.stringify(data);
    
    const response = await fetch(`${this.baseUrl}/redis-json?key=${encodeURIComponent(key)}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: jsonData,
      signal: AbortSignal.timeout(this.timeout)
    });

    if (!response.ok) {
      throw new Error(`Redis cache failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Benchmark SIMD vs Standard JSON parsing
   */
  async benchmark(testData: any, iterations = 1000): Promise<{
    simd_avg_ms: number;
    standard_avg_ms: number;
    speedup_factor: number;
    data_size: number;
  }> {
    const jsonData = JSON.stringify(testData);
    
    // SIMD benchmark
    const simdTimes: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await this.parseJSON(jsonData);
      simdTimes.push(performance.now() - start);
    }
    
    // Standard benchmark
    const standardTimes: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      JSON.parse(jsonData);
      standardTimes.push(performance.now() - start);
    }
    
    const simdAvg = simdTimes.reduce((a, b) => a + b) / simdTimes.length;
    const standardAvg = standardTimes.reduce((a, b) => a + b) / standardTimes.length;
    
    return {
      simd_avg_ms: simdAvg,
      standard_avg_ms: standardAvg,
      speedup_factor: standardAvg / simdAvg,
      data_size: jsonData.length
    };
  }
}

// Singleton instance
export const simdRedisClient = new SIMDRedisClient();