// ...existing imports...
import { GpuAcceleratedJsonParser } from '../wasm/gpu-json-parser';
import { JsonParserBenchmark } from '../wasm/benchmark-json-parser';

// ...existing interfaces...

interface WasmPerformanceMetrics {
    parserInitTime: number;
    averageParseTime: number;
    cacheHitRate: number;
    memoryUsage: number;
    gpuAcceleration: boolean;
}

// ...existing class declaration...

    private jsonParser: GpuAcceleratedJsonParser | null = null;
    private wasmMetrics: WasmPerformanceMetrics = {
        parserInitTime: 0,
        averageParseTime: 0,
        cacheHitRate: 0,
        memoryUsage: 0,
        gpuAcceleration: false
    };

    // ...existing methods...

    /**
     * Initialize WebAssembly JSON parser for high-performance processing
     */
    private async initializeWasmParser(): Promise<void> {
        const startTime = performance.now();

        try {
            this.jsonParser = new GpuAcceleratedJsonParser();

            // Test GPU acceleration availability
            const testJson = '{"test": true, "gpu": "acceleration"}';
            const gpuResult = await this.jsonParser.validateWithGpu(testJson);

            this.wasmMetrics.parserInitTime = performance.now() - startTime;
            this.wasmMetrics.gpuAcceleration = !gpuResult.errors.length;

            this.emit('wasmInitialized', {
                initTime: this.wasmMetrics.parserInitTime,
                gpuAcceleration: this.wasmMetrics.gpuAcceleration
            });

        } catch (error) {
            console.warn('WebAssembly JSON parser initialization failed:', error);
        }
    }

    /**
     * Process large JSON configurations with GPU acceleration
     */
    async processJsonConfig(jsonData: string, options: {
        useCache?: boolean;
        useGpuValidation?: boolean;
        benchmark?: boolean;
    } = {}): Promise<{
        parsed: any;
        metrics: any;
        valid: boolean;
        processingTime: number;
    }> {
        if (!this.jsonParser) {
            await this.initializeWasmParser();
        }

        const startTime = performance.now();
        const { useCache = true, useGpuValidation = true, benchmark = false } = options;

        try {
            // Parse JSON with WebAssembly acceleration
            const parseResult = await this.jsonParser!.parse(jsonData, {
                useCache,
                useWorker: jsonData.length > 100000
            });

            if (!parseResult.success) {
                throw new Error(parseResult.errorMessage || 'JSON parsing failed');
            }

            // Validate with GPU acceleration if available
            let validationResult = { valid: true, errors: [] };
            if (useGpuValidation) {
                validationResult = await this.jsonParser!.validateWithGpu(jsonData);
            }

            // Get performance metrics
            const metrics = await this.jsonParser!.getMetrics();
            const cacheStats = await this.jsonParser!.getCacheStats();

            // Update internal metrics
            this.wasmMetrics.averageParseTime = metrics.parseTime;
            this.wasmMetrics.cacheHitRate = cacheStats.hitRate;
            this.wasmMetrics.memoryUsage = metrics.documentSize;

            const processingTime = performance.now() - startTime;

            // Run benchmark if requested
            if (benchmark && jsonData.length > 10000) {
                const benchmarkSuite = new JsonParserBenchmark();
                const benchmarkResult = await benchmarkSuite.comparePerformance('custom', 10);

                console.log('🚀 JSON Processing Benchmark:', {
                    speedup: benchmarkResult.speedup,
                    efficiency: benchmarkResult.efficiency,
                    cacheHitRate: benchmarkResult.cacheHitRate
                });

                benchmarkSuite.dispose();
            }

            return {
                parsed: parseResult,
                metrics: {
                    ...metrics,
                    cacheStats,
                    validation: validationResult,
                    gpu: this.wasmMetrics.gpuAcceleration
                },
                valid: validationResult.valid,
                processingTime
            };

        } catch (error) {
            console.error('JSON processing failed:', error);
            throw error;
        }
    }

    /**
     * Optimize Docker configuration files with GPU-accelerated JSON processing
     */
    async optimizeDockerConfig(configPath: string): Promise<{
        optimized: any;
        recommendations: string[];
        performance: any;
    }> {
        try {
            // Read Docker configuration
            const configData = await this.readFile(configPath);

            // Process with GPU acceleration
            const result = await this.processJsonConfig(configData, {
                useCache: true,
                useGpuValidation: true,
                benchmark: true
            });

            if (!result.valid) {
                throw new Error('Invalid Docker configuration JSON');
            }

            const config = JSON.parse(configData);
            const recommendations: string[] = [];

            // Analyze and optimize configuration
            if (config.services) {
                for (const [serviceName, service] of Object.entries(config.services as any)) {
                    // Memory optimization
                    if (service.mem_limit && this.parseMemorySize(service.mem_limit) > 2 * 1024 * 1024 * 1024) {
                        recommendations.push(`Consider reducing memory limit for ${serviceName}`);
                    }

                    // CPU optimization
                    if (service.cpus && parseFloat(service.cpus) > 2.0) {
                        recommendations.push(`Consider reducing CPU allocation for ${serviceName}`);
                    }

                    // Add GPU acceleration for compatible services
                    if (serviceName.includes('ai') || serviceName.includes('ml') || serviceName.includes('gpu')) {
                        if (!service.runtime && this.wasmMetrics.gpuAcceleration) {
                            service.runtime = 'nvidia';
                            service.environment = service.environment || [];
                            service.environment.push('NVIDIA_VISIBLE_DEVICES=all');
                            recommendations.push(`Added GPU runtime for ${serviceName}`);
                        }
                    }
                }
            }

            // Optimize for WebAssembly services
            if (config.services?.['wasm-parser']) {
                config.services['wasm-parser'].environment = config.services['wasm-parser'].environment || [];
                config.services['wasm-parser'].environment.push('WASM_THREADS=4');
                config.services['wasm-parser'].environment.push('WASM_MEMORY=512MB');
                recommendations.push('Optimized WebAssembly parser service');
            }

            return {
                optimized: config,
                recommendations,
                performance: {
                    processingTime: result.processingTime,
                    wasmMetrics: this.wasmMetrics,
                    jsonMetrics: result.metrics
                }
            };

        } catch (error) {
            console.error('Docker config optimization failed:', error);
            throw error;
        }
    }

    /**
     * Monitor WebAssembly performance in Docker containers
     */
    async monitorWasmPerformance(): Promise<WasmPerformanceMetrics> {
        if (!this.jsonParser) {
            await this.initializeWasmParser();
        }

        try {
            // Test parsing performance
            const testJson = JSON.stringify({
                timestamp: Date.now(),
                containers: Array.from({ length: 100 }, (_, i) => ({
                    id: `container_${i}`,
                    memory: Math.random() * 1024,
                    cpu: Math.random() * 100,
                    status: Math.random() > 0.5 ? 'running' : 'stopped'
                }))
            });

            const parseStart = performance.now();
            await this.jsonParser!.parse(testJson, { useCache: true });
            const parseTime = performance.now() - parseStart;

            const cacheStats = await this.jsonParser!.getCacheStats();

            this.wasmMetrics.averageParseTime = parseTime;
            this.wasmMetrics.cacheHitRate = cacheStats.hitRate;

            return this.wasmMetrics;

        } catch (error) {
            console.error('WebAssembly performance monitoring failed:', error);
            return this.wasmMetrics;
        }
    }

    // ...existing methods continue...

    /**
     * Get comprehensive system metrics including WebAssembly performance
     */
    async getEnhancedMetrics(): Promise<SystemMetrics & { wasm: WasmPerformanceMetrics }> {
        const baseMetrics = await this.getSystemMetrics();
        const wasmMetrics = await this.monitorWasmPerformance();

        return {
            ...baseMetrics,
            wasm: wasmMetrics
        };
    }

    /**
     * Cleanup WebAssembly resources
     */
    dispose(): void {
        if (this.jsonParser) {
            this.jsonParser.dispose();
            this.jsonParser = null;
        }
        super.dispose?.();
    }