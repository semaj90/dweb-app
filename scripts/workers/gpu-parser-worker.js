import { parentPort, workerData } from 'worker_threads';

const ENABLE_GPU = (() => {
    try {
        const v = process?.env?.ENABLE_GPU;
        if (typeof v === 'string') return v.toLowerCase() !== 'false' && v !== '0';
    } catch (e) { }
    return true;
})();

class GpuParserWorker {
    constructor(config) {
        this.config = config;
        this.workerId = config.workerId;
        this.workerType = config.workerType;
    }

    async processTask(task) {
        console.log(`🔧 ${this.workerType}-${this.workerId} processing task: ${task.type}`);
        if (!ENABLE_GPU && task.type && task.type.toLowerCase().includes('gpu')) {
            // Short-circuit GPU tasks when GPU is disabled to avoid errors in CI/dev without GPU
            return { success: false, error: 'GPU disabled via ENABLE_GPU', result: null };
        }

        try {
            // Simulate GPU parsing work
            await new Promise(resolve => setTimeout(resolve, Math.random() * 1000));

            return {
                success: true,
                result: `Task ${task.id} completed by ${this.workerType}-${this.workerId}`,
                processedAt: new Date().toISOString()
            };
        } catch (error) {
            throw new Error(`Task processing failed: ${error.message}`);
        }
    }
}

const worker = new GpuParserWorker(workerData);

parentPort.on('message', async (task) => {
    try {
        const result = await worker.processTask(task);
        parentPort.postMessage({
            type: 'result',
            taskId: task.id,
            result
        });
    } catch (error) {
        parentPort.postMessage({
            type: 'error',
            taskId: task.id,
            error: error.message
        });
    }
});

parentPort.postMessage({ type: 'ready' });