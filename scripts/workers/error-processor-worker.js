import { parentPort, workerData } from 'worker_threads';

class ErrorProcessorWorker {
    constructor(config) {
        this.config = config;
        this.workerId = config.workerId;
        this.workerType = config.workerType;
    }
    
    async processTask(task) {
        console.log(`🔧 ${this.workerType}-${this.workerId} processing task: ${task.type}`);
        
        try {
            // Simulate error processing work
            await new Promise(resolve => setTimeout(resolve, Math.random() * 500));
            
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

const worker = new ErrorProcessorWorker(workerData);

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