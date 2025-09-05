#!/usr/bin/env node

/**
 * Legal AI Worker Startup Script
 * Starts the BullMQ worker for processing documents via Go server
 */

import { createLegalAIWorker } from './src/lib/workers/legal-ai-worker.ts';

console.log('🚀 Starting Legal AI Worker...');
console.log('📊 Configuration:');
console.log(`   - Go Server: ${process.env.GO_SERVER_URL || 'http://localhost:8080'}`);
console.log(`   - Redis: ${process.env.REDIS_URL || 'redis://localhost:6379'}`);
console.log(`   - Node Environment: ${process.env.NODE_ENV || 'development'}`);
console.log('');

let worker;

try {
	// Create and start the worker
	worker = createLegalAIWorker();

	// Graceful shutdown handling
	const shutdown = async (signal) => {
		console.log(`\n🛑 Received ${signal}, gracefully shutting down...`);
		
		if (worker) {
			try {
				await worker.close();
				console.log('✅ Legal AI Worker closed successfully');
			} catch (error) {
				console.error('❌ Error closing worker:', error);
			}
		}
		
		process.exit(0);
	};

	// Handle shutdown signals
	process.on('SIGINT', () => shutdown('SIGINT'));
	process.on('SIGTERM', () => shutdown('SIGTERM'));

	// Handle uncaught exceptions
	process.on('uncaughtException', (error) => {
		console.error('💥 Uncaught Exception:', error);
		shutdown('uncaughtException');
	});

	process.on('unhandledRejection', (reason, promise) => {
		console.error('💥 Unhandled Rejection at:', promise, 'reason:', reason);
		shutdown('unhandledRejection');
	});

	console.log('✅ Legal AI Worker started successfully');
	console.log('🔄 Waiting for jobs...');
	console.log('Press Ctrl+C to stop');

} catch (error) {
	console.error('❌ Failed to start Legal AI Worker:', error);
	process.exit(1);
}