// workers/evidenceProcessor.ts
import { consumeFromQueue, publishToQueue } from '../sveltekit-frontend/src/lib/server/rabbitmq.js';
import { sendWsMessageToSession, initializeWsBroker } from '../sveltekit-frontend/src/lib/server/wsBroker.js';
import { runOcrForEvidence } from './services/ocr.js';
import { generateEmbeddings } from './services/embeddings.js';
import { runRag } from './services/rag.js';
import { v4 as uuidv4 } from 'uuid';
import { db } from '../sveltekit-frontend/src/lib/server/db.js';

interface ProcessingJob {
  sessionId: string;
  evidenceId: string;
  steps: string[];
  userId: string;
  timestamp: string;
}

interface ControlMessage {
  action: 'cancel' | 'pause' | 'resume';
  sessionId: string;
  timestamp: string;
}

// Global state for tracking active jobs
const activeJobs = new Map<string, { cancelled: boolean; paused: boolean }>();

async function updateProcessStatus(sessionId: string, status: string, error?: string) {
  try {
    const updates: any = {
      status,
      updated_at: new Date()
    };
    
    if (status === 'processing') {
      updates.started_at = new Date();
    } else if (status === 'completed' || status === 'failed' || status === 'cancelled') {
      updates.finished_at = new Date();
    }
    
    if (error) {
      updates.error = error;
    }
    
    await db.update('evidence_process')
      .set(updates)
      .where('id', '=', sessionId);
      
  } catch (err) {
    console.error('❌ Failed to update process status:', err);
  }
}

async function processEvidenceJob(job: ProcessingJob, ack: () => void, nack: () => void) {
  const { sessionId, evidenceId, steps, userId } = job;
  const fileId = evidenceId; // alias for WebSocket messages
  
  console.log(`🚀 Starting evidence processing job: ${sessionId}`);
  
  // Initialize job state
  activeJobs.set(sessionId, { cancelled: false, paused: false });
  
  try {
    // Mark as processing
    await updateProcessStatus(sessionId, 'processing');
    
    sendWsMessageToSession(sessionId, {
      type: 'processing-step',
      fileId,
      step: 'initialization',
      stepProgress: 0
    });
    
    // Process each step sequentially
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      
      // Check for cancellation
      const jobState = activeJobs.get(sessionId);
      if (jobState?.cancelled) {
        console.log(`⏹️ Job cancelled: ${sessionId}`);
        await updateProcessStatus(sessionId, 'cancelled');
        sendWsMessageToSession(sessionId, {
          type: 'error',
          fileId,
          error: { message: 'Processing cancelled by user', code: 'CANCELLED' }
        });
        ack();
        return;
      }
      
      // Wait if paused
      while (jobState?.paused) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      
      console.log(`📋 Processing step ${i + 1}/${steps.length}: ${step} for evidence ${evidenceId}`);
      
      // Announce step start
      sendWsMessageToSession(sessionId, {
        type: 'processing-step',
        fileId,
        step,
        stepProgress: 0
      });
      
      try {
        if (step === 'ocr') {
          console.log('🔍 Running OCR processing...');
          
          const ocrResult = await runOcrForEvidence(evidenceId);
          
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'ocr',
            stepProgress: 50,
            fragment: { textLength: ocrResult.text?.length ?? 0 }
          });
          
          // Persist OCR result
          await db.insert('evidence_ocr').values({
            id: uuidv4(),
            evidence_id: evidenceId,
            text: ocrResult.text,
            confidence: ocrResult.confidence,
            metadata: JSON.stringify(ocrResult.metadata || {}),
            created_at: new Date()
          });
          
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'ocr',
            stepProgress: 100,
            fragment: { 
              textPreview: ocrResult.text?.substring(0, 200) + '...',
              confidence: ocrResult.confidence 
            }
          });
          
        } else if (step === 'embedding') {
          console.log('🧠 Generating embeddings...');
          
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'embedding',
            stepProgress: 10
          });
          
          const embeddingResult = await generateEmbeddings({ 
            evidenceId, 
            model: 'nomic-embed-text' 
          });
          
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'embedding',
            stepProgress: 60
          });
          
          // Store vector into database
          await db.insert('evidence_embeddings').values({
            id: uuidv4(),
            evidence_id: evidenceId,
            model: embeddingResult.model,
            dim: embeddingResult.dim,
            vector: JSON.stringify(embeddingResult.vector),
            metadata: JSON.stringify(embeddingResult.metadata || {}),
            created_at: new Date()
          });
          
          // TODO: Upsert to Qdrant for similarity search
          // await qdrantClient.upsert(collection, { id: evidenceId, vector: embeddingResult.vector });
          
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'embedding',
            stepProgress: 100,
            fragment: { 
              model: embeddingResult.model,
              dimensions: embeddingResult.dim 
            }
          });
          
        } else if (step === 'rag' || step === 'analysis') {
          console.log('📚 Running RAG analysis...');
          
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'rag',
            stepProgress: 0
          });
          
          const ragResult = await runRag({ evidenceId, topK: 5 });
          
          // Stream partial results
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'rag',
            stepProgress: 60,
            fragment: { 
              snippet: ragResult.snippets?.[0],
              relevantDocsCount: ragResult.relevantDocs?.length || 0
            }
          });
          
          // Persist analysis
          await db.insert('evidence_analysis').values({
            id: uuidv4(),
            evidence_id: evidenceId,
            summary: ragResult.summary,
            confidence: ragResult.confidence,
            snippets: JSON.stringify(ragResult.snippets),
            relevant_docs: JSON.stringify(ragResult.relevantDocs),
            created_at: new Date()
          });
          
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'rag',
            stepProgress: 100,
            fragment: { 
              summary: ragResult.summary.substring(0, 200) + '...',
              confidence: ragResult.confidence
            }
          });
          
        } else {
          // Generic step processing
          console.log(`🔧 Processing generic step: ${step}`);
          
          // Simulate processing time
          await new Promise(resolve => setTimeout(resolve, 2000));
          
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step,
            stepProgress: 100
          });
        }
        
      } catch (stepError) {
        console.error(`❌ Error in step ${step}:`, stepError);
        
        sendWsMessageToSession(sessionId, {
          type: 'error',
          fileId,
          error: { 
            message: `Failed in step: ${step}`,
            code: 'STEP_FAILED',
            meta: { step, error: String(stepError) }
          }
        });
        
        // Continue to next step instead of failing entire job
        continue;
      }
    }
    
    // Final completion
    const finalResult = { 
      message: 'Evidence processing completed successfully',
      evidenceId,
      stepsCompleted: steps.length,
      timestamp: new Date().toISOString()
    };
    
    await updateProcessStatus(sessionId, 'completed');
    
    sendWsMessageToSession(sessionId, {
      type: 'processing-complete',
      fileId,
      finalResult
    });
    
    console.log(`✅ Evidence processing completed: ${sessionId}`);
    ack();
    
  } catch (error) {
    console.error(`❌ Evidence processing failed for ${sessionId}:`, error);
    
    await updateProcessStatus(sessionId, 'failed', String(error));
    
    sendWsMessageToSession(sessionId, {
      type: 'error',
      fileId,
      error: { 
        message: String(error),
        code: 'PROCESSING_FAILED'
      }
    });
    
    nack(); // Requeue for retry (based on your retry policy)
    
  } finally {
    // Cleanup job state
    activeJobs.delete(sessionId);
  }
}

async function handleControlMessage(message: ControlMessage, ack: () => void, nack: () => void) {
  const { action, sessionId } = message;
  
  console.log(`🎛️ Control message: ${action} for session ${sessionId}`);
  
  try {
    const jobState = activeJobs.get(sessionId);
    
    if (!jobState) {
      console.log(`⚠️ No active job found for session ${sessionId}`);
      ack();
      return;
    }
    
    switch (action) {
      case 'cancel':
        jobState.cancelled = true;
        console.log(`⏹️ Marked job for cancellation: ${sessionId}`);
        break;
        
      case 'pause':
        jobState.paused = true;
        console.log(`⏸️ Paused job: ${sessionId}`);
        break;
        
      case 'resume':
        jobState.paused = false;
        console.log(`▶️ Resumed job: ${sessionId}`);
        break;
    }
    
    ack();
    
  } catch (error) {
    console.error(`❌ Error handling control message:`, error);
    nack();
  }
}

async function startWorker() {
  console.log('🏭 Starting Evidence Processing Worker...');
  
  try {
    // Initialize WebSocket broker
    await initializeWsBroker();
    
    // Start consuming main processing queue
    await consumeFromQueue('evidence.process.queue', processEvidenceJob);
    
    // Start consuming control messages
    await consumeFromQueue('evidence.process.control', handleControlMessage);
    
    console.log('✅ Evidence processing worker started successfully');
    console.log('📋 Listening for jobs on: evidence.process.queue');
    console.log('🎛️ Listening for control messages on: evidence.process.control');
    
    // Health check endpoint (if needed)
    setInterval(() => {
      console.log(`💓 Worker health: ${activeJobs.size} active jobs`);
    }, 60000);
    
  } catch (error) {
    console.error('❌ Failed to start evidence processing worker:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('🛑 Received SIGINT, shutting down gracefully...');
  
  // Cancel all active jobs
  for (const sessionId of activeJobs.keys()) {
    activeJobs.get(sessionId)!.cancelled = true;
  }
  
  // Wait for jobs to finish (with timeout)
  let timeout = 30000; // 30 seconds
  const interval = 1000;
  
  while (activeJobs.size > 0 && timeout > 0) {
    console.log(`⏳ Waiting for ${activeJobs.size} jobs to finish...`);
    await new Promise(resolve => setTimeout(resolve, interval));
    timeout -= interval;
  }
  
  if (activeJobs.size > 0) {
    console.log(`⚠️ Force shutdown with ${activeJobs.size} jobs still active`);
  }
  
  process.exit(0);
});

// Start the worker
startWorker().catch(error => {
  console.error('❌ Worker startup failed:', error);
  process.exit(1);
});
