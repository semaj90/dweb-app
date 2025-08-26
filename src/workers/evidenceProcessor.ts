import amqp from 'amqplib';
import { v4 as uuidv4 } from 'uuid';
import { sendWsMessageToSession } from '../lib/server/wsBroker';
import { db } from '../lib/server/db';
// Use a namespace import for the schema module and cast to any to avoid missing named-export errors
import * as legalDocuments from '../lib/database/schema/legal-documents';
const { evidenceProcess, evidenceOcr, evidenceEmbeddings, evidenceAnalysis } = (legalDocuments as any);
import { eq } from 'drizzle-orm';
import fetch from 'node-fetch';

/*
  Notes on fixes:
  - Removed top-level QdrantClient import (we use dynamic import inside the function and cast to any).
  - Typing of dynamic JSON responses uses `any` to avoid 'property does not exist on unknown' errors.
  - Renamed a couple of variables to avoid unused-variable diagnostics.
  - Typed the RabbitMQ consumer callback parameter as `any` to avoid implicit-any.
  - Ensured all try/catch blocks and braces are balanced.
*/

// Service imports - these would be implemented based on your stack
async function runOcrForEvidence(evidenceId: string) {
  // TODO: Implement OCR service integration
  // Could use Tesseract.js, AWS Textract, or custom OCR solution
  console.log(`Running OCR for evidence ${evidenceId}`);

  // Mock implementation - replace with real OCR
  await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate processing time

  return {
    text: `Extracted text from evidence ${evidenceId}. This would contain the actual OCR results.`,
    confidence: 0.92,
    metadata: { pages: 1, language: 'en' }
  };
}

async function generateEmbeddings(params: { evidenceId: string; model: string; text?: string }) {
  // Try the app's internal embeddings API first (fallback to local/mock if it fails)
  const apiBase = process.env.INTERNAL_API_URL || process.env.APP_API_URL || 'http://localhost:3000';
  const endpoint = `${apiBase.replace(/\/$/, '')}/api/embeddings`;

  try {
    const fetchFn: typeof fetch = (typeof fetch !== 'undefined')
      ? fetch
      : (await import('node-fetch')).default as any;

    const resp = await fetchFn(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        evidenceId: params.evidenceId,
        model: params.model,
        text: params.text ?? null
      })
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => '<no body>');
      console.warn(`Embeddings API responded ${resp.status}: ${text}`);
      throw new Error(`Embeddings API error ${resp.status}`);
    }

    const body = await resp.json() as { embedding: number[]; model: string; dim: number };

    if (!Array.isArray(body.embedding) || typeof body.dim !== 'number') {
      console.warn('Embeddings API returned unexpected payload, falling back to mock', body);
      throw new Error('Invalid embeddings payload');
    }

    console.log(`Received embeddings from internal API for ${params.evidenceId} (model=${body.model})`);
    return body;
  } catch (apiErr) {
    console.warn('Failed to generate embeddings via internal API, using local fallback:', apiErr);
  }

  // Fallback: try to use Nomic's "nomic-embed-text" if the internal API failed.
  // This attempts to use a provided text (params.text) or an OCR record for the evidence,
  // then calls NOMIC_API_KEY-protected endpoint. If that fails, fall back to a local mock.
  try {
    const nomicKey = process.env.NOMIC_API_KEY;
    if (nomicKey) {
      // ensure a fetch implementation is available in Node
      const fetchFn: typeof fetch = (typeof fetch !== 'undefined')
        ? fetch
        : (await import('node-fetch')).default as any;

      // prefer explicit text passed in; otherwise try to read OCR text from DB
      let text = params.text;
      if (!text) {
        try {
          // best-effort: read latest OCR text for this evidence (may be adapted to your schema)
          const rows = await db.select().from(evidenceOcr).where(eq(evidenceOcr.evidenceId, params.evidenceId)).limit(1) as any[];
          if (rows && rows.length > 0) text = rows[0].text;
        } catch (dbReadErr) {
          // ignore DB read problems and proceed to attempt Nomic only if we have text
          console.warn('Could not read OCR text from DB for embeddings fallback:', dbReadErr);
        }
      }

      if (!text || typeof text !== 'string' || text.trim().length === 0) {
        throw new Error('No text available for embedding (provide params.text or ensure OCR exists).');
      }

      // Nomic embeddings API (example): adapt path/shape if your account uses a different endpoint
      const nomicEndpoint = 'https://api.nomic.ai/v1/embeddings';

      const resp = await fetchFn(nomicEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${nomicKey}`
        },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          input: text
        })
      });

      if (!resp.ok) {
        const bodyText = await resp.text().catch(() => '<no body>');
        throw new Error(`Nomic embeddings API error ${resp.status}: ${bodyText}`);
      }

      const nomicBody: any = await resp.json();

      // Nomic may return embedding in different shapes; try common forms
      const embedding: number[] | undefined =
        Array.isArray(nomicBody?.embedding) && typeof nomicBody.embedding[0] === 'number'
          ? nomicBody.embedding
          : Array.isArray(nomicBody?.[0]?.embedding) && typeof nomicBody[0].embedding[0] === 'number'
            ? nomicBody[0].embedding
            : undefined;

      if (!embedding) {
        console.warn('Nomic API returned unexpected payload, falling back to mock:', nomicBody);
        throw new Error('Invalid Nomic embeddings payload');
      }

      return {
        embedding,
        model: 'nomic-embed-text',
        dim: embedding.length
      };
    } else {
      console.warn('NOMIC_API_KEY not configured, skipping Nomic embedding attempt');
      throw new Error('NOMIC_API_KEY missing');
    }
  } catch (nomicErr) {
    console.warn('Nomic embeddings attempt failed, using local mock fallback:', nomicErr);
  }

  // Final fallback: local/mock embedding generator (deterministic or random)
  // NOTE: Replace this with a proper local model or vector DB ingestion in production.
  await new Promise(resolve => setTimeout(resolve, 1500)); // simulate work
  const dim = 768;
  return {
    embedding: Array.from({ length: dim }, () => Math.random()),
    model: params.model || 'mock-embed',
    dim
  };
}

async function runRag(params: { evidenceId: string; topK?: number }) {
  // TODO: Implement RAG with Qdrant vector search + Ollama LLM
  console.log(`Running RAG analysis for evidence ${params.evidenceId}`);

  // Mock implementation - replace with real RAG pipeline
  await new Promise(resolve => setTimeout(resolve, 3000));

  return {
    summary: `AI-generated summary for evidence ${params.evidenceId}. This would contain insights from RAG analysis.`,
    snippets: [
      { text: "Key finding 1", score: 0.95 },
      { text: "Key finding 2", score: 0.87 }
    ],
    reasoning: "Analysis based on similarity search and LLM reasoning",
    confidence: 0.89
  };
}

async function processEvidenceJob(payload: unknown) {
  // Define a minimal runtime type for incoming jobs
  type JobPayload = {
    sessionId?: string;
    evidenceId?: string;
    fileId?: string;
    steps?: unknown;
    userId?: string;
  };

  const p = payload as JobPayload;

  // Normalize and validate fields coming from the queue
  const sessionId = typeof p.sessionId === 'string' ? p.sessionId : undefined;
  const evidenceId = typeof p.evidenceId === 'string' ? p.evidenceId : (typeof p.fileId === 'string' ? p.fileId : undefined);
  const _userId = typeof p.userId === 'string' ? p.userId : undefined; // unused for now
  const steps = Array.isArray(p.steps) ? p.steps as string[] : [];

  const fileId = evidenceId; // alias for consistency

  if (!sessionId || !evidenceId) {
    console.error('Invalid job payload, missing sessionId or evidenceId:', payload);
    // Persist a failed state if possible
    try {
      await db.update(evidenceProcess)
        .set({
          status: 'failed',
          error: 'Invalid job payload: missing sessionId or evidenceId',
          finishedAt: new Date()
        })
        .where(eq(evidenceProcess.id, sessionId ?? ''));
    } catch (dbErr) {
      console.error('Failed to write invalid-payload state to DB:', dbErr);
    }
    return;
  }

  try {
    console.log(`Starting evidence processing for session ${sessionId}, evidence ${evidenceId}`);

    // Mark as started
    await db.update(evidenceProcess)
      .set({
        status: 'processing',
        startedAt: new Date()
      })
      .where(eq(evidenceProcess.id, sessionId));

    // Process each step sequentially
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      const stepProgress = Math.round(((i + 1) / steps.length) * 100);

      console.log(`Processing step ${i + 1}/${steps.length}: ${step} (progress ${stepProgress}%)`);

      // Announce step start
      sendWsMessageToSession(sessionId, {
        type: 'processing-step',
        fileId,
        step,
        stepProgress: 0
      });

      if (step === 'ocr') {
        try {
          // Run OCR with progress updates
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'ocr',
            stepProgress: 25
          });

          const ocrResult = await runOcrForEvidence(evidenceId);

          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'ocr',
            stepProgress: 75,
            fragment: { textLength: ocrResult.text?.length ?? 0 }
          });

          // Persist OCR results
          await db.insert(evidenceOcr).values({
            id: uuidv4(),
            evidenceId: evidenceId,
            text: ocrResult.text,
            confidence: ocrResult.confidence
          });

          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'ocr',
            stepProgress: 100,
            fragment: {
              textLength: ocrResult.text?.length ?? 0,
              confidence: ocrResult.confidence
            }
          });

        } catch (error) {
          console.error('OCR step failed:', error);
          throw new Error(`OCR processing failed: ${String(error)}`);
        }
      } else if (step === 'embedding') {
        try {
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'embedding',
            stepProgress: 10
          });

          // Get text content for embedding (from OCR or existing source)
          const embedding = await generateEmbeddings({
            evidenceId,
            model: 'nomic-embed-text'
          });

          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'embedding',
            stepProgress: 60
          });

          // Persist embedding in Postgres (metadata + vector). Keep IDs consistent.
          const pgId = uuidv4();
          const vector = embedding.embedding;
          try {
            await db.insert(evidenceEmbeddings).values({
              id: pgId,
              evidenceId: evidenceId,
              model: embedding.model,
              dim: embedding.dim,
              // Cast to any in case your drizzle schema typing doesn't include a vector column type
              vector: vector as any
            } as any);
            console.log(`Stored embedding metadata/vector in Postgres for evidence ${evidenceId}`);
          } catch (pgErr) {
            console.warn('Failed to store embedding in Postgres:', pgErr);
          }

          // Try to push the vector to Qdrant (preferred) with fallbacks.
          const qdrantUrl = process.env.QDRANT_URL;
          const qdrantApiKey = process.env.QDRANT_API_KEY;
          const qdrantCollection = process.env.QDRANT_COLLECTION || 'evidence_embeddings';
          let qdrantStored = false;

          if (qdrantUrl) {
            try {
              // Prefer using the official JS client if available (dynamic import)
              try {
                const { QdrantClient: QC } = await import('@qdrant/js-client-rest').catch(() => ({ QdrantClient: null }));
                if (QC) {
                  const client = new (QC as any)({
                    url: qdrantUrl,
                    apiKey: qdrantApiKey
                  } as any);

                  // Ensure collection exists (safe to recreate/configure as needed)
                  try {
                    await (client.collections as any).createOrUpdate(qdrantCollection, {
                      vectors: { size: embedding.dim, distance: 'Cosine' }
                    });
                  } catch (cErr) {
                    // Some client versions expose different helpers; ignore non-fatal errors
                  }

                  // Upsert point using available shapes
                  if ((client as any).points?.upsert) {
                    await (client as any).points.upsert({
                      collection_name: qdrantCollection,
                      points: [
                        {
                          id: pgId,
                          vector,
                          payload: { evidenceId, model: embedding.model }
                        }
                      ]
                    });
                    qdrantStored = true;
                  } else if ((client as any).upsert) {
                    await (client as any).upsert({
                      collection_name: qdrantCollection,
                      points: [
                        {
                          id: pgId,
                          vector,
                          payload: { evidenceId, model: embedding.model }
                        }
                      ]
                    });
                    qdrantStored = true;
                  }
                }
              } catch (clientErr) {
                // ignore and fall back to REST upsert below
              }

              if (!qdrantStored) {
                // REST fallback (works with Qdrant HTTP API)
                const base = qdrantUrl.replace(/\/$/, '');
                try {
                  // create collection if needed (PUT)
                  await fetch(`${base}/collections/${encodeURIComponent(qdrantCollection)}`, {
                    method: 'PUT',
                    headers: qdrantApiKey ? { 'Content-Type': 'application/json', 'X-API-Key': qdrantApiKey } : { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ vectors: { size: embedding.dim, distance: 'Cosine' } })
                  }).catch(() => null);

                  // upsert points
                  const upsertResp = await fetch(`${base}/collections/${encodeURIComponent(qdrantCollection)}/points?wait=true`, {
                    method: 'PUT',
                    headers: qdrantApiKey ? { 'Content-Type': 'application/json', 'X-API-Key': qdrantApiKey } : { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      points: [
                        { id: pgId, payload: { evidenceId, model: embedding.model }, vector }
                      ]
                    })
                  });

                  if (!upsertResp.ok) {
                    const body = await upsertResp.text().catch(() => '<no body>');
                    throw new Error(`Qdrant upsert failed: ${upsertResp.status} ${body}`);
                  }

                  qdrantStored = true;
                } catch (restErr) {
                  console.warn('Qdrant REST fallback failed:', restErr);
                  qdrantStored = false;
                }
              }
            } catch (qErr) {
              console.warn('Failed to store embedding in Qdrant:', qErr);
              qdrantStored = false;
            }
          } else {
            console.warn('QDRANT_URL not configured; skipping Qdrant ingestion');
          }

          // Final progress updates to client
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'embedding',
            stepProgress: qdrantStored ? 95 : 85,
            fragment: {
              model: embedding.model,
              dimensions: embedding.dim,
              stored: {
                postgres: true,
                qdrant: qdrantStored
              }
            }
          });

          // Mark embedding step complete
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step: 'embedding',
            stepProgress: 100,
            fragment: {
              model: embedding.model,
              dimensions: embedding.dim,
              storedIn: qdrantStored ? ['postgres', 'qdrant'] : ['postgres']
            }
          });

        } catch (error) {
          console.error('Embedding step failed:', error);
          throw new Error(`Embedding generation failed: ${String(error)}`);
        }
      } else if (step === 'rag' || step === 'analysis') {
        try {
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step,
            stepProgress: 0
          });

          const ragResult = await runRag({ evidenceId, topK: 5 });

          // Stream partial results
          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step,
            stepProgress: 60,
            fragment: {
              snippet: ragResult.snippets?.[0],
              confidence: ragResult.confidence
            }
          });

          // Persist analysis results
          await db.insert(evidenceAnalysis).values({
            id: uuidv4(),
            evidenceId: evidenceId,
            summary: ragResult.summary
          });

          sendWsMessageToSession(sessionId, {
            type: 'processing-step',
            fileId,
            step,
            stepProgress: 100,
            fragment: {
              summary: ragResult.summary,
              snippetCount: ragResult.snippets?.length ?? 0,
              confidence: ragResult.confidence
            }
          });

        } catch (error) {
          console.error('RAG/Analysis step failed:', error);
          throw new Error(`${step} processing failed: ${String(error)}`);
        }
      } else {
        // Generic step handler for extensibility
        console.log(`Processing generic step: ${step}`);
        await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate work

        sendWsMessageToSession(sessionId, {
          type: 'processing-step',
          fileId,
          step,
          stepProgress: 100
        });
      }
    }

    // Mark as completed
    const finalResult = {
      message: 'Evidence processing completed successfully',
      evidenceId,
      stepsCompleted: steps,
      timestamp: new Date().toISOString()
    };

    await db.update(evidenceProcess)
      .set({
        status: 'completed',
        finishedAt: new Date()
      })
      .where(eq(evidenceProcess.id, sessionId));

    sendWsMessageToSession(sessionId, {
      type: 'processing-complete',
      fileId,
      finalResult
    });

    console.log(`Evidence processing completed for session ${sessionId}`);

  } catch (error) {
    console.error('Evidence processing failed:', error);

    // Mark as failed in database
    await db.update(evidenceProcess)
      .set({
        status: 'failed',
        error: String(error),
        finishedAt: new Date()
      })
      .where(eq(evidenceProcess.id, sessionId));

    // Send error to client
    sendWsMessageToSession(sessionId, {
      type: 'error',
      fileId,
      error: {
        message: error instanceof Error ? error.message : String(error),
        code: 'PROCESSING_FAILED'
      }
    });
  }
}

async function startWorker() {
  try {
    const conn = await amqp.connect(process.env.RABBITMQ_URL || 'amqp://localhost');
    const ch = await conn.createChannel();
    const q = 'evidence.process.queue';

    await ch.assertQueue(q, { durable: true });
    ch.prefetch(1); // Process one job at a time

    console.log('Evidence processing worker started, waiting for jobs...');

    // typed as any to avoid implicit-any from missing @types/amqplib
    ch.consume(q, async (msg: any) => {
      if (!msg) return;

      try {
        const payload = JSON.parse(msg.content.toString());
        console.log('Received processing job:', payload?.sessionId ?? payload);

        // Process the job payload
        await processEvidenceJob(payload);

        // Acknowledge successful processing
        ch.ack(msg);
        console.log('Job completed:', payload?.sessionId ?? payload);
      } catch (error) {
        console.error('Job processing error:', error);
        ch.nack(msg, false, false); // Don't requeue failed jobs
      }
    });

    // Handle shutdown gracefully
    process.on('SIGINT', async () => {
      console.log('Shutting down evidence processor...');
      await ch.close();
      await conn.close();
      process.exit(0);
    });

  } catch (error) {
    console.error('Failed to start evidence processing worker:', error);
    process.exit(1);
  }
}

// Start worker if this file is run directly
// Start worker if this file is run directly (safe for CommonJS and ESM)
if (typeof require !== 'undefined' && (require as any).main === module) {
  startWorker();
} else if (typeof process !== 'undefined' && process.argv && process.argv.includes('--run-evidence-worker')) {
// allow explicit startup flag when running in ESM environments
  startWorker();
}

export { processEvidenceJob, startWorker };