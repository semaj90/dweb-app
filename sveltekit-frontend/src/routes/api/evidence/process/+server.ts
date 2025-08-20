// Simplified evidence processing endpoint for embedding jobs.
// POST: process next queued ingestion job
// GET:  ?jobId=... returns status

import { json, type RequestHandler } from '@sveltejs/kit';
import { processNextJob, getJobStatus } from '$lib/server/embedding/pgvector-embedding-repository';

export const POST: RequestHandler = async () => {
  const status = await processNextJob();
  return json({ processed: !!status, status });
};

export const GET: RequestHandler = async ({ url }) => {
  const jobId = url.searchParams.get('jobId');
  if (!jobId) return json({ error: 'jobId required' }, { status: 400 });
  const status = getJobStatus(jobId);
  if (!status) return json({ error: 'not found' }, { status: 404 });
  return json(status);
};

export const DELETE: RequestHandler = async () => {
  return json({ message: 'Cancellation not implemented in minimal queue' }, { status: 501 });
};
