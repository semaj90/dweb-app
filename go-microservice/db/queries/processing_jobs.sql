-- name: CreateProcessingJob :one
INSERT INTO processing_jobs (
    job_type, status, priority, input_data, processing_options,
    scheduled_at, expires_at, created_by, assigned_worker, resource_requirements
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
) RETURNING *;

-- name: GetProcessingJob :one
SELECT * FROM processing_jobs WHERE id = $1;

-- name: GetJobsByStatus :many
SELECT * FROM processing_jobs 
WHERE status = $1
ORDER BY priority DESC, created_at ASC
LIMIT $2 OFFSET $3;

-- name: GetJobsByWorker :many
SELECT * FROM processing_jobs 
WHERE assigned_worker = $1
ORDER BY created_at DESC
LIMIT $2 OFFSET $3;

-- name: GetQueuedJobs :many
SELECT * FROM processing_jobs 
WHERE status = 'queued' 
AND (scheduled_at IS NULL OR scheduled_at <= NOW())
AND (expires_at IS NULL OR expires_at > NOW())
ORDER BY priority DESC, created_at ASC
LIMIT $1;

-- name: UpdateJobStatus :exec
UPDATE processing_jobs 
SET status = $2, current_stage = $3, progress_percentage = $4,
    started_at = CASE WHEN $2 = 'processing' AND started_at IS NULL THEN NOW() ELSE started_at END,
    completed_at = CASE WHEN $2 IN ('completed', 'failed', 'cancelled') THEN NOW() ELSE completed_at END
WHERE id = $1;

-- name: UpdateJobProgress :exec
UPDATE processing_jobs 
SET progress_percentage = $2, current_stage = $3
WHERE id = $1;

-- name: UpdateJobResult :exec
UPDATE processing_jobs 
SET status = $2, output_data = $3, processing_time_ms = $4,
    gpu_memory_used_mb = $5, completed_at = NOW()
WHERE id = $1;

-- name: UpdateJobError :exec
UPDATE processing_jobs 
SET status = 'failed', error_message = $2, retry_count = retry_count + 1,
    completed_at = NOW()
WHERE id = $1;

-- name: AssignJobToWorker :one
UPDATE processing_jobs 
SET status = 'processing', assigned_worker = $2, started_at = NOW()
WHERE id = $1 AND status = 'queued'
RETURNING *;

-- name: CancelJob :exec
UPDATE processing_jobs 
SET status = 'cancelled', completed_at = NOW()
WHERE id = $1 AND status IN ('queued', 'processing');

-- name: RetryJob :exec
UPDATE processing_jobs 
SET status = 'queued', error_message = NULL, started_at = NULL,
    completed_at = NULL, assigned_worker = NULL, progress_percentage = 0,
    current_stage = NULL, scheduled_at = NOW() + INTERVAL '5 minutes'
WHERE id = $1 AND retry_count < max_retries;

-- name: GetJobStatistics :one
SELECT 
    COUNT(*) as total_jobs,
    COUNT(*) FILTER (WHERE status = 'queued') as queued_jobs,
    COUNT(*) FILTER (WHERE status = 'processing') as processing_jobs,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_jobs,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_jobs,
    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled_jobs,
    AVG(processing_time_ms) FILTER (WHERE processing_time_ms IS NOT NULL) as avg_processing_time_ms,
    MAX(processing_time_ms) as max_processing_time_ms,
    MIN(processing_time_ms) FILTER (WHERE processing_time_ms > 0) as min_processing_time_ms
FROM processing_jobs;

-- name: GetJobsByDateRange :many
SELECT * FROM processing_jobs 
WHERE created_at BETWEEN $1 AND $2
ORDER BY created_at DESC
LIMIT $3 OFFSET $4;

-- name: CleanupCompletedJobs :exec
DELETE FROM processing_jobs 
WHERE status IN ('completed', 'failed', 'cancelled')
AND completed_at < NOW() - INTERVAL '1 day' * $1;

-- name: GetExpiredJobs :many
SELECT * FROM processing_jobs 
WHERE expires_at IS NOT NULL 
AND expires_at < NOW() 
AND status NOT IN ('completed', 'failed', 'cancelled');