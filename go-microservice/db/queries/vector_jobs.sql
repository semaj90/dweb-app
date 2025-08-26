-- name: CreateVectorJob :one
INSERT INTO vector_jobs (
    job_type, owner_type, owner_id, metadata, input_vector,
    priority, legal_domain, scheduled_at
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8
) RETURNING *;

-- name: GetVectorJob :one
SELECT * FROM vector_jobs WHERE id = $1;

-- name: GetPendingJobs :many
SELECT * FROM vector_jobs 
WHERE status = 'pending' 
AND (scheduled_at IS NULL OR scheduled_at <= NOW())
ORDER BY priority DESC, created_at ASC 
LIMIT $1;

-- name: UpdateJobStatus :exec
UPDATE vector_jobs 
SET status = $2, updated_at = NOW()
WHERE id = $1;

-- name: CompleteJob :exec
UPDATE vector_jobs 
SET status = 'completed', 
    output_vector = $2,
    gpu_name = $3,
    processing_time_ms = $4,
    memory_usage_mb = $5,
    confidence_score = $6,
    completed_at = NOW(),
    updated_at = NOW()
WHERE id = $1;

-- name: FailJob :exec
UPDATE vector_jobs 
SET status = 'failed',
    attempts = attempts + 1,
    error_message = $2,
    error_details = $3,
    updated_at = NOW()
WHERE id = $1;

-- name: GetJobsByOwner :many
SELECT * FROM vector_jobs 
WHERE owner_type = $1 AND owner_id = $2
ORDER BY created_at DESC
LIMIT $3 OFFSET $4;

-- name: SearchSimilarVectors :many
SELECT id, job_type, owner_type, owner_id, metadata, 
       input_vector, output_vector, confidence_score,
       1 - (input_vector <=> $1::vector) AS similarity
FROM vector_jobs 
WHERE input_vector IS NOT NULL
AND legal_domain = $2
ORDER BY input_vector <=> $1::vector
LIMIT $3;

-- name: GetJobMetrics :one
SELECT 
    COUNT(*) as total_jobs,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_jobs,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_jobs,
    COUNT(*) FILTER (WHERE status = 'pending') as pending_jobs,
    AVG(processing_time_ms) FILTER (WHERE processing_time_ms IS NOT NULL) as avg_processing_time,
    AVG(confidence_score) FILTER (WHERE confidence_score IS NOT NULL) as avg_confidence
FROM vector_jobs 
WHERE created_at >= NOW() - INTERVAL '24 hours';

-- name: CleanOldJobs :exec
DELETE FROM vector_jobs 
WHERE status IN ('completed', 'failed') 
AND created_at < NOW() - INTERVAL '30 days';

-- name: GetLegalDomainStats :many
SELECT 
    legal_domain,
    COUNT(*) as job_count,
    AVG(processing_time_ms) as avg_processing_time,
    AVG(confidence_score) as avg_confidence,
    COUNT(*) FILTER (WHERE status = 'completed') as success_rate
FROM vector_jobs 
WHERE legal_domain IS NOT NULL
AND created_at >= NOW() - INTERVAL '7 days'
GROUP BY legal_domain
ORDER BY job_count DESC;