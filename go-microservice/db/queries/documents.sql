-- name: CreateDocument :one
INSERT INTO documents (
    title, content, content_hash, embedding_768, embedding_384, 
    metadata, document_type, jurisdiction, language, created_by
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
) RETURNING *;

-- name: GetDocument :one
SELECT * FROM documents WHERE id = $1;

-- name: GetDocumentByHash :one
SELECT * FROM documents WHERE content_hash = $1;

-- name: UpdateDocument :one
UPDATE documents 
SET title = $2, content = $3, content_hash = $4, 
    embedding_768 = $5, embedding_384 = $6, metadata = $7,
    document_type = $8, jurisdiction = $9, updated_by = $10
WHERE id = $1 
RETURNING *;

-- name: UpdateDocumentProcessingStatus :exec
UPDATE documents 
SET processing_status = $2, processing_error = $3, updated_at = NOW()
WHERE id = $1;

-- name: DeleteDocument :exec
DELETE FROM documents WHERE id = $1;

-- name: ListDocuments :many
SELECT * FROM documents 
ORDER BY created_at DESC 
LIMIT $1 OFFSET $2;

-- name: ListDocumentsByType :many
SELECT * FROM documents 
WHERE document_type = $1
ORDER BY created_at DESC 
LIMIT $2 OFFSET $3;

-- name: ListDocumentsByJurisdiction :many
SELECT * FROM documents 
WHERE jurisdiction = $1
ORDER BY created_at DESC 
LIMIT $2 OFFSET $3;

-- name: SearchDocumentsByContent :many
SELECT *, ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) as rank
FROM documents 
WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
ORDER BY rank DESC 
LIMIT $2 OFFSET $3;

-- name: FindSimilarDocuments768 :many
SELECT id, title, metadata, embedding_768 <=> $1 as distance
FROM documents 
WHERE embedding_768 IS NOT NULL
ORDER BY embedding_768 <=> $1 
LIMIT $2;

-- name: FindSimilarDocuments384 :many
SELECT id, title, metadata, embedding_384 <=> $1 as distance
FROM documents 
WHERE embedding_384 IS NOT NULL
ORDER BY embedding_384 <=> $1 
LIMIT $2;

-- name: FindSimilarDocumentsWithThreshold768 :many
SELECT id, title, metadata, embedding_768 <=> $1 as distance
FROM documents 
WHERE embedding_768 IS NOT NULL 
AND embedding_768 <=> $1 < $3
ORDER BY embedding_768 <=> $1 
LIMIT $2;

-- name: FindSimilarDocumentsWithThreshold384 :many
SELECT id, title, metadata, embedding_384 <=> $1 as distance
FROM documents 
WHERE embedding_384 IS NOT NULL 
AND embedding_384 <=> $1 < $3
ORDER BY embedding_384 <=> $1 
LIMIT $2;

-- name: FindSimilarDocumentsByTypeAndJurisdiction768 :many
SELECT id, title, metadata, document_type, jurisdiction, embedding_768 <=> $1 as distance
FROM documents 
WHERE embedding_768 IS NOT NULL 
AND ($2 = '' OR document_type = $2)
AND ($3 = '' OR jurisdiction = $3)
ORDER BY embedding_768 <=> $1 
LIMIT $4;

-- name: UpdateDocumentAccessStats :exec
UPDATE documents 
SET access_count = access_count + 1, last_accessed = NOW()
WHERE id = $1;

-- name: GetDocumentStats :one
SELECT 
    COUNT(*) as total_documents,
    COUNT(*) FILTER (WHERE processing_status = 'completed') as completed_documents,
    COUNT(*) FILTER (WHERE processing_status = 'processing') as processing_documents,
    COUNT(*) FILTER (WHERE processing_status = 'pending') as pending_documents,
    COUNT(*) FILTER (WHERE processing_status = 'error') as error_documents,
    COUNT(*) FILTER (WHERE embedding_768 IS NOT NULL) as documents_with_768_embedding,
    COUNT(*) FILTER (WHERE embedding_384 IS NOT NULL) as documents_with_384_embedding
FROM documents;