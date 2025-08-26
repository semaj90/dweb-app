-- name: CreateLegalEntity :one
INSERT INTO legal_entities (
    document_id, entity_type, entity_name, entity_role,
    confidence_score, start_position, end_position, context, metadata
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9
) RETURNING *;

-- name: GetLegalEntity :one
SELECT * FROM legal_entities WHERE id = $1;

-- name: GetEntitiesByDocument :many
SELECT * FROM legal_entities 
WHERE document_id = $1
ORDER BY confidence_score DESC, start_position ASC;

-- name: GetEntitiesByType :many
SELECT * FROM legal_entities 
WHERE entity_type = $1
ORDER BY confidence_score DESC
LIMIT $2 OFFSET $3;

-- name: GetEntitiesByName :many
SELECT * FROM legal_entities 
WHERE entity_name ILIKE '%' || $1 || '%'
ORDER BY confidence_score DESC
LIMIT $2 OFFSET $3;

-- name: GetEntitiesByRole :many
SELECT * FROM legal_entities 
WHERE entity_role = $1
ORDER BY confidence_score DESC
LIMIT $2 OFFSET $3;

-- name: UpdateLegalEntity :one
UPDATE legal_entities 
SET entity_type = $2, entity_name = $3, entity_role = $4,
    confidence_score = $5, start_position = $6, end_position = $7,
    context = $8, metadata = $9
WHERE id = $1 
RETURNING *;

-- name: DeleteLegalEntity :exec
DELETE FROM legal_entities WHERE id = $1;

-- name: DeleteEntitiesByDocument :exec
DELETE FROM legal_entities WHERE document_id = $1;

-- name: GetEntityStatistics :one
SELECT 
    COUNT(*) as total_entities,
    COUNT(DISTINCT entity_name) as unique_entities,
    COUNT(DISTINCT entity_type) as unique_types,
    COUNT(DISTINCT entity_role) as unique_roles,
    AVG(confidence_score) as avg_confidence_score
FROM legal_entities;

-- name: GetTopEntitiesByType :many
SELECT entity_name, entity_type, COUNT(*) as occurrence_count, 
       AVG(confidence_score) as avg_confidence
FROM legal_entities 
WHERE entity_type = $1
GROUP BY entity_name, entity_type
ORDER BY occurrence_count DESC, avg_confidence DESC
LIMIT $2;

-- name: GetEntitiesByDocumentAndType :many
SELECT * FROM legal_entities 
WHERE document_id = $1 AND entity_type = $2
ORDER BY confidence_score DESC;