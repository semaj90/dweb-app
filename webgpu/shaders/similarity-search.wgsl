// Legal Document Similarity Search Shader
// GPU-accelerated cosine similarity for legal document search

@group(0) @binding(0) var<storage, read> document_embeddings: array<f32>;
@group(0) @binding(1) var<storage, read> query_embedding: array<f32>;
@group(0) @binding(2) var<storage, read_write> similarity_scores: array<f32>;
@group(0) @binding(3) var<uniform> config: SimilarityConfig;

struct SimilarityConfig {
    embedding_dim: u32,
    num_documents: u32,
    temperature: f32,
    threshold: f32,
}

@compute @workgroup_size(256)
fn similarity_search(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let doc_idx = global_id.x;
    if (doc_idx >= config.num_documents) { return; }
    
    var dot_product = 0.0;
    var norm_doc = 0.0;
    var norm_query = 0.0;
    
    // Calculate cosine similarity with legal document weighting
    for (var i = 0u; i < config.embedding_dim; i++) {
        let doc_val = document_embeddings[doc_idx * config.embedding_dim + i];
        let query_val = query_embedding[i];
        
        dot_product += doc_val * query_val;
        norm_doc += doc_val * doc_val;
        norm_query += query_val * query_val;
    }
    
    let similarity = dot_product / (sqrt(norm_doc) * sqrt(norm_query));
    
    // Apply temperature scaling for legal relevance ranking
    similarity_scores[doc_idx] = tanh(similarity * config.temperature);
}
