// Legal Document Embedding Generation Shader
// Specialized embedding generation for legal documents

@group(0) @binding(0) var<storage, read> legal_tokens: array<u32>;
@group(0) @binding(1) var<storage, read> embedding_weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_embeddings: array<f32>;
@group(0) @binding(3) var<uniform> embed_config: EmbedConfig;

struct EmbedConfig {
    vocab_size: u32,
    embedding_dim: u32,
    sequence_length: u32,
    legal_boost_factor: f32,
}

// Legal term weighting - boost importance of legal terminology
fn get_legal_weight(token_id: u32) -> f32 {
    // Legal keywords get boosted relevance
    let legal_terms = array<u32, 20>(
        1234u, 5678u, 9012u, 3456u, 7890u,  // contract, plaintiff, defendant, liability, statute
        2345u, 6789u, 0123u, 4567u, 8901u,  // jurisdiction, precedent, tort, equity, remedy
        3456u, 7890u, 1234u, 5678u, 9012u,  // evidence, testimony, appeal, motion, brief
        4567u, 8901u, 2345u, 6789u, 0123u   // discovery, settlement, injunction, damages, court
    );
    
    for (var i = 0u; i < 20u; i++) {
        if (token_id == legal_terms[i]) {
            return embed_config.legal_boost_factor;
        }
    }
    return 1.0;
}

@compute @workgroup_size(256)
fn generate_legal_embedding(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim_idx = global_id.x;
    if (dim_idx >= embed_config.embedding_dim) { return; }
    
    var embedding_value = 0.0;
    
    // Process all tokens in sequence
    for (var seq_idx = 0u; seq_idx < embed_config.sequence_length; seq_idx++) {
        let token_id = legal_tokens[seq_idx];
        if (token_id < embed_config.vocab_size) {
            let weight_idx = token_id * embed_config.embedding_dim + dim_idx;
            let token_weight = embedding_weights[weight_idx];
            let legal_boost = get_legal_weight(token_id);
            
            embedding_value += token_weight * legal_boost;
        }
    }
    
    // Normalize and store
    output_embeddings[dim_idx] = tanh(embedding_value / f32(embed_config.sequence_length));
}
