// Tensor Cache Management Shader
// High-speed tensor caching with LRU eviction

@group(0) @binding(0) var<storage, read_write> cache_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache_metadata: array<CacheEntry>;
@group(0) @binding(2) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(3) var<uniform> cache_config: CacheConfig;

struct CacheEntry {
    key_hash: u32,
    timestamp: u32,
    size: u32,
    offset: u32,
    access_count: u32,
    is_valid: u32,
}

struct CacheConfig {
    cache_size: u32,
    entry_count: u32,
    tensor_dim: u32,
    current_time: u32,
}

fn hash_key(data: ptr<function, array<f32, 768>>) -> u32 {
    var hash = 0u;
    for (var i = 0u; i < 768u; i++) {
        hash = hash * 31u + u32((*data)[i] * 1000.0);
    }
    return hash;
}

@compute @workgroup_size(64)
fn cache_lookup(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    if (thread_id >= cache_config.entry_count) { return; }
    
    // Hash input tensor
    var tensor_data: array<f32, 768>;
    for (var i = 0u; i < 768u; i++) {
        tensor_data[i] = input_tensor[i];
    }
    let key = hash_key(&tensor_data);
    
    // Check cache entry
    if (cache_metadata[thread_id].key_hash == key && cache_metadata[thread_id].is_valid == 1u) {
        // Cache hit - update access metadata
        cache_metadata[thread_id].access_count += 1u;
        cache_metadata[thread_id].timestamp = cache_config.current_time;
    }
}

@compute @workgroup_size(64)
fn cache_evict_lru(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    if (thread_id >= cache_config.entry_count) { return; }
    
    var oldest_time = 0xFFFFFFFFu;
    var oldest_idx = 0u;
    
    // Find LRU entry for eviction
    for (var i = 0u; i < cache_config.entry_count; i++) {
        if (cache_metadata[i].is_valid == 1u && cache_metadata[i].timestamp < oldest_time) {
            oldest_time = cache_metadata[i].timestamp;
            oldest_idx = i;
        }
    }
    
    // Evict oldest entry
    if (thread_id == 0u) {
        cache_metadata[oldest_idx].is_valid = 0u;
    }
}
