// Matrix Operations Shader
// Optimized matrix operations for neural network computations

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_result: array<f32>;
@group(0) @binding(3) var<uniform> matrix_config: MatrixConfig;

struct MatrixConfig {
    rows_a: u32,
    cols_a: u32,
    cols_b: u32,
    activation_type: u32, // 0=none, 1=relu, 2=gelu, 3=tanh
}

fn apply_activation(value: f32, activation_type: u32) -> f32 {
    switch activation_type {
        case 1u: { return max(0.0, value); }           // ReLU
        case 2u: { return value * 0.5 * (1.0 + tanh(sqrt(2.0 / 3.14159) * (value + 0.044715 * pow(value, 3.0)))); } // GELU
        case 3u: { return tanh(value); }               // Tanh
        default: { return value; }                     // Linear
    }
}

@compute @workgroup_size(16, 16)
fn matrix_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    if (row >= matrix_config.rows_a || col >= matrix_config.cols_b) { return; }
    
    var sum = 0.0;
    
    // Compute dot product with loop unrolling for performance
    var k = 0u;
    for (; k + 3u < matrix_config.cols_a; k += 4u) {
        sum += matrix_a[row * matrix_config.cols_a + k] * matrix_b[k * matrix_config.cols_b + col];
        sum += matrix_a[row * matrix_config.cols_a + k + 1u] * matrix_b[(k + 1u) * matrix_config.cols_b + col];
        sum += matrix_a[row * matrix_config.cols_a + k + 2u] * matrix_b[(k + 2u) * matrix_config.cols_b + col];
        sum += matrix_a[row * matrix_config.cols_a + k + 3u] * matrix_b[(k + 3u) * matrix_config.cols_b + col];
    }
    
    // Handle remaining elements
    for (; k < matrix_config.cols_a; k++) {
        sum += matrix_a[row * matrix_config.cols_a + k] * matrix_b[k * matrix_config.cols_b + col];
    }
    
    // Apply activation and store result
    matrix_result[row * matrix_config.cols_b + col] = apply_activation(sum, matrix_config.activation_type);
}
