module yorha-legal-ai-neural

go 1.21

require (
	github.com/gin-gonic/gin v1.9.1
	github.com/redis/go-redis/v9 v9.3.0
	github.com/jackc/pgx/v5 v5.5.1
	github.com/bytedance/sonic v1.10.2
	github.com/tidwall/gjson v1.17.0
	github.com/fsnotify/fsnotify v1.7.0
)

require (
	github.com/bytedance/sonic/loader v0.1.1 // indirect
	github.com/cloudwego/base64x v0.1.4 // indirect
	github.com/cloudwego/iasm v0.2.0 // indirect
	github.com/gabriel-vasile/mimetype v1.4.3 // indirect
	github.com/gin-contrib/sse v0.1.0 // indirect
	github.com/go-playground/locales v0.14.1 // indirect
	github.com/go-playground/universal-translator v0.18.1 // indirect
	github.com/go-playground/validator/v10 v10.16.0 // indirect
	github.com/goccy/go-json v0.10.2 // indirect
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20231201235250-de7065d80cb9 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/klauspost/cpuid/v2 v2.2.6 // indirect
	github.com/leodido/go-urn v1.2.4 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.2 // indirect
	github.com/pelletier/go-toml/v2 v2.1.1 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/twitchyliquid64/golang-asm v0.15.1 // indirect
	github.com/ugorji/go/codec v1.2.12 // indirect
	golang.org/x/arch v0.6.0 // indirect
	golang.org/x/crypto v0.17.0 // indirect
	golang.org/x/net v0.19.0 // indirect
	golang.org/x/sys v0.15.0 // indirect
	golang.org/x/text v0.14.0 // indirect
	google.golang.org/protobuf v1.31.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

// YoRHa-specific modules for neural processing
require (
	github.com/NVIDIA/gpu-monitoring-tools/bindings/go v0.0.0-20231204194309-b35ac5e3e5c4
)

// Optional GPU acceleration dependencies (commented out for CPU-only builds)
// Uncomment these when building with CUDA support:
// replace github.com/gorgonia/cu => github.com/gorgonia/cu v0.9.4
// replace github.com/gorgonia/gorgonia => github.com/gorgonia/gorgonia v0.9.17

// YoRHa build configuration
// Build tags available:
// - cuda: Enable CUDA acceleration
// - cublas: Enable cuBLAS linear algebra
// - tensorrt: Enable TensorRT optimization
// - debug: Enable detailed debug logging

// Example build commands:
// Standard CPU build:     go build -o yorha-processor-cpu.exe
// CUDA build:             go build -tags=cuda -o yorha-processor-gpu.exe
// Full GPU build:         go build -tags=cuda,cublas,tensorrt -o yorha-processor-full.exe

// YoRHa system information
// Neural Network Version: 3.0.0
// Target Architecture: amd64
// CUDA Compute Capability: 6.0+
// Memory Requirements: 4GB+ (8GB+ for GPU acceleration)
