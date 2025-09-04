Set-StrictMode -Off
Set-Location -Path "$PSScriptRoot"
$env:CGO_ENABLED = '1'
$env:CGO_CFLAGS = '-O3 -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include" -mavx2 -mfma'
$env:CGO_LDFLAGS = '-L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64" -lcudart_static -lcublas'
Write-Output "Environment variables set. Running go build..."
go build -tags "cuda avx2" -o bin/simd-gpu.exe simd_gpu_parser.go
Write-Output "go build exit code: $LASTEXITCODE"
