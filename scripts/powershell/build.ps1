$env:CC = "clang"
$env:CXX = "clang++"
$env:CGO_ENABLED = "1"
$env:CGO_CFLAGS = "-IC:/Progra~1/NVIDIA~2/CUDA/v12.9/include"
$env:CGO_LDFLAGS = "-LC:/Progra~1/NVIDIA~2/CUDA/v12.9/lib/x64 -lcudart -lcublas"

Set-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice"

go build -o ai-microservice.exe .

if (Test-Path ai-microservice.exe) {
    Stop-Process -Name ai-microservice -Force -ErrorAction SilentlyContinue
    Start-Process -FilePath ".\ai-microservice.exe" -WindowStyle Hidden
    Start-Sleep -Seconds 2
    Invoke-WebRequest http://localhost:8080/health
} else {
    Write-Host "Build failed"
}
