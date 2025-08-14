@echo off
set MINIO_ROOT_USER=admin
set MINIO_ROOT_PASSWORD=password123
echo Starting MinIO server on port 9000 with console on 9001...
minio.exe server minio-data --console-address ":9001"
pause