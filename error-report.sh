#!/bin/bash
# Error Report & Fix Commands

## Build Errors Found:

# 1. MSVC receiving GCC flags (D8021: invalid '/Werror')
# 2. Sonic and go-nvml still in go.mod 
# 3. Toolchain mismatch (go 1.23 vs 1.24.5)

## Fix Sequence:

cd go-microservice

# Clean corrupted modules
go clean -modcache
rm go.mod go.sum

# Reinitialize with correct versions
go mod init microservice
go get github.com/gin-gonic/gin@v1.10.1
go get github.com/valyala/fastjson@v1.6.4
go get github.com/minio/simdjson-go@v0.4.5
go get github.com/jackc/pgx/v5@v5.7.2
go get github.com/go-redis/redis/v8@v8.11.5
go get github.com/gorilla/websocket@v1.5.3

# Build without CGO to bypass compiler issues
$env:CGO_ENABLED="0"
go build -o service.exe main.go

# Verify
./service.exe
