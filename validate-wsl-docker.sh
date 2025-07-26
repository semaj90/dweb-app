#!/bin/bash
# WSL Docker Desktop validation for Windows 10
echo "🔍 WSL + Docker Desktop Validation"
echo "================================="

# Check WSL environment
if [[ -d "/mnt/c" ]]; then
    echo "✅ Running in WSL"
else
    echo "❌ Not in WSL environment"
    exit 1
fi

# Check Docker integration
if docker version &>/dev/null; then
    echo "✅ Docker available in WSL"
    docker version --format "{{.Server.Version}}"
else
    echo "❌ Docker not integrated with WSL"
    echo "Fix: Enable WSL integration in Docker Desktop settings"
    exit 1
fi

# Check project path accessibility
PROJECT_PATH="/mnt/c/Users/james/Desktop/deeds-web/deeds-web-app"
if [[ -d "$PROJECT_PATH" ]]; then
    echo "✅ Project accessible from WSL"
    echo "Path: $PROJECT_PATH"
else
    echo "❌ Project path not accessible"
    exit 1
fi

# Test Docker Compose
cd "$PROJECT_PATH"
if docker-compose --version &>/dev/null; then
    echo "✅ Docker Compose available"
else
    echo "❌ Docker Compose not available"
fi

echo ""
echo "🚀 WSL + Docker Desktop ready for production phases!"
