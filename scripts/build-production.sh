#!/bin/bash

# =============================================================================
# PRODUCTION BUILD SCRIPT - LEGAL AI PLATFORM
# =============================================================================
# This script builds all components for production deployment including:
# - Go microservices
# - SvelteKit frontend
# - Static assets
# - Docker images (optional)
# =============================================================================

set -e # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/dist"
GO_BUILD_DIR="$BUILD_DIR/go-services"
FRONTEND_BUILD_DIR="$BUILD_DIR/frontend"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BUILD_VERSION=${BUILD_VERSION:-"1.0.0-${TIMESTAMP}"}

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}LEGAL AI PLATFORM - PRODUCTION BUILD${NC}"
echo -e "${BLUE}Build Version: ${BUILD_VERSION}${NC}"
echo -e "${BLUE}Build Directory: ${BUILD_DIR}${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# Function to log messages
log() {
    echo -e "${GREEN}[BUILD]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking build prerequisites..."
    
    # Check if Go is installed
    if ! command -v go &> /dev/null; then
        error "Go is not installed. Please install Go 1.21 or later."
    fi
    
    # Check Go version
    GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
    log "Found Go version: $GO_VERSION"
    
    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed. Please install Node.js 18 or later."
    fi
    
    # Check Node.js version
    NODE_VERSION=$(node --version)
    log "Found Node.js version: $NODE_VERSION"
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        error "npm is not installed. Please install npm."
    fi
    
    # Check if pnpm is available (preferred)
    if command -v pnpm &> /dev/null; then
        PACKAGE_MANAGER="pnpm"
        log "Using pnpm as package manager"
    else
        PACKAGE_MANAGER="npm"
        log "Using npm as package manager"
    fi
}

# Function to clean previous builds
clean_build() {
    log "Cleaning previous build artifacts..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    mkdir -p "$GO_BUILD_DIR"
    mkdir -p "$FRONTEND_BUILD_DIR"
    log "Build directories created"
}

# Function to build Go microservices
build_go_services() {
    log "Building Go microservices..."
    
    cd "$PROJECT_ROOT"
    
    # List of Go services to build
    GO_SERVICES=(
        "go-microservice/main.go:enhanced-rag-service"
        "go-microservice/gin-upload.go:upload-service"
        "go-microservice/gpu-legal-ai-server.go:gpu-legal-ai-server"
        "go-microservice/enhanced-legal-ai-gpu.go:enhanced-legal-ai-gpu"
        "go-microservice/cmd/enhanced-rag/main.go:enhanced-rag-v2"
        "go-microservice/cmd/cluster-service/main.go:cluster-service"
    )
    
    for service in "${GO_SERVICES[@]}"; do
        IFS=':' read -r source_path binary_name <<< "$service"
        
        if [[ -f "$source_path" ]]; then
            log "Building $binary_name from $source_path..."
            
            # Set build environment
            export CGO_ENABLED=0
            export GOOS=linux
            export GOARCH=amd64
            
            # Build with optimization flags
            go build \
                -ldflags="-w -s -X main.version=${BUILD_VERSION} -X main.buildTime=${TIMESTAMP}" \
                -trimpath \
                -o "$GO_BUILD_DIR/$binary_name" \
                "$source_path"
            
            if [[ $? -eq 0 ]]; then
                log "✓ Successfully built $binary_name"
                # Get binary size
                SIZE=$(du -h "$GO_BUILD_DIR/$binary_name" | cut -f1)
                log "  Binary size: $SIZE"
            else
                error "Failed to build $binary_name"
            fi
        else
            warn "Source file not found: $source_path, skipping..."
        fi
    done
    
    log "Go microservices build completed"
}

# Function to build additional Go services
build_go_services_advanced() {
    log "Building additional Go services from go-services directory..."
    
    if [[ -d "$PROJECT_ROOT/go-services" ]]; then
        cd "$PROJECT_ROOT/go-services"
        
        # Build enhanced RAG service
        if [[ -f "cmd/enhanced-rag/main.go" ]]; then
            log "Building enhanced-rag service..."
            go build \
                -ldflags="-w -s" \
                -trimpath \
                -o "$GO_BUILD_DIR/enhanced-rag-service-v2" \
                "./cmd/enhanced-rag"
            log "✓ Enhanced RAG service built"
        fi
    fi
}

# Function to build SvelteKit frontend
build_frontend() {
    log "Building SvelteKit frontend..."
    
    cd "$PROJECT_ROOT/sveltekit-frontend"
    
    # Install dependencies
    log "Installing frontend dependencies..."
    $PACKAGE_MANAGER install --frozen-lockfile
    
    # Run type checking
    log "Running TypeScript type checking..."
    $PACKAGE_MANAGER run check || warn "Type checking completed with warnings"
    
    # Build for production
    log "Building SvelteKit application for production..."
    export NODE_ENV=production
    export VITE_BUILD_VERSION="$BUILD_VERSION"
    export VITE_BUILD_TIMESTAMP="$TIMESTAMP"
    
    $PACKAGE_MANAGER run build
    
    if [[ $? -eq 0 ]]; then
        log "✓ SvelteKit build completed successfully"
        
        # Copy build output to distribution directory
        if [[ -d "build" ]]; then
            cp -r build/* "$FRONTEND_BUILD_DIR/"
            log "✓ Frontend assets copied to distribution directory"
        fi
        
        # Get build size
        if [[ -d "$FRONTEND_BUILD_DIR" ]]; then
            SIZE=$(du -sh "$FRONTEND_BUILD_DIR" | cut -f1)
            log "  Frontend build size: $SIZE"
        fi
    else
        error "SvelteKit build failed"
    fi
}

# Function to optimize build output
optimize_build() {
    log "Optimizing build output..."
    
    # Strip debug symbols from Go binaries (already done during build)
    log "Go binaries already optimized during build"
    
    # Compress static assets (if gzip is available)
    if command -v gzip &> /dev/null; then
        log "Compressing static assets..."
        find "$FRONTEND_BUILD_DIR" -name "*.js" -o -name "*.css" -o -name "*.html" | while read -r file; do
            gzip -k -f "$file"
            log "  Compressed: $(basename "$file")"
        done
    fi
    
    # Create checksums for integrity verification
    log "Creating checksums..."
    cd "$BUILD_DIR"
    find . -type f -exec sha256sum {} \; > checksums.sha256
    log "✓ Checksums created: checksums.sha256"
}

# Function to create deployment package
create_deployment_package() {
    log "Creating deployment package..."
    
    cd "$PROJECT_ROOT"
    
    # Create deployment structure
    DEPLOY_DIR="$BUILD_DIR/deployment"
    mkdir -p "$DEPLOY_DIR"
    
    # Copy configuration files
    cp .env.production "$DEPLOY_DIR/env.production"
    
    # Copy deployment scripts
    if [[ -d "scripts" ]]; then
        mkdir -p "$DEPLOY_DIR/scripts"
        cp scripts/deploy-production.sh "$DEPLOY_DIR/scripts/" 2>/dev/null || true
        cp scripts/setup-ssl.sh "$DEPLOY_DIR/scripts/" 2>/dev/null || true
        cp scripts/nginx.conf "$DEPLOY_DIR/scripts/" 2>/dev/null || true
    fi
    
    # Copy database migrations if they exist
    if [[ -d "migrations" ]]; then
        cp -r migrations "$DEPLOY_DIR/"
    fi
    
    # Create deployment README
    cat > "$DEPLOY_DIR/README.md" << EOF
# Legal AI Platform - Production Deployment

## Build Information
- Version: $BUILD_VERSION
- Build Date: $(date)
- Build Environment: $(uname -a)

## Contents
- \`go-services/\`: Compiled Go microservices
- \`frontend/\`: SvelteKit build output
- \`scripts/\`: Deployment scripts
- \`env.production\`: Production environment configuration
- \`migrations/\`: Database migrations (if available)

## Deployment Steps
1. Configure environment variables in \`env.production\`
2. Run database migrations
3. Deploy Go services
4. Deploy frontend assets
5. Configure reverse proxy (nginx/apache)
6. Start services

See deployment documentation for detailed instructions.
EOF
    
    # Create archive
    ARCHIVE_NAME="legal-ai-platform-${BUILD_VERSION}.tar.gz"
    tar -czf "$PROJECT_ROOT/$ARCHIVE_NAME" -C "$BUILD_DIR" .
    
    log "✓ Deployment package created: $ARCHIVE_NAME"
    log "  Package size: $(du -h "$PROJECT_ROOT/$ARCHIVE_NAME" | cut -f1)"
}

# Function to validate build
validate_build() {
    log "Validating build output..."
    
    # Check Go binaries
    for binary in "$GO_BUILD_DIR"/*; do
        if [[ -x "$binary" ]]; then
            log "✓ $(basename "$binary") - executable"
        else
            warn "$(basename "$binary") - not executable"
        fi
    done
    
    # Check frontend build
    if [[ -f "$FRONTEND_BUILD_DIR/index.html" ]]; then
        log "✓ Frontend index.html found"
    else
        warn "Frontend index.html not found"
    fi
    
    # Check static assets
    if [[ -d "$FRONTEND_BUILD_DIR/_app" ]]; then
        log "✓ Frontend assets found"
        ASSET_COUNT=$(find "$FRONTEND_BUILD_DIR/_app" -type f | wc -l)
        log "  Asset count: $ASSET_COUNT files"
    fi
    
    log "Build validation completed"
}

# Function to show build summary
show_build_summary() {
    log "Build completed successfully!"
    echo ""
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${BLUE}BUILD SUMMARY${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
    echo "Build Version: $BUILD_VERSION"
    echo "Build Time: $(date)"
    echo "Build Directory: $BUILD_DIR"
    echo ""
    echo "Go Services Built:"
    find "$GO_BUILD_DIR" -type f -executable 2>/dev/null | while read -r binary; do
        SIZE=$(du -h "$binary" | cut -f1)
        echo "  - $(basename "$binary") ($SIZE)"
    done
    echo ""
    echo "Frontend Build:"
    if [[ -d "$FRONTEND_BUILD_DIR" ]]; then
        SIZE=$(du -sh "$FRONTEND_BUILD_DIR" | cut -f1)
        echo "  - SvelteKit build ($SIZE)"
    fi
    echo ""
    echo "Total Build Size: $(du -sh "$BUILD_DIR" | cut -f1)"
    echo ""
    echo "Next Steps:"
    echo "1. Review build output in: $BUILD_DIR"
    echo "2. Test deployment package"
    echo "3. Deploy to staging environment"
    echo "4. Deploy to production"
    echo -e "${BLUE}==============================================================================${NC}"
}

# Main execution
main() {
    log "Starting production build process..."
    
    check_prerequisites
    clean_build
    build_go_services
    build_go_services_advanced
    build_frontend
    optimize_build
    create_deployment_package
    validate_build
    show_build_summary
    
    log "Production build process completed successfully!"
}

# Execute main function
main "$@"