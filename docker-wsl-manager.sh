#!/bin/bash

# Legal AI Assistant - WSL2 Docker Manager
# Optimized for WSL2 and Docker CLI workflows

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_COMPOSE_FILE="docker-compose.yml"
GPU_COMPOSE_FILE="docker-compose.gpu.yml"
PROD_COMPOSE_FILE="docker-compose.production.yml"
OPTIMIZED_COMPOSE_FILE="docker-compose.optimized.yml"

# Command line options
ACTION=""
SERVICE=""
COMPOSE_FILE="$DEFAULT_COMPOSE_FILE"
FORCE=false
FOLLOW_LOGS=false

# Functions
print_header() {
    echo -e "${CYAN}🚀 $1${NC}"
    echo -e "${CYAN}$(printf '=%.0s' $(seq 1 $((${#1} + 3))))${NC}"
    echo
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Docker is available and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker Desktop."
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon not running or not accessible"
        print_info "Make sure Docker Desktop is running and WSL integration is enabled"
        exit 1
    fi

    print_success "Docker is available and running"
}

# Check if docker-compose is available
check_compose() {
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        print_error "Docker Compose not found"
        exit 1
    fi

    print_success "Docker Compose is available: $COMPOSE_CMD"
}

# Select compose file based on options
select_compose_file() {
    if [[ "$ACTION" == *"gpu"* ]] || [[ "$*" == *"--gpu"* ]]; then
        COMPOSE_FILE="$GPU_COMPOSE_FILE"
    elif [[ "$ACTION" == *"prod"* ]] || [[ "$*" == *"--production"* ]]; then
        COMPOSE_FILE="$PROD_COMPOSE_FILE"
    elif [[ "$*" == *"--optimized"* ]]; then
        COMPOSE_FILE="$OPTIMIZED_COMPOSE_FILE"
    fi

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        print_warning "Compose file not found: $COMPOSE_FILE"
        print_info "Using default: $DEFAULT_COMPOSE_FILE"
        COMPOSE_FILE="$DEFAULT_COMPOSE_FILE"
    fi

    print_info "Using compose file: $COMPOSE_FILE"
}

# Start services
start_services() {
    print_header "Starting Legal AI Services"

    check_docker
    check_compose
    select_compose_file "$@"

    print_info "Pulling latest images..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" pull

    print_info "Starting services..."
    if [[ -n "$SERVICE" ]]; then
        $COMPOSE_CMD -f "$COMPOSE_FILE" up -d "$SERVICE"
    else
        $COMPOSE_CMD -f "$COMPOSE_FILE" up -d
    fi

    if [[ $? -eq 0 ]]; then
        print_success "Services started successfully"
        sleep 5
        show_status
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Stop services
stop_services() {
    print_header "Stopping Legal AI Services"

    check_compose
    select_compose_file "$@"

    if [[ -n "$SERVICE" ]]; then
        print_info "Stopping service: $SERVICE"
        $COMPOSE_CMD -f "$COMPOSE_FILE" stop "$SERVICE"
    else
        print_info "Stopping all services..."
        if [[ "$FORCE" == true ]]; then
            $COMPOSE_CMD -f "$COMPOSE_FILE" down -v --remove-orphans
        else
            $COMPOSE_CMD -f "$COMPOSE_FILE" down --remove-orphans
        fi
    fi

    print_success "Services stopped"
}

# Restart services
restart_services() {
    print_header "Restarting Legal AI Services"
    stop_services "$@"
    sleep 3
    start_services "$@"
}

# Show status
show_status() {
    print_header "Legal AI Services Status"

    check_docker
    check_compose
    select_compose_file "$@"

    echo -e "${CYAN}Docker Compose Services:${NC}"
    $COMPOSE_CMD -f "$COMPOSE_FILE" ps
    echo

    echo -e "${CYAN}Service Health Check:${NC}"
    declare -A services=(
        ["PostgreSQL"]=5432
        ["Redis"]=6379
        ["Ollama"]=11434
        ["Qdrant"]=6333
        ["Neo4j"]=7474
        ["SvelteKit"]=5173
        ["RabbitMQ"]=15672
    )

    for service in "${!services[@]}"; do
        port=${services[$service]}
        if nc -z localhost $port 2>/dev/null; then
            print_success "$service responding on port $port"
        else
            print_warning "$service not responding on port $port"
        fi
    done

    echo
    echo -e "${CYAN}Docker System Info:${NC}"
    echo "Docker Version: $(docker --version)"
    echo "Running Containers: $(docker ps --format '{{.Names}}' | wc -l)"
    echo "Available Images: $(docker images --format '{{.Repository}}' | wc -l)"
}

# Show logs
show_logs() {
    print_header "Legal AI Services Logs"

    check_compose
    select_compose_file "$@"

    if [[ -n "$SERVICE" ]]; then
        print_info "Showing logs for: $SERVICE"
        if [[ "$FOLLOW_LOGS" == true ]]; then
            $COMPOSE_CMD -f "$COMPOSE_FILE" logs -f --tail=100 "$SERVICE"
        else
            $COMPOSE_CMD -f "$COMPOSE_FILE" logs --tail=100 "$SERVICE"
        fi
    else
        print_info "Showing logs for all services..."
        if [[ "$FOLLOW_LOGS" == true ]]; then
            $COMPOSE_CMD -f "$COMPOSE_FILE" logs -f --tail=50
        else
            $COMPOSE_CMD -f "$COMPOSE_FILE" logs --tail=50
        fi
    fi
}

# Clean Docker system
clean_docker() {
    print_header "Cleaning Docker System"

    check_docker

    if [[ "$FORCE" == true ]]; then
        response="y"
    else
        echo -e "${YELLOW}This will remove unused containers, networks, images, and build cache${NC}"
        read -p "Continue? (y/N): " response
    fi

    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Cleaning Docker system..."
        docker system prune -af
        docker volume prune -f
        print_success "Docker system cleaned"
    else
        print_info "Cleanup cancelled"
    fi
}

# Rebuild services
rebuild_services() {
    print_header "Rebuilding Legal AI Services"

    check_compose
    select_compose_file "$@"

    print_info "Stopping services..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" down

    print_info "Building images..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" build --no-cache

    print_info "Starting services..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d

    print_success "Services rebuilt and started"
    sleep 5
    show_status
}

# Health check
health_check() {
    print_header "Legal AI Health Check"

    # Run health check script if available
    if [[ -f "scripts/health-check.js" ]]; then
        print_info "Running health check script..."
        node scripts/health-check.js
    else
        show_status "$@"
    fi

    # Check Ollama models
    if nc -z localhost 11434 2>/dev/null; then
        echo
        echo -e "${CYAN}Ollama Models:${NC}"
        if docker ps --format '{{.Names}}' | grep -q ollama; then
            ollama_container=$(docker ps --format '{{.Names}}' | grep ollama | head -1)
            docker exec "$ollama_container" ollama list 2>/dev/null || print_warning "Could not retrieve Ollama models"
        fi
    fi
}

# Show help
show_help() {
    print_header "Legal AI WSL2 Docker Manager"

    cat << EOF
USAGE:
    ./docker-wsl-manager.sh <action> [options]

ACTIONS:
    start           Start services
    stop            Stop services
    restart         Restart services
    status          Show service status
    logs            Show service logs
    clean           Clean Docker system
    rebuild         Rebuild and restart services
    health          Run health checks
    help            Show this help

OPTIONS:
    --service <name>    Target specific service
    --gpu               Use GPU-accelerated configuration
    --production        Use production configuration
    --optimized         Use memory-optimized configuration
    --force             Force operation (skip confirmations)
    --follow            Follow logs in real-time

EXAMPLES:
    ./docker-wsl-manager.sh start --gpu
    ./docker-wsl-manager.sh stop --service ollama
    ./docker-wsl-manager.sh logs --service sveltekit --follow
    ./docker-wsl-manager.sh clean --force
    ./docker-wsl-manager.sh restart --optimized

ACCESS POINTS:
    • SvelteKit App:     http://localhost:5173
    • Neo4j Browser:     http://localhost:7474  (admin/password)
    • RabbitMQ Mgmt:     http://localhost:15672 (guest/guest)
    • Qdrant Dashboard:  http://localhost:6333/dashboard

QUICK COMMANDS:
    npm run wsl:start          # Start WSL workflow
    npm run docker:status      # Check status
    npm run docker:logs        # View logs
    npm run health             # Health check

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --service)
            SERVICE="$2"
            shift 2
            ;;
        --gpu)
            COMPOSE_FILE="$GPU_COMPOSE_FILE"
            shift
            ;;
        --production)
            COMPOSE_FILE="$PROD_COMPOSE_FILE"
            shift
            ;;
        --optimized)
            COMPOSE_FILE="$OPTIMIZED_COMPOSE_FILE"
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --follow)
            FOLLOW_LOGS=true
            shift
            ;;
        *)
            if [[ -z "$ACTION" ]]; then
                ACTION="$1"
            fi
            shift
            ;;
    esac
done

# Execute action
case "$ACTION" in
    start)
        start_services "$@"
        ;;
    stop)
        stop_services "$@"
        ;;
    restart)
        restart_services "$@"
        ;;
    status)
        show_status "$@"
        ;;
    logs)
        show_logs "$@"
        ;;
    clean)
        clean_docker "$@"
        ;;
    rebuild)
        rebuild_services "$@"
        ;;
    health)
        health_check "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        print_error "No action specified"
        show_help
        exit 1
        ;;
    *)
        print_error "Unknown action: $ACTION"
        show_help
        exit 1
        ;;
esac
