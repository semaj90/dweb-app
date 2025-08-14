# ================================================================================
# COMPLETE RABBITMQ & NEO4J INTEGRATION SETUP
# ================================================================================
# RabbitMQ • Neo4j • Kratos • Event-Driven Architecture • Graph Database
# ================================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet('Install', 'Start', 'Stop', 'Status', 'Configure')]
    [string]$Command = 'Install'
)

Write-Host "🐰 RABBITMQ & NEO4J INTEGRATION SETUP" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# ============================================================================
# RABBITMQ INSTALLATION & SETUP
# ============================================================================

function Install-RabbitMQ {
    Write-Host "`n📦 Installing RabbitMQ..." -ForegroundColor Yellow
    
    # Check if RabbitMQ is already installed
    $rabbitService = Get-Service -Name "RabbitMQ" -ErrorAction SilentlyContinue
    if ($rabbitService) {
        Write-Host "✅ RabbitMQ already installed" -ForegroundColor Green
        return
    }
    
    # Download and install RabbitMQ
    Write-Host "📥 Downloading RabbitMQ installer..." -ForegroundColor Cyan
    $rabbitUrl = "https://github.com/rabbitmq/rabbitmq-server/releases/download/v3.12.4/rabbitmq-server-3.12.4.exe"
    $installerPath = "$env:TEMP\rabbitmq-installer.exe"
    
    try {
        Invoke-WebRequest -Uri $rabbitUrl -OutFile $installerPath -UseBasicParsing
        Write-Host "✅ Downloaded RabbitMQ installer" -ForegroundColor Green
        
        # Install RabbitMQ
        Write-Host "🔧 Installing RabbitMQ (this may take a few minutes)..." -ForegroundColor Cyan
        Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait
        
        # Wait for service to be available
        Start-Sleep 10
        
        # Enable management plugin
        & "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.12.4\sbin\rabbitmq-plugins.bat" enable rabbitmq_management
        
        Write-Host "✅ RabbitMQ installed successfully" -ForegroundColor Green
        
    } catch {
        Write-Host "❌ Failed to install RabbitMQ: $_" -ForegroundColor Red
        return $false
    }
    
    return $true
}

function Install-Neo4j {
    Write-Host "`n📊 Installing Neo4j..." -ForegroundColor Yellow
    
    # Check if Neo4j is already installed
    if (Test-Path "C:\Program Files\Neo4j") {
        Write-Host "✅ Neo4j already installed" -ForegroundColor Green
        return $true
    }
    
    # Download Neo4j Community Edition
    Write-Host "📥 Downloading Neo4j Community Edition..." -ForegroundColor Cyan
    $neo4jUrl = "https://dist.neo4j.org/neo4j-community-5.12.0-windows.zip"
    $zipPath = "$env:TEMP\neo4j-community.zip"
    $extractPath = "C:\Neo4j"
    
    try {
        Invoke-WebRequest -Uri $neo4jUrl -OutFile $zipPath -UseBasicParsing
        Write-Host "✅ Downloaded Neo4j" -ForegroundColor Green
        
        # Extract Neo4j
        Write-Host "📂 Extracting Neo4j..." -ForegroundColor Cyan
        Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
        
        # Rename directory
        $neo4jDir = Get-ChildItem -Path $extractPath -Directory | Where-Object { $_.Name -like "neo4j-*" } | Select-Object -First 1
        if ($neo4jDir) {
            Rename-Item -Path $neo4jDir.FullName -NewName "neo4j-community"
        }
        
        # Install as Windows service
        $neo4jPath = "$extractPath\neo4j-community\bin\neo4j.bat"
        if (Test-Path $neo4jPath) {
            & $neo4jPath install-service
            Write-Host "✅ Neo4j installed as Windows service" -ForegroundColor Green
        }
        
        # Set initial password
        $neo4jAdmin = "$extractPath\neo4j-community\bin\neo4j-admin.bat"
        if (Test-Path $neo4jAdmin) {
            & $neo4jAdmin set-default-admin password123
            Write-Host "✅ Neo4j default password set to 'password123'" -ForegroundColor Green
        }
        
        Write-Host "✅ Neo4j installed successfully" -ForegroundColor Green
        return $true
        
    } catch {
        Write-Host "❌ Failed to install Neo4j: $_" -ForegroundColor Red
        return $false
    }
}

function Install-ErlangOTP {
    Write-Host "`n🔧 Installing Erlang/OTP (required for RabbitMQ)..." -ForegroundColor Yellow
    
    # Check if Erlang is already installed
    $erlangPath = Get-ItemProperty -Path "HKLM:\SOFTWARE\Ericsson\Erlang" -ErrorAction SilentlyContinue
    if ($erlangPath) {
        Write-Host "✅ Erlang/OTP already installed" -ForegroundColor Green
        return $true
    }
    
    # Download Erlang/OTP
    $erlangUrl = "https://erlang.org/download/otp_win64_26.0.exe"
    $installerPath = "$env:TEMP\erlang-installer.exe"
    
    try {
        Invoke-WebRequest -Uri $erlangUrl -OutFile $installerPath -UseBasicParsing
        Write-Host "✅ Downloaded Erlang/OTP" -ForegroundColor Green
        
        # Install Erlang
        Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait
        Write-Host "✅ Erlang/OTP installed successfully" -ForegroundColor Green
        
        return $true
        
    } catch {
        Write-Host "❌ Failed to install Erlang/OTP: $_" -ForegroundColor Red
        return $false
    }
}

# ============================================================================
# SERVICE MANAGEMENT
# ============================================================================

function Start-AllServices {
    Write-Host "`n🚀 Starting all services..." -ForegroundColor Yellow
    
    # Start Erlang/RabbitMQ
    Write-Host "🐰 Starting RabbitMQ..." -ForegroundColor Cyan
    try {
        Start-Service -Name "RabbitMQ" -ErrorAction Stop
        Write-Host "✅ RabbitMQ started" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ RabbitMQ start failed: $_" -ForegroundColor Yellow
    }
    
    # Start Neo4j
    Write-Host "📊 Starting Neo4j..." -ForegroundColor Cyan
    try {
        Start-Service -Name "Neo4j" -ErrorAction Stop
        Write-Host "✅ Neo4j started" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Neo4j start failed: $_" -ForegroundColor Yellow
        
        # Try alternative start method
        $neo4jBat = "C:\Neo4j\neo4j-community\bin\neo4j.bat"
        if (Test-Path $neo4jBat) {
            Start-Process -FilePath $neo4jBat -ArgumentList "start" -WindowStyle Hidden
            Start-Sleep 5
            Write-Host "✅ Neo4j started via batch file" -ForegroundColor Green
        }
    }
}

function Stop-AllServices {
    Write-Host "`n🛑 Stopping all services..." -ForegroundColor Yellow
    
    # Stop RabbitMQ
    try {
        Stop-Service -Name "RabbitMQ" -Force -ErrorAction SilentlyContinue
        Write-Host "✅ RabbitMQ stopped" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ RabbitMQ stop failed" -ForegroundColor Yellow
    }
    
    # Stop Neo4j
    try {
        Stop-Service -Name "Neo4j" -Force -ErrorAction SilentlyContinue
        Write-Host "✅ Neo4j stopped" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Neo4j stop failed" -ForegroundColor Yellow
    }
}

function Show-ServiceStatus {
    Write-Host "`n📊 Service Status:" -ForegroundColor Yellow
    
    # Check RabbitMQ
    $rabbitService = Get-Service -Name "RabbitMQ" -ErrorAction SilentlyContinue
    if ($rabbitService) {
        $status = if ($rabbitService.Status -eq "Running") { "✅ Running" } else { "❌ Stopped" }
        Write-Host "  🐰 RabbitMQ: $status" -ForegroundColor $(if ($rabbitService.Status -eq "Running") { "Green" } else { "Red" })
        
        # Test RabbitMQ connectivity
        if ($rabbitService.Status -eq "Running") {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:15672/api/overview" -Credential (New-Object PSCredential("guest", (ConvertTo-SecureString "guest" -AsPlainText -Force))) -ErrorAction SilentlyContinue
                Write-Host "    📡 Management UI: http://localhost:15672 (guest/guest)" -ForegroundColor Gray
            } catch {
                Write-Host "    ⚠️ Management UI not accessible" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "  🐰 RabbitMQ: ❌ Not installed" -ForegroundColor Red
    }
    
    # Check Neo4j
    $neo4jService = Get-Service -Name "Neo4j" -ErrorAction SilentlyContinue
    if ($neo4jService) {
        $status = if ($neo4jService.Status -eq "Running") { "✅ Running" } else { "❌ Stopped" }
        Write-Host "  📊 Neo4j: $status" -ForegroundColor $(if ($neo4jService.Status -eq "Running") { "Green" } else { "Red" })
        
        # Test Neo4j connectivity
        if ($neo4jService.Status -eq "Running") {
            Start-Sleep 2
            try {
                $testConnection = Test-NetConnection -ComputerName localhost -Port 7474 -InformationLevel Quiet -WarningAction SilentlyContinue
                if ($testConnection) {
                    Write-Host "    📡 Browser UI: http://localhost:7474 (neo4j/password123)" -ForegroundColor Gray
                    Write-Host "    🔌 Bolt: bolt://localhost:7687" -ForegroundColor Gray
                } else {
                    Write-Host "    ⚠️ Neo4j not responding on port 7474" -ForegroundColor Yellow
                }
            } catch {
                Write-Host "    ⚠️ Neo4j connectivity test failed" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "  📊 Neo4j: ❌ Not installed" -ForegroundColor Red
    }
    
    # Check other services
    $services = @(
        @{Name="PostgreSQL"; Port=5432; Service="postgresql-x64-17"},
        @{Name="Redis"; Port=6379; Service="Redis"},
        @{Name="Ollama"; Port=11434; Service="OllamaService"},
        @{Name="MinIO"; Port=9000; Service="MinIO"}
    )
    
    foreach ($svc in $services) {
        $testPort = Test-NetConnection -ComputerName localhost -Port $svc.Port -InformationLevel Quiet -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
        $status = if ($testPort) { "✅ Running" } else { "❌ Stopped" }
        $color = if ($testPort) { "Green" } else { "Red" }
        Write-Host "  🔧 $($svc.Name): $status" -ForegroundColor $color
    }
}

# ============================================================================
# CONFIGURATION SETUP
# ============================================================================

function Configure-RabbitMQ {
    Write-Host "`n⚙️ Configuring RabbitMQ..." -ForegroundColor Yellow
    
    # Create RabbitMQ configuration
    $rabbitConfig = @"
listeners.tcp.default = 5672
management.listener.port = 15672
management.listener.ssl = false

# Enable management plugin
management.load_definitions = /etc/rabbitmq/definitions.json

# Logging
log.console = true
log.console.level = info

# Memory and disk limits
vm_memory_high_watermark.relative = 0.6
disk_free_limit.relative = 2.0

# Clustering (for future scaling)
cluster_formation.peer_discovery_backend = rabbit_peer_discovery_classic_config
cluster_formation.classic_config.nodes.1 = rabbit@localhost

# Message TTL
default_vhost = /
default_user = guest
default_pass = guest
default_user_tags.administrator = true
default_permissions.configure = .*
default_permissions.write = .*
default_permissions.read = .*
"@

    $configPath = "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.12.4\etc\rabbitmq\rabbitmq.conf"
    $configDir = Split-Path $configPath -Parent
    
    if (!(Test-Path $configDir)) {
        New-Item -Path $configDir -ItemType Directory -Force | Out-Null
    }
    
    $rabbitConfig | Out-File -FilePath $configPath -Encoding UTF8
    Write-Host "✅ RabbitMQ configuration created" -ForegroundColor Green
    
    # Create definitions for queues and exchanges
    $definitions = @{
        "rabbitmq_version" = "3.12.4"
        "rabbit_version" = "3.12.4"
        "product_name" = "RabbitMQ"
        "product_version" = "3.12.4"
        "users" = @(
            @{
                "name" = "legal_ai"
                "password_hash" = "gTSF+WBX5l7sJKkPE6tSqRD1FJ8C5ZhS9Zm6+GtUBfE4+xE4"
                "hashing_algorithm" = "rabbit_password_hashing_sha256"
                "tags" = @("administrator")
                "limits" = @{}
            }
        )
        "vhosts" = @(
            @{
                "name" = "/"
            },
            @{
                "name" = "/legal-ai"
                "description" = "Legal AI Virtual Host"
            }
        )
        "permissions" = @(
            @{
                "user" = "legal_ai"
                "vhost" = "/legal-ai"
                "configure" = ".*"
                "write" = ".*"
                "read" = ".*"
            }
        )
        "topic_permissions" = @()
        "parameters" = @()
        "global_parameters" = @()
        "policies" = @()
        "queues" = @(
            @{
                "name" = "document.analysis"
                "vhost" = "/legal-ai"
                "durable" = $true
                "auto_delete" = $false
                "arguments" = @{
                    "x-message-ttl" = 3600000
                    "x-max-length" = 10000
                }
            },
            @{
                "name" = "vector.search"
                "vhost" = "/legal-ai"
                "durable" = $true
                "auto_delete" = $false
                "arguments" = @{
                    "x-message-ttl" = 1800000
                }
            },
            @{
                "name" = "chat.processing"
                "vhost" = "/legal-ai"
                "durable" = $true
                "auto_delete" = $false
                "arguments" = @{
                    "x-message-ttl" = 300000
                }
            },
            @{
                "name" = "gpu.computation"
                "vhost" = "/legal-ai"
                "durable" = $true
                "auto_delete" = $false
                "arguments" = @{
                    "x-message-ttl" = 600000
                }
            }
        )
        "exchanges" = @(
            @{
                "name" = "legal.ai.direct"
                "vhost" = "/legal-ai"
                "type" = "direct"
                "durable" = $true
                "auto_delete" = $false
                "internal" = $false
                "arguments" = @{}
            },
            @{
                "name" = "legal.ai.topic"
                "vhost" = "/legal-ai"
                "type" = "topic"
                "durable" = $true
                "auto_delete" = $false
                "internal" = $false
                "arguments" = @{}
            },
            @{
                "name" = "legal.ai.fanout"
                "vhost" = "/legal-ai"
                "type" = "fanout"
                "durable" = $true
                "auto_delete" = $false
                "internal" = $false
                "arguments" = @{}
            }
        )
        "bindings" = @(
            @{
                "source" = "legal.ai.topic"
                "vhost" = "/legal-ai"
                "destination" = "document.analysis"
                "destination_type" = "queue"
                "routing_key" = "document.analysis"
                "arguments" = @{}
            },
            @{
                "source" = "legal.ai.topic"
                "vhost" = "/legal-ai"
                "destination" = "vector.search"
                "destination_type" = "queue"
                "routing_key" = "vector.search"
                "arguments" = @{}
            }
        )
    }
    
    $definitionsPath = "$configDir\definitions.json"
    $definitions | ConvertTo-Json -Depth 10 | Out-File -FilePath $definitionsPath -Encoding UTF8
    Write-Host "✅ RabbitMQ definitions created" -ForegroundColor Green
}

function Configure-Neo4j {
    Write-Host "`n⚙️ Configuring Neo4j..." -ForegroundColor Yellow
    
    $neo4jConfigPath = "C:\Neo4j\neo4j-community\conf\neo4j.conf"
    
    if (!(Test-Path $neo4jConfigPath)) {
        Write-Host "❌ Neo4j configuration file not found" -ForegroundColor Red
        return
    }
    
    # Backup original config
    Copy-Item $neo4jConfigPath "$neo4jConfigPath.backup" -Force
    
    # Neo4j configuration for legal AI
    $neo4jConfig = @"
# Network connector configuration
server.default_listen_address=0.0.0.0
server.default_advertised_address=localhost

# Bolt connector
server.bolt.enabled=true
server.bolt.listen_address=:7687

# HTTP Connector
server.http.enabled=true
server.http.listen_address=:7474

# HTTPS Connector
server.https.enabled=false

# Database configuration
dbms.default_database=legal-ai

# Memory settings for RTX 3060 Ti system
server.memory.heap.initial_size=2G
server.memory.heap.max_size=4G
server.memory.pagecache.size=2G

# Transaction timeout
db.transaction.timeout=60s

# Query timeout
db.transaction.bookmark_ready_timeout=30s

# Logging
server.logs.user.rotation.keep_number=5
server.logs.user.rotation.size=20M

# Security
dbms.security.auth_enabled=true
dbms.security.procedures.unrestricted=gds.*,apoc.*

# APOC configuration
dbms.security.procedures.allowlist=gds.*,apoc.*

# Performance tuning
cypher.default_language_version=5
cypher.render_plan_description=true

# Legal AI specific settings
# Enable full-text search
dbms.security.procedures.unrestricted=apoc.*,gds.*,n10s.*

# Graph data science library
dbms.security.procedures.whitelist=gds.*

# Custom procedures path
dbms.directories.plugins=plugins
"@

    $neo4jConfig | Out-File -FilePath $neo4jConfigPath -Encoding UTF8
    Write-Host "✅ Neo4j configuration updated" -ForegroundColor Green
    
    # Create initialization Cypher script
    $cypherScript = @"
// Legal AI Database Initialization
// Create constraints and indexes for optimal performance

// User constraints
CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT user_email_unique IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE;

// Document constraints
CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE;

// Legal entity constraints
CREATE CONSTRAINT precedent_id_unique IF NOT EXISTS FOR (p:Precedent) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT statute_id_unique IF NOT EXISTS FOR (s:Statute) REQUIRE s.id IS UNIQUE;

// Performance indexes
CREATE INDEX document_content_text IF NOT EXISTS FOR (d:Document) ON (d.content);
CREATE INDEX case_type_index IF NOT EXISTS FOR (c:Case) ON (c.type);
CREATE INDEX precedent_year_index IF NOT EXISTS FOR (p:Precedent) ON (p.year);
CREATE INDEX document_timestamp_index IF NOT EXISTS FOR (d:Document) ON (d.created_at);

// Vector similarity index (for embeddings)
CREATE VECTOR INDEX document_embeddings IF NOT EXISTS FOR (d:Document) ON (d.embedding)
OPTIONS {
  indexConfig: {
    'vector.dimensions': 384,
    'vector.similarity_function': 'cosine'
  }
};

// Full-text search indexes
CREATE FULLTEXT INDEX legal_document_search IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.content];
CREATE FULLTEXT INDEX case_search IF NOT EXISTS FOR (c:Case) ON EACH [c.title, c.description];
CREATE FULLTEXT INDEX precedent_search IF NOT EXISTS FOR (p:Precedent) ON EACH [p.title, p.summary];

// Sample data for testing
MERGE (u:User {id: 'user-001', email: 'admin@legalai.com', name: 'Legal AI Admin'})
MERGE (c:Case {id: 'case-001', title: 'Contract Dispute Example', type: 'contract', status: 'active'})
MERGE (d:Document {id: 'doc-001', title: 'Sample Contract', type: 'contract', content: 'This is a sample legal contract...'})
MERGE (p:Precedent {id: 'prec-001', title: 'Smith v. Jones', year: 2023, court: 'Supreme Court'})

// Create relationships
MERGE (u)-[:OWNS]->(c)
MERGE (c)-[:CONTAINS]->(d)
MERGE (d)-[:CITES]->(p)

// Legal knowledge graph structure
MERGE (concept1:LegalConcept {name: 'Contract Law', category: 'civil'})
MERGE (concept2:LegalConcept {name: 'Consideration', category: 'contract-element'})
MERGE (concept3:LegalConcept {name: 'Breach of Contract', category: 'contract-violation'})

MERGE (concept1)-[:INCLUDES]->(concept2)
MERGE (concept1)-[:INCLUDES]->(concept3)

RETURN 'Legal AI database initialized successfully' AS result;
"@

    $cypherPath = "C:\Neo4j\neo4j-community\import\legal-ai-init.cypher"
    $cypherScript | Out-File -FilePath $cypherPath -Encoding UTF8
    Write-Host "✅ Neo4j initialization script created" -ForegroundColor Green
}

# ============================================================================
# KRATOS INTEGRATION SETUP
# ============================================================================

function Install-Kratos {
    Write-Host "`n🔐 Setting up Ory Kratos integration..." -ForegroundColor Yellow
    
    # Create Kratos configuration
    $kratosConfig = @{
        version = "v1.0.0"
        dsn = "postgresql://kratos:kratos123@localhost:5432/kratos_db?sslmode=disable"
        
        serve = @{
            public = @{
                base_url = "http://localhost:4433"
                cors = @{
                    enabled = $true
                    allowed_origins = @("http://localhost:5173", "http://localhost:3000")
                    allowed_methods = @("GET", "POST", "PUT", "DELETE", "OPTIONS")
                    allowed_headers = @("Authorization", "Content-Type", "X-Session-Token")
                }
            }
            admin = @{
                base_url = "http://localhost:4434"
            }
        }
        
        selfservice = @{
            default_browser_return_url = "http://localhost:5173/"
            allowed_return_urls = @(
                "http://localhost:5173/",
                "http://localhost:5173/dashboard",
                "http://localhost:5173/auth/callback"
            )
            
            methods = @{
                password = @{ enabled = $true }
                totp = @{ enabled = $true }
                lookup_secret = @{ enabled = $true }
                webauthn = @{ enabled = $true }
                link = @{ enabled = $true }
                code = @{ enabled = $true }
            }
            
            flows = @{
                error = @{
                    ui_url = "http://localhost:5173/auth/error"
                }
                
                settings = @{
                    ui_url = "http://localhost:5173/auth/settings"
                    privileged_session_max_age = "15m"
                    required_aal = "highest_available"
                }
                
                recovery = @{
                    enabled = $true
                    ui_url = "http://localhost:5173/auth/recovery"
                    use = "code"
                }
                
                verification = @{
                    enabled = $true
                    ui_url = "http://localhost:5173/auth/verification"
                    use = "code"
                }
                
                logout = @{
                    after = @{
                        default_browser_return_url = "http://localhost:5173/auth/login"
                    }
                }
                
                login = @{
                    ui_url = "http://localhost:5173/auth/login"
                    lifespan = "10m"
                }
                
                registration = @{
                    lifespan = "10m"
                    ui_url = "http://localhost:5173/auth/registration"
                    
                    after = @{
                        password = @{
                            hooks = @(
                                @{
                                    hook = "session"
                                }
                            )
                        }
                    }
                }
            }
        }
        
        log = @{
            level = "debug"
            format = "text"
            leak_sensitive_values = $true
        }
        
        secrets = @{
            cookie = @("PLEASE-CHANGE-ME-I-AM-VERY-INSECURE")
            cipher = @("32-LONG-SECRET-NOT-SECURE-AT-ALL")
        }
        
        ciphers = @{
            algorithm = "xchacha20-poly1305"
        }
        
        hashers = @{
            algorithm = "bcrypt"
            bcrypt = @{
                cost = 8
            }
        }
        
        identity = @{
            default_schema_id = "default"
            schemas = @(
                @{
                    id = "default"
                    url = "file:///etc/kratos/identity.schema.json"
                }
            )
        }
        
        courier = @{
            smtp = @{
                connection_uri = "smtps://test:test@mailslurper:1025/?skip_ssl_verify=true"
            }
        }
    }
    
    # Create Kratos directory
    $kratosDir = ".\kratos"
    if (!(Test-Path $kratosDir)) {
        New-Item -Path $kratosDir -ItemType Directory | Out-Null
    }
    
    # Save Kratos configuration
    $kratosConfig | ConvertTo-Json -Depth 10 | Out-File -FilePath "$kratosDir\kratos.json" -Encoding UTF8
    Write-Host "✅ Kratos configuration created" -ForegroundColor Green
    
    # Create identity schema
    $identitySchema = @{
        '$id' = "https://schemas.ory.sh/presets/kratos/identity.email.schema.json"
        '$schema' = "http://json-schema.org/draft-07/schema#"
        title = "Legal AI User"
        type = "object"
        properties = @{
            traits = @{
                type = "object"
                properties = @{
                    email = @{
                        type = "string"
                        format = "email"
                        title = "E-Mail"
                        'ory.sh/kratos' = @{
                            credentials = @{
                                password = @{
                                    identifier = $true
                                }
                                totp = @{
                                    account_name = $true
                                }
                                webauthn = @{
                                    identifier = $true
                                }
                            }
                            verification = @{
                                via = "email"
                            }
                            recovery = @{
                                via = "email"
                            }
                        }
                    }
                    name = @{
                        type = "object"
                        properties = @{
                            first = @{
                                type = "string"
                                title = "First Name"
                            }
                            last = @{
                                type = "string"
                                title = "Last Name"
                            }
                        }
                        required = @("first", "last")
                    }
                    role = @{
                        type = "string"
                        enum = @("user", "admin", "legal_expert")
                        title = "Role"
                        default = "user"
                    }
                    organization = @{
                        type = "string"
                        title = "Organization"
                    }
                }
                required = @("email", "name")
                additionalProperties = $false
            }
        }
    }
    
    $identitySchema | ConvertTo-Json -Depth 10 | Out-File -FilePath "$kratosDir\identity.schema.json" -Encoding UTF8
    Write-Host "✅ Kratos identity schema created" -ForegroundColor Green
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

switch ($Command) {
    'Install' {
        Write-Host "`n🔧 Installing all components..." -ForegroundColor Green
        
        Install-ErlangOTP
        Install-RabbitMQ
        Install-Neo4j
        Install-Kratos
        
        Write-Host "`n✅ Installation completed!" -ForegroundColor Green
        Write-Host "Run with -Command Configure to setup configurations" -ForegroundColor Cyan
    }
    
    'Start' {
        Start-AllServices
        Show-ServiceStatus
    }
    
    'Stop' {
        Stop-AllServices
    }
    
    'Status' {
        Show-ServiceStatus
    }
    
    'Configure' {
        Configure-RabbitMQ
        Configure-Neo4j
        
        Write-Host "`n✅ Configuration completed!" -ForegroundColor Green
        Write-Host "Run with -Command Start to start all services" -ForegroundColor Cyan
    }
}

Write-Host "`n🎯 Quick Access URLs:" -ForegroundColor Cyan
Write-Host "  🐰 RabbitMQ Management: http://localhost:15672 (guest/guest)" -ForegroundColor White
Write-Host "  📊 Neo4j Browser: http://localhost:7474 (neo4j/password123)" -ForegroundColor White
Write-Host "  🔐 Kratos Admin: http://localhost:4434" -ForegroundColor White
Write-Host "  🔐 Kratos Public: http://localhost:4433" -ForegroundColor White
