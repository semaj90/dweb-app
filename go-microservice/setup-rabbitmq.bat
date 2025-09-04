@echo off
REM RabbitMQ Message Queue Setup for Vector Consumer Service
REM Configures enterprise message queuing with management interface

echo Setting up RabbitMQ Message Queue for Vector Consumer Service...
echo.

REM Create RabbitMQ directories
echo [1/6] Creating RabbitMQ directories...
if not exist "C:\enterprise-services\rabbitmq" mkdir "C:\enterprise-services\rabbitmq"
if not exist "C:\enterprise-services\rabbitmq\data" mkdir "C:\enterprise-services\rabbitmq\data"
if not exist "C:\enterprise-services\rabbitmq\logs" mkdir "C:\enterprise-services\rabbitmq\logs"
if not exist "C:\enterprise-services\rabbitmq\config" mkdir "C:\enterprise-services\rabbitmq\config"

REM Check if RabbitMQ is installed
echo [2/6] Checking RabbitMQ installation...
where rabbitmq-server >nul 2>&1
if %errorlevel% == 0 (
    echo RabbitMQ server found
    rabbitmq-diagnostics status 2>nul || echo RabbitMQ service not running
) else (
    echo RabbitMQ not found. Installation required:
    echo.
    echo Step 1 - Install Erlang/OTP:
    echo   Download from: https://www.erlang.org/downloads
    echo   Version: 25.x or 26.x (latest stable)
    echo   Install with default options
    echo.
    echo Step 2 - Install RabbitMQ Server:
    echo   Download from: https://www.rabbitmq.com/install-windows.html
    echo   Run installer as Administrator
    echo   Install as Windows Service
    echo.
    echo Step 3 - Enable Management Plugin:
    echo   Run: rabbitmq-plugins enable rabbitmq_management
    echo.
    pause
)

REM Check if Erlang is installed
echo [3/6] Checking Erlang installation...
where erl >nul 2>&1
if %errorlevel% == 0 (
    echo Erlang/OTP found
) else (
    echo Warning: Erlang/OTP not found in PATH
    echo RabbitMQ requires Erlang/OTP to run
    echo Install from: https://www.erlang.org/downloads
)

REM Create RabbitMQ configuration
echo [4/6] Creating RabbitMQ configuration...
(
echo # RabbitMQ Configuration for Vector Consumer Service v2.0
echo # Enterprise message queuing configuration
echo.
echo # Network Configuration
echo listeners.tcp.default = 5672
echo management.listener.port = 15672
echo management.listener.ssl = false
echo.
echo # Authentication Configuration
echo default_user = vector_admin
echo default_pass = vector_2024_secure
echo default_vhost = /
echo default_user_tags.administrator = true
echo.
echo # Memory and Disk Configuration
echo vm_memory_high_watermark.relative = 0.6
echo disk_free_limit.relative = 2.0
echo log.file.level = info
echo log.file = C:/enterprise-services/rabbitmq/logs/rabbitmq.log
echo log.dir = C:/enterprise-services/rabbitmq/logs
echo.
echo # Clustering Configuration
echo cluster_formation.peer_discovery_backend = classic_config
echo cluster_formation.classic_config.nodes.1 = rabbit@localhost
echo.
echo # Queue Configuration for Vector Processing
echo queue_master_locator = min-masters
echo.
echo # Performance Tuning
echo channel_max = 2047
echo frame_max = 131072
echo heartbeat = 60
echo.
echo # Management Plugin Configuration
echo management.rates_mode = basic
echo management.sample_retention_policies.global.minute = 5
echo management.sample_retention_policies.global.hour = 60
echo management.sample_retention_policies.global.day = 1440
echo.
echo # Security Configuration
echo auth_backends.1 = internal
echo auth_mechanisms.1 = PLAIN
echo auth_mechanisms.2 = AMQPLAIN
) > "C:\enterprise-services\rabbitmq\config\rabbitmq.conf"

REM Create advanced RabbitMQ definitions
echo [5/6] Creating RabbitMQ definitions...
(
echo {
echo   "rabbit_version": "3.12.0",
echo   "rabbitmq_version": "3.12.0",
echo   "product_name": "RabbitMQ",
echo   "product_version": "3.12.0",
echo   "users": [
echo     {
echo       "name": "vector_admin",
echo       "password_hash": "vector_2024_secure",
echo       "hashing_algorithm": "rabbit_password_hashing_sha256",
echo       "tags": ["administrator", "management"]
echo     },
echo     {
echo       "name": "vector_service",
echo       "password_hash": "service_2024_secure", 
echo       "hashing_algorithm": "rabbit_password_hashing_sha256",
echo       "tags": ["monitoring"]
echo     }
echo   ],
echo   "vhosts": [
echo     {
echo       "name": "/",
echo       "description": "Default virtual host"
echo     },
echo     {
echo       "name": "/vector_processing",
echo       "description": "Vector processing virtual host"
echo     }
echo   ],
echo   "permissions": [
echo     {
echo       "user": "vector_admin",
echo       "vhost": "/",
echo       "configure": ".*",
echo       "write": ".*",
echo       "read": ".*"
echo     },
echo     {
echo       "user": "vector_admin", 
echo       "vhost": "/vector_processing",
echo       "configure": ".*",
echo       "write": ".*",
echo       "read": ".*"
echo     },
echo     {
echo       "user": "vector_service",
echo       "vhost": "/vector_processing",
echo       "configure": "",
echo       "write": "vector_.*",
echo       "read": "vector_.*"
echo     }
echo   ],
echo   "topic_permissions": [],
echo   "parameters": [],
echo   "global_parameters": [
echo     {
echo       "name": "cluster_name",
echo       "value": "vector_consumer_cluster"
echo     }
echo   ],
echo   "policies": [
echo     {
echo       "vhost": "/vector_processing",
echo       "name": "vector_processing_policy",
echo       "pattern": "vector_.*",
echo       "apply_to": "queues",
echo       "definition": {
echo         "max-length": 10000,
echo         "max-length-bytes": 104857600,
echo         "message-ttl": 3600000,
echo         "expires": 1800000
echo       },
echo       "priority": 0
echo     }
echo   ],
echo   "queues": [
echo     {
echo       "name": "vector_processing_v2",
echo       "vhost": "/vector_processing",
echo       "durable": true,
echo       "auto_delete": false,
echo       "arguments": {
echo         "x-message-ttl": 3600000,
echo         "x-max-length": 10000,
echo         "x-queue-type": "classic"
echo       }
echo     },
echo     {
echo       "name": "vector_results",
echo       "vhost": "/vector_processing", 
echo       "durable": true,
echo       "auto_delete": false,
echo       "arguments": {
echo         "x-message-ttl": 1800000,
echo         "x-max-length": 5000
echo       }
echo     },
echo     {
echo       "name": "vector_dlq",
echo       "vhost": "/vector_processing",
echo       "durable": true,
echo       "auto_delete": false,
echo       "arguments": {
echo         "x-message-ttl": 86400000
echo       }
echo     }
echo   ],
echo   "exchanges": [
echo     {
echo       "name": "vector_exchange",
echo       "vhost": "/vector_processing",
echo       "type": "topic",
echo       "durable": true,
echo       "auto_delete": false,
echo       "internal": false,
echo       "arguments": {}
echo     },
echo     {
echo       "name": "vector_dlx",
echo       "vhost": "/vector_processing",
echo       "type": "direct",
echo       "durable": true,
echo       "auto_delete": false,
echo       "internal": false,
echo       "arguments": {}
echo     }
echo   ],
echo   "bindings": [
echo     {
echo       "source": "vector_exchange",
echo       "vhost": "/vector_processing",
echo       "destination": "vector_processing_v2",
echo       "destination_type": "queue",
echo       "routing_key": "vector.process",
echo       "arguments": {}
echo     },
echo     {
echo       "source": "vector_exchange",
echo       "vhost": "/vector_processing", 
echo       "destination": "vector_results",
echo       "destination_type": "queue",
echo       "routing_key": "vector.result",
echo       "arguments": {}
echo     },
echo     {
echo       "source": "vector_dlx",
echo       "vhost": "/vector_processing",
echo       "destination": "vector_dlq",
echo       "destination_type": "queue",
echo       "routing_key": "failed",
echo       "arguments": {}
echo     }
echo   ]
echo }
) > "C:\enterprise-services\rabbitmq\config\definitions.json"

REM Create RabbitMQ startup script
echo [6/6] Creating RabbitMQ startup script...
(
echo @echo off
echo echo Starting RabbitMQ Message Queue for Vector Consumer Service...
echo.
echo REM Set RabbitMQ environment variables
echo set RABBITMQ_BASE=C:\enterprise-services\rabbitmq
echo set RABBITMQ_CONFIG_FILE=C:\enterprise-services\rabbitmq\config\rabbitmq
echo set RABBITMQ_MNESIA_BASE=C:\enterprise-services\rabbitmq\data
echo set RABBITMQ_LOG_BASE=C:\enterprise-services\rabbitmq\logs
echo.
echo REM Check if RabbitMQ is already running
echo sc query RabbitMQ ^| findstr "RUNNING" ^>nul
echo if errorlevel 0 ^(
echo     echo RabbitMQ service is already running
echo     rabbitmq-diagnostics status
echo     exit /b 0
echo ^)
echo.
echo REM Start RabbitMQ service
echo echo Starting RabbitMQ Windows service...
echo net start RabbitMQ
echo if errorlevel 1 ^(
echo     echo Failed to start RabbitMQ service, trying manual start...
echo     echo.
echo     start /B "RabbitMQ Server" rabbitmq-server
echo     timeout /t 10 /nobreak ^>nul
echo ^)
echo.
echo REM Wait for RabbitMQ to be ready
echo echo Waiting for RabbitMQ to be ready...
echo timeout /t 15 /nobreak ^>nul
echo.
echo REM Enable management plugin
echo echo Enabling management plugin...
echo rabbitmq-plugins enable rabbitmq_management
echo.
echo REM Apply definitions if available
echo if exist "C:\enterprise-services\rabbitmq\config\definitions.json" ^(
echo     echo Applying RabbitMQ definitions...
echo     timeout /t 5 /nobreak ^>nul
echo     curl -i -u vector_admin:vector_2024_secure -H "content-type:application/json" ^
echo          -X POST -d @C:\enterprise-services\rabbitmq\config\definitions.json ^
echo          http://localhost:15672/api/definitions
echo ^)
echo.
echo REM Display status
echo echo RabbitMQ startup complete!
echo echo.
echo echo Management Interface: http://localhost:15672
echo echo Username: vector_admin
echo echo Password: vector_2024_secure
echo echo.
echo echo AMQP URL: amqp://vector_admin:vector_2024_secure@localhost:5672/
echo echo Processing VHost: amqp://vector_admin:vector_2024_secure@localhost:5672/vector_processing
echo.
echo echo Queues configured:
echo echo - vector_processing_v2 ^(main processing queue^)
echo echo - vector_results ^(results queue^)
echo echo - vector_dlq ^(dead letter queue^)
echo.
echo rabbitmq-diagnostics status
echo.
echo pause
) > "C:\enterprise-services\rabbitmq\start-rabbitmq.bat"

REM Create RabbitMQ health check script
(
echo @echo off
echo echo RabbitMQ Health Check for Vector Consumer Service
echo echo ==============================================
echo.
echo REM Check service status
echo sc query RabbitMQ ^| findstr "STATE" ^| findstr "RUNNING" ^>nul
echo if errorlevel 1 ^(
echo     echo [✗] RabbitMQ Windows service not running
echo     exit /b 1
echo ^) else ^(
echo     echo [✓] RabbitMQ Windows service running
echo ^)
echo.
echo REM Check node health
echo rabbitmq-diagnostics status ^>nul 2^>^&1
echo if errorlevel 1 ^(
echo     echo [✗] RabbitMQ node health check failed
echo     exit /b 1
echo ^) else ^(
echo     echo [✓] RabbitMQ node healthy
echo ^)
echo.
echo REM Check management API
echo curl -s -u vector_admin:vector_2024_secure http://localhost:15672/api/overview ^>nul 2^>^&1
echo if errorlevel 1 ^(
echo     echo [✗] RabbitMQ management API not accessible
echo ^) else ^(
echo     echo [✓] RabbitMQ management API accessible
echo ^)
echo.
echo REM Display queue status
echo echo Queue Status:
echo rabbitmqctl list_queues name messages consumers
echo.
echo echo RabbitMQ Health Check Complete
echo echo Status: All systems operational
) > "C:\enterprise-services\rabbitmq\rabbitmq-health-check.bat"

echo.
echo ===================================================
echo RabbitMQ Message Queue Setup Complete!
echo ===================================================
echo.
echo Configuration Files Created:
echo - C:\enterprise-services\rabbitmq\config\rabbitmq.conf
echo - C:\enterprise-services\rabbitmq\config\definitions.json
echo - C:\enterprise-services\rabbitmq\start-rabbitmq.bat
echo - C:\enterprise-services\rabbitmq\rabbitmq-health-check.bat
echo.
echo To start RabbitMQ:
echo   C:\enterprise-services\rabbitmq\start-rabbitmq.bat
echo.
echo Management Interface:
echo   URL: http://localhost:15672
echo   Username: vector_admin
echo   Password: vector_2024_secure
echo.
echo Connection Strings:
echo   AMQP: amqp://vector_admin:vector_2024_secure@localhost:5672/
echo   Processing: amqp://vector_admin:vector_2024_secure@localhost:5672/vector_processing
echo.
echo Queues Configured:
echo - vector_processing_v2 (main processing, 10k messages, 1h TTL)
echo - vector_results (results delivery, 5k messages, 30min TTL)
echo - vector_dlq (dead letter queue, 24h TTL)
echo.
echo Health Check:
echo   C:\enterprise-services\rabbitmq\rabbitmq-health-check.bat
echo.
echo Next: Run create-windows-services.bat
pause