@echo off
echo 🚀 Installing RabbitMQ as Native Windows Service
echo.

:: Create RabbitMQ directory
if not exist "rabbitmq-server" (
    echo 📁 Creating RabbitMQ directory...
    mkdir rabbitmq-server
)

cd rabbitmq-server

:: Download Erlang (required for RabbitMQ)
echo 📥 Checking Erlang installation...
where erl >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Erlang not found. Please install Erlang first from:
    echo https://www.erlang.org/downloads
    echo.
    echo Or use chocolatey: choco install erlang
    pause
    exit /b 1
)

:: Download RabbitMQ Server
echo 📥 Downloading RabbitMQ Server...
if not exist "rabbitmq_server-3.12.10.exe" (
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/rabbitmq/rabbitmq-server/releases/download/v3.12.10/rabbitmq-server-3.12.10.exe' -OutFile 'rabbitmq-server-3.12.10.exe'"
)

:: Install RabbitMQ silently
echo 🔧 Installing RabbitMQ Server...
rabbitmq-server-3.12.10.exe /S

:: Enable RabbitMQ Management Plugin
echo 🔌 Enabling RabbitMQ Management Plugin...
timeout /t 5 /nobreak >nul
"C:\Program Files\RabbitMQ Server\rabbitmq_server-3.12.10\sbin\rabbitmq-plugins.bat" enable rabbitmq_management

:: Start RabbitMQ Service
echo ▶️  Starting RabbitMQ Service...
net start RabbitMQ

:: Create default virtual host and user for AI system
echo 👤 Setting up AI user...
"C:\Program Files\RabbitMQ Server\rabbitmq_server-3.12.10\sbin\rabbitmqctl.bat" add_user ai_user ai_password_2024
"C:\Program Files\RabbitMQ Server\rabbitmq_server-3.12.10\sbin\rabbitmqctl.bat" set_user_tags ai_user administrator
"C:\Program Files\RabbitMQ Server\rabbitmq_server-3.12.10\sbin\rabbitmqctl.bat" set_permissions -p / ai_user ".*" ".*" ".*"

echo.
echo ✅ RabbitMQ Installation Complete!
echo.
echo 📊 Management Interface: http://localhost:15672
echo 👤 Username: ai_user
echo 🔑 Password: ai_password_2024
echo.
echo 🔌 Connection URL: amqp://ai_user:ai_password_2024@localhost:5672/
echo.

cd ..
pause