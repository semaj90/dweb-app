@echo off
REM =============================================================================
REM SSL/TLS SETUP SCRIPT - LEGAL AI PLATFORM (WINDOWS)
REM =============================================================================
REM This script configures SSL/TLS certificates for secure HTTPS deployment
REM Supports both self-signed certificates and Let's Encrypt certificates
REM =============================================================================

setlocal EnableDelayedExpansion

REM Configuration
set "SSL_DIR=C:\LegalAI\ssl"
set "CERT_DIR=%SSL_DIR%\certs"
set "PRIVATE_DIR=%SSL_DIR%\private"
set "DOMAIN=yourdomain.com"
set "COUNTRY=US"
set "STATE=YourState"
set "CITY=YourCity"
set "ORG=Legal AI Platform"
set "OU=IT Department"

echo ================================================================================
echo SSL/TLS CERTIFICATE SETUP - LEGAL AI PLATFORM
echo ================================================================================

REM Check if running as administrator
net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] This script must be run as Administrator
    echo [ERROR] Right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo [SSL] Starting SSL/TLS certificate setup...

REM Create SSL directory structure
call :create_ssl_directories
if errorlevel 1 goto :error

REM Show certificate options
call :show_certificate_options

REM Get user choice
set /p CERT_CHOICE="Enter your choice (1-3): "

if "%CERT_CHOICE%"=="1" (
    call :create_self_signed_certificate
) else if "%CERT_CHOICE%"=="2" (
    call :setup_lets_encrypt
) else if "%CERT_CHOICE%"=="3" (
    call :import_existing_certificate
) else (
    echo [ERROR] Invalid choice
    goto :error
)

if errorlevel 1 goto :error

REM Configure web server
call :configure_web_server
if errorlevel 1 goto :error

REM Test SSL configuration
call :test_ssl_configuration
if errorlevel 1 goto :error

echo [SSL] SSL/TLS setup completed successfully!
call :show_ssl_summary
goto :end

:create_ssl_directories
echo [SSL] Creating SSL directory structure...

if not exist "%SSL_DIR%" mkdir "%SSL_DIR%"
if not exist "%CERT_DIR%" mkdir "%CERT_DIR%"
if not exist "%PRIVATE_DIR%" mkdir "%PRIVATE_DIR%"

REM Set secure permissions on private directory
icacls "%PRIVATE_DIR%" /inheritance:d
icacls "%PRIVATE_DIR%" /grant:r Administrators:F
icacls "%PRIVATE_DIR%" /remove Users
icacls "%PRIVATE_DIR%" /remove "Authenticated Users"

echo [SSL] ✓ SSL directories created with secure permissions
exit /b 0

:show_certificate_options
echo.
echo Select SSL certificate option:
echo 1. Create self-signed certificate (for development/testing)
echo 2. Set up Let's Encrypt certificate (for production)
echo 3. Import existing certificate
echo.
exit /b 0

:create_self_signed_certificate
echo [SSL] Creating self-signed certificate...

REM Check if OpenSSL is available
openssl version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] OpenSSL is not installed or not in PATH
    echo [ERROR] Please install OpenSSL from https://slproweb.com/products/Win32OpenSSL.html
    exit /b 1
)

echo [SSL] OpenSSL found, generating self-signed certificate...

REM Generate private key
echo [SSL] Generating private key...
openssl genpkey -algorithm RSA -out "%PRIVATE_DIR%\%DOMAIN%.key" -pkcs8 -aes256
if errorlevel 1 (
    echo [ERROR] Failed to generate private key
    exit /b 1
)

REM Generate certificate signing request
echo [SSL] Generating certificate signing request...
(
echo %COUNTRY%
echo %STATE%
echo %CITY%
echo %ORG%
echo %OU%
echo %DOMAIN%
echo admin@%DOMAIN%
echo.
echo.
) | openssl req -new -key "%PRIVATE_DIR%\%DOMAIN%.key" -out "%SSL_DIR%\%DOMAIN%.csr"

if errorlevel 1 (
    echo [ERROR] Failed to generate certificate signing request
    exit /b 1
)

REM Generate self-signed certificate
echo [SSL] Generating self-signed certificate (valid for 365 days)...
openssl x509 -req -days 365 -in "%SSL_DIR%\%DOMAIN%.csr" -signkey "%PRIVATE_DIR%\%DOMAIN%.key" -out "%CERT_DIR%\%DOMAIN%.crt"
if errorlevel 1 (
    echo [ERROR] Failed to generate self-signed certificate
    exit /b 1
)

REM Create certificate bundle
copy "%CERT_DIR%\%DOMAIN%.crt" "%CERT_DIR%\%DOMAIN%.pem"

echo [SSL] ✓ Self-signed certificate created successfully
echo [SSL] Certificate: %CERT_DIR%\%DOMAIN%.crt
echo [SSL] Private Key: %PRIVATE_DIR%\%DOMAIN%.key
echo [SSL] Certificate Bundle: %CERT_DIR%\%DOMAIN%.pem

exit /b 0

:setup_lets_encrypt
echo [SSL] Setting up Let's Encrypt certificate...

REM Check if Certbot is installed
certbot --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Certbot is not installed
    echo [INFO] Please install Certbot from https://certbot.eff.org/
    echo [INFO] Or use Chocolatey: choco install certbot
    exit /b 1
)

REM Prompt for domain and email
set /p DOMAIN="Enter your domain name (e.g., yourdomain.com): "
set /p EMAIL="Enter your email address for Let's Encrypt notifications: "

echo [SSL] Requesting Let's Encrypt certificate for %DOMAIN%...
echo [SSL] This requires port 80 to be accessible from the internet

REM Request certificate using standalone authenticator
certbot certonly --standalone --email "%EMAIL%" --agree-tos --no-eff-email -d "%DOMAIN%"
if errorlevel 1 (
    echo [ERROR] Failed to obtain Let's Encrypt certificate
    echo [INFO] Make sure:
    echo [INFO] 1. Domain %DOMAIN% points to this server
    echo [INFO] 2. Port 80 is accessible from the internet
    echo [INFO] 3. No other web server is running on port 80
    exit /b 1
)

REM Copy certificates to our SSL directory
set "LE_CERT_DIR=C:\Certbot\live\%DOMAIN%"
if exist "%LE_CERT_DIR%\fullchain.pem" (
    copy "%LE_CERT_DIR%\fullchain.pem" "%CERT_DIR%\%DOMAIN%.pem"
    copy "%LE_CERT_DIR%\privkey.pem" "%PRIVATE_DIR%\%DOMAIN%.key"
    copy "%LE_CERT_DIR%\cert.pem" "%CERT_DIR%\%DOMAIN%.crt"
    echo [SSL] ✓ Let's Encrypt certificate installed successfully
) else (
    echo [ERROR] Let's Encrypt certificate files not found
    exit /b 1
)

REM Set up automatic renewal
echo [SSL] Setting up automatic certificate renewal...
schtasks /create /tn "Legal AI - SSL Renewal" /tr "certbot renew --quiet" /sc daily /st 02:00 /f
if errorlevel 1 (
    echo [WARN] Failed to create automatic renewal task
) else (
    echo [SSL] ✓ Automatic renewal scheduled for 2:00 AM daily
)

exit /b 0

:import_existing_certificate
echo [SSL] Importing existing certificate...

echo.
echo Please provide the paths to your existing certificate files:
set /p EXISTING_CERT="Certificate file (.crt/.pem): "
set /p EXISTING_KEY="Private key file (.key): "
set /p EXISTING_CA="CA bundle (optional, press Enter to skip): "

REM Validate certificate file
if not exist "%EXISTING_CERT%" (
    echo [ERROR] Certificate file not found: %EXISTING_CERT%
    exit /b 1
)

REM Validate private key file
if not exist "%EXISTING_KEY%" (
    echo [ERROR] Private key file not found: %EXISTING_KEY%
    exit /b 1
)

REM Copy certificate files
copy "%EXISTING_CERT%" "%CERT_DIR%\%DOMAIN%.crt"
copy "%EXISTING_KEY%" "%PRIVATE_DIR%\%DOMAIN%.key"

REM Copy CA bundle if provided
if not "%EXISTING_CA%"=="" (
    if exist "%EXISTING_CA%" (
        copy "%EXISTING_CA%" "%CERT_DIR%\ca-bundle.crt"
        REM Create certificate chain
        type "%CERT_DIR%\%DOMAIN%.crt" "%CERT_DIR%\ca-bundle.crt" > "%CERT_DIR%\%DOMAIN%.pem"
        echo [SSL] ✓ Certificate chain created
    ) else (
        echo [WARN] CA bundle file not found, skipping...
        copy "%CERT_DIR%\%DOMAIN%.crt" "%CERT_DIR%\%DOMAIN%.pem"
    )
) else (
    copy "%CERT_DIR%\%DOMAIN%.crt" "%CERT_DIR%\%DOMAIN%.pem"
)

echo [SSL] ✓ Existing certificate imported successfully

exit /b 0

:configure_web_server
echo [SSL] Configuring web server for SSL...

REM Create nginx configuration if nginx is available
nginx -v >nul 2>&1
if not errorlevel 1 (
    call :create_nginx_ssl_config
    exit /b 0
)

REM Create IIS configuration if IIS is available
if exist "%SystemRoot%\System32\inetsrv\iisreset.exe" (
    call :create_iis_ssl_config
    exit /b 0
)

REM Create generic SSL configuration
call :create_generic_ssl_config

exit /b 0

:create_nginx_ssl_config
echo [SSL] Creating nginx SSL configuration...

set "NGINX_CONF_DIR=C:\nginx\conf"
if not exist "%NGINX_CONF_DIR%" set "NGINX_CONF_DIR=C:\ProgramData\nginx\conf"

(
echo server {
echo     listen 80;
echo     server_name %DOMAIN% www.%DOMAIN%;
echo     return 301 https://$server_name$request_uri;
echo }
echo.
echo server {
echo     listen 443 ssl http2;
echo     server_name %DOMAIN% www.%DOMAIN%;
echo.
echo     ssl_certificate %CERT_DIR%\%DOMAIN%.pem;
echo     ssl_certificate_key %PRIVATE_DIR%\%DOMAIN%.key;
echo.
echo     ssl_protocols TLSv1.2 TLSv1.3;
echo     ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
echo     ssl_prefer_server_ciphers off;
echo.
echo     ssl_session_cache shared:SSL:1m;
echo     ssl_session_timeout 5m;
echo.
echo     # HSTS
echo     add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
echo.
echo     # Security headers
echo     add_header X-Frame-Options DENY;
echo     add_header X-Content-Type-Options nosniff;
echo     add_header X-XSS-Protection "1; mode=block";
echo.
echo     root C:/LegalAI/www;
echo     index index.html;
echo.
echo     # Proxy pass to Go services
echo     location /api/ {
echo         proxy_pass http://localhost:8094;
echo         proxy_set_header Host $host;
echo         proxy_set_header X-Real-IP $remote_addr;
echo         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
echo         proxy_set_header X-Forwarded-Proto $scheme;
echo     }
echo.
echo     location / {
echo         try_files $uri $uri/ /index.html;
echo     }
echo }
) > "%NGINX_CONF_DIR%\sites-available\legal-ai-ssl.conf"

echo [SSL] ✓ Nginx SSL configuration created
echo [SSL] Configuration file: %NGINX_CONF_DIR%\sites-available\legal-ai-ssl.conf
echo [SSL] Please include this configuration in your main nginx.conf

exit /b 0

:create_iis_ssl_config
echo [SSL] Creating IIS SSL configuration...

REM Import certificate to Windows certificate store
echo [SSL] Importing certificate to Windows certificate store...
certlm.msc

echo [SSL] Please manually configure IIS SSL binding using IIS Manager:
echo 1. Open IIS Manager
echo 2. Select your site
echo 3. Click "Bindings" in Actions panel
echo 4. Add HTTPS binding on port 443
echo 5. Select the imported certificate
echo [SSL] IIS SSL configuration requires manual setup

exit /b 0

:create_generic_ssl_config
echo [SSL] Creating generic SSL configuration files...

REM Create configuration template
(
echo # Legal AI Platform - SSL Configuration Template
echo.
echo ## Certificate Files
echo - Certificate: %CERT_DIR%\%DOMAIN%.crt
echo - Private Key: %PRIVATE_DIR%\%DOMAIN%.key
echo - Certificate Bundle: %CERT_DIR%\%DOMAIN%.pem
echo.
echo ## Web Server Configuration
echo.
echo ### Nginx Configuration
echo ```nginx
echo server {
echo     listen 443 ssl http2;
echo     server_name %DOMAIN%;
echo.
echo     ssl_certificate %CERT_DIR%\%DOMAIN%.pem;
echo     ssl_certificate_key %PRIVATE_DIR%\%DOMAIN%.key;
echo.
echo     # SSL Security Configuration
echo     ssl_protocols TLSv1.2 TLSv1.3;
echo     ssl_prefer_server_ciphers off;
echo     ssl_session_cache shared:SSL:1m;
echo.
echo     # Proxy to Legal AI services
echo     location /api/ {
echo         proxy_pass http://localhost:8094;
echo         proxy_set_header X-Forwarded-Proto https;
echo     }
echo }
echo ```
echo.
echo ### Apache Configuration
echo ```apache
echo ^<VirtualHost *:443^>
echo     ServerName %DOMAIN%
echo     DocumentRoot "C:/LegalAI/www"
echo.
echo     SSLEngine on
echo     SSLCertificateFile "%CERT_DIR%\%DOMAIN%.crt"
echo     SSLCertificateKeyFile "%PRIVATE_DIR%\%DOMAIN%.key"
echo.
echo     ProxyPass /api/ http://localhost:8094/
echo     ProxyPassReverse /api/ http://localhost:8094/
echo ^</VirtualHost^>
echo ```
) > "%SSL_DIR%\ssl-configuration-guide.md"

echo [SSL] ✓ Generic SSL configuration template created
echo [SSL] Configuration guide: %SSL_DIR%\ssl-configuration-guide.md

exit /b 0

:test_ssl_configuration
echo [SSL] Testing SSL configuration...

REM Test certificate validity
if exist "%CERT_DIR%\%DOMAIN%.crt" (
    echo [SSL] Certificate file exists: ✓
) else (
    echo [SSL] Certificate file missing: ✗
    exit /b 1
)

REM Test private key
if exist "%PRIVATE_DIR%\%DOMAIN%.key" (
    echo [SSL] Private key file exists: ✓
) else (
    echo [SSL] Private key file missing: ✗
    exit /b 1
)

REM Test certificate with OpenSSL if available
openssl version >nul 2>&1
if not errorlevel 1 (
    echo [SSL] Validating certificate with OpenSSL...
    openssl x509 -in "%CERT_DIR%\%DOMAIN%.crt" -text -noout > "%SSL_DIR%\certificate-info.txt"
    if errorlevel 1 (
        echo [SSL] Certificate validation failed: ✗
        exit /b 1
    ) else (
        echo [SSL] Certificate validation passed: ✓
    )
    
    REM Check certificate expiration
    for /f "tokens=2" %%i in ('openssl x509 -in "%CERT_DIR%\%DOMAIN%.crt" -noout -enddate ^| find "notAfter"') do (
        echo [SSL] Certificate expires: %%i
    )
)

echo [SSL] ✓ SSL configuration test completed

exit /b 0

:show_ssl_summary
echo.
echo ================================================================================
echo SSL/TLS SETUP SUMMARY
echo ================================================================================
echo Setup Date: %date% %time%
echo Domain: %DOMAIN%
echo.
echo Certificate Files:
echo - Certificate: %CERT_DIR%\%DOMAIN%.crt
echo - Private Key: %PRIVATE_DIR%\%DOMAIN%.key
echo - Certificate Bundle: %CERT_DIR%\%DOMAIN%.pem
echo.
echo Security Configuration:
echo - SSL Directory: %SSL_DIR%
echo - Private key directory secured with restricted permissions
echo - Configuration templates created
echo.
echo Next Steps:
echo 1. Configure your web server to use the SSL certificates
echo 2. Update firewall rules to allow HTTPS traffic
echo 3. Test HTTPS access to your application
echo 4. Set up certificate monitoring for expiration
echo 5. Configure automatic certificate renewal (if using Let's Encrypt)
echo.
echo Testing:
echo - Test HTTPS access: https://%DOMAIN%
echo - Check certificate: https://www.ssllabs.com/ssltest/
echo - Verify security headers: https://securityheaders.com/
echo ================================================================================

exit /b 0

:error
echo [ERROR] SSL setup failed!
echo Check the error messages above for details.
pause
exit /b 1

:end
echo [SSL] SSL setup completed. Press any key to exit.
pause