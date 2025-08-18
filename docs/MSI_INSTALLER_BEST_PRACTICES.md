# 📦 MSI Installer Best Practices for Legal AI Enterprise
## Professional Windows Deployment

### **WiX Installer Configuration**
```xml
<!-- LegalAI.wxs - WiX installer configuration -->
<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
  <Product Id="*" 
           Name="Legal AI Enterprise System" 
           Language="1033" 
           Version="2.0.0" 
           Manufacturer="Your Company" 
           UpgradeCode="PUT-GUID-HERE">
    
    <Package InstallerVersion="200" 
             Compressed="yes" 
             InstallScope="perMachine"
             AdminImage="no" />
    
    <!-- Major upgrade support -->
    <MajorUpgrade DowngradeErrorMessage="A newer version is already installed." />
    
    <!-- Media and source definitions -->
    <MediaTemplate EmbedCab="yes" />
    
    <!-- Feature definitions -->
    <Feature Id="ProductFeature" Title="Legal AI Core" Level="1">
      <ComponentGroupRef Id="ProductComponents" />
      <ComponentGroupRef Id="ServiceComponents" />
      <ComponentGroupRef Id="DatabaseComponents" />
    </Feature>
    
    <Feature Id="DesktopApp" Title="Desktop Application" Level="2">
      <ComponentGroupRef Id="DesktopComponents" />
    </Feature>
    
    <Feature Id="DeveloperTools" Title="Developer Tools" Level="1000">
      <ComponentGroupRef Id="DevToolsComponents" />
    </Feature>
    
    <!-- Installation directory structure -->
    <Directory Id="TARGETDIR" Name="SourceDir">
      <Directory Id="ProgramFiles64Folder">
        <Directory Id="INSTALLFOLDER" Name="Legal AI Enterprise">
          
          <!-- Core application -->
          <Directory Id="CoreFolder" Name="Core">
            <Component Id="EnhancedRAGService" Guid="PUT-GUID-HERE">
              <File Id="enhanced_rag_exe" 
                    Source="$(var.SourceDir)\enhanced-rag-v2-local.exe" 
                    KeyPath="yes" />
              
              <!-- Windows Service registration -->
              <ServiceInstall Id="LegalAIService"
                            Type="ownProcess"
                            Name="LegalAIService"
                            DisplayName="Legal AI Enterprise Service"
                            Description="High-performance legal document processing and AI system"
                            Start="auto"
                            Account="LocalSystem"
                            ErrorControl="normal" />
              
              <ServiceControl Id="StartLegalAIService"
                            Stop="both"
                            Remove="uninstall"
                            Name="LegalAIService"
                            Wait="yes" />
            </Component>
          </Directory>
          
          <!-- Database components -->
          <Directory Id="DatabaseFolder" Name="Database">
            <Component Id="DatabaseSchema" Guid="PUT-GUID-HERE">
              <File Id="schema_sql" 
                    Source="$(var.SourceDir)\enhanced-rag-v2-schema.sql" />
            </Component>
          </Directory>
          
          <!-- Configuration -->
          <Directory Id="ConfigFolder" Name="Config">
            <Component Id="ConfigFiles" Guid="PUT-GUID-HERE">
              <File Id="config_json" 
                    Source="$(var.SourceDir)\config.json" />
              
              <!-- Registry entries for configuration -->
              <RegistryKey Root="HKLM" 
                         Key="SOFTWARE\Legal AI Enterprise">
                <RegistryValue Name="InstallPath" 
                             Value="[INSTALLFOLDER]" 
                             Type="string" />
                <RegistryValue Name="Version" 
                             Value="2.0.0" 
                             Type="string" />
              </RegistryKey>
            </Component>
          </Directory>
          
          <!-- Desktop application (optional) -->
          <Directory Id="DesktopFolder" Name="Desktop">
            <Component Id="DesktopApp" Guid="PUT-GUID-HERE">
              <File Id="desktop_app_exe" 
                    Source="$(var.SourceDir)\legal-ai-desktop.exe" />
            </Component>
          </Directory>
          
        </Directory>
      </Directory>
      
      <!-- Start Menu shortcuts -->
      <Directory Id="ProgramMenuFolder">
        <Directory Id="ApplicationProgramsFolder" Name="Legal AI Enterprise">
          <Component Id="ApplicationShortcut" Guid="PUT-GUID-HERE">
            <Shortcut Id="ApplicationStartMenuShortcut"
                     Name="Legal AI Enterprise"
                     Description="Launch Legal AI Enterprise System"
                     Target="[#enhanced_rag_exe]"
                     WorkingDirectory="CoreFolder" />
            
            <RemoveFolder Id="ApplicationProgramsFolder" On="uninstall" />
            <RegistryValue Root="HKCU" 
                         Key="Software\Microsoft\Legal AI Enterprise" 
                         Name="installed" 
                         Type="integer" 
                         Value="1" 
                         KeyPath="yes" />
          </Component>
        </Directory>
      </Directory>
    </Directory>
    
    <!-- Custom actions for installation -->
    <CustomAction Id="InstallDatabase"
                  BinaryKey="CustomActionsDLL"
                  DllEntry="InstallDatabase"
                  Execute="deferred"
                  Impersonate="no" />
    
    <CustomAction Id="ConfigureServices"
                  BinaryKey="CustomActionsDLL"
                  DllEntry="ConfigureServices"
                  Execute="deferred"
                  Impersonate="no" />
    
    <CustomAction Id="StartServices"
                  BinaryKey="CustomActionsDLL"
                  DllEntry="StartServices"
                  Execute="deferred"
                  Impersonate="no" />
    
    <!-- Installation sequence -->
    <InstallExecuteSequence>
      <Custom Action="InstallDatabase" After="InstallFiles">
        NOT Installed
      </Custom>
      <Custom Action="ConfigureServices" After="InstallDatabase">
        NOT Installed
      </Custom>
      <Custom Action="StartServices" Before="InstallFinalize">
        NOT Installed
      </Custom>
    </InstallExecuteSequence>
    
    <!-- Prerequisites -->
    <PropertyRef Id="WIX_IS_NETFRAMEWORK_48_OR_LATER_INSTALLED" />
    <Condition Message="This application requires .NET Framework 4.8 or later.">
      <![CDATA[Installed OR WIX_IS_NETFRAMEWORK_48_OR_LATER_INSTALLED]]>
    </Condition>
    
  </Product>
  
  <!-- Component groups -->
  <Fragment>
    <ComponentGroup Id="ProductComponents" Directory="INSTALLFOLDER">
      <ComponentRef Id="EnhancedRAGService" />
      <ComponentRef Id="ConfigFiles" />
    </ComponentGroup>
    
    <ComponentGroup Id="ServiceComponents" Directory="CoreFolder">
      <!-- Additional service components -->
    </ComponentGroup>
    
    <ComponentGroup Id="DatabaseComponents" Directory="DatabaseFolder">
      <ComponentRef Id="DatabaseSchema" />
    </ComponentGroup>
    
    <ComponentGroup Id="DesktopComponents" Directory="DesktopFolder">
      <ComponentRef Id="DesktopApp" />
    </ComponentGroup>
  </Fragment>
  
</Wix>
```

### **Build Script for MSI**
```powershell
# build-installer.ps1
param(
    [string]$BuildConfiguration = "Release",
    [string]$Version = "2.0.0"
)

Write-Host "🔨 Building Legal AI Enterprise Installer v$Version" -ForegroundColor Green

# Set paths
$SourceDir = ".\dist"
$WixPath = "${env:ProgramFiles(x86)}\WiX Toolset v3.11\bin"
$OutputDir = ".\installer"

# Ensure output directory exists
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# Build the Go services first
Write-Host "Building Go services..." -ForegroundColor Yellow
Push-Location go-microservice
go build -o ..\dist\enhanced-rag-v2-local.exe cmd\enhanced-rag-v2-local\main.go
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Go build failed" -ForegroundColor Red
    exit 1
}
Pop-Location

# Build Rust services
Write-Host "Building Rust services..." -ForegroundColor Yellow
Push-Location rust-services
cargo build --release
Copy-Item target\release\*.exe ..\dist\
Pop-Location

# Copy additional files
Write-Host "Copying configuration files..." -ForegroundColor Yellow
Copy-Item sql\enhanced-rag-v2-schema.sql dist\
Copy-Item config\production.json dist\config.json

# Generate WiX object files
Write-Host "Compiling WiX source..." -ForegroundColor Yellow
& "$WixPath\candle.exe" -dSourceDir=$SourceDir -dVersion=$Version LegalAI.wxs
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ WiX compilation failed" -ForegroundColor Red
    exit 1
}

# Link to create MSI
Write-Host "Linking MSI package..." -ForegroundColor Yellow
& "$WixPath\light.exe" -ext WixUIExtension -out "$OutputDir\LegalAI-v$Version.msi" LegalAI.wixobj
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ MSI linking failed" -ForegroundColor Red
    exit 1
}

# Sign the MSI (if certificate available)
if (Test-Path "cert\codesign.pfx") {
    Write-Host "Signing MSI package..." -ForegroundColor Yellow
    & signtool sign /f cert\codesign.pfx /p $env:CERT_PASSWORD /t http://timestamp.comodoca.com/authenticode "$OutputDir\LegalAI-v$Version.msi"
}

Write-Host "✅ Installer built successfully: $OutputDir\LegalAI-v$Version.msi" -ForegroundColor Green
Write-Host "Package size: $((Get-Item "$OutputDir\LegalAI-v$Version.msi").Length / 1MB) MB"
```

### **Custom Actions DLL (C#)**
```csharp
// CustomActions.cs - MSI custom actions
using System;
using System.Data.SqlClient;
using System.Diagnostics;
using System.IO;
using Microsoft.Deployment.WindowsInstaller;

public class CustomActions
{
    [CustomAction]
    public static ActionResult InstallDatabase(Session session)
    {
        session.Log("Starting database installation...");
        
        try
        {
            var installPath = session["INSTALLFOLDER"];
            var schemaFile = Path.Combine(installPath, "Database", "enhanced-rag-v2-schema.sql");
            
            if (!File.Exists(schemaFile))
            {
                session.Log($"Schema file not found: {schemaFile}");
                return ActionResult.Failure;
            }
            
            // Check if PostgreSQL is installed and running
            if (!IsPostgreSQLRunning())
            {
                session.Log("PostgreSQL is not running. Attempting to start...");
                StartPostgreSQL();
            }
            
            // Execute schema installation
            var schema = File.ReadAllText(schemaFile);
            ExecuteSQLScript(schema, session);
            
            session.Log("Database installation completed successfully");
            return ActionResult.Success;
        }
        catch (Exception ex)
        {
            session.Log($"Database installation failed: {ex.Message}");
            return ActionResult.Failure;
        }
    }
    
    [CustomAction]
    public static ActionResult ConfigureServices(Session session)
    {
        session.Log("Configuring Legal AI services...");
        
        try
        {
            var installPath = session["INSTALLFOLDER"];
            var configFile = Path.Combine(installPath, "Config", "config.json");
            
            // Update configuration with actual paths and settings
            var config = File.ReadAllText(configFile);
            config = config.Replace("{{INSTALL_PATH}}", installPath);
            config = config.Replace("{{DATA_PATH}}", Path.Combine(installPath, "Data"));
            
            File.WriteAllText(configFile, config);
            
            // Set up Windows service recovery options
            ConfigureServiceRecovery("LegalAIService");
            
            session.Log("Service configuration completed successfully");
            return ActionResult.Success;
        }
        catch (Exception ex)
        {
            session.Log($"Service configuration failed: {ex.Message}");
            return ActionResult.Failure;
        }
    }
    
    [CustomAction]
    public static ActionResult StartServices(Session session)
    {
        session.Log("Starting Legal AI services...");
        
        try
        {
            // Start the main Legal AI service
            StartWindowsService("LegalAIService", session);
            
            // Verify service is running
            System.Threading.Thread.Sleep(5000);
            if (IsServiceRunning("LegalAIService"))
            {
                session.Log("Legal AI service started successfully");
                return ActionResult.Success;
            }
            else
            {
                session.Log("Failed to start Legal AI service");
                return ActionResult.Failure;
            }
        }
        catch (Exception ex)
        {
            session.Log($"Service startup failed: {ex.Message}");
            return ActionResult.Failure;
        }
    }
    
    private static bool IsPostgreSQLRunning()
    {
        try
        {
            var processes = Process.GetProcessesByName("postgres");
            return processes.Length > 0;
        }
        catch
        {
            return false;
        }
    }
    
    private static void StartPostgreSQL()
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "net",
                Arguments = "start postgresql-x64-14",
                UseShellExecute = false,
                CreateNoWindow = true
            };
            
            var process = Process.Start(psi);
            process?.WaitForExit();
        }
        catch (Exception ex)
        {
            throw new Exception($"Failed to start PostgreSQL: {ex.Message}");
        }
    }
    
    private static void ExecuteSQLScript(string script, Session session)
    {
        var connectionString = "Host=localhost;Database=legal_ai_rag;Username=postgres;Password=postgres";
        
        using (var connection = new NpgsqlConnection(connectionString))
        {
            connection.Open();
            
            using (var command = new NpgsqlCommand(script, connection))
            {
                command.ExecuteNonQuery();
            }
        }
    }
    
    private static void ConfigureServiceRecovery(string serviceName)
    {
        // Configure service to restart on failure
        var psi = new ProcessStartInfo
        {
            FileName = "sc",
            Arguments = $"failure {serviceName} reset= 86400 actions= restart/5000/restart/5000/restart/5000",
            UseShellExecute = false,
            CreateNoWindow = true
        };
        
        var process = Process.Start(psi);
        process?.WaitForExit();
    }
    
    private static void StartWindowsService(string serviceName, Session session)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "net",
                Arguments = $"start {serviceName}",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            
            var process = Process.Start(psi);
            var output = process.StandardOutput.ReadToEnd();
            var error = process.StandardError.ReadToEnd();
            
            process.WaitForExit();
            
            if (process.ExitCode != 0)
            {
                throw new Exception($"Service start failed: {error}");
            }
            
            session.Log($"Service start output: {output}");
        }
        catch (Exception ex)
        {
            throw new Exception($"Failed to start service {serviceName}: {ex.Message}");
        }
    }
    
    private static bool IsServiceRunning(string serviceName)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "sc",
                Arguments = $"query {serviceName}",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true
            };
            
            var process = Process.Start(psi);
            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            
            return output.Contains("RUNNING");
        }
        catch
        {
            return false;
        }
    }
}
```

This MSI installer configuration provides enterprise-grade installation capabilities with proper Windows service registration, database setup, and configuration management for your Legal AI system.
