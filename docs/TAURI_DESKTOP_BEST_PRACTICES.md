# 🖥️ Tauri Desktop App Integration Best Practices
## Native Windows Desktop Application for Legal AI

### **Tauri Configuration**
```json
// src-tauri/tauri.conf.json
{
  "package": {
    "productName": "Legal AI Enterprise",
    "version": "2.0.0"
  },
  "build": {
    "distDir": "../sveltekit-frontend/build",
    "devPath": "http://localhost:5173",
    "beforeDevCommand": "cd sveltekit-frontend && npm run dev",
    "beforeBuildCommand": "cd sveltekit-frontend && npm run build"
  },
  "tauri": {
    "allowlist": {
      "all": false,
      "shell": {
        "all": false,
        "open": true
      },
      "fs": {
        "all": true,
        "scope": ["$APPDATA/legal-ai/**", "$DOCUMENT/**"]
      },
      "dialog": {
        "all": true
      },
      "notification": {
        "all": true
      },
      "os": {
        "all": true
      },
      "path": {
        "all": true
      },
      "process": {
        "relaunch": true,
        "exit": true
      },
      "window": {
        "all": true
      }
    },
    "bundle": {
      "active": true,
      "targets": ["msi", "nsis"],
      "identifier": "com.yourcompany.legal-ai",
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png", 
        "icons/128x128@2x.png",
        "icons/icon.icns",
        "icons/icon.ico"
      ],
      "resources": [
        "resources/*"
      ],
      "externalBin": [
        "bin/enhanced-rag-v2-local"
      ],
      "copyright": "© 2024 Your Company. All rights reserved.",
      "category": "Productivity",
      "shortDescription": "Enterprise Legal AI System",
      "longDescription": "High-performance legal document processing and AI analysis system with advanced vector search capabilities."
    },
    "security": {
      "csp": "default-src 'self'; connect-src 'self' ws://localhost:* http://localhost:*"
    },
    "updater": {
      "active": true,
      "endpoints": [
        "https://your-update-server.com/legal-ai/{{target}}/{{arch}}/{{current_version}}"
      ],
      "dialog": true,
      "pubkey": "YOUR_PUBLIC_KEY_HERE"
    },
    "windows": [
      {
        "fullscreen": false,
        "height": 800,
        "resizable": true,
        "title": "Legal AI Enterprise",
        "width": 1200,
        "minHeight": 600,
        "minWidth": 900,
        "center": true,
        "decorations": true,
        "alwaysOnTop": false,
        "skipTaskbar": false,
        "theme": "Dark"
      }
    ],
    "systemTray": {
      "iconPath": "icons/tray-icon.png",
      "iconAsTemplate": true,
      "menuOnLeftClick": false,
      "title": "Legal AI"
    }
  }
}
```

### **Tauri Backend Commands**
```rust
// src-tauri/src/main.rs
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use tauri::{
    CustomMenuItem, SystemTray, SystemTrayEvent, SystemTrayMenu, 
    Manager, State, Window, WindowEvent
};
use std::sync::Mutex;
use tokio::sync::mpsc;

#[derive(Default)]
struct AppState {
    service_status: Mutex<ServiceStatus>,
    rag_service: Mutex<Option<RAGServiceHandle>>,
}

#[derive(Debug, Clone)]
struct ServiceStatus {
    running: bool,
    health: String,
    last_check: String,
    error_count: u32,
}

#[tauri::command]
async fn start_legal_ai_service(state: State<'_, AppState>) -> Result<String, String> {
    let mut service = state.rag_service.lock().unwrap();
    
    if service.is_none() {
        let handle = RAGServiceHandle::new().map_err(|e| e.to_string())?;
        handle.start().await.map_err(|e| e.to_string())?;
        *service = Some(handle);
        
        // Update status
        let mut status = state.service_status.lock().unwrap();
        status.running = true;
        status.health = "healthy".to_string();
        status.last_check = chrono::Utc::now().to_rfc3339();
        
        Ok("Legal AI service started successfully".to_string())
    } else {
        Err("Service is already running".to_string())
    }
}

#[tauri::command]
async fn stop_legal_ai_service(state: State<'_, AppState>) -> Result<String, String> {
    let mut service = state.rag_service.lock().unwrap();
    
    if let Some(handle) = service.take() {
        handle.stop().await.map_err(|e| e.to_string())?;
        
        // Update status
        let mut status = state.service_status.lock().unwrap();
        status.running = false;
        status.health = "stopped".to_string();
        
        Ok("Legal AI service stopped successfully".to_string())
    } else {
        Err("Service is not running".to_string())
    }
}

#[tauri::command]
async fn get_service_status(state: State<'_, AppState>) -> Result<ServiceStatus, String> {
    let status = state.service_status.lock().unwrap();
    Ok(status.clone())
}

#[tauri::command]
async fn check_service_health(state: State<'_, AppState>) -> Result<ServiceStatus, String> {
    let service = state.rag_service.lock().unwrap();
    
    if let Some(handle) = service.as_ref() {
        match handle.health_check().await {
            Ok(health) => {
                let mut status = state.service_status.lock().unwrap();
                status.health = health;
                status.last_check = chrono::Utc::now().to_rfc3339();
                Ok(status.clone())
            },
            Err(e) => {
                let mut status = state.service_status.lock().unwrap();
                status.health = "unhealthy".to_string();
                status.error_count += 1;
                Err(e.to_string())
            }
        }
    } else {
        Err("Service is not running".to_string())
    }
}

#[tauri::command]
async fn open_logs_folder() -> Result<(), String> {
    let logs_path = std::env::current_dir()
        .map_err(|e| e.to_string())?
        .join("logs");
    
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(logs_path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    
    Ok(())
}

#[tauri::command]
async fn export_diagnostic_info() -> Result<String, String> {
    // Collect system diagnostic information
    let diagnostic = DiagnosticInfo {
        timestamp: chrono::Utc::now(),
        system_info: get_system_info(),
        service_status: get_all_service_status().await,
        recent_logs: get_recent_logs(100),
        performance_metrics: get_performance_metrics().await,
    };
    
    let json = serde_json::to_string_pretty(&diagnostic)
        .map_err(|e| e.to_string())?;
    
    Ok(json)
}

fn create_system_tray() -> SystemTray {
    let quit = CustomMenuItem::new("quit".to_string(), "Quit");
    let hide = CustomMenuItem::new("hide".to_string(), "Hide");
    let show = CustomMenuItem::new("show".to_string(), "Show");
    let status = CustomMenuItem::new("status".to_string(), "Service Status");
    
    let tray_menu = SystemTrayMenu::new()
        .add_item(status)
        .add_native_item(tauri::SystemTrayMenuItem::Separator)
        .add_item(show)
        .add_item(hide)
        .add_native_item(tauri::SystemTrayMenuItem::Separator)
        .add_item(quit);
    
    SystemTray::new().with_menu(tray_menu)
}

fn handle_system_tray_event(app: &tauri::AppHandle, event: SystemTrayEvent) {
    match event {
        SystemTrayEvent::LeftClick { position: _, size: _, .. } => {
            let window = app.get_window("main").unwrap();
            window.show().unwrap();
            window.set_focus().unwrap();
        },
        SystemTrayEvent::MenuItemClick { id, .. } => {
            let window = app.get_window("main").unwrap();
            match id.as_str() {
                "quit" => {
                    std::process::exit(0);
                },
                "hide" => {
                    window.hide().unwrap();
                },
                "show" => {
                    window.show().unwrap();
                    window.set_focus().unwrap();
                },
                "status" => {
                    window.emit("show-status-modal", {}).unwrap();
                },
                _ => {}
            }
        },
        _ => {}
    }
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .system_tray(create_system_tray())
        .on_system_tray_event(handle_system_tray_event)
        .invoke_handler(tauri::generate_handler![
            start_legal_ai_service,
            stop_legal_ai_service, 
            get_service_status,
            check_service_health,
            open_logs_folder,
            export_diagnostic_info
        ])
        .on_window_event(|event| match event.event() {
            WindowEvent::CloseRequested { api, .. } => {
                // Hide instead of close
                event.window().hide().unwrap();
                api.prevent_close();
            },
            _ => {}
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### **RAG Service Handle Implementation**
```rust
// src-tauri/src/rag_service.rs
use std::process::{Command, Child, Stdio};
use std::sync::Arc;
use tokio::sync::Mutex;
use reqwest::Client;
use serde_json::Value;

pub struct RAGServiceHandle {
    process: Arc<Mutex<Option<Child>>>,
    client: Client,
    base_url: String,
}

impl RAGServiceHandle {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            process: Arc::new(Mutex::new(None)),
            client: Client::new(),
            base_url: "http://localhost:8097".to_string(),
        })
    }
    
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut process_guard = self.process.lock().await;
        
        if process_guard.is_some() {
            return Err("Service is already running".into());
        }
        
        // Start the enhanced RAG service
        let child = Command::new("enhanced-rag-v2-local.exe")
            .current_dir("./")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        
        *process_guard = Some(child);
        
        // Wait for service to be ready
        self.wait_for_service_ready().await?;
        
        Ok(())
    }
    
    pub async fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut process_guard = self.process.lock().await;
        
        if let Some(mut child) = process_guard.take() {
            child.kill()?;
            child.wait()?;
        }
        
        Ok(())
    }
    
    pub async fn health_check(&self) -> Result<String, Box<dyn std::error::Error>> {
        let response = self.client
            .get(&format!("{}/health", self.base_url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await?;
        
        if response.status().is_success() {
            let health_data: Value = response.json().await?;
            Ok(health_data["status"].as_str().unwrap_or("unknown").to_string())
        } else {
            Err(format!("Health check failed with status: {}", response.status()).into())
        }
    }
    
    async fn wait_for_service_ready(&self) -> Result<(), Box<dyn std::error::Error>> {
        let max_attempts = 30; // 30 seconds timeout
        let mut attempts = 0;
        
        while attempts < max_attempts {
            if let Ok(_) = self.health_check().await {
                return Ok(());
            }
            
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            attempts += 1;
        }
        
        Err("Service failed to start within timeout period".into())
    }
}
```

### **Frontend Integration (SvelteKit)**
```typescript
// src/lib/tauri-integration.ts
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';
import { sendNotification } from '@tauri-apps/api/notification';

export interface ServiceStatus {
  running: boolean;
  health: string;
  last_check: string;
  error_count: number;
}

export class TauriLegalAIService {
  private statusCheckInterval: number | null = null;
  
  async startService(): Promise<string> {
    try {
      const result = await invoke<string>('start_legal_ai_service');
      await sendNotification({
        title: 'Legal AI',
        body: 'Service started successfully'
      });
      this.startStatusMonitoring();
      return result;
    } catch (error) {
      await sendNotification({
        title: 'Legal AI Error',
        body: `Failed to start service: ${error}`
      });
      throw error;
    }
  }
  
  async stopService(): Promise<string> {
    try {
      this.stopStatusMonitoring();
      const result = await invoke<string>('stop_legal_ai_service');
      await sendNotification({
        title: 'Legal AI',
        body: 'Service stopped successfully'
      });
      return result;
    } catch (error) {
      await sendNotification({
        title: 'Legal AI Error',
        body: `Failed to stop service: ${error}`
      });
      throw error;
    }
  }
  
  async getStatus(): Promise<ServiceStatus> {
    return await invoke<ServiceStatus>('get_service_status');
  }
  
  async checkHealth(): Promise<ServiceStatus> {
    return await invoke<ServiceStatus>('check_service_health');
  }
  
  async openLogsFolder(): Promise<void> {
    await invoke('open_logs_folder');
  }
  
  async exportDiagnostics(): Promise<string> {
    return await invoke<string>('export_diagnostic_info');
  }
  
  private startStatusMonitoring(): void {
    this.statusCheckInterval = window.setInterval(async () => {
      try {
        const status = await this.checkHealth();
        if (status.health === 'unhealthy') {
          await sendNotification({
            title: 'Legal AI Warning',
            body: 'Service health check failed'
          });
        }
      } catch (error) {
        console.error('Health check failed:', error);
      }
    }, 30000); // Check every 30 seconds
  }
  
  private stopStatusMonitoring(): void {
    if (this.statusCheckInterval) {
      window.clearInterval(this.statusCheckInterval);
      this.statusCheckInterval = null;
    }
  }
  
  async initializeEventListeners(): Promise<void> {
    // Listen for system tray events
    await listen('show-status-modal', () => {
      // Show status modal in your SvelteKit app
      this.showStatusModal();
    });
    
    // Listen for service events
    await listen('service-status-changed', (event) => {
      console.log('Service status changed:', event.payload);
    });
  }
  
  private showStatusModal(): void {
    // Implement modal showing in your SvelteKit app
    // This could dispatch a custom event or update a store
    window.dispatchEvent(new CustomEvent('show-service-status'));
  }
}

// Export singleton instance
export const tauriService = new TauriLegalAIService();
```

### **Build Scripts**
```powershell
# build-tauri-app.ps1
param(
    [string]$BuildMode = "release"
)

Write-Host "🔨 Building Tauri Legal AI Desktop App" -ForegroundColor Green

# Ensure Tauri CLI is installed
if (!(Get-Command "cargo-tauri" -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Tauri CLI..." -ForegroundColor Yellow
    cargo install tauri-cli
}

# Build the SvelteKit frontend first
Write-Host "Building SvelteKit frontend..." -ForegroundColor Yellow
Push-Location sveltekit-frontend
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Frontend build failed" -ForegroundColor Red
    exit 1
}
Pop-Location

# Copy Legal AI binaries to Tauri resources
Write-Host "Copying Legal AI binaries..." -ForegroundColor Yellow
$binPath = "src-tauri\bin"
if (!(Test-Path $binPath)) {
    New-Item -ItemType Directory -Path $binPath | Out-Null
}

Copy-Item "go-microservice\bin\enhanced-rag-v2-local.exe" "$binPath\"

# Build Tauri app
Write-Host "Building Tauri application..." -ForegroundColor Yellow
Push-Location src-tauri

if ($BuildMode -eq "dev") {
    cargo tauri dev
} else {
    cargo tauri build
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Tauri build failed" -ForegroundColor Red
    exit 1
}

Pop-Location

Write-Host "✅ Tauri application built successfully" -ForegroundColor Green

if ($BuildMode -eq "release") {
    $outputPath = "src-tauri\target\release\bundle"
    Write-Host "📦 Installer packages created in: $outputPath" -ForegroundColor Cyan
    Get-ChildItem $outputPath -Recurse -Include "*.msi", "*.exe" | ForEach-Object {
        Write-Host "  - $($_.Name) ($([math]::Round($_.Length / 1MB, 2)) MB)"
    }
}
```

This comprehensive Tauri implementation provides a professional desktop application with native Windows integration, system tray support, service management, and seamless integration with your existing Legal AI infrastructure.
