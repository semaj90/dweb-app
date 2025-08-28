# VS Code Usage Guide - Legal AI Platform

## 🚀 Quick Start

### 1. Open the Project
```bash
code C:\Users\james\Desktop\deeds-web\deeds-web-app\deeds-web-app.code-workspace
```

### 2. Start Development (Choose One Method)

#### Method A: Use VS Code Tasks (Recommended)
- **Ctrl+Shift+P** → "Tasks: Run Task"
- Select **"🚀 Start Legal AI Platform (Quick)"** for full system
- Select **"⚡ Start SvelteKit Only (Fast)"** for frontend only

#### Method B: Use VS Code Terminal
- **Ctrl+`** to open terminal
- Run: `npm run dev:full` (full system) or `cd sveltekit-frontend && npm run dev` (frontend only)

#### Method C: Use Debug/Launch
- **F5** or **Ctrl+F5**
- Select **"🚀 Launch Legal AI Platform"**

## 🛠️ Available Tasks

| Task | Description | Keyboard Shortcut |
|------|-------------|-------------------|
| 🚀 Start Legal AI Platform (Quick) | Full system startup | **Ctrl+Shift+P** → Tasks: Run Task |
| ⚡ Start SvelteKit Only (Fast) | Frontend development only | **Ctrl+Shift+P** → Tasks: Run Task |
| Dev: Full Stack (dev:full, tee logs) | Full system with logging | From task list |

## 📱 Access Points After Starting

- **Frontend**: http://localhost:5173
- **Enhanced RAG**: http://localhost:8094  
- **Upload Service**: http://localhost:8093
- **Health Dashboard**: http://localhost:5173/system/health

## 🔧 VS Code Features Configured

### IntelliSense & Language Support
- ✅ Svelte 5 support with proper syntax highlighting
- ✅ TypeScript support with auto-imports
- ✅ Go language support with CGO/CUDA
- ✅ SQL syntax highlighting for database files

### Debugging
- ✅ Node.js debugging for SvelteKit
- ✅ Launch configurations for common tasks
- ✅ Integrated terminal for Go services

### Extensions Enabled
- ✅ **Svelte for VS Code** - Svelte 5 support
- ✅ **TypeScript and JavaScript** - Language features
- ✅ **Go** - Go language support
- ✅ **Tailwind CSS IntelliSense** - CSS utilities

## 📁 Workspace Structure

```
deeds-web-app/
├── .vscode/                    # VS Code configuration
├── sveltekit-frontend/         # Main frontend application
├── go-microservice/            # Go backend services
├── build-go-services.ps1       # Service build script
└── deeds-web-app.code-workspace # Main workspace file
```

## 🐛 Troubleshooting

### Task Not Working?
1. **Ctrl+Shift+P** → "Developer: Reload Window"
2. Try running from terminal: `npm run dev:full`

### TypeScript Errors?
1. **Ctrl+Shift+P** → "TypeScript: Restart TS Server" 
2. Run task: **"TypeScript Check"**

### Extensions Issues?
1. **Ctrl+Shift+X** → Check for extension updates
2. Restart VS Code

## ⌨️ Essential Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Open Command Palette | **Ctrl+Shift+P** |
| Open Terminal | **Ctrl+`** |
| Run Task | **Ctrl+Shift+P** → "Tasks: Run Task" |
| Debug/Launch | **F5** |
| Quick File Open | **Ctrl+P** |
| Go to Symbol | **Ctrl+Shift+O** |
| Find in Files | **Ctrl+Shift+F** |

## 🎯 Recommended Workflow

1. **Open workspace**: Use the `.code-workspace` file
2. **Start development**: Use the "🚀 Start Legal AI Platform (Quick)" task
3. **Edit files**: Use the integrated editor with IntelliSense
4. **View results**: http://localhost:5173
5. **Debug issues**: Use integrated terminal and debugging tools

This setup provides a complete development environment for the YoRHa Legal AI Platform!