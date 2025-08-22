# Source Code Development Rules - Claude.md

## 🚀 Go Build Optimization Rules

**CRITICAL: Always use preexisting Go binaries to avoid unnecessary recompilation**

### **Rule 1: Binary Location Map**
```bash
# Enhanced RAG Services (AI Engine - Port 8094):
../go-microservice/cmd/enhanced-rag/main.go → builds to enhanced-rag.exe
../go-microservice/cmd/enhanced-rag-v2/main.go → builds to enhanced-rag-v2.exe
../go-microservice/cmd/production-rag/main.go → builds to production-rag.exe

# Kratos Services (Legal gRPC - Port 50051):
../go-services/cmd/kratos-server/main.go → builds to kratos-server.exe
../backups/go-services-backup-*/cmd/kratos-server/main.go → backup versions

# Upload Services (File Handler - Port 8093):
../go-microservice/cmd/upload-service/main.go → builds to upload-service.exe
```

### **Rule 2: Pre-build Check Commands**
```bash
# Check for existing binaries BEFORE building:
find ../go-microservice -name "*.exe" -type f | grep -E "(enhanced-rag|upload-service)"
find ../go-services -name "*.exe" -type f | grep "kratos-server"

# If found, use directly:
../go-microservice/cmd/enhanced-rag/enhanced-rag.exe &
../go-services/cmd/kratos-server/kratos-server.exe &
```

### **Rule 3: Smart Build Pattern**
```bash
# Pattern: Check → Use existing OR Build → Run
start_enhanced_rag() {
  if [ -f "../go-microservice/cmd/enhanced-rag/enhanced-rag.exe" ]; then
    echo "✅ Using existing enhanced-rag.exe"
    ../go-microservice/cmd/enhanced-rag/enhanced-rag.exe
  else
    echo "🔨 Building enhanced-rag.exe..."
    cd ../go-microservice && go build -o ./cmd/enhanced-rag/enhanced-rag.exe ./cmd/enhanced-rag && ./cmd/enhanced-rag/enhanced-rag.exe
  fi
}
```

### **Rule 4: Service Integration Architecture**

#### **agentShellMachine.ts → Go Services Flow**
```typescript
// src/lib/machines/agentShellMachine.ts
export const agentShellMachine = createMachine({
  states: {
    processing: {
      invoke: {
        src: "callAgent", // Calls Go enhanced-rag service (8094)
        onDone: "idle"
      },
      on: {
        ACCEPT_PATCH: "acceptPatchAction",     // Kratos gRPC (50051)
        SEMANTIC_SEARCH: "semanticSearchAction", // Enhanced RAG (8094)
        FILE_UPLOAD: "fileUploadAction"        // Upload service (8093)
      }
    }
  }
});
```

#### **Agent Integration Points**
```typescript
// agents/claude-agent.ts integration:
export class ClaudeAgent {
  async execute(request: ClaudeAgentRequest): Promise<ClaudeAgentResponse> {
    // 1. Enhanced RAG Context (Port 8094)
    const ragContext = await fetch('http://localhost:8094/api/rag', {
      method: 'POST',
      body: JSON.stringify({ query: request.prompt })
    });

    // 2. Context7 Analysis (existing)
    if (request.options?.includeContext7) {
      const analysis = await context7Service.analyzeComponent('sveltekit', 'legal-ai');
    }

    // 3. GPU Processing via Kratos (Port 50051)
    if (request.options?.useGPU) {
      const gpuResult = await grpcClient.call('kratos.GPUCompute', {
        data: request.context
      });
    }
  }
}
```

### **Rule 5: Current Service Status**

#### **Active Services (as of current session):**
- ✅ SvelteKit Frontend: http://localhost:5175
- ✅ MCP Multi-cluster: localhost:40000 (4 worker threads)
- ⚠️ Enhanced RAG: 8094 (needs binary check)
- ⚠️ Kratos Server: 50051 (build issues - use existing)
- ⚠️ Upload Service: 8093 (pending)

#### **Ollama Models Available:**
- ✅ `gemma3-legal:latest` (7.3 GB) - primary legal AI
- ✅ `nomic-embed-text:latest` (274 MB) - embeddings
- ✅ `deeds-web:latest` (3.0 GB) - document processing

### **Rule 6: Development Workflow**
1. **Check existing binaries** before any go build command
2. **Use SvelteKit as main entry point** (port 5175)
3. **MCP server handles multi-cluster coordination** (port 40000)
4. **Go services provide backend AI/processing** (ports 8093, 8094, 50051)
5. **agentShellMachine.ts orchestrates service communication**

### **Rule 7: Error Prevention**
- ❌ **NEVER run `go build` without checking for existing binaries**
- ❌ **NEVER ignore relative import errors in Go modules**
- ✅ **ALWAYS use find commands to locate existing executables**
- ✅ **ALWAYS check service ports before starting new instances**

---

## 🎯 agentShellMachine.ts Analysis

The XState machine in `src/lib/machines/agentShellMachine.ts` is the **central orchestrator** for:

### **State Management:**
- `idle` → `processing` → back to `idle`
- Handles agent calls via `callAgent` service
- Processes events: `ACCEPT_PATCH`, `RATE_SUGGESTION`, `SEMANTIC_SEARCH`

### **Integration Points:**
- **Claude Agent**: `agents/claude-agent.ts` (Context7 + AutoFix)
- **CrewAI Agent**: `agents/crewai-agent.ts`
- **AutoGen Agent**: `agents/autogen-agent.ts`

### **Service Communication Flow:**
```
agentShellMachine.ts → Go Services
├── Enhanced RAG (8094) ← semantic search, AI analysis
├── Kratos Server (50051) ← gRPC legal services, GPU compute
├── Upload Service (8093) ← file processing, storage
└── MCP Server (4001) ← multi-agent coordination
```

This architecture ensures **optimal resource usage** by reusing compiled Go binaries and coordinating services through the XState machine.