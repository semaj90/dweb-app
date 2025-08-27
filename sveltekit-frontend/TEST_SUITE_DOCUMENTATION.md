# 🧪 Comprehensive Test Suite Documentation

## **Test Suite Overview**

This test suite provides comprehensive coverage for the Legal AI Platform's XState machine integration, route health monitoring, SSR/hydration compatibility, and real-time re-validation systems.

---

## 🎯 **Test Categories**

### 1. **XState Machine Integration Tests** 
**File:** `src/lib/tests/integration/xstate-machine.test.ts`

**Coverage:**
- ✅ **Initial State Validation** - Machine starts in correct idle state with proper context
- ✅ **Case Loading Flow** - LOAD_CASE event handling and async operations
- ✅ **Case Creation** - Form data processing and CREATE_CASE workflow
- ✅ **Evidence Management** - File upload queue and evidence selection
- ✅ **AI Analysis Flow** - Analysis progress tracking and completion
- ✅ **Navigation and UI State** - Tab switching and workflow stage management
- ✅ **Search Functionality** - Query handling and result management
- ✅ **Error Handling** - Retry logic and error dismissal
- ✅ **Selectors** - State derivation and computed values
- ✅ **SSR/Hydration Compatibility** - Server-safe initialization

**Key Test Scenarios:**
```typescript
// Case loading with mock API
actor.send({ type: 'LOAD_CASE', caseId: 'test-case-123' });

// Evidence file handling
const mockFiles = [new File(['content'], 'evidence.pdf')];
actor.send({ type: 'ADD_EVIDENCE', files: mockFiles });

// AI analysis progress
actor.send({ type: 'AI_ANALYSIS_PROGRESS', progress: 50 });
```

### 2. **Route Health Endpoint Tests**
**File:** `src/lib/tests/integration/route-health.test.ts`

**Coverage:**
- ✅ **Health Check Endpoint** - `/api/health` comprehensive status
- ✅ **Database Connection** - PostgreSQL + pgvector health validation
- ✅ **Redis Connection** - Cache service availability
- ✅ **Ollama Integration** - AI model service health
- ✅ **Debug Logs Endpoint** - `/api/debug/logs` with filtering
- ✅ **AI Chat Endpoint** - Streaming response handling
- ✅ **Vector Search** - `/api/ai/vector-search` with similarity
- ✅ **File Upload** - `/api/upload` validation and processing
- ✅ **Performance Monitoring** - Response time validation
- ✅ **Error Scenarios** - Malformed requests and service failures

**Health Check Response:**
```json
{
  "status": "healthy",
  "services": {
    "database": { "status": "ok", "responseTime": 50 },
    "redis": { "status": "ok", "responseTime": 5 },
    "ollama": { "status": "ok", "responseTime": 100 }
  },
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

### 3. **SSR Context and Session Integrity**
**File:** `src/lib/tests/integration/ssr-session-integrity.test.ts`

**Coverage:**
- ✅ **Server Hooks Integration** - `hooks.server.ts` functionality
- ✅ **Session Cookie Validation** - Lucia auth integration
- ✅ **Context Preservation** - Middleware chain integrity
- ✅ **Security Features** - Secure cookies, CSRF protection
- ✅ **Session Expiration** - Token refresh and cleanup
- ✅ **Page Load Context** - User data serialization
- ✅ **API Authentication** - Protected route enforcement
- ✅ **XState SSR Integration** - Machine state serialization
- ✅ **Error Boundaries** - Graceful failure handling
- ✅ **Performance Caching** - Session validation optimization

**Session Flow:**
```typescript
// Validate session during SSR
mockLucia.validateSession.mockResolvedValue({
  session: { id: sessionId, userId: 'user-123' },
  user: { id: 'user-123', email: 'test@example.com' }
});

// Context preserved across middleware
expect(mockEvent.locals.user).toBeTruthy();
expect(mockEvent.locals.session).toBeTruthy();
```

### 4. **UI Hydration with XState**
**File:** `src/lib/tests/integration/ui-hydration.test.ts`

**Coverage:**
- ✅ **Component Hydration** - CaseManagerXState without errors
- ✅ **State Persistence** - Server state preservation during hydration
- ✅ **Hydration Mismatch** - Client/server state differences
- ✅ **LocalStorage Integration** - XState context persistence
- ✅ **Event Handler Preservation** - Post-hydration interactivity
- ✅ **Form Submissions** - Hydration-safe form handling
- ✅ **File Upload Handling** - Client-side file processing
- ✅ **Route Parameter Integration** - Case ID from URL params
- ✅ **WebSocket Reconnection** - Real-time connection restoration
- ✅ **Performance Optimization** - Hydration timing and layout stability
- ✅ **Error Recovery** - Machine initialization failures

**Hydration Test:**
```typescript
// Preserve server-rendered state
const serverState = {
  value: 'loaded',
  context: {
    case: { id: 'case-123', title: 'Hydration Test Case' },
    evidence: [{ id: 'evidence-1', title: 'Test Evidence' }]
  }
};

// Verify state preservation
expect(container.textContent).toContain('Hydration Test Case');
```

### 5. **Interval-Based Re-validation**
**File:** `src/lib/tests/integration/interval-revalidation.test.ts`

**Coverage:**
- ✅ **AI Analysis Progress** - Automated progress polling at 5-second intervals
- ✅ **Evidence Upload Tracking** - File upload progress monitoring
- ✅ **Case Status Synchronization** - 30-second server sync intervals
- ✅ **Conflict Detection** - Version control and merge resolution
- ✅ **Offline Scenarios** - Exponential backoff retry logic
- ✅ **Real-time Collaboration** - WebSocket message processing
- ✅ **Typing Indicators** - Collaborative editing feedback
- ✅ **Performance Monitoring** - Metrics collection at 1-minute intervals
- ✅ **Health Check Intervals** - Service monitoring at 2-minute intervals
- ✅ **Background Sync Optimization** - Visibility-based frequency adjustment
- ✅ **Memory Management** - Interval cleanup and leak prevention

**Interval Configuration:**
```typescript
// AI Analysis Progress: 5 seconds
vi.advanceTimersByTime(5000);

// Case Synchronization: 30 seconds  
vi.advanceTimersByTime(30000);

// Performance Metrics: 60 seconds
vi.advanceTimersByTime(60000);

// Health Checks: 120 seconds
vi.advanceTimersByTime(120000);
```

---

## 🚀 **Running Tests**

### **Individual Test Suites**
```bash
# XState machine tests
npm run test:xstate

# Route health tests  
npm run test:routes

# SSR session integrity
npm run test:ssr

# UI hydration tests
npm run test:hydration  

# Interval re-validation
npm run test:revalidation
```

### **Complete Integration Testing**
```bash
# Run all integration tests
npm run test:integration

# Watch mode with hot reload
npm run test:integration:watch

# UI test runner with browser interface
npm run test:integration:ui

# Complete smoke test suite
npm run test:smoke
```

### **Coverage and Reporting**
```bash
# Generate coverage report
npm run test:coverage

# View coverage in browser
# Open coverage/index.html after running coverage
```

---

## 📊 **Coverage Targets**

| Component | Target | Current |
|-----------|--------|---------|
| XState Machines | 90% | ✅ |
| API Routes | 85% | ✅ |
| SSR Hooks | 80% | ✅ |
| UI Components | 75% | ✅ |
| Service Integration | 80% | ✅ |

---

## 🔧 **Test Configuration**

### **Vitest Configuration**
**File:** `vitest.config.integration.ts`

**Features:**
- ✅ **JSDOM Environment** - Browser-like testing environment
- ✅ **Global Test APIs** - No need for imports in test files
- ✅ **SvelteKit Integration** - Full SvelteKit plugin support
- ✅ **Path Aliases** - Proper `$lib`, `$app`, `$routes` resolution
- ✅ **Mock Configuration** - Automatic dependency mocking
- ✅ **Coverage Thresholds** - Minimum 70% coverage enforcement
- ✅ **Timeout Configuration** - 10-second test timeouts for async operations

### **Test Setup**
**File:** `src/lib/tests/setup/integration-setup.ts`

**Global Mocks:**
- ✅ **SvelteKit Environment** - `$app/environment`, `$app/navigation`
- ✅ **Web APIs** - `fetch`, `WebSocket`, `localStorage`, `crypto`
- ✅ **Browser APIs** - `IntersectionObserver`, `ResizeObserver`, `performance`
- ✅ **File APIs** - `File`, `FileReader`, `URL.createObjectURL`
- ✅ **Default Responses** - Comprehensive API endpoint mocking

---

## 🎯 **Test Scenarios by User Flow**

### **Case Management Workflow**
1. **Load Case** → Test XState machine case loading
2. **Add Evidence** → Test file upload and progress tracking  
3. **Start AI Analysis** → Test interval-based progress updates
4. **Real-time Updates** → Test WebSocket collaboration
5. **Session Management** → Test SSR context preservation

### **Error Recovery Flows**
1. **Network Failures** → Test offline scenarios and retry logic
2. **Session Expiration** → Test authentication refresh
3. **Hydration Errors** → Test client/server state mismatch
4. **Service Outages** → Test health check escalation

### **Performance Validation**
1. **Hydration Speed** → Test < 1 second hydration time
2. **API Response Times** → Test < 5 second timeout limits
3. **Memory Usage** → Test interval cleanup and leak prevention
4. **Background Sync** → Test visibility-based optimization

---

## 📈 **Continuous Integration**

### **Pre-commit Hooks**
```bash
# Run smoke tests before commit
git commit → npm run test:smoke

# Type check + integration tests
npm run check:all && npm run test:integration
```

### **CI/CD Pipeline**
```yaml
# GitHub Actions / CI Pipeline
test:
  runs-on: ubuntu-latest
  steps:
    - name: Install dependencies
      run: npm ci
    - name: Run integration tests
      run: npm run test:integration
    - name: Generate coverage
      run: npm run test:coverage
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
```

---

## 🛠️ **Development Workflow**

### **TDD Approach**
1. **Write failing test** for new feature
2. **Implement minimum code** to pass
3. **Refactor** while maintaining test coverage
4. **Add integration test** for full workflow

### **Debugging Tests**
```bash
# Run single test file with verbose output
npx vitest src/lib/tests/integration/xstate-machine.test.ts --reporter=verbose

# Debug with browser DevTools
npm run test:integration:ui

# Watch specific test pattern
npx vitest --watch --grep "AI Analysis"
```

---

## ✅ **Test Suite Success Criteria**

### **Automated Validation**
- ✅ **XState machine state transitions** work correctly
- ✅ **API route health checks** return expected responses  
- ✅ **SSR session management** preserves user context
- ✅ **UI hydration** maintains state without errors
- ✅ **Interval re-validation** provides real-time updates

### **Performance Benchmarks**
- ✅ **Hydration time** < 1 second
- ✅ **API response time** < 5 seconds
- ✅ **Test execution time** < 30 seconds for full suite
- ✅ **Memory usage** bounded during long-running intervals

### **Error Recovery**
- ✅ **Network failures** handled with exponential backoff
- ✅ **Session expiration** triggers proper re-authentication  
- ✅ **Service outages** show appropriate user feedback
- ✅ **Hydration mismatches** recover gracefully

---

**🎉 Test Suite Status: PRODUCTION READY**

The comprehensive test suite validates all critical user flows, ensures SSR/hydration compatibility, and provides confidence for continuous deployment of the Legal AI Platform.