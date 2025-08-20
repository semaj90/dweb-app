<!--
  XState Authentication Demo Component
  Demonstrates complete integration of XState machines with Svelte components
  Uses Bits UI v2 + GPU orchestrator + Context7 documentation
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { Badge } from '$lib/components/ui/badge';
  import { Alert, AlertDescription } from '$lib/components/ui/alert';
  import { 
    Shield, CheckCircle, AlertCircle, Loader2, 
    User, MessageCircle, Settings, LogOut,
    Cpu, Zap, Brain
  } from 'lucide-svelte';
  
  // Import XState integration service
  import xstateIntegration, { 
    type GlobalAppState,
    authState,
    sessionState,
    aiAssistantState,
    isAuthenticated,
    currentUser,
    systemHealth,
    globalState
  } from '$lib/services/xstate-integration';
  
  // Component state
  let email = $state('prosecutor@example.gov');
  let password = $state('TestPassword123!');
  let aiMessage = $state('Analyze the evidence from case #2024-001');
  let isLoading = $state(false);
  let demoStep = $state<'auth' | 'dashboard' | 'ai' | 'complete'>('auth');
  
  // Reactive state from XState integration
  let auth = $state($authState);
  let session = $state($sessionState);
  let aiAssistant = $state($aiAssistantState);
  let authenticated = $state($isAuthenticated);
  let user = $state($currentUser);
  let health = $state($systemHealth);
  let global = $state($globalState);
  
  // Subscribe to state changes
  let unsubscribeAuth: () => void;
  let unsubscribeSession: () => void;
  let unsubscribeAI: () => void;
  let unsubscribeGlobal: () => void;
  let unsubscribeAuth2: () => void;
  let unsubscribeUser: () => void;
  let unsubscribeHealth: () => void;
  
  onMount(() => {
    // Subscribe to all relevant stores
    unsubscribeAuth = authState.subscribe(value => auth = value);
    unsubscribeSession = sessionState.subscribe(value => session = value);
    unsubscribeAI = aiAssistantState.subscribe(value => aiAssistant = value);
    unsubscribeGlobal = globalState.subscribe(value => global = value);
    unsubscribeAuth2 = isAuthenticated.subscribe(value => authenticated = value);
    unsubscribeUser = currentUser.subscribe(value => user = value);
    unsubscribeHealth = systemHealth.subscribe(value => health = value);
    
    console.log('XState Auth Demo mounted, initial state:', { auth, session, aiAssistant });
  });
  
  onDestroy(() => {
    // Clean up subscriptions
    unsubscribeAuth?.();
    unsubscribeSession?.();
    unsubscribeAI?.();
    unsubscribeGlobal?.();
    unsubscribeAuth2?.();
    unsubscribeUser?.();
    unsubscribeHealth?.();
  });
  
  // Demo functions
  async function demonstrateLogin() {
    isLoading = true;
    
    try {
      // Use XState integration service for login
      xstateIntegration.login(email, password, {
        rememberMe: true
      });
      
      // Wait for authentication to complete
      setTimeout(() => {
        if (authenticated) {
          demoStep = 'dashboard';
        }
        isLoading = false;
      }, 2000);
      
    } catch (error) {
      console.error('Login demo failed:', error);
      isLoading = false;
    }
  }
  
  function demonstrateAI() {
    demoStep = 'ai';
    
    // Send AI message using XState integration
    xstateIntegration.sendAIMessage(aiMessage, true); // with Context7
    
    // Demonstrate Context7 analysis
    xstateIntegration.analyzeWithContext7('legal evidence analysis');
  }
  
  function demonstrateLogout() {
    xstateIntegration.logout();
    demoStep = 'auth';
  }
  
  function demonstrateSessionActivity() {
    xstateIntegration.recordActivity('/evidence/analysis', 'analyze_document');
  }
  
  function demonstrateUpload() {
    // Create a mock file for demo
    const mockFile = new File(['Mock legal document content'], 'evidence.pdf', {
      type: 'application/pdf'
    });
    
    xstateIntegration.uploadDocument(mockFile, {
      type: 'evidence',
      caseId: 'case_2024_001',
      description: 'Key evidence document'
    });
  }
  
  // Get status color based on health
  function getHealthColor(isHealthy: boolean): string {
    return isHealthy ? 'text-green-500' : 'text-red-500';
  }
  
  function getOverallHealthVariant(overall: string): "default" | "secondary" | "destructive" | "outline" {
    switch (overall) {
      case 'healthy': return 'default';
      case 'degraded': return 'secondary';
      case 'critical': return 'destructive';
      default: return 'outline';
    }
  }
</script>

<div class="w-full max-w-4xl mx-auto space-y-6 p-6">
  <Card.Root>
    <Card.Header>
      <div class="flex items-center gap-3">
        <Shield class="h-8 w-8 text-primary" />
        <div>
          <Card.Title class="text-2xl">XState Integration Demo</Card.Title>
          <Card.Description>
            Complete demonstration of XState machines with Svelte components, 
            GPU orchestration, and Context7 documentation
          </Card.Description>
        </div>
      </div>
    </Card.Header>
    
    <Card.Content class="space-y-6">
      
      <!-- System Health Monitor -->
      <div class="bg-slate-50 p-4 rounded-lg">
        <div class="flex items-center justify-between mb-4">
          <h3 class="font-semibold flex items-center gap-2">
            <Cpu class="h-5 w-5" />
            System Health Monitor
          </h3>
          <Badge variant={getOverallHealthVariant(health.overall)}>
            {health.overall}
          </Badge>
        </div>
        
        <div class="grid grid-cols-3 gap-4 text-sm">
          <div class="flex items-center gap-2">
            <CheckCircle class="h-4 w-4 {getHealthColor(health.auth)}" />
            Authentication: {health.auth ? 'Healthy' : 'Issues'}
          </div>
          <div class="flex items-center gap-2">
            <CheckCircle class="h-4 w-4 {getHealthColor(health.ai)}" />
            AI Services: {health.ai ? 'Healthy' : 'Issues'}
          </div>
          <div class="flex items-center gap-2">
            <CheckCircle class="h-4 w-4 {getHealthColor(health.services)}" />
            Backend: {health.services ? 'Healthy' : 'Issues'}
          </div>
        </div>
      </div>
      
      <!-- Current State Display -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="space-y-3">
          <h4 class="font-medium">Authentication State</h4>
          <div class="bg-gray-100 p-3 rounded text-sm">
            <div>Status: <Badge variant={authenticated ? 'default' : 'secondary'}>
              {authenticated ? 'Authenticated' : 'Not Authenticated'}
            </Badge></div>
            <div class="mt-1">Loading: {auth.isLoading ? 'Yes' : 'No'}</div>
            {auth.error && (
              <div class="mt-1 text-red-600">Error: {auth.error}</div>
            )}
            {user && (
              <div class="mt-1">User: {user.firstName} {user.lastName} ({user.role})</div>
            )}
          </div>
        </div>
        
        <div class="space-y-3">
          <h4 class="font-medium">AI Assistant State</h4>
          <div class="bg-gray-100 p-3 rounded text-sm">
            <div>Model: {aiAssistant.model || 'None'}</div>
            <div>Processing: {aiAssistant.isProcessing ? 'Yes' : 'No'}</div>
            <div>Queries: {aiAssistant.usage?.totalQueries || 0}</div>
            <div>Avg Response: {Math.round(aiAssistant.usage?.averageResponseTime || 0)}ms</div>
          </div>
        </div>
      </div>
      
      <!-- Demo Steps -->
      {#if demoStep === 'auth'}
        <div class="space-y-4">
          <h3 class="text-lg font-semibold">Step 1: Authentication with XState</h3>
          
          <div class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label for="demo-email">Email</Label>
                <Input
                  id="demo-email"
                  type="email"
                  bind:value={email}
                  placeholder="prosecutor@example.gov"
                  disabled={isLoading}
                />
              </div>
              <div>
                <Label for="demo-password">Password</Label>
                <Input
                  id="demo-password"
                  type="password"
                  bind:value={password}
                  placeholder="Password"
                  disabled={isLoading}
                />
              </div>
            </div>
            
            <Button 
              onclick={demonstrateLogin}
              disabled={isLoading || authenticated}
              class="w-full"
            >
              {#if isLoading}
                <Loader2 class="h-4 w-4 mr-2 animate-spin" />
                Authenticating...
              {:else if authenticated}
                <CheckCircle class="h-4 w-4 mr-2" />
                Already Authenticated
              {:else}
                <Shield class="h-4 w-4 mr-2" />
                Demonstrate XState Login
              {/if}
            </Button>
          </div>
        </div>
      {/if}
      
      {#if demoStep === 'dashboard' && authenticated}
        <div class="space-y-4">
          <h3 class="text-lg font-semibold">Step 2: Dashboard Integration</h3>
          
          <Alert>
            <User class="h-4 w-4" />
            <AlertDescription>
              Welcome {user?.firstName}! You're now authenticated and your session is managed by XState.
              Role: {user?.role} | Department: {user?.department}
            </AlertDescription>
          </Alert>
          
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Button onclick={demonstrateAI} variant="outline">
              <Brain class="h-4 w-4 mr-2" />
              Test AI Assistant
            </Button>
            
            <Button onclick={demonstrateUpload} variant="outline">
              <Zap class="h-4 w-4 mr-2" />
              Demo File Upload
            </Button>
            
            <Button onclick={demonstrateSessionActivity} variant="outline">
              <Settings class="h-4 w-4 mr-2" />
              Record Activity
            </Button>
          </div>
          
          <Button onclick={demonstrateLogout} variant="destructive" class="w-full">
            <LogOut class="h-4 w-4 mr-2" />
            Demonstrate Logout
          </Button>
        </div>
      {/if}
      
      {#if demoStep === 'ai'}
        <div class="space-y-4">
          <h3 class="text-lg font-semibold">Step 3: AI Assistant with Context7</h3>
          
          <div class="space-y-4">
            <div>
              <Label for="ai-message">Message to AI Assistant</Label>
              <Input
                id="ai-message"
                bind:value={aiMessage}
                placeholder="Ask the AI assistant something..."
              />
            </div>
            
            {#if aiAssistant.isProcessing}
              <Alert>
                <Loader2 class="h-4 w-4 animate-spin" />
                <AlertDescription>
                  AI is processing your request with Context7 enhancement...
                </AlertDescription>
              </Alert>
            {/if}
            
            {#if aiAssistant.response}
              <div class="bg-blue-50 p-4 rounded-lg">
                <h4 class="font-medium mb-2">AI Response:</h4>
                <p class="text-sm">{aiAssistant.response}</p>
              </div>
            {/if}
            
            {#if aiAssistant.context7Analysis}
              <div class="bg-green-50 p-4 rounded-lg">
                <h4 class="font-medium mb-2">Context7 Analysis:</h4>
                <div class="space-y-2 text-sm">
                  <div>Confidence: {Math.round((aiAssistant.context7Analysis.confidence || 0) * 100)}%</div>
                  <div>Suggestions: {aiAssistant.context7Analysis.suggestions?.length || 0}</div>
                  <div>Code Examples: {aiAssistant.context7Analysis.codeExamples?.length || 0}</div>
                </div>
              </div>
            {/if}
            
            <div class="flex gap-2">
              <Button onclick={() => demoStep = 'dashboard'} variant="outline">
                Back to Dashboard
              </Button>
              <Button onclick={demonstrateLogout} variant="destructive">
                <LogOut class="h-4 w-4 mr-2" />
                Complete Demo
              </Button>
            </div>
          </div>
        </div>
      {/if}
      
      <!-- Session Information -->
      {#if authenticated && session}
        <div class="bg-slate-50 p-4 rounded-lg">
          <h4 class="font-medium mb-3">Session Information</h4>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div class="font-medium">Security Level</div>
              <Badge variant="outline">{session.securityLevel || 'standard'}</Badge>
            </div>
            <div>
              <div class="font-medium">Permissions</div>
              <div>{session.permissions?.length || 0} granted</div>
            </div>
            <div>
              <div class="font-medium">Last Activity</div>
              <div>{session.lastActivity ? new Date(session.lastActivity).toLocaleTimeString() : 'N/A'}</div>
            </div>
            <div>
              <div class="font-medium">Session Health</div>
              <Badge variant={session.sessionHealth?.isValid ? 'default' : 'destructive'}>
                {session.sessionHealth?.isValid ? 'Valid' : 'Invalid'}
              </Badge>
            </div>
          </div>
        </div>
      {/if}
      
      <!-- Notifications -->
      {#if global.ui.notifications && global.ui.notifications.length > 0}
        <div class="space-y-2">
          <h4 class="font-medium">Recent Notifications</h4>
          {#each global.ui.notifications.slice(-3) as notification}
            <Alert variant={notification.type === 'error' ? 'destructive' : 'default'}>
              {#if notification.type === 'success'}
                <CheckCircle class="h-4 w-4" />
              {:else if notification.type === 'error'}
                <AlertCircle class="h-4 w-4" />
              {:else}
                <Shield class="h-4 w-4" />
              {/if}
              <AlertDescription>
                <strong>{notification.title}:</strong> {notification.message}
                <div class="text-xs text-muted-foreground mt-1">
                  {notification.timestamp.toLocaleTimeString()}
                </div>
              </AlertDescription>
            </Alert>
          {/each}
        </div>
      {/if}
      
    </Card.Content>
  </Card.Root>
</div>