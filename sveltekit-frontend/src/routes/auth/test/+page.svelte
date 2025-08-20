<!--
  Authentication Flow Test Page - Legal AI Platform
  Tests GPU accelerated authentication, XState integration, and Context7 documentation
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { authStore } from '$lib/stores/auth-store';
  import { mcpGPUOrchestrator } from '$lib/services/mcp-gpu-orchestrator';
  import { getSvelte5Docs, getBitsUIv2Docs, getXStateDocs } from '$lib/mcp-context72-get-library-docs';
  import LoginForm from '$lib/components/auth/LoginForm.svelte';
  import RegisterForm from '$lib/components/auth/RegisterForm.svelte';
  import * as Card from '$lib/components/ui/card';
  import * as Alert from '$lib/components/ui/alert';
  import * as Tabs from '$lib/components/ui/tabs';
  import { Button } from '$lib/components/ui/button';
  import { Badge } from '$lib/components/ui/badge';
  import { Separator } from '$lib/components/ui/separator';
  import { 
    Shield, Zap, Cpu, Database, Brain, 
    CheckCircle, AlertCircle, Clock, Users,
    Activity, Server, Eye, Code
  } from 'lucide-svelte';

  // Test state
  let testResults = $state<Record<string, any>>({});
  let testRunning = $state(false);
  let currentTab = $state('auth');
  
  // GPU cluster status
  let clusterStatus = $state<any>(null);
  let context7Docs = $state<any>(null);
  
  // Mock data for testing
  const testCredentials = {
    login: {
      email: 'prosecutor@legal-ai.com',
      password: 'SecurePass123!@#'
    },
    register: {
      email: 'investigator@police.gov',
      firstName: 'Jane',
      lastName: 'Smith',
      password: 'SecurePass123!@#',
      confirmPassword: 'SecurePass123!@#',
      role: 'investigator',
      department: 'Metropolitan Police Department',
      jurisdiction: 'Washington DC',
      badgeNumber: 'MPD-4567',
      enableTwoFactor: true,
      agreeToTerms: true,
      agreeToPrivacy: true
    }
  };

  onMount(async () => {
    await runInitialTests();
  });

  async function runInitialTests() {
    testRunning = true;
    testResults = {};
    
    try {
      // Test 1: GPU Cluster Status
      console.log('🧪 Testing GPU cluster status...');
      const clusterTest = await mcpGPUOrchestrator.getClusterStatus();
      testResults.cluster = {
        success: true,
        data: clusterTest,
        timestamp: new Date().toISOString()
      };
      clusterStatus = clusterTest;
      
      // Test 2: Context7 Documentation
      console.log('🧪 Testing Context7 documentation retrieval...');
      const docsTest = await Promise.allSettled([
        getSvelte5Docs('authentication|stores'),
        getBitsUIv2Docs('forms|components'),
        getXStateDocs('machines|actors')
      ]);
      
      testResults.context7 = {
        success: docsTest.every(result => result.status === 'fulfilled'),
        data: docsTest.map(result => result.status === 'fulfilled' ? result.value : null),
        timestamp: new Date().toISOString()
      };
      context7Docs = testResults.context7.data;
      
      // Test 3: Security Analysis
      console.log('🧪 Testing GPU security analysis...');
      const securityTest = await mcpGPUOrchestrator.dispatchGPUTask({
        id: `test_security_${Date.now()}`,
        type: 'security_analysis',
        priority: 'medium',
        data: {
          email: 'test@legal-ai.com',
          timestamp: new Date().toISOString(),
          userAgent: navigator.userAgent,
          fingerprint: btoa(JSON.stringify({ test: true }))
        },
        context: {
          action: 'test_analysis',
          enhancedSecurity: true
        },
        config: {
          useGPU: true,
          model: 'gemma3-legal',
          protocol: 'auto'
        }
      });
      
      testResults.security = {
        success: securityTest.success,
        data: securityTest,
        timestamp: new Date().toISOString()
      };
      
      // Test 4: Legal Professional Validation
      console.log('🧪 Testing legal professional validation...');
      const validationTest = await mcpGPUOrchestrator.dispatchGPUTask({
        id: `test_validation_${Date.now()}`,
        type: 'security_validation',
        priority: 'medium',
        data: {
          email: 'prosecutor@da.gov',
          firstName: 'John',
          lastName: 'Doe',
          role: 'prosecutor',
          department: 'District Attorney Office',
          jurisdiction: 'Los Angeles County',
          badgeNumber: 'DA-123'
        },
        context: {
          action: 'test_validation',
          legalProfessionalCheck: true
        },
        config: {
          useGPU: true,
          model: 'gemma3-legal',
          protocol: 'auto'
        }
      });
      
      testResults.validation = {
        success: validationTest.success,
        data: validationTest,
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      console.error('Test failed:', error);
      testResults.error = {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      };
    } finally {
      testRunning = false;
    }
  }

  function getTestStatusIcon(testKey: string) {
    const test = testResults[testKey];
    if (!test) return Clock;
    return test.success ? CheckCircle : AlertCircle;
  }

  function getTestStatusColor(testKey: string) {
    const test = testResults[testKey];
    if (!test) return 'text-gray-400';
    return test.success ? 'text-green-500' : 'text-red-500';
  }

  // Mock form data for components
  const mockFormData = {
    login: { email: '', password: '' },
    register: { 
      email: '', firstName: '', lastName: '', password: '', 
      confirmPassword: '', role: 'prosecutor', department: '', 
      jurisdiction: '', badgeNumber: '', enableTwoFactor: false,
      agreeToTerms: false, agreeToPrivacy: false
    }
  };

  function populateTestData(form: 'login' | 'register') {
    if (form === 'login') {
      // Fill login form with test data
      const emailInput = document.getElementById('email') as HTMLInputElement;
      const passwordInput = document.getElementById('password') as HTMLInputElement;
      
      if (emailInput) emailInput.value = testCredentials.login.email;
      if (passwordInput) passwordInput.value = testCredentials.login.password;
    } else {
      // Fill register form with test data
      Object.entries(testCredentials.register).forEach(([key, value]) => {
        const input = document.getElementById(key) as HTMLInputElement;
        if (input) {
          if (input.type === 'checkbox') {
            input.checked = value as boolean;
          } else {
            input.value = value as string;
          }
        }
      });
    }
  }
</script>

<svelte:head>
  <title>Authentication Test - Legal AI Platform</title>
</svelte:head>

<div class="container mx-auto p-6 space-y-6">
  <div class="text-center space-y-4">
    <div class="flex items-center justify-center gap-3">
      <Shield class="h-8 w-8 text-primary" />
      <h1 class="text-3xl font-bold">Legal AI Authentication Test Suite</h1>
    </div>
    <p class="text-muted-foreground max-w-2xl mx-auto">
      Comprehensive testing of GPU-accelerated authentication, XState integration, 
      Context7 documentation, and production service clients for the Legal AI Platform.
    </p>
  </div>

  <!-- Test Status Overview -->
  <Card.Root>
    <Card.Header>
      <Card.Title class="flex items-center gap-2">
        <Activity class="h-5 w-5" />
        System Test Status
      </Card.Title>
      <Card.Description>
        Real-time testing of core authentication and AI services
      </Card.Description>
    </Card.Header>
    <Card.Content>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <!-- GPU Cluster Status -->
        <div class="flex items-center gap-3 p-3 border rounded-lg">
          <svelte:component 
            this={getTestStatusIcon('cluster')} 
            class="h-5 w-5 {getTestStatusColor('cluster')}" 
          />
          <div>
            <div class="font-medium">GPU Cluster</div>
            <div class="text-sm text-muted-foreground">
              {clusterStatus?.available ? 'Online' : 'Offline'}
            </div>
          </div>
        </div>

        <!-- Context7 Documentation -->
        <div class="flex items-center gap-3 p-3 border rounded-lg">
          <svelte:component 
            this={getTestStatusIcon('context7')} 
            class="h-5 w-5 {getTestStatusColor('context7')}" 
          />
          <div>
            <div class="font-medium">Context7 Docs</div>
            <div class="text-sm text-muted-foreground">
              {testResults.context7?.success ? 'Available' : 'Loading...'}
            </div>
          </div>
        </div>

        <!-- Security Analysis -->
        <div class="flex items-center gap-3 p-3 border rounded-lg">
          <svelte:component 
            this={getTestStatusIcon('security')} 
            class="h-5 w-5 {getTestStatusColor('security')}" 
          />
          <div>
            <div class="font-medium">Security Analysis</div>
            <div class="text-sm text-muted-foreground">
              {testResults.security?.success ? 'Operational' : 'Testing...'}
            </div>
          </div>
        </div>

        <!-- Legal Validation -->
        <div class="flex items-center gap-3 p-3 border rounded-lg">
          <svelte:component 
            this={getTestStatusIcon('validation')} 
            class="h-5 w-5 {getTestStatusColor('validation')}" 
          />
          <div>
            <div class="font-medium">Legal Validation</div>
            <div class="text-sm text-muted-foreground">
              {testResults.validation?.success ? 'Active' : 'Testing...'}
            </div>
          </div>
        </div>
      </div>

      <div class="flex gap-2">
        <Button 
          onclick={runInitialTests} 
          disabled={testRunning}
          variant="outline"
        >
          {#if testRunning}
            <Clock class="mr-2 h-4 w-4 animate-spin" />
            Running Tests...
          {:else}
            <Activity class="mr-2 h-4 w-4" />
            Refresh Tests
          {/if}
        </Button>
      </div>
    </Card.Content>
  </Card.Root>

  <!-- Main Testing Interface -->
  <Tabs.Root bind:value={currentTab}>
    <Tabs.List class="grid w-full grid-cols-4">
      <Tabs.Trigger value="auth">Authentication</Tabs.Trigger>
      <Tabs.Trigger value="gpu">GPU Testing</Tabs.Trigger>
      <Tabs.Trigger value="context7">Context7 Docs</Tabs.Trigger>
      <Tabs.Trigger value="results">Test Results</Tabs.Trigger>
    </Tabs.List>

    <!-- Authentication Testing -->
    <Tabs.Content value="auth" class="space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Login Testing -->
        <Card.Root>
          <Card.Header>
            <Card.Title class="flex items-center gap-2">
              <Shield class="h-5 w-5" />
              Login Form Test
            </Card.Title>
            <Card.Description>
              Test login with GPU-enhanced security analysis
            </Card.Description>
          </Card.Header>
          <Card.Content class="space-y-4">
            <Button 
              onclick={() => populateTestData('login')}
              variant="outline"
              size="sm"
            >
              Fill Test Data
            </Button>
            
            <LoginForm 
              data={mockFormData.login}
              enableGPUAuth={true}
              showRegistration={false}
            />
          </Card.Content>
        </Card.Root>

        <!-- Registration Testing -->
        <Card.Root>
          <Card.Header>
            <Card.Title class="flex items-center gap-2">
              <Users class="h-5 w-5" />
              Registration Form Test
            </Card.Title>
            <Card.Description>
              Test registration with legal professional validation
            </Card.Description>
          </Card.Header>
          <Card.Content class="space-y-4">
            <Button 
              onclick={() => populateTestData('register')}
              variant="outline"
              size="sm"
            >
              Fill Test Data
            </Button>
            
            <RegisterForm 
              data={mockFormData.register}
              enableGPUValidation={true}
              showLogin={false}
            />
          </Card.Content>
        </Card.Root>
      </div>

      <!-- Auth Store Status -->
      <Card.Root>
        <Card.Header>
          <Card.Title class="flex items-center gap-2">
            <Database class="h-5 w-5" />
            XState Auth Store Status
          </Card.Title>
        </Card.Header>
        <Card.Content>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="space-y-2">
              <div class="text-sm font-medium">User Status</div>
              <Badge variant={authStore.isAuthenticated ? 'default' : 'secondary'}>
                {authStore.isAuthenticated ? 'Authenticated' : 'Not Authenticated'}
              </Badge>
            </div>
            <div class="space-y-2">
              <div class="text-sm font-medium">Machine State</div>
              <Badge variant="outline">{authStore.machineState}</Badge>
            </div>
            <div class="space-y-2">
              <div class="text-sm font-medium">Session Status</div>
              <Badge variant={authStore.session ? 'default' : 'secondary'}>
                {authStore.session ? 'Active' : 'No Session'}
              </Badge>
            </div>
          </div>
        </Card.Content>
      </Card.Root>
    </Tabs.Content>

    <!-- GPU Testing -->
    <Tabs.Content value="gpu" class="space-y-6">
      <Card.Root>
        <Card.Header>
          <Card.Title class="flex items-center gap-2">
            <Cpu class="h-5 w-5" />
            MCP GPU Orchestrator Status
          </Card.Title>
        </Card.Header>
        <Card.Content>
          {#if clusterStatus}
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div class="space-y-4">
                <h4 class="font-medium">Cluster Metrics</h4>
                <div class="space-y-2 text-sm">
                  <div class="flex justify-between">
                    <span>Available:</span>
                    <Badge variant={clusterStatus.available ? 'default' : 'destructive'}>
                      {clusterStatus.available ? 'Yes' : 'No'}
                    </Badge>
                  </div>
                  <div class="flex justify-between">
                    <span>Active Nodes:</span>
                    <span>{clusterStatus.activeNodes}/{clusterStatus.totalNodes}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>Queue Length:</span>
                    <span>{clusterStatus.queueLength}</span>
                  </div>
                </div>
              </div>
              
              <div class="space-y-4">
                <h4 class="font-medium">Ollama Status</h4>
                <div class="space-y-2 text-sm">
                  <div class="flex justify-between">
                    <span>Status:</span>
                    <Badge variant={clusterStatus.ollama.status === 'online' ? 'default' : 'destructive'}>
                      {clusterStatus.ollama.status}
                    </Badge>
                  </div>
                  <div class="flex justify-between">
                    <span>Models:</span>
                    <span>{clusterStatus.ollama.models.length}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>Loaded:</span>
                    <span>{clusterStatus.ollama.loadedModels.length}</span>
                  </div>
                </div>
              </div>
            </div>
          {:else}
            <div class="text-center py-8 text-muted-foreground">
              <Cpu class="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>Loading cluster status...</p>
            </div>
          {/if}
        </Card.Content>
      </Card.Root>
    </Tabs.Content>

    <!-- Context7 Documentation -->
    <Tabs.Content value="context7" class="space-y-6">
      <Card.Root>
        <Card.Header>
          <Card.Title class="flex items-center gap-2">
            <Code class="h-5 w-5" />
            Context7 Documentation Status
          </Card.Title>
        </Card.Header>
        <Card.Content>
          {#if context7Docs}
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
              {#each ['Svelte 5', 'Bits UI v2', 'XState'] as lib, index}
                <div class="border rounded-lg p-4">
                  <h4 class="font-medium mb-2">{lib}</h4>
                  {#if context7Docs[index]}
                    <div class="space-y-2 text-sm">
                      <div class="flex justify-between">
                        <span>Library:</span>
                        <span>{context7Docs[index].metadata.library}</span>
                      </div>
                      <div class="flex justify-between">
                        <span>Tokens:</span>
                        <span>{context7Docs[index].metadata.tokenCount}</span>
                      </div>
                      <Badge variant="default">Available</Badge>
                    </div>
                  {:else}
                    <Badge variant="destructive">Failed</Badge>
                  {/if}
                </div>
              {/each}
            </div>
          {:else}
            <div class="text-center py-8 text-muted-foreground">
              <Code class="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>Loading Context7 documentation...</p>
            </div>
          {/if}
        </Card.Content>
      </Card.Root>
    </Tabs.Content>

    <!-- Test Results -->
    <Tabs.Content value="results" class="space-y-6">
      <Card.Root>
        <Card.Header>
          <Card.Title class="flex items-center gap-2">
            <Eye class="h-5 w-5" />
            Detailed Test Results
          </Card.Title>
        </Card.Header>
        <Card.Content>
          <div class="space-y-4">
            {#each Object.entries(testResults) as [testKey, testData]}
              <div class="border rounded-lg p-4">
                <div class="flex items-center justify-between mb-2">
                  <h4 class="font-medium capitalize">{testKey} Test</h4>
                  <Badge variant={testData.success ? 'default' : 'destructive'}>
                    {testData.success ? 'Passed' : 'Failed'}
                  </Badge>
                </div>
                <div class="text-xs text-muted-foreground mb-2">
                  {testData.timestamp}
                </div>
                <pre class="text-xs bg-muted p-2 rounded overflow-auto max-h-32">
{JSON.stringify(testData.data, null, 2)}
                </pre>
              </div>
            {/each}
          </div>
        </Card.Content>
      </Card.Root>
    </Tabs.Content>
  </Tabs.Root>
</div>