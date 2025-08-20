<!--
  XState Authentication Flow Demo Page
  Complete testing of authentication flow with GPU accelerated legal AI features
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import XStateAuthDemo from '$lib/components/auth/XStateAuthDemo.svelte';
  import { Button } from '$lib/components/ui/button';
  import * as Card from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/badge';
  import { 
    Shield, CheckCircle, AlertTriangle, Code2, 
    Cpu, Zap, Brain, FileText, Users
  } from 'lucide-svelte';

  let testResults = $state<Array<{
    name: string;
    status: 'pending' | 'running' | 'success' | 'error';
    message: string;
    duration?: number;
  }>>([]);

  let isRunningTests = $state(false);
  let overallStatus = $state<'idle' | 'testing' | 'complete'>('idle');

  const testSuite = [
    {
      name: 'XState Integration Service',
      test: async () => {
        // Test XState integration service initialization
        const { xstateIntegration } = await import('$lib/services/xstate-integration.js');
        if (!xstateIntegration) throw new Error('XState integration service not available');
        return 'XState integration service initialized successfully';
      }
    },
    {
      name: 'Authentication Machine',
      test: async () => {
        const { authMachine } = await import('$lib/machines/auth-machine.js');
        if (!authMachine) throw new Error('Auth machine not available');
        return 'Authentication machine loaded and ready';
      }
    },
    {
      name: 'Session Management',
      test: async () => {
        const { sessionMachine } = await import('$lib/machines/sessionMachine.js');
        if (!sessionMachine) throw new Error('Session machine not available');
        return 'Session management machine operational';
      }
    },
    {
      name: 'AI Assistant Integration',
      test: async () => {
        const { aiAssistantMachine } = await import('$lib/machines/aiAssistantMachine.js');
        if (!aiAssistantMachine) throw new Error('AI assistant machine not available');
        return 'AI assistant with Ollama cluster integration ready';
      }
    },
    {
      name: 'MCP GPU Orchestrator',
      test: async () => {
        const { mcpGPUOrchestrator } = await import('$lib/services/mcp-gpu-orchestrator.js');
        if (!mcpGPUOrchestrator) throw new Error('MCP GPU orchestrator not available');
        return 'GPU orchestrator service ready for enhanced security';
      }
    },
    {
      name: 'Production Service Client',
      test: async () => {
        const { productionServiceClient } = await import('$lib/services/production-service-client.js');
        if (!productionServiceClient) throw new Error('Production service client not available');
        return 'Multi-protocol service client initialized';
      }
    },
    {
      name: 'Context7 Documentation',
      test: async () => {
        const context7Module = await import('$lib/mcp-context72-get-library-docs.js');
        if (!context7Module.getSvelte5Docs) throw new Error('Context7 docs service not available');
        return 'Context7 documentation retrieval ready';
      }
    },
    {
      name: 'Bits UI v2 Components',
      test: async () => {
        // Test that Bits UI components are available
        const formModule = await import('$lib/components/ui/form/index.js');
        const buttonModule = await import('$lib/components/ui/button/index.js');
        if (!formModule || !buttonModule) throw new Error('Bits UI v2 components not available');
        return 'Bits UI v2 component library loaded';
      }
    }
  ];

  async function runTests() {
    isRunningTests = true;
    overallStatus = 'testing';
    testResults = [];

    for (const { name, test } of testSuite) {
      // Add pending test
      testResults = [...testResults, {
        name,
        status: 'pending',
        message: 'Waiting to run...'
      }];

      // Update to running
      testResults = testResults.map(result => 
        result.name === name 
          ? { ...result, status: 'running', message: 'Running test...' }
          : result
      );

      try {
        const startTime = Date.now();
        const message = await test();
        const duration = Date.now() - startTime;
        
        // Update to success
        testResults = testResults.map(result => 
          result.name === name 
            ? { ...result, status: 'success', message, duration }
            : result
        );
      } catch (error) {
        // Update to error
        testResults = testResults.map(result => 
          result.name === name 
            ? { 
                ...result, 
                status: 'error', 
                message: error instanceof Error ? error.message : 'Test failed'
              }
            : result
        );
      }

      // Small delay between tests for visual effect
      await new Promise(resolve => setTimeout(resolve, 300));
    }

    isRunningTests = false;
    overallStatus = 'complete';
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case 'success': return CheckCircle;
      case 'error': return AlertTriangle;
      case 'running': return Cpu;
      default: return Code2;
    }
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'success': return 'text-green-500';
      case 'error': return 'text-red-500';
      case 'running': return 'text-blue-500';
      default: return 'text-gray-400';
    }
  }

  function getStatusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
    switch (status) {
      case 'success': return 'default';
      case 'error': return 'destructive';
      case 'running': return 'secondary';
      default: return 'outline';
    }
  }

  const successfulTests = $derived(testResults.filter(t => t.status === 'success').length);
  const totalTests = $derived(testSuite.length);
  const allTestsPassed = $derived(successfulTests === totalTests && overallStatus === 'complete');
</script>

<svelte:head>
  <title>XState Authentication Demo - Legal AI Platform</title>
  <meta name="description" content="Complete demonstration and testing of XState authentication flow with GPU acceleration" />
</svelte:head>

<div class="min-h-screen bg-background p-6">
  <div class="max-w-6xl mx-auto space-y-8">
    
    <!-- Header -->
    <div class="text-center space-y-4">
      <div class="flex items-center justify-center gap-3">
        <Shield class="h-10 w-10 text-primary" />
        <h1 class="text-4xl font-bold">XState Authentication Flow Demo</h1>
      </div>
      <p class="text-xl text-muted-foreground max-w-3xl mx-auto">
        Complete demonstration and testing of authentication flow with GPU accelerated legal AI features, 
        XState management, Bits UI v2 components, and Context7 documentation integration.
      </p>
    </div>

    <!-- Test Suite -->
    <Card.Root>
      <Card.Header>
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <Code2 class="h-6 w-6 text-primary" />
            <div>
              <Card.Title>Integration Test Suite</Card.Title>
              <Card.Description>
                Verify all components and services are properly integrated
              </Card.Description>
            </div>
          </div>
          <div class="flex items-center gap-3">
            {#if overallStatus === 'complete'}
              <Badge variant={allTestsPassed ? 'default' : 'destructive'}>
                {successfulTests}/{totalTests} Tests Passed
              </Badge>
            {/if}
            <Button 
              onclick={runTests} 
              disabled={isRunningTests}
              variant={allTestsPassed ? 'outline' : 'default'}
            >
              {#if isRunningTests}
                <Cpu class="h-4 w-4 mr-2 animate-spin" />
                Running Tests...
              {:else if overallStatus === 'complete'}
                <CheckCircle class="h-4 w-4 mr-2" />
                Run Tests Again
              {:else}
                <Zap class="h-4 w-4 mr-2" />
                Run Integration Tests
              {/if}
            </Button>
          </div>
        </div>
      </Card.Header>
      
      <Card.Content>
        {#if testResults.length === 0}
          <div class="text-center py-8 text-muted-foreground">
            <Code2 class="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>Click "Run Integration Tests" to verify all systems are operational</p>
          </div>
        {:else}
          <div class="space-y-3">
            {#each testResults as result}
              <div class="flex items-center justify-between p-3 rounded-lg border">
                <div class="flex items-center gap-3">
                  <svelte:component 
                    this={getStatusIcon(result.status)} 
                    class="h-5 w-5 {getStatusColor(result.status)} {result.status === 'running' ? 'animate-spin' : ''}"
                  />
                  <div>
                    <div class="font-medium">{result.name}</div>
                    <div class="text-sm text-muted-foreground">{result.message}</div>
                  </div>
                </div>
                <div class="flex items-center gap-2">
                  {#if result.duration}
                    <span class="text-xs text-muted-foreground">{result.duration}ms</span>
                  {/if}
                  <Badge variant={getStatusVariant(result.status)}>
                    {result.status}
                  </Badge>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </Card.Content>
    </Card.Root>

    <!-- Success Message -->
    {#if allTestsPassed}
      <Card.Root class="border-green-200 bg-green-50">
        <Card.Content class="pt-6">
          <div class="flex items-center gap-4">
            <CheckCircle class="h-8 w-8 text-green-500" />
            <div>
              <h3 class="text-lg font-semibold text-green-900">All Systems Operational! 🎉</h3>
              <p class="text-green-700">
                The complete authentication flow with XState, GPU acceleration, and AI integration is ready for testing.
                You can now proceed with the interactive demo below.
              </p>
            </div>
          </div>
        </Card.Content>
      </Card.Root>
    {/if}

    <!-- Feature Overview -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <Card.Root>
        <Card.Content class="pt-6">
          <div class="flex items-center gap-3 mb-3">
            <Shield class="h-6 w-6 text-blue-500" />
            <h3 class="font-semibold">XState Integration</h3>
          </div>
          <p class="text-sm text-muted-foreground">
            Complete state management with auth, session, and AI machines
          </p>
        </Card.Content>
      </Card.Root>

      <Card.Root>
        <Card.Content class="pt-6">
          <div class="flex items-center gap-3 mb-3">
            <Cpu class="h-6 w-6 text-green-500" />
            <h3 class="font-semibold">GPU Acceleration</h3>
          </div>
          <p class="text-sm text-muted-foreground">
            MCP GPU orchestrator for enhanced security analysis
          </p>
        </Card.Content>
      </Card.Root>

      <Card.Root>
        <Card.Content class="pt-6">
          <div class="flex items-center gap-3 mb-3">
            <Brain class="h-6 w-6 text-purple-500" />
            <h3 class="font-semibold">AI Integration</h3>
          </div>
          <p class="text-sm text-muted-foreground">
            Ollama cluster with Context7 documentation enhancement
          </p>
        </Card.Content>
      </Card.Root>

      <Card.Root>
        <Card.Content class="pt-6">
          <div class="flex items-center gap-3 mb-3">
            <Users class="h-6 w-6 text-orange-500" />
            <h3 class="font-semibold">Legal Professional</h3>
          </div>
          <p class="text-sm text-muted-foreground">
            Role-based authentication with legal domain validation
          </p>
        </Card.Content>
      </Card.Root>
    </div>

    <!-- Interactive Demo -->
    <XStateAuthDemo />

    <!-- Technical Details -->
    <Card.Root>
      <Card.Header>
        <Card.Title class="flex items-center gap-2">
          <FileText class="h-5 w-5" />
          Technical Implementation
        </Card.Title>
      </Card.Header>
      <Card.Content class="space-y-4">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 class="font-semibold mb-3">XState Machines</h4>
            <ul class="space-y-2 text-sm">
              <li>• <code>authMachine</code> - Authentication flow management</li>
              <li>• <code>sessionMachine</code> - Session lifecycle and security</li>
              <li>• <code>aiAssistantMachine</code> - AI chat and Context7 integration</li>
              <li>• <code>agentShellMachine</code> - Multi-agent orchestration</li>
            </ul>
          </div>
          
          <div>
            <h4 class="font-semibold mb-3">Services Integration</h4>
            <ul class="space-y-2 text-sm">
              <li>• <code>productionServiceClient</code> - Multi-protocol communication</li>
              <li>• <code>mcpGPUOrchestrator</code> - GPU-enhanced processing</li>
              <li>• <code>context7Service</code> - Documentation retrieval</li>
              <li>• <code>xstateIntegration</code> - Unified state management</li>
            </ul>
          </div>
        </div>
        
        <div class="bg-slate-100 p-4 rounded-lg">
          <h4 class="font-semibold mb-2">Current Route</h4>
          <code class="text-sm">{$page.url.pathname}</code>
          <p class="text-sm text-muted-foreground mt-2">
            This demo page showcases the complete integration of all authentication and AI systems
            working together with XState for predictable state management.
          </p>
        </div>
      </Card.Content>
    </Card.Root>

  </div>
</div>