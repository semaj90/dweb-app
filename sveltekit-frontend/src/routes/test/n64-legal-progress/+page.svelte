<script lang="ts">
  import { onMount } from 'svelte';
  import N64ProgressBar from '$lib/components/ui/gaming/n64/N64ProgressBar.svelte';
  import N64LoadingRing from '$lib/components/ui/gaming/n64/N64LoadingRing.svelte';
  import N64EvolutionLoader from '$lib/components/ui/gaming/n64/N64EvolutionLoader.svelte';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Badge } from '$lib/components/ui/badge';
  import { Brain, Zap, Cpu, Database } from 'lucide-svelte';

  let mockData = $state({
    aiConfidence: 0,
    tokensPerSecond: 0,
    cacheHitRate: 0,
    gpuUtilization: 0,
    vramUsage: 0,
    responseTime: 5000,
    gpuLayers: 0,
    isProcessing: false,
    evolutionStage: 'nes' as 'nes' | 'snes' | 'n64' | 'modern'
  });

  let animationInterval: NodeJS.Timeout | null = null;

  function startAnimation() {
    if (animationInterval) return;
    
    mockData.isProcessing = true;
    
    animationInterval = setInterval(() => {
      // Simulate AI confidence building up
      mockData.aiConfidence = Math.min(mockData.aiConfidence + Math.random() * 5, 95);
      
      // Simulate tokens per second ramping up
      mockData.tokensPerSecond = Math.min(mockData.tokensPerSecond + Math.random() * 10, 150);
      
      // Simulate cache hit rate improving
      mockData.cacheHitRate = Math.min(mockData.cacheHitRate + Math.random() * 3, 85);
      
      // Simulate GPU utilization
      mockData.gpuUtilization = Math.min(mockData.gpuUtilization + Math.random() * 8, 92);
      
      // Simulate VRAM usage
      mockData.vramUsage = Math.min(mockData.vramUsage + Math.random() * 0.2, 6.8);
      
      // Improve response time
      mockData.responseTime = Math.max(mockData.responseTime - Math.random() * 100, 250);
      
      // GPU layers loading
      mockData.gpuLayers = Math.min(mockData.gpuLayers + Math.random() * 2, 35);
      
      // Check if we should evolve
      if (mockData.aiConfidence > 80 && mockData.evolutionStage === 'nes') {
        mockData.evolutionStage = 'snes';
      } else if (mockData.tokensPerSecond > 100 && mockData.evolutionStage === 'snes') {
        mockData.evolutionStage = 'n64';
      } else if (mockData.gpuUtilization > 80 && mockData.evolutionStage === 'n64') {
        mockData.evolutionStage = 'modern';
      }
    }, 200);

    // Stop after 15 seconds
    setTimeout(() => {
      stopAnimation();
    }, 15000);
  }

  function stopAnimation() {
    if (animationInterval) {
      clearInterval(animationInterval);
      animationInterval = null;
    }
    mockData.isProcessing = false;
  }

  function resetData() {
    stopAnimation();
    mockData = {
      aiConfidence: 0,
      tokensPerSecond: 0,
      cacheHitRate: 0,
      gpuUtilization: 0,
      vramUsage: 0,
      responseTime: 5000,
      gpuLayers: 0,
      isProcessing: false,
      evolutionStage: 'nes'
    };
  }
</script>

<div class="container mx-auto p-6 space-y-6">
  <!-- Header -->
  <Card>
    <CardHeader>
      <CardTitle class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <Brain class="h-6 w-6" />
          N64 Progress Components - Legal AI Integration Test
        </div>
        <div class="flex gap-2">
          <Button on:click={startAnimation} disabled={mockData.isProcessing}>
            Start AI Processing
          </Button>
          <Button variant="outline" on:click={resetData}>
            Reset
          </Button>
        </div>
      </CardTitle>
    </CardHeader>
  </Card>

  <!-- Processing Status -->
  {#if mockData.isProcessing}
    <Card>
      <CardContent class="p-6">
        <div class="flex items-center gap-4">
          <N64LoadingRing 
            size="lg" 
            theme="classic" 
            speed="medium"
            showPercentage={false}
          />
          <div class="flex flex-col">
            <span class="text-lg font-medium">Legal AI Processing Active</span>
            <span class="text-muted-foreground">Evolution Stage: {mockData.evolutionStage.toUpperCase()}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Evolution Loader -->
  <Card>
    <CardHeader>
      <CardTitle class="flex items-center gap-2">
        <Zap class="h-5 w-5" />
        N64 Evolution Loader
      </CardTitle>
    </CardHeader>
    <CardContent>
      <N64EvolutionLoader 
        stage={mockData.evolutionStage}
        showStageText={true}
        size="lg"
        animateTransitions={true}
      />
    </CardContent>
  </Card>

  <!-- AI Performance Metrics -->
  <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
    <!-- AI Confidence -->
    <Card>
      <CardHeader>
        <CardTitle class="text-sm flex items-center gap-2">
          <Brain class="h-4 w-4" />
          AI Analysis Confidence
        </CardTitle>
      </CardHeader>
      <CardContent class="space-y-4">
        <div>
          <div class="flex justify-between text-sm mb-2">
            <span>Current Confidence</span>
            <span class="font-medium">{mockData.aiConfidence.toFixed(1)}%</span>
          </div>
          <N64ProgressBar 
            value={mockData.aiConfidence} 
            max={100}
            size="md"
            theme="gold"
            animated={mockData.isProcessing}
            showPercentage={true}
            sparkle={mockData.aiConfidence > 80}
            retro={true}
          />
        </div>
      </CardContent>
    </Card>

    <!-- Performance Metrics -->
    <Card>
      <CardHeader>
        <CardTitle class="text-sm flex items-center gap-2">
          <Cpu class="h-4 w-4" />
          Performance Metrics
        </CardTitle>
      </CardHeader>
      <CardContent class="space-y-4">
        <div>
          <div class="flex justify-between text-sm mb-2">
            <span>Tokens/sec</span>
            <span class="font-medium">{mockData.tokensPerSecond.toFixed(0)}</span>
          </div>
          <N64ProgressBar 
            value={mockData.tokensPerSecond} 
            max={150}
            size="sm"
            theme="green"
            animated={mockData.isProcessing}
            showPercentage={false}
            sparkle={mockData.tokensPerSecond > 120}
          />
        </div>
        
        <div>
          <div class="flex justify-between text-sm mb-2">
            <span>Cache Hit Rate</span>
            <span class="font-medium">{mockData.cacheHitRate.toFixed(1)}%</span>
          </div>
          <N64ProgressBar 
            value={mockData.cacheHitRate} 
            max={100}
            size="sm"
            theme="blue"
            animated={mockData.isProcessing}
            showPercentage={false}
          />
        </div>
      </CardContent>
    </Card>

    <!-- GPU Status -->
    <Card>
      <CardHeader>
        <CardTitle class="text-sm flex items-center gap-2">
          <Zap class="h-4 w-4" />
          RTX 3060 Ti Status
        </CardTitle>
      </CardHeader>
      <CardContent class="space-y-4">
        <div>
          <div class="flex justify-between text-sm mb-2">
            <span>GPU Layers</span>
            <span class="font-medium">{Math.round(mockData.gpuLayers)}/35</span>
          </div>
          <N64ProgressBar 
            value={mockData.gpuLayers} 
            max={35}
            size="sm"
            theme="purple"
            animated={mockData.isProcessing}
            showPercentage={false}
            sparkle={mockData.gpuLayers >= 35}
          />
        </div>
        
        <div>
          <div class="flex justify-between text-sm mb-2">
            <span>VRAM Usage</span>
            <span class="font-medium">{mockData.vramUsage.toFixed(1)}GB / 8.0GB</span>
          </div>
          <N64ProgressBar 
            value={mockData.vramUsage} 
            max={8}
            size="sm"
            theme={mockData.vramUsage < 6 ? 'green' : mockData.vramUsage < 7 ? 'gold' : 'red'}
            animated={mockData.isProcessing}
            showPercentage={false}
          />
        </div>
        
        <div>
          <div class="flex justify-between text-sm mb-2">
            <span>GPU Utilization</span>
            <span class="font-medium">{mockData.gpuUtilization.toFixed(0)}%</span>
          </div>
          <N64ProgressBar 
            value={mockData.gpuUtilization} 
            max={100}
            size="sm"
            theme="classic"
            animated={mockData.isProcessing}
            showPercentage={false}
            sparkle={mockData.gpuUtilization > 90}
          />
        </div>
      </CardContent>
    </Card>

    <!-- Response Time -->
    <Card>
      <CardHeader>
        <CardTitle class="text-sm flex items-center gap-2">
          <Database class="h-4 w-4" />
          System Response
        </CardTitle>
      </CardHeader>
      <CardContent class="space-y-4">
        <div>
          <div class="flex justify-between text-sm mb-2">
            <span>Response Time</span>
            <span class="font-medium">{Math.round(mockData.responseTime)}ms</span>
          </div>
          <N64ProgressBar 
            value={Math.max(0, 5000 - mockData.responseTime)} 
            max={5000}
            size="md"
            theme={mockData.responseTime < 1000 ? 'green' : 
                   mockData.responseTime < 3000 ? 'gold' : 'red'}
            animated={mockData.isProcessing}
            showPercentage={false}
            sparkle={mockData.responseTime < 500}
            retro={true}
          />
        </div>
      </CardContent>
    </Card>
  </div>

  <!-- Color Theme Showcase -->
  <Card>
    <CardHeader>
      <CardTitle>N64 Controller Color Themes</CardTitle>
    </CardHeader>
    <CardContent>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {#each ['classic', 'gold', 'red', 'blue', 'green', 'purple'] as theme}
          <div class="space-y-2">
            <Badge variant="outline" class="w-full justify-center">
              {theme.charAt(0).toUpperCase() + theme.slice(1)}
            </Badge>
            <N64ProgressBar 
              value={75} 
              max={100}
              size="sm"
              theme={theme}
              animated={true}
              showPercentage={true}
              sparkle={theme === 'gold'}
            />
          </div>
        {/each}
      </div>
    </CardContent>
  </Card>
</div>

<style>
  :global(.container) {
    max-width: 1200px;
  }
</style>