<script lang="ts">
  import ProductionAIChat from '$lib/components/ProductionAIChat.svelte';
  import { onMount } from 'svelte';

  let orchestratorStatus = $state('checking...');
  let systemInfo = $state<any>(null);

  onMount(() => {
    checkOrchestrator();
    
    // Periodic health checks
    const healthInterval = setInterval(checkOrchestrator, 10000);
    
    return () => {
      clearInterval(healthInterval);
    };
  });

  async function checkOrchestrator() {
    try {
      const response = await fetch('http://localhost:40000/health');
      if (response.ok) {
        systemInfo = await response.json();
        orchestratorStatus = systemInfo.status;
      } else {
        orchestratorStatus = 'not responding';
      }
    } catch (error) {
      orchestratorStatus = 'not available';
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': return '#48bb78';
      case 'degraded': return '#ed8936';
      case 'checking...': return '#4299e1';
      default: return '#f56565';
    }
  }
</script>

<svelte:head>
  <title>Production Legal AI Assistant</title>
  <meta name="description" content="Production-ready GPU-accelerated legal AI with Context7 MCP integration" />
</svelte:head>

<div class="production-page">
  <div class="page-header">
    <h1>⚡ Production Legal AI Assistant</h1>
    <p class="subtitle">
      Enterprise-grade GPU-accelerated legal AI with multi-core Context7 MCP orchestration,
      Go microservices, and real-time WebSocket communication.
    </p>
  </div>

  <!-- System Status Dashboard -->
  <div class="status-dashboard">
    <h2>🔧 System Architecture Status</h2>
    
    <div class="status-grid">
      <div class="status-card" style="border-left-color: {getStatusColor(orchestratorStatus)}">
        <div class="status-header">
          <span class="status-icon">🎯</span>
          <h3>Production Orchestrator</h3>
        </div>
        <div class="status-value" style="color: {getStatusColor(orchestratorStatus)}">
          {orchestratorStatus}
        </div>
        {#if systemInfo}
          <div class="status-details">
            <span>Worker: {systemInfo.workerId}</span>
            <span>Connections: {systemInfo.connections}</span>
            <span>Multi-core: {systemInfo.multiCore.enabled ? '✅' : '❌'}</span>
          </div>
        {/if}
      </div>

      {#if systemInfo}
        <div class="status-card" style="border-left-color: #48bb78">
          <div class="status-header">
            <span class="status-icon">🧠</span>
            <h3>AI Services</h3>
          </div>
          <div class="status-value" style="color: #48bb78">
            {systemInfo.services?.ollama ? 'Connected' : 'Disconnected'}
          </div>
          <div class="status-details">
            <span>Ollama: {systemInfo.services.ollama}</span>
            <span>Go Service: {systemInfo.services.goService}</span>
          </div>
        </div>

        <div class="status-card" style="border-left-color: #4299e1">
          <div class="status-header">
            <span class="status-icon">⚡</span>
            <h3>Worker Pool</h3>
          </div>
          <div class="status-value" style="color: #4299e1">
            {systemInfo.metrics.workerPool.busyWorkers}/{systemInfo.metrics.workerPool.totalWorkers} Active
          </div>
          <div class="status-details">
            <span>Queue: {systemInfo.metrics.workerPool.queueLength}</span>
            <span>Jobs: {systemInfo.metrics.workerPool.activeJobs}</span>
          </div>
        </div>

        <div class="status-card" style="border-left-color: #ed8936">
          <div class="status-header">
            <span class="status-icon">📊</span>
            <h3>Performance</h3>
          </div>
          <div class="status-value" style="color: #ed8936">
            {Math.round(systemInfo.metrics.averageResponseTime)}ms avg
          </div>
          <div class="status-details">
            <span>Processed: {systemInfo.metrics.requestsProcessed}</span>
            <span>Error Rate: {(systemInfo.metrics.errorRate * 100).toFixed(2)}%</span>
          </div>
        </div>
      {/if}
    </div>
  </div>

  <!-- Architecture Overview -->
  <div class="architecture-section">
    <h2>🏗️ Production Architecture</h2>
    
    <div class="architecture-grid">
      <div class="arch-component primary">
        <h3>🎯 Context7 MCP Orchestrator</h3>
        <p>Multi-core Node.js cluster with worker thread pools for parallel processing</p>
        <ul>
          <li>Port 40000 - REST API & WebSocket</li>
          <li>8 worker processes maximum</li>
          <li>4 worker threads per process</li>
          <li>Real-time health monitoring</li>
        </ul>
      </div>

      <div class="arch-component secondary">
        <h3>🦙 Go Llama Integration</h3>
        <p>High-performance Go microservices for AI workload distribution</p>
        <ul>
          <li>Ports 4100+ - Worker services</li>
          <li>Job queue management</li>
          <li>WebSocket broadcasting</li>
          <li>Context7 integration</li>
        </ul>
      </div>

      <div class="arch-component accent">
        <h3>🖥️ SvelteKit 2 Frontend</h3>
        <p>Modern reactive UI with Svelte 5 runes and real-time updates</p>
        <ul>
          <li>Port 5174 - Development server</li>
          <li>WebSocket client integration</li>
          <li>Performance monitoring</li>
          <li>Professional UI components</li>
        </ul>
      </div>

      <div class="arch-component info">
        <h3>🤖 AI Model Stack</h3>
        <p>Production AI models with GPU acceleration and legal specialization</p>
        <ul>
          <li>Ollama (Port 11434)</li>
          <li>Gemma3-Legal (11.8B parameters)</li>
          <li>NoMic-Embed-Text (384d vectors)</li>
          <li>PostgreSQL + pgvector storage</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- Interactive Chat Interface -->
  <div class="chat-section">
    <h2>💬 Interactive Legal AI Chat</h2>
    
    {#if orchestratorStatus === 'healthy'}
      <ProductionAIChat />
    {:else}
      <div class="error-state">
        <h3>⚠️ Production Orchestrator Not Available</h3>
        <p>Status: <strong style="color: {getStatusColor(orchestratorStatus)}">{orchestratorStatus}</strong></p>
        
        <div class="startup-instructions">
          <h4>🚀 Quick Start Instructions:</h4>
          <div class="code-block">
            <code>
              # Start the production orchestrator<br>
              node production-ai-orchestrator.js<br><br>
              # Or with environment variables<br>
              MCP_DEBUG=true MCP_MULTICORE=true node production-ai-orchestrator.js
            </code>
          </div>
          
          <p>The orchestrator will start on <strong>port 40000</strong> with multi-core processing enabled.</p>
        </div>
      </div>
    {/if}
  </div>

  <!-- Technical Specifications -->
  <div class="specs-section">
    <h2>🔧 Technical Specifications</h2>
    
    <div class="specs-grid">
      <div class="spec-category">
        <h3>⚡ Performance</h3>
        <ul>
          <li><strong>Multi-core:</strong> Up to 8 worker processes</li>
          <li><strong>Concurrency:</strong> 4 worker threads per process</li>
          <li><strong>Queue:</strong> 100 concurrent jobs per worker</li>
          <li><strong>Timeout:</strong> 30-second request timeout</li>
          <li><strong>WebSocket:</strong> Real-time bidirectional communication</li>
        </ul>
      </div>

      <div class="spec-category">
        <h3>🏗️ Architecture</h3>
        <ul>
          <li><strong>Framework:</strong> Node.js cluster + Express</li>
          <li><strong>State Management:</strong> Worker threads with Redis</li>
          <li><strong>API:</strong> REST + WebSocket + Go gRPC</li>
          <li><strong>Frontend:</strong> SvelteKit 2 + Svelte 5</li>
          <li><strong>Database:</strong> PostgreSQL + pgvector</li>
        </ul>
      </div>

      <div class="spec-category">
        <h3>🤖 AI Integration</h3>
        <ul>
          <li><strong>Models:</strong> Gemma3-Legal, Llama3.2, NoMic-Embed</li>
          <li><strong>Processing:</strong> Parallel embedding generation</li>
          <li><strong>RAG:</strong> Vector similarity search</li>
          <li><strong>Context:</strong> Legal document retrieval</li>
          <li><strong>Streaming:</strong> Real-time response generation</li>
        </ul>
      </div>

      <div class="spec-category">
        <h3>🔒 Production Features</h3>
        <ul>
          <li><strong>Security:</strong> CORS, Helmet, rate limiting</li>
          <li><strong>Monitoring:</strong> Health checks, metrics tracking</li>
          <li><strong>Reliability:</strong> Graceful shutdown, error recovery</li>
          <li><strong>Scalability:</strong> Horizontal worker scaling</li>
          <li><strong>Observability:</strong> Real-time performance data</li>
        </ul>
      </div>
    </div>
  </div>
</div>

<style>
  .production-page {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
    background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%);
    min-height: 100vh;
    color: white;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  .page-header {
    text-align: center;
    margin-bottom: 3rem;
  }

  .page-header h1 {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4299e1 0%, #48bb78 50%, #ed8936 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
  }

  .subtitle {
    font-size: 1.25rem;
    color: #a0aec0;
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.6;
  }

  .status-dashboard,
  .architecture-section,
  .chat-section,
  .specs-section {
    background: rgba(45, 55, 72, 0.3);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 3rem;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .status-dashboard h2,
  .architecture-section h2,
  .chat-section h2,
  .specs-section h2 {
    margin-bottom: 2rem;
    font-size: 1.75rem;
    font-weight: 700;
    color: #f7fafc;
  }

  .status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
  }

  .status-card {
    background: rgba(26, 32, 44, 0.6);
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid;
    transition: transform 0.2s, box-shadow 0.2s;
  }

  .status-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
  }

  .status-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .status-icon {
    font-size: 1.5rem;
  }

  .status-header h3 {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
  }

  .status-value {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
  }

  .status-details {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.85rem;
    color: #a0aec0;
  }

  .architecture-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
  }

  .arch-component {
    background: rgba(26, 32, 44, 0.6);
    border-radius: 12px;
    padding: 2rem;
    border-left: 4px solid;
    transition: transform 0.2s;
  }

  .arch-component:hover {
    transform: translateY(-4px);
  }

  .arch-component.primary { border-left-color: #4299e1; }
  .arch-component.secondary { border-left-color: #48bb78; }
  .arch-component.accent { border-left-color: #ed8936; }
  .arch-component.info { border-left-color: #9f7aea; }

  .arch-component h3 {
    margin-bottom: 1rem;
    font-size: 1.25rem;
    font-weight: 600;
    color: #f7fafc;
  }

  .arch-component p {
    margin-bottom: 1.5rem;
    color: #a0aec0;
    line-height: 1.6;
  }

  .arch-component ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .arch-component li {
    padding: 0.5rem 0;
    color: #e2e8f0;
    position: relative;
    padding-left: 1.5rem;
    font-size: 0.9rem;
  }

  .arch-component li:before {
    content: '▶';
    position: absolute;
    left: 0;
    color: #4299e1;
    font-size: 0.7rem;
  }

  .error-state {
    background: rgba(245, 101, 101, 0.1);
    border: 1px solid rgba(245, 101, 101, 0.3);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
  }

  .error-state h3 {
    margin-bottom: 1rem;
    color: #fc8181;
  }

  .startup-instructions {
    margin-top: 2rem;
    text-align: left;
  }

  .startup-instructions h4 {
    margin-bottom: 1rem;
    color: #4299e1;
  }

  .code-block {
    background: #1a202c;
    border: 1px solid #4a5568;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  }

  .code-block code {
    color: #68d391;
    font-size: 0.9rem;
    line-height: 1.6;
  }

  .specs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
  }

  .spec-category {
    background: rgba(26, 32, 44, 0.6);
    border-radius: 12px;
    padding: 2rem;
  }

  .spec-category h3 {
    margin-bottom: 1.5rem;
    font-size: 1.25rem;
    font-weight: 600;
    color: #f7fafc;
  }

  .spec-category ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .spec-category li {
    padding: 0.75rem 0;
    color: #e2e8f0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 0.9rem;
    line-height: 1.5;
  }

  .spec-category li:last-child {
    border-bottom: none;
  }

  .spec-category strong {
    color: #4299e1;
    font-weight: 600;
  }

  @media (max-width: 768px) {
    .production-page {
      padding: 1rem;
    }

    .page-header h1 {
      font-size: 2.5rem;
    }

    .subtitle {
      font-size: 1rem;
    }

    .status-grid,
    .architecture-grid,
    .specs-grid {
      grid-template-columns: 1fr;
    }

    .status-dashboard,
    .architecture-section,
    .chat-section,
    .specs-section {
      padding: 1.5rem;
    }
  }
</style>