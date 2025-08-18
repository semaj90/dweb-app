<script lang="ts">
  import GPUAcceleratedChat from '$lib/components/GPUAcceleratedChat.svelte';
  import { onMount } from 'svelte';

  let orchestratorStatus = 'checking...';

  onMount(async () => {
    // Check if the GPU orchestrator is running
    try {
      const response = await fetch('http://localhost:4001/health');
      if (response.ok) {
        const health = await response.json();
        orchestratorStatus = health.status;
      } else {
        orchestratorStatus = 'not running';
      }
    } catch (error) {
      orchestratorStatus = 'not available';
    }
  });
</script>

<svelte:head>
  <title>GPU-Accelerated Legal AI Chat</title>
  <meta name="description" content="Test the GPU-accelerated legal AI chat system" />
</svelte:head>

<div class="page-container">
  <div class="page-header">
    <h1>🎯 GPU-Accelerated Legal AI Chat</h1>
    <p class="subtitle">
      Experience the power of GPU-accelerated legal AI with Gemma3 and real-time WebSocket communication
    </p>
  </div>

  <div class="status-card">
    <h3>🔧 System Status</h3>
    <div class="status-grid">
      <div class="status-item">
        <span class="status-label">GPU Orchestrator:</span>
        <span class="status-value" class:success={orchestratorStatus === 'healthy'} 
              class:warning={orchestratorStatus === 'degraded'}
              class:error={orchestratorStatus === 'not running' || orchestratorStatus === 'not available'}>
          {orchestratorStatus}
        </span>
      </div>
      <div class="status-item">
        <span class="status-label">AI Models:</span>
        <span class="status-value">Gemma3-Legal, NoMic-Embed-Text</span>
      </div>
      <div class="status-item">
        <span class="status-label">Vector Database:</span>
        <span class="status-value">PostgreSQL + pgvector</span>
      </div>
      <div class="status-item">
        <span class="status-label">Communication:</span>
        <span class="status-value">WebSocket + REST API</span>
      </div>
    </div>
  </div>

  <div class="architecture-info">
    <h3>🏗️ Architecture Overview</h3>
    <div class="architecture-grid">
      <div class="arch-component">
        <h4>🎯 GPU Orchestrator</h4>
        <p>Node.js server with XState managing the AI pipeline on port 4001</p>
      </div>
      <div class="arch-component">
        <h4>🤖 AI Models</h4>
        <p>Ollama serving Gemma3-Legal (11.8B) with 35 GPU layers</p>
      </div>
      <div class="arch-component">
        <h4>🔍 Vector Search</h4>
        <p>PostgreSQL with pgvector for semantic document retrieval</p>
      </div>
      <div class="arch-component">
        <h4>⚡ Real-time</h4>
        <p>WebSocket connections for instant AI responses</p>
      </div>
    </div>
  </div>

  <div class="instructions">
    <h3>💡 How to Use</h3>
    <ul>
      <li>Ask any legal question in natural language</li>
      <li>The system will generate embeddings using NoMic-Embed-Text</li>
      <li>Relevant legal context is retrieved from the vector database</li>
      <li>Gemma3-Legal generates a comprehensive response</li>
      <li>Monitor real-time processing metrics in the chat interface</li>
    </ul>
  </div>

  {#if orchestratorStatus === 'not running' || orchestratorStatus === 'not available'}
    <div class="error-message">
      <h3>⚠️ GPU Orchestrator Not Available</h3>
      <p>The GPU orchestrator is not running. To start it:</p>
      <code>cd gpu-orchestrator && npm start</code>
      <p>Or check if the service is running on port 4001.</p>
    </div>
  {:else}
    <div class="chat-container">
      <GPUAcceleratedChat />
    </div>
  {/if}

  <div class="technical-specs">
    <h3>🔧 Technical Specifications</h3>
    <div class="specs-grid">
      <div class="spec-section">
        <h4>GPU Acceleration</h4>
        <ul>
          <li>RTX 3060 Ti (8GB VRAM)</li>
          <li>CUDA 12.8 integration</li>
          <li>35 GPU layers for Gemma3</li>
          <li>Parallel embedding processing</li>
        </ul>
      </div>
      <div class="spec-section">
        <h4>AI Pipeline</h4>
        <ul>
          <li>Text → Embedding (384d)</li>
          <li>Vector similarity search</li>
          <li>Context retrieval</li>
          <li>LLM generation</li>
        </ul>
      </div>
      <div class="spec-section">
        <h4>Performance</h4>
        <ul>
          <li>Sub-second embeddings</li>
          <li>Real-time WebSocket</li>
          <li>Concurrent request handling</li>
          <li>GPU memory optimization</li>
        </ul>
      </div>
    </div>
  </div>
</div>

<style>
  .page-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    min-height: 100vh;
  }

  .page-header {
    text-align: center;
    margin-bottom: 3rem;
  }

  .page-header h1 {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
  }

  .subtitle {
    font-size: 1.2rem;
    color: #4a5568;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
  }

  .status-card {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  }

  .status-card h3 {
    margin-bottom: 1.5rem;
    color: #2d3748;
    font-size: 1.5rem;
  }

  .status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
  }

  .status-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: #f7fafc;
    border-radius: 8px;
  }

  .status-label {
    font-weight: 600;
    color: #4a5568;
  }

  .status-value {
    font-weight: 500;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
  }

  .status-value.success {
    background: #c6f6d5;
    color: #22543d;
  }

  .status-value.warning {
    background: #fefcbf;
    color: #744210;
  }

  .status-value.error {
    background: #fed7d7;
    color: #822727;
  }

  .architecture-info {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  }

  .architecture-info h3 {
    margin-bottom: 1.5rem;
    color: #2d3748;
    font-size: 1.5rem;
  }

  .architecture-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
  }

  .arch-component {
    padding: 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
  }

  .arch-component h4 {
    margin-bottom: 0.75rem;
    font-size: 1.1rem;
  }

  .arch-component p {
    font-size: 0.9rem;
    line-height: 1.5;
    opacity: 0.9;
  }

  .instructions {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  }

  .instructions h3 {
    margin-bottom: 1.5rem;
    color: #2d3748;
    font-size: 1.5rem;
  }

  .instructions ul {
    list-style: none;
    padding: 0;
  }

  .instructions li {
    padding: 0.75rem 0;
    border-bottom: 1px solid #e2e8f0;
    position: relative;
    padding-left: 2rem;
  }

  .instructions li:before {
    content: '✓';
    position: absolute;
    left: 0;
    color: #667eea;
    font-weight: bold;
  }

  .error-message {
    background: #fed7d7;
    border: 1px solid #fc8181;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
    color: #822727;
  }

  .error-message h3 {
    margin-bottom: 1rem;
  }

  .error-message code {
    background: #822727;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-family: 'Courier New', monospace;
    display: block;
    margin: 1rem 0;
  }

  .chat-container {
    margin: 2rem 0;
  }

  .technical-specs {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  }

  .technical-specs h3 {
    margin-bottom: 1.5rem;
    color: #2d3748;
    font-size: 1.5rem;
  }

  .specs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
  }

  .spec-section h4 {
    color: #667eea;
    margin-bottom: 1rem;
    font-size: 1.1rem;
  }

  .spec-section ul {
    list-style: none;
    padding: 0;
  }

  .spec-section li {
    padding: 0.5rem 0;
    color: #4a5568;
    position: relative;
    padding-left: 1.5rem;
  }

  .spec-section li:before {
    content: '•';
    position: absolute;
    left: 0;
    color: #667eea;
    font-weight: bold;
  }

  @media (max-width: 768px) {
    .page-header h1 {
      font-size: 2rem;
    }

    .status-grid,
    .architecture-grid,
    .specs-grid {
      grid-template-columns: 1fr;
    }
  }
</style>