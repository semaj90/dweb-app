<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card/index.js';
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';

  // Demo page categories with golden ratio proportions
  const demoCategories = [
    {
      title: 'Core System Demos',
      description: 'Production AI and legal processing systems',
      color: 'yorha-primary',
      pages: [
        { name: 'Production AI Chat', path: '/production-ai', icon: '🎯' },
        { name: 'GPU Orchestrator', path: '/gpu-orchestrator', icon: '⚡' },
        { name: 'Semantic 3D', path: '/semantic-3d', icon: '🧊' },
        { name: 'Test Gemma3', path: '/test-gemma3', icon: '🤖' }
      ]
    },
    {
      title: 'Authentication & User Management',
      description: 'Registration, login, and user workflows',
      color: 'yorha-secondary',
      pages: [
        { name: 'User Registration', path: '/auth/register', icon: '👤' },
        { name: 'User Login', path: '/auth/login', icon: '🔑' },
        { name: 'User Profile', path: '/auth/profile', icon: '📋' },
        { name: 'Role Management', path: '/auth/roles', icon: '🔐' }
      ]
    },
    {
      title: 'Legal Case Management',
      description: 'CRUD operations for legal cases and evidence',
      color: 'yorha-accent',
      pages: [
        { name: 'Create Case', path: '/cases/create', icon: '📁' },
        { name: 'Case Dashboard', path: '/cases', icon: '📊' },
        { name: 'Evidence Upload', path: '/evidence/upload', icon: '📎' },
        { name: 'Case Analysis', path: '/cases/analysis', icon: '🔍' }
      ]
    },
    {
      title: 'AI & Analytics',
      description: 'AI-powered analysis and recommendations',
      color: 'yorha-warning',
      pages: [
        { name: 'AI Find Modal', path: '/ai/find', icon: '🎯' },
        { name: 'Document Analysis', path: '/ai/analyze', icon: '📄' },
        { name: 'Vector Search', path: '/ai/search', icon: '🔍' },
        { name: 'Chat Interface', path: '/ai/chat', icon: '💬' }
      ]
    },
    {
      title: 'Components & UI',
      description: 'UI component demonstrations and testing',
      color: 'yorha-info',
      pages: [
        { name: 'Bits UI Demo', path: '/ui/bits-demo', icon: '🎨' },
        { name: 'Gaming Interface', path: '/ui/gaming', icon: '🎮' },
        { name: 'Enhanced Canvas', path: '/ui/canvas', icon: '🖼️' },
        { name: 'Data Visualization', path: '/ui/dataviz', icon: '📈' }
      ]
    },
    {
      title: 'Development Tools',
      description: 'MCP tools, testing, and development utilities',
      color: 'yorha-success',
      pages: [
        { name: 'MCP Tools', path: '/dev/mcp-tools', icon: '🔧' },
        { name: 'Context7 Demo', path: '/dev/context7', icon: '🧠' },
        { name: 'System Health', path: '/dev/health', icon: '💓' },
        { name: 'Performance Monitor', path: '/dev/performance', icon: '📊' }
      ]
    }
  ];

  let selectedCategory = $state<string | null>(null);
  let animationClass = $state('');

  onMount(() => {
    animationClass = 'animate-fade-in';
  });

  function navigateToPage(path: string) {
    goto(path);
  }

  function getButtonVariant(color: string) {
    const variants = {
      'yorha-primary': 'yorha-primary',
      'yorha-secondary': 'yorha-secondary', 
      'yorha-accent': 'yorha-accent',
      'yorha-warning': 'yorha-warning',
      'yorha-info': 'yorha-info',
      'yorha-success': 'yorha-success'
    };
    return variants[color] || 'default';
  }
</script>

<svelte:head>
  <title>YoRHa Demo Navigation - Legal AI System</title>
  <meta name="description" content="Comprehensive navigation for all legal AI system demos and components" />
</svelte:head>

<div class="yorha-demo-container {animationClass}">
  <!-- Header with YoRHa styling -->
  <header class="yorha-header">
    <div class="yorha-title-container">
      <h1 class="yorha-title">YoRHa Demo Navigation System</h1>
      <p class="yorha-subtitle">Legal AI Platform - Component Demonstrations</p>
    </div>
    <div class="yorha-system-info">
      <div class="system-status">
        <span class="status-indicator active"></span>
        <span class="status-text">All Systems Operational</span>
      </div>
    </div>
  </header>

  <!-- Main navigation grid with golden ratio proportions -->
  <main class="demo-grid">
    {#each demoCategories as category, categoryIndex}
      <Card class="category-card {category.color}">
        <CardHeader class="category-header">
          <CardTitle class="category-title">{category.title}</CardTitle>
          <CardDescription class="category-description">
            {category.description}
          </CardDescription>
        </CardHeader>
        
        <CardContent class="category-content">
          <div class="demo-buttons-grid">
            {#each category.pages as page, pageIndex}
              <Button
                variant={getButtonVariant(category.color)}
                size="lg"
                class="demo-button"
                onclick={() => navigateToPage(page.path)}
              >
                <span class="button-icon">{page.icon}</span>
                <span class="button-text">{page.name}</span>
                <span class="button-arrow">→</span>
              </Button>
            {/each}
          </div>
        </CardContent>
      </Card>
    {/each}
  </main>

  <!-- Footer with system information -->
  <footer class="yorha-footer">
    <div class="footer-grid">
      <div class="footer-section">
        <h3>System Architecture</h3>
        <ul>
          <li>SvelteKit 2.0 + Svelte 5</li>
          <li>GPU-Accelerated AI (RTX 3060 Ti)</li>
          <li>Context7 MCP Integration</li>
          <li>PostgreSQL + pgvector</li>
        </ul>
      </div>
      <div class="footer-section">
        <h3>UI Framework</h3>
        <ul>
          <li>bits-ui Components</li>
          <li>UnoCSS Styling</li>
          <li>YoRHa Design System</li>
          <li>Golden Ratio Layout</li>
        </ul>
      </div>
      <div class="footer-section">
        <h3>AI Models</h3>
        <ul>
          <li>gemma3-legal:latest</li>
          <li>nomic-embed-text</li>
          <li>Ollama Integration</li>
          <li>RAG Pipeline</li>
        </ul>
      </div>
    </div>
  </footer>
</div>

<style>
  /* YoRHa Design System with Golden Ratio Proportions */
  .yorha-demo-container {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
    color: #e8e6e3;
    font-family: 'Rajdhani', 'Roboto Mono', monospace;
    position: relative;
    overflow-x: hidden;
  }

  .yorha-demo-container::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      radial-gradient(circle at 20% 80%, rgba(255, 206, 84, 0.03) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(159, 122, 234, 0.03) 0%, transparent 50%),
      radial-gradient(circle at 40% 40%, rgba(66, 153, 225, 0.02) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }

  .yorha-header {
    position: relative;
    z-index: 10;
    padding: 2rem 2rem 1rem;
    border-bottom: 2px solid #d69e2e;
    background: rgba(26, 26, 26, 0.8);
    backdrop-filter: blur(10px);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .yorha-title-container {
    flex: 1;
  }

  .yorha-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #d69e2e;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 0 0 0.5rem 0;
    text-shadow: 0 0 20px rgba(214, 158, 46, 0.3);
  }

  .yorha-subtitle {
    font-size: 1.125rem;
    color: #a0aec0;
    margin: 0;
    font-weight: 300;
  }

  .yorha-system-info {
    display: flex;
    align-items: center;
  }

  .system-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    background: rgba(68, 200, 245, 0.1);
    border: 1px solid #44c8f5;
    border-radius: 0;
    text-transform: uppercase;
    font-size: 0.875rem;
    font-weight: 600;
  }

  .status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #68d391;
    box-shadow: 0 0 10px rgba(104, 211, 145, 0.5);
    animation: pulse 2s ease-in-out infinite alternate;
  }

  /* Main Grid using Golden Ratio (1.618) */
  .demo-grid {
    position: relative;
    z-index: 10;
    padding: 2rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1.618rem;
    max-width: 1400px;
    margin: 0 auto;
  }

  /* Category Cards */
  .category-card {
    background: rgba(45, 55, 72, 0.6);
    border: 1px solid rgba(214, 158, 46, 0.3);
    border-radius: 0;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
  }

  .category-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #d69e2e, #ed8936);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.3s ease;
  }

  .category-card:hover::before {
    transform: scaleX(1);
  }

  .category-card:hover {
    transform: translateY(-4px);
    border-color: #d69e2e;
    box-shadow: 0 10px 30px rgba(214, 158, 46, 0.2);
  }

  .category-header {
    padding: 1.5rem 1.5rem 1rem;
  }

  .category-title {
    color: #d69e2e;
    font-size: 1.5rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
  }

  .category-description {
    color: #a0aec0;
    font-size: 0.95rem;
    line-height: 1.6;
  }

  .category-content {
    padding: 1rem 1.5rem 1.5rem;
  }

  /* Demo Buttons Grid */
  .demo-buttons-grid {
    display: grid;
    gap: 0.75rem;
    grid-template-columns: 1fr;
  }

  /* YoRHa Styled Buttons */
  :global(.demo-button) {
    width: 100%;
    height: auto;
    min-height: 3.5rem;
    background: rgba(26, 32, 44, 0.8);
    border: 1px solid #4a5568;
    color: #e2e8f0;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    border-radius: 0;
  }

  :global(.demo-button::before) {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(214, 158, 46, 0.1), transparent);
    transition: left 0.5s ease;
  }

  :global(.demo-button:hover::before) {
    left: 100%;
  }

  :global(.demo-button:hover) {
    border-color: #d69e2e;
    color: #d69e2e;
    transform: translateX(4px);
    box-shadow: 0 0 20px rgba(214, 158, 46, 0.2);
  }

  .button-icon {
    font-size: 1.25rem;
    margin-right: 0.75rem;
  }

  .button-text {
    flex: 1;
    text-align: left;
  }

  .button-arrow {
    font-size: 1.125rem;
    transition: transform 0.3s ease;
  }

  :global(.demo-button:hover) .button-arrow {
    transform: translateX(4px);
  }

  /* Color Variants */
  .yorha-primary {
    border-left: 4px solid #4299e1;
  }

  .yorha-secondary {
    border-left: 4px solid #9f7aea;
  }

  .yorha-accent {
    border-left: 4px solid #ed8936;
  }

  .yorha-warning {
    border-left: 4px solid #f6e05e;
  }

  .yorha-info {
    border-left: 4px solid #63b3ed;
  }

  .yorha-success {
    border-left: 4px solid #68d391;
  }

  /* Footer */
  .yorha-footer {
    position: relative;
    z-index: 10;
    margin-top: 3rem;
    padding: 2rem;
    border-top: 2px solid #4a5568;
    background: rgba(26, 26, 26, 0.9);
    backdrop-filter: blur(10px);
  }

  .footer-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
    max-width: 1200px;
    margin: 0 auto;
  }

  .footer-section h3 {
    color: #d69e2e;
    font-size: 1.125rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 1rem;
    letter-spacing: 1px;
  }

  .footer-section ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .footer-section li {
    color: #a0aec0;
    font-size: 0.875rem;
    padding: 0.25rem 0;
    position: relative;
    padding-left: 1rem;
  }

  .footer-section li::before {
    content: '▶';
    position: absolute;
    left: 0;
    color: #4a5568;
    font-size: 0.75rem;
  }

  /* Animations */
  @keyframes fade-in {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes pulse {
    from {
      opacity: 1;
    }
    to {
      opacity: 0.5;
    }
  }

  .animate-fade-in {
    animation: fade-in 0.8s ease-out;
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .yorha-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .yorha-title {
      font-size: 2rem;
    }

    .demo-grid {
      grid-template-columns: 1fr;
      padding: 1rem;
      gap: 1rem;
    }

    .category-card {
      margin: 0;
    }

    .footer-grid {
      grid-template-columns: 1fr;
      gap: 1.5rem;
    }
  }

  @media (max-width: 480px) {
    .yorha-title {
      font-size: 1.75rem;
    }

    .yorha-subtitle {
      font-size: 1rem;
    }

    .demo-buttons-grid {
      gap: 0.5rem;
    }

    :global(.demo-button) {
      min-height: 3rem;
      padding: 0.75rem 1rem;
    }
  }
</style>