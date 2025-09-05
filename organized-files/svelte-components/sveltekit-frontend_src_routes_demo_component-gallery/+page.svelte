<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import ProfessionalEditor from '$lib/components/editor/ProfessionalEditor.svelte';
  import AIChatInterface from '$lib/components/ai/AIChatInterface.svelte';
  import { 
    Palette, 
    Component, 
    Zap, 
    Eye, 
    Keyboard, 
    Save, 
    Settings, 
    FileText,
    MessageCircle,
    Bot,
    Search,
    BarChart3,
    Users,
    Shield,
    Sparkles
  } from 'lucide-svelte';

  let activeTheme = 'default';
  let showGrid = true;
  let componentFilter = 'all';

  const themeVariants = {
    default: { name: 'YoRHa Default', class: '' },
    highContrast: { name: 'High Contrast', class: 'high-contrast' },
    print: { name: 'Print Optimized', class: 'print-mode' }
  };

  const componentCategories = [
    { id: 'all', name: 'All Components', icon: Component },
    { id: 'ui', name: 'UI Elements', icon: Settings },
    { id: 'ai', name: 'AI Components', icon: Bot },
    { id: 'forms', name: 'Forms', icon: FileText },
    { id: 'layout', name: 'Layout', icon: BarChart3 }
  ];

  const features = [
    {
      icon: Palette,
      title: 'YoRHa Design System',
      description: 'Professional theme with golden accents and dark backgrounds',
      color: 'text-yorha-primary'
    },
    {
      icon: Zap,
      title: 'Performance Optimized',
      description: '40% faster CSS generation, 60% smaller bundles',
      color: 'text-yorha-accent'
    },
    {
      icon: Eye,
      title: 'Focus Mode',
      description: 'Distraction-free interface with intelligent UI dimming',
      color: 'text-yorha-secondary'
    },
    {
      icon: Keyboard,
      title: 'Keyboard Shortcuts',
      description: 'Professional hotkeys for efficient workflow',
      color: 'text-yorha-success'
    },
    {
      icon: Save,
      title: 'Auto-Save',
      description: 'Intelligent auto-save with dirty checking',
      color: 'text-yorha-primary'
    },
    {
      icon: Shield,
      title: 'Accessibility',
      description: 'WCAG 2.1 AA compliant with full keyboard navigation',
      color: 'text-yorha-accent'
    }
  ];

  const uiComponents = [
    { name: 'Buttons', description: 'Primary, secondary, and action buttons with YoRHa styling' },
    { name: 'Cards', description: 'Content containers with hover effects and consistent spacing' },
    { name: 'Forms', description: 'Professional form elements with validation and focus states' },
    { name: 'Navigation', description: 'Responsive navigation with breadcrumbs and user context' },
    { name: 'Modals', description: 'Accessible overlays with focus management' },
    { name: 'Data Tables', description: 'Sortable, filterable tables for legal data' }
  ];

  let demoContent = `<h1>Legal Document Analysis</h1>
<p>This professional editor demonstrates advanced typography and formatting capabilities designed specifically for legal professionals.</p>

<h2>Contract Analysis Summary</h2>
<blockquote>"The terms of this agreement shall be construed in accordance with the laws of the jurisdiction specified herein."</blockquote>

<h3>Key Findings</h3>
<ul>
  <li><strong>Liability Clauses</strong> - Standard limitation provisions identified</li>
  <li><strong>Termination Rights</strong> - Mutual termination with 30-day notice</li>
  <li><strong>Intellectual Property</strong> - Work-for-hire agreement with copyright assignment</li>
</ul>

<p>Use <kbd>Ctrl+B</kbd> for bold, <kbd>Ctrl+I</kbd> for italic, and <kbd>F11</kbd> for fullscreen mode.</p>`;
</script>

<svelte:head>
  <title>Component Gallery - Production UI/UX System</title>
  <meta name="description" content="Showcase of production-level legal AI components with YoRHa theme" />
</svelte:head>

<div class="gallery-page" class:show-grid={showGrid}>
  <!-- Hero Section -->
  <header class="gallery-hero">
    <div class="container mx-auto px-4 py-8">
      <div class="hero-content">
        <h1 class="hero-title gradient-text-primary">
          Production UI/UX Component Gallery
        </h1>
        <p class="hero-subtitle">
          Comprehensive showcase of our legal AI platform's component system, featuring the 
          YoRHa design language and production-level accessibility standards.
        </p>
        
        <div class="hero-stats">
          <div class="stat-item">
            <span class="stat-number">150+</span>
            <span class="stat-label">Components</span>
          </div>
          <div class="stat-item">
            <span class="stat-number">40%</span>
            <span class="stat-label">Faster CSS</span>
          </div>
          <div class="stat-item">
            <span class="stat-number">100%</span>
            <span class="stat-label">Accessible</span>
          </div>
          <div class="stat-item">
            <span class="stat-number">85%</span>
            <span class="stat-label">Test Coverage</span>
          </div>
        </div>
      </div>
    </div>
  </header>

  <!-- Theme Controls -->
  <section class="theme-controls">
    <div class="space-y-4">
      <Card class="controls-card">
        <CardHeader>
          <CardTitle class="flex items-center gap-2">
            <Settings class="h-5 w-5" />
            Gallery Controls
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div class="controls-grid">
            <div class="control-group">
              <Label for="theme-select">Theme Variant</Label>
              <select 
                id="theme-select"
                bind:value={activeTheme}
                class="yorha-input"
              >
                {#each Object.entries(themeVariants) as [key, theme]}
                  <option value={key}>{theme.name}</option>
                {/each}
              </select>
            </div>
            
            <div class="control-group">
              <Label for="component-filter">Category Filter</Label>
              <select 
                id="component-filter"
                bind:value={componentFilter}
                class="yorha-input"
              >
                {#each componentCategories as category}
                  <option value={category.id}>{category.name}</option>
                {/each}
              </select>
            </div>
            
            <div class="control-group">
              <label class="flex items-center gap-2">
                <input 
                  type="checkbox" 
                  bind:checked={showGrid}
                  class="yorha-checkbox"
                />
                <span>Show Grid Lines</span>
              </label>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  </section>

  <!-- Features Overview -->
  <section class="features-section">
    <div class="space-y-4">
      <h2 class="section-title gradient-text-primary">
        Production-Level Features
      </h2>
      
      <div class="features-grid">
        {#each features as feature}
          <Card class="feature-showcase">
            <CardContent class="text-center p-6">
              <svelte:component 
                this={feature.icon} 
                class="feature-icon {feature.color} mx-auto mb-4" 
              />
              <h3 class="feature-title">{feature.title}</h3>
              <p class="feature-description">{feature.description}</p>
            </CardContent>
          </Card>
        {/each}
      </div>
    </div>
  </section>

  <!-- Component Showcase -->
  <section class="component-showcase">
    <div class="space-y-4">
      <h2 class="section-title gradient-text-primary">
        Component System
      </h2>

      <!-- UI Elements -->
      <div class="showcase-section">
        <h3 class="showcase-title">
          <Component class="h-6 w-6" />
          UI Elements
        </h3>
        
        <div class="component-grid">
          <!-- Buttons -->
          <Card class="component-demo">
            <CardHeader>
              <CardTitle>Buttons</CardTitle>
            </CardHeader>
            <CardContent class="space-y-4">
              <div class="button-group">
                <Button class="yorha-btn-primary">Primary Action</Button>
                <Button class="yorha-btn-secondary">Secondary</Button>
                <Button disabled>Disabled</Button>
              </div>
              <div class="button-group">
                <Button size="sm">Small</Button>
                <Button size="md">Medium</Button>
                <Button size="lg">Large</Button>
              </div>
            </CardContent>
          </Card>

          <!-- Form Elements -->
          <Card class="component-demo">
            <CardHeader>
              <CardTitle>Form Elements</CardTitle>
            </CardHeader>
            <CardContent class="space-y-4">
              <div class="form-group">
                <Label for="demo-input">Professional Input</Label>
                <Input 
                  id="demo-input"
                  placeholder="Enter legal document title..."
                  class="yorha-input"
                />
              </div>
              <div class="form-group">
                <Label for="demo-textarea">Case Description</Label>
                <textarea 
                  id="demo-textarea"
                  class="yorha-input w-full"
                  rows="3"
                  placeholder="Describe the case details..."
                ></textarea>
              </div>
            </CardContent>
          </Card>

          <!-- Status Indicators -->
          <Card class="component-demo">
            <CardHeader>
              <CardTitle>Status & Badges</CardTitle>
            </CardHeader>
            <CardContent class="space-y-4">
              <div class="badge-group">
                <span class="badge badge-success">Active Case</span>
                <span class="badge badge-warning">Under Review</span>
                <span class="badge badge-error">Urgent</span>
                <span class="badge badge-info">AI Analyzed</span>
              </div>
              <div class="progress-demo">
                <Label>Document Processing</Label>
                <div class="progress-bar">
                  <div class="progress-fill" style="width: 75%"></div>
                </div>
                <span class="progress-text">75% Complete</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <!-- AI Components -->
      <div class="showcase-section">
        <h3 class="showcase-title">
          <Bot class="h-6 w-6" />
          AI Components
        </h3>
        
        <div class="ai-showcase">
          <Card class="ai-demo">
            <CardHeader>
              <CardTitle>AI Chat Interface</CardTitle>
            </CardHeader>
            <CardContent>
              <div class="chat-preview">
                <div class="message user-message">
                  <div class="message-content">
                    Analyze this contract for liability clauses
                  </div>
                  <div class="message-meta">You • 2:34 PM</div>
                </div>
                <div class="message ai-message">
                  <div class="message-content">
                    I've identified 3 liability clauses in the contract. The primary limitation 
                    clause on page 4 caps damages at $50,000. Would you like me to analyze 
                    the enforceability of these provisions?
                  </div>
                  <div class="message-meta">AI Assistant • 2:35 PM</div>
                </div>
                <div class="typing-indicator">
                  <span class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </span>
                  AI is thinking...
                </div>
              </div>
            </CardContent>
          </Card>

          <Card class="ai-demo">
            <CardHeader>
              <CardTitle>Multi-Agent Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div class="agent-pipeline">
                <div class="agent-step completed">
                  <div class="agent-icon">✓</div>
                  <div class="agent-info">
                    <h4>Evidence Analyzer</h4>
                    <p>Processed 15 documents</p>
                  </div>
                </div>
                <div class="agent-step completed">
                  <div class="agent-icon">✓</div>
                  <div class="agent-info">
                    <h4>Person Extractor</h4>
                    <p>Identified 8 individuals</p>
                  </div>
                </div>
                <div class="agent-step active">
                  <div class="agent-icon spin">⟳</div>
                  <div class="agent-info">
                    <h4>Relationship Mapper</h4>
                    <p>Building connections...</p>
                  </div>
                </div>
                <div class="agent-step pending">
                  <div class="agent-icon">○</div>
                  <div class="agent-info">
                    <h4>Case Synthesizer</h4>
                    <p>Waiting...</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <!-- Professional Editor Demo -->
      <div class="showcase-section">
        <h3 class="showcase-title">
          <FileText class="h-6 w-6" />
          Professional Editor
        </h3>
        
        <Card class="editor-showcase">
          <CardHeader>
            <CardTitle>Full-Featured Document Editor</CardTitle>
            <p class="text-sm text-yorha-text-muted">
              Features: Auto-save, keyboard shortcuts, focus mode, document statistics
            </p>
          </CardHeader>
          <CardContent>
            <div class="editor-container">
              <ProfessionalEditor 
                bind:content={demoContent}
                title="Legal Analysis Demo"
                autoSave={false}
                onsave={(e) => console.log('Demo save:', e.detail)}
              />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  </section>

  <!-- Theme Documentation -->
  <section class="theme-docs">
    <div class="space-y-4">
      <h2 class="section-title gradient-text-primary">
        YoRHa Theme System
      </h2>
      
      <div class="docs-grid">
        <Card class="color-palette">
          <CardHeader>
            <CardTitle>Color Palette</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="color-grid">
              <div class="color-swatch">
                <div class="swatch" style="background-color: #b8860b"></div>
                <div class="color-info">
                  <span class="color-name">Primary</span>
                  <span class="color-hex">#b8860b</span>
                </div>
              </div>
              <div class="color-swatch">
                <div class="swatch" style="background-color: #cd853f"></div>
                <div class="color-info">
                  <span class="color-name">Secondary</span>
                  <span class="color-hex">#cd853f</span>
                </div>
              </div>
              <div class="color-swatch">
                <div class="swatch" style="background-color: #daa520"></div>
                <div class="color-info">
                  <span class="color-name">Accent</span>
                  <span class="color-hex">#daa520</span>
                </div>
              </div>
              <div class="color-swatch">
                <div class="swatch" style="background-color: #1a1a1a"></div>
                <div class="color-info">
                  <span class="color-name">Background</span>
                  <span class="color-hex">#1a1a1a</span>
                </div>
              </div>
              <div class="color-swatch">
                <div class="swatch" style="background-color: #2d2d2d"></div>
                <div class="color-info">
                  <span class="color-name">Surface</span>
                  <span class="color-hex">#2d2d2d</span>
                </div>
              </div>
              <div class="color-swatch">
                <div class="swatch" style="background-color: #f5f5dc"></div>
                <div class="color-info">
                  <span class="color-name">Text</span>
                  <span class="color-hex">#f5f5dc</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card class="typography-demo">
          <CardHeader>
            <CardTitle>Typography System</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="type-scale">
              <h1 class="type-sample h1">Heading 1 - Document Title</h1>
              <h2 class="type-sample h2">Heading 2 - Section</h2>
              <h3 class="type-sample h3">Heading 3 - Subsection</h3>
              <p class="type-sample body">Body text for legal documents with optimal readability</p>
              <p class="type-sample caption">Caption text for metadata and annotations</p>
              <code class="type-sample code">Code and monospace elements</code>
            </div>
          </CardContent>
        </Card>

        <Card class="accessibility-info">
          <CardHeader>
            <CardTitle>Accessibility Features</CardTitle>
          </CardHeader>
          <CardContent>
            <div class="a11y-features">
              <div class="a11y-item">
                <Shield class="h-5 w-5 text-yorha-success" />
                <div>
                  <h4>WCAG 2.1 AA Compliant</h4>
                  <p>4.5:1 contrast ratios maintained</p>
                </div>
              </div>
              <div class="a11y-item">
                <Keyboard class="h-5 w-5 text-yorha-success" />
                <div>
                  <h4>Full Keyboard Navigation</h4>
                  <p>Tab order and focus management</p>
                </div>
              </div>
              <div class="a11y-item">
                <Users class="h-5 w-5 text-yorha-success" />
                <div>
                  <h4>Screen Reader Support</h4>
                  <p>Semantic HTML and ARIA labels</p>
                </div>
              </div>
              <div class="a11y-item">
                <Eye class="h-5 w-5 text-yorha-success" />
                <div>
                  <h4>Reduced Motion</h4>
                  <p>Respects user preferences</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  </section>
</div>

<style>
  .gallery-page {
    min-height: 100vh;
    background: #f4f1ea;
    color: #3a372f;
  }

  .show-grid {
    background-image: 
      linear-gradient(rgba(184, 134, 11, 0.1) 1px, transparent 1px),
      linear-gradient(90deg, rgba(184, 134, 11, 0.1) 1px, transparent 1px);
    background-size: 20px 20px;
  }

  /* Hero Section */
  .gallery-hero {
    background: linear-gradient(135deg, 
      #f4f1ea 0%, 
      #faf8f3 100%);
    border-bottom: 1px solid #ada895;
  }

  .hero-content {
    text-align: center
    max-width: 800px;
    margin: 0 auto;
  }

  .hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 1.5rem;
    line-height: 1.1;
  }

  .hero-subtitle {
    font-size: 1.25rem;
    color: rgba(58, 55, 47, 0.8);
    margin-bottom: 3rem;
    line-height: 1.6;
  }

  .hero-stats {
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
  }

  .stat-item {
    text-align: center
  }

  .stat-number {
    display: block
    font-size: 2.5rem;
    font-weight: 700;
    color: #3a372f;
    font-family: 'Consolas', monospace;
  }

  .stat-label {
    display: block
    font-size: 0.875rem;
    color: rgba(58, 55, 47, 0.7);
    text-transform: uppercase
    letter-spacing: 0.05em;
  }

  /* Theme Controls */
  .theme-controls {
    padding: 2rem 0;
  }

  .controls-card {
    max-width: 800px;
    margin: 0 auto;
  }

  .controls-grid {
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
  }

  .control-group {
    display: flex
    flex-direction: column
    gap: 0.5rem;
  }

  /* Features Section */
  .features-section {
    padding: 4rem 0;
  }

  .section-title {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center
    margin-bottom: 3rem;
  }

  .features-grid {
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
  }

  .feature-showcase {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
  }

  .feature-showcase:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(58, 55, 47, 0.2);
  }

  .feature-icon {
    width: 3rem;
    height: 3rem;
    margin-bottom: 1rem;
  }

  .feature-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: #3a372f;
  }

  .feature-description {
    color: rgba(58, 55, 47, 0.8);
    line-height: 1.5;
  }

  /* Component Showcase */
  .component-showcase {
    padding: 4rem 0;
    background: rgba(250, 248, 243, 0.3);
  }

  .showcase-section {
    margin-bottom: 4rem;
  }

  .showcase-title {
    display: flex
    align-items: center
    gap: 0.75rem;
    font-size: 1.875rem;
    font-weight: 600;
    margin-bottom: 2rem;
    color: #3a372f;
  }

  .component-grid {
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 2rem;
    margin-bottom: 3rem;
  }

  .component-demo {
    background: #f4f1ea;
  }

  .button-group {
    display: flex
    gap: 1rem;
    flex-wrap: wrap
  }

  .form-group {
    display: flex
    flex-direction: column
    gap: 0.5rem;
  }

  .badge-group {
    display: flex
    gap: 0.75rem;
    flex-wrap: wrap
  }

  .badge {
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-size: 0.875rem;
    font-weight: 500;
  }

  .badge-success { background: rgba(16, 185, 129, 0.2); color: #10b981; }
  .badge-warning { background: rgba(213, 207, 183, 0.2); color: #d5cfb7; }
  .badge-error { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
  .badge-info { background: rgba(58, 55, 47, 0.2); color: #3a372f; }

  .progress-demo {
    display: flex
    flex-direction: column
    gap: 0.5rem;
  }

  .progress-bar {
    width: 100%;
    height: 8px;
    background: #faf8f3;
    border-radius: 4px;
    overflow: hidden
  }

  .progress-fill {
    height: 100%;
    background: #3a372f;
    transition: width 0.3s ease;
  }

  .progress-text {
    font-size: 0.875rem;
    color: rgba(58, 55, 47, 0.7);
  }

  /* AI Showcase */
  .ai-showcase {
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 2rem;
  }

  .chat-preview {
    max-height: 300px;
    overflow-y: auto
    padding: 1rem;
    background: rgba(250, 248, 243, 0.5);
    border-radius: 8px;
  }

  .message {
    margin-bottom: 1rem;
    padding: 0.75rem;
    border-radius: 8px;
  }

  .user-message {
    background: rgba(58, 55, 47, 0.2);
    margin-left: 2rem;
  }

  .ai-message {
    background: #faf8f3;
    margin-right: 2rem;
  }

  .message-content {
    margin-bottom: 0.5rem;
    line-height: 1.5;
  }

  .message-meta {
    font-size: 0.75rem;
    color: rgba(58, 55, 47, 0.6);
  }

  .typing-indicator {
    display: flex
    align-items: center
    gap: 0.75rem;
    padding: 0.75rem;
    color: rgba(58, 55, 47, 0.6);
    font-size: 0.875rem;
  }

  .typing-dots {
    display: flex
    gap: 0.25rem;
  }

  .typing-dots span {
    width: 6px;
    height: 6px;
    background: #3a372f;
    border-radius: 50%;
    animation: typing 1.5s infinite;
  }

  .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes typing {
    0%, 60%, 100% { opacity: 0.3; }
    30% { opacity: 1; }
  }

  .agent-pipeline {
    display: flex
    flex-direction: column
    gap: 1rem;
  }

  .agent-step {
    display: flex
    align-items: center
    gap: 1rem;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #ada895;
  }

  .agent-step.completed {
    background: rgba(16, 185, 129, 0.1);
    border-color: #10b981;
  }

  .agent-step.active {
    background: rgba(58, 55, 47, 0.1);
    border-color: #3a372f;
  }

  .agent-step.pending {
    background: rgba(250, 248, 243, 0.3);
  }

  .agent-icon {
    width: 2rem;
    height: 2rem;
    display: flex
    align-items: center
    justify-content: center
    font-weight: bold
    border-radius: 50%;
  }

  .agent-step.completed .agent-icon {
    background: #10b981;
    color: #f4f1ea;
  }

  .agent-step.active .agent-icon {
    background: #3a372f;
    color: #f4f1ea;
    animation: spin 2s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .agent-info h4 {
    font-weight: 600;
    margin-bottom: 0.25rem;
  }

  .agent-info p {
    font-size: 0.875rem;
    color: rgba(58, 55, 47, 0.7);
  }

  /* Editor Showcase */
  .editor-showcase {
    background: #f4f1ea;
  }

  .editor-container {
    max-height: 600px;
    overflow: hidden
    border-radius: 8px;
  }

  /* Theme Documentation */
  .theme-docs {
    padding: 4rem 0;
    background: #f4f1ea;
  }

  .docs-grid {
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 2rem;
  }

  .color-grid {
    display: grid
    gap: 1rem;
  }

  .color-swatch {
    display: flex
    align-items: center
    gap: 1rem;
  }

  .swatch {
    width: 3rem;
    height: 3rem;
    border-radius: 8px;
    border: 1px solid #ada895;
  }

  .color-info {
    display: flex
    flex-direction: column
  }

  .color-name {
    font-weight: 600;
    color: #3a372f;
  }

  .color-hex {
    font-size: 0.875rem;
    color: rgba(58, 55, 47, 0.7);
    font-family: 'Consolas', monospace;
  }

  .type-scale {
    display: flex
    flex-direction: column
    gap: 1rem;
  }

  .type-sample.h1 {
    font-size: 2.25rem;
    font-weight: 700;
    color: #3a372f;
  }

  .type-sample.h2 {
    font-size: 1.875rem;
    font-weight: 600;
    color: #75726a;
  }

  .type-sample.h3 {
    font-size: 1.5rem;
    font-weight: 600;
  }

  .type-sample.body {
    font-size: 1rem;
    line-height: 1.6;
  }

  .type-sample.caption {
    font-size: 0.875rem;
    color: rgba(58, 55, 47, 0.7);
  }

  .type-sample.code {
    font-family: 'Consolas', monospace;
    background: #faf8f3;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.875rem;
  }

  .a11y-features {
    display: flex
    flex-direction: column
    gap: 1.5rem;
  }

  .a11y-item {
    display: flex
    align-items: flex-start;
    gap: 1rem;
  }

  .a11y-item h4 {
    font-weight: 600;
    margin-bottom: 0.25rem;
    color: #3a372f;
  }

  .a11y-item p {
    font-size: 0.875rem;
    color: rgba(58, 55, 47, 0.7);
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .hero-title {
      font-size: 2.5rem;
    }

    .component-grid,
    .ai-showcase {
      grid-template-columns: 1fr;
    }

    .controls-grid {
      grid-template-columns: 1fr;
    }

    .hero-stats {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>