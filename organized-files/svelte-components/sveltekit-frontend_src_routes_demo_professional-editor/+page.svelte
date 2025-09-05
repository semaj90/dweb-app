<script lang="ts">
  import ProfessionalEditor from '$lib/components/editor/ProfessionalEditor.svelte';
  import { FileText, Zap, Keyboard, Eye, Save, BarChart3 } from 'lucide-svelte';

  let editorContent = `<h1>Professional Document Editor</h1>

<p>This is a demonstration of the professional document editor with advanced features designed for legal professionals and content creators.</p>

<h2>Key Features</h2>

<ul>
  <li><strong>Full-screen immersive experience</strong> - Press F11 or click the maximize button</li>
  <li><strong>Keyboard shortcuts</strong> - Press Ctrl+/ to see all available shortcuts</li>
  <li><strong>Auto-save functionality</strong> - Your work is automatically saved every 5 seconds</li>
  <li><strong>Focus mode</strong> - Press F10 to dim the UI and focus on writing</li>
  <li><strong>Professional typography</strong> - Optimized for readability and professional documents</li>
  <li><strong>Real-time statistics</strong> - Word count, character count, and reading time</li>
</ul>

<h2>Sample Legal Document</h2>

<blockquote>
"The law is reason, free from passion." - Aristotle
</blockquote>

<p>In the matter of contract interpretation, courts generally apply the principle of <em>contra proferentem</em>, which states that ambiguous terms should be construed against the party who drafted the contract.</p>

<h3>Contract Elements</h3>

<ol>
  <li>Offer and acceptance</li>
  <li>Consideration</li>
  <li>Capacity to contract</li>
  <li>Legality of purpose</li>
</ol>

<p>Try using the keyboard shortcuts to format this text, or switch to focus mode for distraction-free writing.</p>`;

  let documentTitle = 'Professional Editor Demo';

  function handleSave(event: CustomEvent) {
    console.log('Document saved:', event.detail);
    // Here you would typically save to your backend
    alert('Document saved successfully!');
  }
</script>

<svelte:head>
  <title>Professional Editor Demo - Legal AI Platform</title>
  <meta name="description" content="Demonstration of professional document editor with advanced features" />
</svelte:head>

<div class="demo-page">
  <!-- Hero Section -->
  <header class="demo-header">
    <div class="space-y-4">
      <div class="hero-content">
        <div class="hero-text">
          <h1 class="hero-title gradient-text-primary">
            Professional Document Editor
          </h1>
          <p class="hero-subtitle">
            A powerful, feature-rich editor designed for legal professionals 
            and content creators who demand excellence in their writing tools.
          </p>
        </div>
        
        <div class="features-grid">
          <div class="feature-card yorha-card">
            <Zap class="feature-icon text-yorha-primary" />
            <h3>Full-Screen Experience</h3>
            <p>Immersive writing environment with F11 fullscreen support</p>
          </div>
          
          <div class="feature-card yorha-card">
            <Keyboard class="feature-icon text-yorha-secondary" />
            <h3>Keyboard Shortcuts</h3>
            <p>Professional hotkeys for efficient document editing</p>
          </div>
          
          <div class="feature-card yorha-card">
            <Save class="feature-icon text-yorha-accent" />
            <h3>Auto-Save</h3>
            <p>Never lose your work with intelligent auto-save functionality</p>
          </div>
          
          <div class="feature-card yorha-card">
            <Eye class="feature-icon text-yorha-success" />
            <h3>Focus Mode</h3>
            <p>Distraction-free writing with dimmed UI elements</p>
          </div>
          
          <div class="feature-card yorha-card">
            <FileText class="feature-icon text-yorha-primary" />
            <h3>Professional Typography</h3>
            <p>Optimized fonts and spacing for legal documents</p>
          </div>
          
          <div class="feature-card yorha-card">
            <BarChart3 class="feature-icon text-yorha-secondary" />
            <h3>Document Statistics</h3>
            <p>Real-time word count, character count, and reading time</p>
          </div>
        </div>
      </div>
    </div>
  </header>

  <!-- Editor Section -->
  <main class="demo-main">
    <div class="space-y-4">
      <section class="editor-section">
        <div class="editor-intro">
          <h2 class="section-title gradient-text-primary">
            Try the Editor
          </h2>
          <p class="section-subtitle">
            Experience all the professional features in action. The editor below 
            includes sample content to demonstrate formatting capabilities.
          </p>
          
          <div class="shortcuts-reminder">
            <h4>Quick Shortcuts to Try:</h4>
            <div class="shortcuts-list">
              <kbd>Ctrl + S</kbd> Save document
              <kbd>Ctrl + B</kbd> Bold text
              <kbd>Ctrl + I</kbd> Italic text
              <kbd>F11</kbd> Toggle full-screen
              <kbd>F10</kbd> Toggle focus mode
              <kbd>Ctrl + /</kbd> Show all shortcuts
            </div>
          </div>
        </div>

        <!-- Professional Editor Component -->
        <div class="editor-wrapper">
          <ProfessionalEditor 
            bind:content={editorContent}
            bind:title={documentTitle}
            autoSave={true}
            autoSaveInterval={5000}
            onsave={handleSave}
          />
        </div>
      </section>

      <!-- Usage Instructions -->
      <section class="instructions-section">
        <div class="instructions-grid">
          <div class="instruction-card yorha-card">
            <h3>Getting Started</h3>
            <ol>
              <li>Click in the editor area to start typing</li>
              <li>Use the formatting toolbar for basic styling</li>
              <li>Press <kbd>Ctrl + /</kbd> to see all keyboard shortcuts</li>
              <li>Try <kbd>F11</kbd> for full-screen writing</li>
            </ol>
          </div>
          
          <div class="instruction-card yorha-card">
            <h3>Professional Features</h3>
            <ul>
              <li>Auto-save every 5 seconds (configurable)</li>
              <li>Real-time document statistics</li>
              <li>Professional typography for legal documents</li>
              <li>Focus mode to minimize distractions</li>
            </ul>
          </div>
          
          <div class="instruction-card yorha-card">
            <h3>Keyboard Shortcuts</h3>
            <div class="shortcuts-table">
              <div class="shortcut-row">
                <kbd>Ctrl + S</kbd>
                <span>Save document</span>
              </div>
              <div class="shortcut-row">
                <kbd>Ctrl + B</kbd>
                <span>Bold text</span>
              </div>
              <div class="shortcut-row">
                <kbd>Ctrl + I</kbd>
                <span>Italic text</span>
              </div>
              <div class="shortcut-row">
                <kbd>F11</kbd>
                <span>Toggle full-screen</span>
              </div>
              <div class="shortcut-row">
                <kbd>F10</kbd>
                <span>Toggle focus mode</span>
              </div>
              <div class="shortcut-row">
                <kbd>Ctrl + /</kbd>
                <span>Show all shortcuts</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  </main>
</div>

<style>
  .demo-page {
    min-height: 100vh;
    background: #f4f1ea;
    color: #3a372f;
  }

  /* Hero Section */
  .demo-header {
    padding: 4rem 0;
    background: linear-gradient(135deg, 
      #f4f1ea 0%, 
      #faf8f3 100%);
    border-bottom: 1px solid #ada895;
  }

  .hero-content {
    text-align: center
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
    max-width: 600px;
    margin: 0 auto 3rem auto;
    line-height: 1.6;
  }

  .features-grid {
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin-top: 3rem;
  }

  .feature-card {
    padding: 2rem;
    text-align: center
    transition: transform 0.3s ease, box-shadow 0.3s ease;
  }

  .feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(58, 55, 47, 0.2);
  }

  .feature-icon {
    width: 3rem;
    height: 3rem;
    margin: 0 auto 1rem auto;
  }

  .feature-card h3 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: #3a372f;
  }

  .feature-card p {
    color: rgba(58, 55, 47, 0.8);
    line-height: 1.5;
  }

  /* Main Section */
  .demo-main {
    padding: 4rem 0;
  }

  .editor-section {
    margin-bottom: 4rem;
  }

  .editor-intro {
    text-align: center
    margin-bottom: 3rem;
  }

  .section-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
  }

  .section-subtitle {
    font-size: 1.125rem;
    color: rgba(58, 55, 47, 0.8);
    max-width: 600px;
    margin: 0 auto 2rem auto;
    line-height: 1.6;
  }

  .shortcuts-reminder {
    background: #faf8f3;
    border: 1px solid #ada895;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 2rem auto;
    max-width: 600px;
  }

  .shortcuts-reminder h4 {
    color: #3a372f;
    margin-bottom: 1rem;
    font-weight: 600;
  }

  .shortcuts-list {
    display: flex
    flex-wrap: wrap
    gap: 1rem;
    align-items: center
  }

  .shortcuts-list kbd {
    background: #f4f1ea;
    border: 1px solid #ada895;
    border-radius: 4px;
    padding: 0.25rem 0.5rem;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.875rem;
    color: #3a372f;
  }

  .editor-wrapper {
    margin: 2rem 0;
  }

  /* Instructions Section */
  .instructions-section {
    margin-top: 4rem;
  }

  .instructions-grid {
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 2rem;
  }

  .instruction-card {
    padding: 2rem;
  }

  .instruction-card h3 {
    color: #3a372f;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
  }

  .instruction-card ol,
  .instruction-card ul {
    padding-left: 1.5rem;
  }

  .instruction-card li {
    margin-bottom: 0.75rem;
    line-height: 1.6;
    color: rgba(58, 55, 47, 0.9);
  }

  .shortcuts-table {
    display: grid
    gap: 0.75rem;
  }

  .shortcut-row {
    display: flex
    align-items: center
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(173, 168, 149, 0.3);
  }

  .shortcut-row kbd {
    background: #faf8f3;
    border: 1px solid #ada895;
    border-radius: 4px;
    padding: 0.25rem 0.5rem;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.875rem;
    color: #3a372f;
  }

  .shortcut-row span {
    color: rgba(58, 55, 47, 0.9);
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .hero-title {
      font-size: 2.5rem;
    }

    .features-grid {
      grid-template-columns: 1fr;
    }

    .demo-header,
    .demo-main {
      padding: 2rem 0;
    }

    .shortcuts-list {
      justify-content: center
    }

    .instructions-grid {
      grid-template-columns: 1fr;
    }
  }

  /* Animation for focus */
  .feature-card,
  .instruction-card {
    animation: fadeInUp 0.6s ease-out;
  }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>