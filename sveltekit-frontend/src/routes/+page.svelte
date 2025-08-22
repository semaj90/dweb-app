<script lang="ts">
  import { onMount } from 'svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Card from '$lib/components/ui/Card.svelte';

  let systemInfo = {
    uptime: '6 hours, 23 minutes',
    activeServices: 7,
    lastSync: '2 minutes ago'
  };

  let showAuth = false;
  let isLogin = true;
  let email = '';
  let password = '';
  let firstName = '';
  let lastName = '';
  let loading = false;
  let message = '';
  let error = '';

  onMount(() => {
    console.log('YoRHa Legal AI Platform loaded');
  });

  async function handleAuth(event) {
    event.preventDefault();
    if (!email || !password) {
      error = 'Email and password are required';
      return;
    }

    loading = true;
    error = '';
    message = '';

    try {
      const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
      const body = isLogin 
        ? { email, password }
        : { email, password, firstName, lastName };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      const result = await response.json();

      if (result.success) {
        message = result.message;
        if (!isLogin) {
          // Switch to login after successful registration
          isLogin = true;
          message = 'Registration successful! You can now login.';
        } else {
          // Redirect after successful login
          window.location.href = '/dashboard';
        }
      } else {
        error = result.error || 'An error occurred';
      }
    } catch (err) {
      error = 'Network error occurred';
      console.error(err);
    }

    loading = false;
  }
</script>

<svelte:head>
  <title>YoRHa Legal AI Platform</title>
</svelte:head>

<div class="home-page">
  <div class="hero-section">
    <h1>YoRHa Legal AI Platform</h1>
    <p class="subtitle">Advanced evidence processing with AI-powered analysis</p>
  </div>
  <!-- Authentication Section -->
  <div class="auth-section" style="text-align: center; margin: 2rem 0;">
    <Button variant="legal" onclick={() => showAuth = !showAuth}>
      {showAuth ? 'Hide' : 'Show'} Authentication
    </Button>
  </div>

  {#if showAuth}
    <Card class="max-w-md mx-auto">
      <h2 class="text-xl font-bold mb-4">{isLogin ? 'Login' : 'Register'}</h2>
      
      <form onsubmit={handleAuth} class="space-y-4">
        <div>
          <label for="email" class="block text-sm font-medium mb-1">Email</label>
          <input
            id="email"
            type="email"
            bind:value={email}
            required
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter your email"
          />
        </div>

        <div>
          <label for="password" class="block text-sm font-medium mb-1">Password</label>
          <input
            id="password"
            type="password"
            bind:value={password}
            required
            minlength="8"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter your password"
          />
        </div>

        {#if !isLogin}
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label for="firstName" class="block text-sm font-medium mb-1">First Name</label>
              <input
                id="firstName"
                type="text"
                bind:value={firstName}
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="First name"
              />
            </div>
            <div>
              <label for="lastName" class="block text-sm font-medium mb-1">Last Name</label>
              <input
                id="lastName"
                type="text"
                bind:value={lastName}
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Last name"
              />
            </div>
          </div>
        {/if}

        {#if error}
          <div class="text-red-600 text-sm">{error}</div>
        {/if}

        {#if message}
          <div class="text-green-600 text-sm">{message}</div>
        {/if}

        <Button 
          type="submit" 
          variant="legal" 
          class="w-full"
          {loading}
          loadingText={isLogin ? 'Signing in...' : 'Creating account...'}
        >
          {isLogin ? 'Sign In' : 'Create Account'}
        </Button>
      </form>

      <div class="mt-4 text-center">
        <button
          type="button"
          class="text-blue-600 hover:underline text-sm"
          onclick={() => {
            isLogin = !isLogin;
            error = '';
            message = '';
          }}
        >
          {isLogin ? 'Need an account? Sign up' : 'Already have an account? Sign in'}
        </button>
      </div>
    </Card>
  {/if}

  <!-- Quick access CTA: All Routes -->
  <div class="home-cta" style="text-align:center; margin: 1.5rem 0;">
    <a href="/all-routes" class="primary-cta">📚 View All Routes</a>
  </div>

  <div class="quick-stats">
    <div class="stat-card">
      <h3>System Uptime</h3>
      <p class="stat-value">{systemInfo.uptime}</p>
    </div>
    <div class="stat-card">
      <h3>Active Services</h3>
      <p class="stat-value">{systemInfo.activeServices}</p>
    </div>
    <div class="stat-card">
      <h3>Last Sync</h3>
      <p class="stat-value">{systemInfo.lastSync}</p>
    </div>
  </div>

  <!-- Auth Section -->
  <div class="auth-section">
    <h2>Access Platform</h2>
    <div class="auth-buttons">
      <a href="/auth" class="auth-card unified">
        <h3>🔐 Login / Register</h3>
        <p>Unified authentication experience - login or create account</p>
      </a>
    </div>
    <!-- Legacy buttons for testing -->
    <div class="auth-buttons legacy">
      <a href="/auth/login" class="auth-card login">
        <h3>🔐 Login</h3>
        <p>Access your account and continue your work</p>
      </a>
      <a href="/auth/register" class="auth-card register">
        <h3>📝 Register</h3>
        <p>Create a new account to get started</p>
      </a>
    </div>
  </div>

  <div class="action-grid">
    <a href="/all-routes" class="action-card featured">
      <h3>📚 All Routes</h3>
      <p>Comprehensive navigation for all 80+ available routes</p>
    </a>
    <a href="/endpoints" class="action-card">
      <h3>📊 Service Status</h3>
      <p>Monitor backend services and API endpoints</p>
    </a>
    <a href="/demo/enhanced-rag-semantic" class="action-card">
      <h3>🤖 Enhanced RAG Demo</h3>
      <p>Test AI semantic search capabilities</p>
    </a>
    <a href="/evidence" class="action-card">
      <h3>📁 Evidence Manager</h3>
      <p>Upload and analyze legal documents</p>
    </a>
    <a href="/chat" class="action-card">
      <h3>💬 AI Assistant</h3>
      <p>Interactive legal AI chat interface</p>
    </a>
    <a href="/ai-assistant" class="action-card">
      <h3>🎯 AI Assistant</h3>
      <p>Advanced AI-powered legal analysis</p>
    </a>
  </div>
</div>

<style>
  .home-page {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }

  .hero-section {
    text-align: center;
    margin-bottom: 3rem;
  }

  .hero-section h1 {
    font-size: 2.5rem;
    color: #ffd700;
    margin-bottom: 1rem;
    text-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
  }

  .subtitle {
    font-size: 1.2rem;
    color: #b0b0b0;
    margin-bottom: 0;
  }

  .quick-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 3rem;
  }

  .stat-card {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
  }

  .stat-card h3 {
    margin: 0 0 0.5rem 0;
    color: #ffd700;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .stat-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #00ff41;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
  }

  .action-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
  }

  .action-card {
    background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
    border: 1px solid #444;
    border-radius: 8px;
    padding: 2rem;
    text-decoration: none;
    color: inherit;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
  }

  .action-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.1), transparent);
    transition: left 0.5s;
  }

  .action-card:hover::before {
    left: 100%;
  }

  .action-card:hover {
    border-color: #ffd700;
    box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
    transform: translateY(-2px);
  }

  .action-card h3 {
    margin: 0 0 1rem 0;
    color: #ffd700;
    font-size: 1.2rem;
  }

  .action-card p {
    margin: 0;
    color: #b0b0b0;
    line-height: 1.5;
  }

  .action-card.featured {
    background: linear-gradient(135deg, #ffd700 0%, #ff8c00 100%);
    color: #1a1a1a;
    border-color: #ffd700;
    box-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
  }

  .action-card.featured h3 {
    color: #1a1a1a;
    text-shadow: none;
  }

  .action-card.featured p {
    color: #333;
  }

  /* Auth Section Styles with UnoCSS/PostCSS YoRHa Theme */
  .auth-section {
    margin-bottom: 3rem;
    text-align: center;
  }

  .auth-section h2 {
    color: #ffd700;
    font-size: 2rem;
    margin: 0 0 2rem 0;
    text-transform: uppercase;
    letter-spacing: 2px;
    text-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
  }

  .auth-buttons {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 2rem;
    max-width: 800px;
    margin: 0 auto;
  }

  .auth-card {
    background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
    border: 2px solid #444;
    border-radius: 12px;
    padding: 2.5rem;
    text-decoration: none;
    color: inherit;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }

  .auth-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.15), transparent);
    transition: left 0.6s;
  }

  .auth-card:hover::before {
    left: 100%;
  }

  .auth-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(255, 215, 0, 0.25);
  }

  .auth-card.login:hover {
    border-color: #00ff41;
    box-shadow: 0 8px 30px rgba(0, 255, 65, 0.25);
  }

  .auth-card.register:hover {
    border-color: #ff6b35;
    box-shadow: 0 8px 30px rgba(255, 107, 53, 0.25);
  }

  .auth-card h3 {
    margin: 0 0 1rem 0;
    font-size: 1.5rem;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .auth-card.login h3 {
    color: #00ff41;
    text-shadow: 0 0 8px rgba(0, 255, 65, 0.4);
  }

  .auth-card.register h3 {
    color: #ff6b35;
    text-shadow: 0 0 8px rgba(255, 107, 53, 0.4);
  }

  .auth-card p {
    margin: 0;
    color: #b0b0b0;
    font-size: 1rem;
    line-height: 1.4;
    text-align: center;
  }

  .auth-card:hover p {
    color: #e0e0e0;
  }

  /* Unified Auth Button Styling */
  .auth-card.unified {
    background: linear-gradient(135deg, #ffd700 0%, #ff8c00 100%);
    color: #1a1a1a;
    border-color: #ffd700;
    box-shadow: 0 8px 30px rgba(255, 215, 0, 0.3);
    min-height: 140px;
    margin-bottom: 2rem;
  }

  .auth-card.unified h3 {
    color: #1a1a1a !important;
    text-shadow: none !important;
    font-size: 1.8rem;
  }

  .auth-card.unified p {
    color: #333 !important;
    font-weight: 500;
  }

  .auth-card.unified:hover {
    background: linear-gradient(135deg, #ffed4e 0%, #ff9a00 100%);
    border-color: #ffed4e;
    box-shadow: 0 12px 40px rgba(255, 215, 0, 0.4);
    transform: translateY(-5px);
  }

  .auth-card.unified:hover p {
    color: #222 !important;
  }

  /* Legacy buttons styling */
  .auth-buttons.legacy {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #444;
    opacity: 0.7;
  }

  .auth-buttons.legacy .auth-card {
    min-height: 100px;
    padding: 1.5rem;
  }

  .auth-buttons.legacy .auth-card h3 {
    font-size: 1.2rem;
  }

  .auth-buttons.legacy .auth-card p {
    font-size: 0.9rem;
  }

  /* Single column for unified button */
  .auth-buttons:not(.legacy) {
    grid-template-columns: 1fr;
    max-width: 600px;
  }

  /* Home CTA Button */
  .home-cta .primary-cta {
    display: inline-block;
    padding: 0.75rem 1.25rem;
    background: linear-gradient(90deg, #ffd700, #ff8c00);
    color: #1a1a1a;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 700;
    box-shadow: 0 6px 18px rgba(255, 160, 0, 0.15);
    transition: all 0.3s ease;
  }

  .home-cta .primary-cta:hover {
    background: linear-gradient(90deg, #ffed4e, #ff9a00);
    box-shadow: 0 8px 25px rgba(255, 160, 0, 0.3);
    transform: translateY(-2px);
  }
</style>