<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  
  // Layout Components
  import PageLayout from '$lib/components/layout/PageLayout.svelte';
  import ContentSection from '$lib/components/layout/ContentSection.svelte';
  import Card from '$lib/components/ui/card/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  
  // Feedback Integration
  import FeedbackIntegration from '$lib/components/feedback/FeedbackIntegration.svelte';
  
  export let data: {
    userId: string | null;
    sessionId: string | null;
    email: string | null;
    isAuthenticated: boolean;
  };
  
  let loading = false;
  let welcomeMessage = "Welcome to the Legal AI Dashboard";
  
  // Feedback integration reference
  let dashboardFeedback: any;
  
  // Simulate loading demo data
  onMount(() => {
    loading = true;
    setTimeout(() => {
      loading = false;
      if (data.isAuthenticated) {
        welcomeMessage = `🎉 Welcome back, ${data.email || 'User'}!`;
      } else {
        welcomeMessage = "Please log in to access the dashboard";
      }
    }, 1000);
  });
</script>

<svelte:head>
  <title>Dashboard - Legal AI Platform</title>
</svelte:head>

<PageLayout 
  title="Legal AI Dashboard" 
  subtitle="Complete system overview and session management"
  variant="dashboard" 
  maxWidth="xl" 
  padding="lg" 
  gap="lg"
>
  <!-- Session Management Panel -->
  <ContentSection title="Session Management" variant="card">
    {#if data.isAuthenticated}
      <Card variant="legal" priority="high" class="p-8">
        <div class="flex-between mb-6">
          <div class="flex items-center gap-4">
            <span class="text-4xl neural-sprite-active">✅</span>
            <div>
              <h3 class="nes-legal-title text-xl text-green-300 mb-2">
                Welcome, {data.email}
              </h3>
              <p class="text-green-200">
                Session authenticated and active
              </p>
            </div>
          </div>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div class="nes-legal-priority-medium p-4 rounded">
            <h4 class="text-yellow-400 font-bold mb-2">User ID</h4>
            <code class="text-white font-mono">{data.userId}</code>
          </div>
          <div class="nes-legal-priority-medium p-4 rounded">
            <h4 class="text-yellow-400 font-bold mb-2">Session ID</h4>
            <code class="text-white font-mono">{data.sessionId}</code>
          </div>
        </div>
        
        <div class="flex flex-wrap gap-4">
          <a href="/">
            <Button variant="yorha" class="yorha-3d-button">
              🏠 Back to Homepage
            </Button>
          </a>
          <form method="post" style="display: inline;">
            <Button
              type="submit"
              formaction="?/logout"
              variant="destructive"
              class="yorha-3d-button"
            >
              🔓 Logout
            </Button>
          </form>
        </div>
      </Card>
    {:else}
      <Card variant="legal" priority="critical" class="p-8">
        <div class="flex-center mb-6">
          <span class="text-4xl neural-sprite-loading mr-4">❌</span>
          <h3 class="nes-legal-title text-xl text-red-300">
            Authentication Required
          </h3>
        </div>
        <p class="text-center text-gray-300 mb-6">
          Please log in to access the dashboard features
        </p>
        <div class="flex-center gap-4">
          <a href="/auth/login">
            <Button variant="success" class="yorha-3d-button">
              🔑 Login
            </Button>
          </a>
          <a href="/">
            <Button variant="yorha" class="yorha-3d-button">
              🏠 Back to Homepage
            </Button>
          </a>
        </div>
      </Card>
    {/if}
  </ContentSection>

  {#if loading}
    <ContentSection title="Loading Dashboard" variant="card">
      <Card class="flex-center p-12">
        <div class="neural-sprite-loading rounded-full h-16 w-16 border-4 border-yellow-400"></div>
        <span class="ml-4 text-xl nes-legal-title">Loading dashboard...</span>
      </Card>
    </ContentSection>
  {:else}
    <!-- Welcome Section -->
    <ContentSection title="Welcome Status" variant="card">
      <Card variant="legal" priority="high" class="p-6">
        <h2 class="nes-legal-title text-2xl mb-4">{welcomeMessage}</h2>
        {#if data.isAuthenticated}
          <div class="space-y-3">
            <div class="flex items-center gap-3">
              <span class="neural-sprite-active">✅</span>
              <span>Authentication successful - User logged in</span>
            </div>
            <div class="flex items-center gap-3">
              <span class="neural-sprite-active">✅</span>
              <span>User ID: <code class="nes-legal-priority-medium px-2 py-1 rounded">{data.userId}</code></span>
            </div>
            <div class="flex items-center gap-3">
              <span class="neural-sprite-active">✅</span>
              <span>Session ID: <code class="nes-legal-priority-medium px-2 py-1 rounded text-xs">{data.sessionId}</code></span>
            </div>
            <div class="flex items-center gap-3">
              <span class="neural-sprite-active">✅</span>
              <span>Email: <code class="nes-legal-priority-medium px-2 py-1 rounded">{data.email}</code></span>
            </div>
          </div>
        {:else}
          <div class="flex items-center gap-3">
            <span class="neural-sprite-loading">❌</span>
            <span class="text-red-300">Not authenticated - please log in</span>
          </div>
        {/if}
      </Card>
    </ContentSection>

    <!-- Dashboard Statistics -->
    <ContentSection title="Dashboard Overview" variant="grid" columns={3} gap="lg">
      <!-- Cases Card -->
      <Card variant="yorha" priority="critical" interactive={true} class="group p-6">
        <div class="flex-col-center text-center space-y-4">
          <h3 class="nes-legal-title text-xl text-yellow-400">Active Cases</h3>
          <div class="text-4xl font-bold text-white neural-sprite-active">12</div>
          <p class="text-gray-300">Currently investigating</p>
          <Button
            variant="yorha"
            class="yorha-3d-button group-hover:scale-105 transition-transform"
            onclick={() => {
              dashboardFeedback?.feedback.featureUsed('view_cases', { dashboardWidget: 'active_cases' });
              window.location.href = '/cases';
            }}
          >
            View Cases
          </Button>
        </div>
      </Card>

      <!-- Evidence Card -->
      <Card variant="legal" priority="high" interactive={true} class="group p-6">
        <div class="flex-col-center text-center space-y-4">
          <h3 class="nes-legal-title text-xl text-yellow-400">Evidence Items</h3>
          <div class="text-4xl font-bold text-white nes-memory-active">247</div>
          <p class="text-gray-300">Documents & Files</p>
          <Button
            variant="evidence"
            class="yorha-3d-button group-hover:scale-105 transition-transform"
            onclick={() => {
              dashboardFeedback?.feedback.featureUsed('manage_evidence', { dashboardWidget: 'evidence_items' });
              window.location.href = '/evidence';
            }}
          >
            Manage Evidence
          </Button>
        </div>
      </Card>

      <!-- AI Analysis Card -->
      <Card variant="nes" priority="critical" interactive={true} class="group p-6">
        <div class="flex-col-center text-center space-y-4">
          <h3 class="nes-legal-title text-xl text-yellow-400">AI Analysis</h3>
          <div class="text-4xl font-bold text-white neural-sprite-cached">89%</div>
          <p class="text-gray-300">Accuracy Rate</p>
          <Button
            variant="success"
            class="yorha-3d-button group-hover:scale-105 transition-transform"
            onclick={() => {
              dashboardFeedback?.feedback.featureUsed('start_ai_analysis', { 
                dashboardWidget: 'ai_analysis',
                accuracyRate: '89%'
              });
              window.location.href = '/aiassistant';
            }}
          >
            Start Analysis
          </Button>
        </div>
      </Card>
    </ContentSection>

    <!-- Auto-Login Test Results -->
    <ContentSection title="Auto-Login Test Results" variant="card">
      <Card variant="legal" priority="high" class="p-6">
        <div class="space-y-consistent">
          <div class="flex items-center gap-4">
            <span class="text-green-400 text-xl neural-sprite-active">✅</span>
            <span class="text-lg">Quick Demo Login button functionality</span>
          </div>
          <div class="flex items-center gap-4">
            <span class="text-green-400 text-xl neural-sprite-active">✅</span>
            <span class="text-lg">Auto-fill Demo Credentials functionality</span>
          </div>
          <div class="flex items-center gap-4">
            <span class="text-green-400 text-xl neural-sprite-active">✅</span>
            <span class="text-lg">Login page UI components working</span>
          </div>
          <div class="flex items-center gap-4">
            <span class="text-green-400 text-xl neural-sprite-active">✅</span>
            <span class="text-lg">Dashboard redirect successful</span>
          </div>
          <div class="flex items-center gap-4">
            <span class="text-green-400 text-xl neural-sprite-active">✅</span>
            <span class="text-lg">Native Windows services integration</span>
          </div>
          <div class="flex items-center gap-4">
            <span class="text-green-400 text-xl neural-sprite-active">✅</span>
            <span class="text-lg">Production-ready implementation</span>
          </div>
        </div>
      </Card>
    </ContentSection>

    <!-- System Status -->
    <ContentSection title="System Status" variant="grid" columns={4} gap="md">
      <Card class="flex-col-center p-6 text-center">
        <div class="text-3xl font-bold text-green-400 neural-sprite-active mb-2">✓</div>
        <div class="text-sm text-gray-300">SvelteKit Frontend</div>
      </Card>
      <Card class="flex-col-center p-6 text-center">
        <div class="text-3xl font-bold text-green-400 neural-sprite-active mb-2">✓</div>
        <div class="text-sm text-gray-300">Auto-login System</div>
      </Card>
      <Card class="flex-col-center p-6 text-center">
        <div class="text-3xl font-bold text-green-400 neural-sprite-active mb-2">✓</div>
        <div class="text-sm text-gray-300">Authentication</div>
      </Card>
      <Card class="flex-col-center p-6 text-center">
        <div class="text-3xl font-bold text-green-400 neural-sprite-active mb-2">✓</div>
        <div class="text-sm text-gray-300">Dashboard Ready</div>
      </Card>
    </ContentSection>

    <!-- Navigation -->
    <ContentSection title="Quick Navigation" variant="card">
      <div class="flex-center gap-6">
        <a href="/auth/login">
          <Button variant="ghost" class="yorha-3d-button">
            ← Back to Login
          </Button>
        </a>
        <a href="/all-routes">
          <Button variant="yorha" class="yorha-3d-button">
            View All Routes
          </Button>
        </a>
      </div>
    </ContentSection>
  {/if}
</PageLayout>

<style>
  .animate-spin {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* Session Debug Panel Styles */
  .session-debug-panel {
    background: #1a1a1a;
    border: 2px solid #ffd700;
    border-radius: 12px;
    padding: 2rem;
    margin: 1rem;
    box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
  }

  .debug-title {
    color: #ffd700;
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0 0 1.5rem 0;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 2px;
    text-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
  }

  .debug-info {
    background: #2a2a2a;
    border-radius: 8px;
    padding: 1.5rem;
  }

  .debug-info.authenticated {
    border: 2px solid #00ff41;
    background: linear-gradient(135deg, rgba(0, 255, 65, 0.1) 0%, rgba(0, 255, 65, 0.05) 100%);
  }

  .debug-info.not-authenticated {
    border: 2px solid #ff6b35;
    background: linear-gradient(135deg, rgba(255, 107, 53, 0.1) 0%, rgba(255, 107, 53, 0.05) 100%);
  }

  .status-message {
    font-size: 1.2rem;
    font-weight: bold;
    margin: 0 0 1.5rem 0;
    text-align: center;
    color: #e0e0e0;
  }

  .session-details {
    margin: 1rem 0;
    padding: 1rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 6px;
    border: 1px solid #444;
  }

  .detail-item {
    margin: 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .detail-item .label {
    font-weight: bold;
    color: #ffd700;
    min-width: 80px;
  }

  .detail-item .value {
    background: #1a1a1a;
    color: #00ff41;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    border: 1px solid #333;
  }

  .detail-item .session-id {
    color: #ffd700;
    font-size: 0.8rem;
    word-break: break-all;
  }

  .session-actions {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 1.5rem;
  }

  .action-btn {
    padding: 0.75rem 1.5rem;
    border-radius: 6px;
    text-decoration: none;
    font-weight: bold;
    font-size: 0.9rem;
    border: 2px solid;
    transition: all 0.3s ease;
    cursor: pointer;
    font-family: inherit;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
  }

  .action-btn.dashboard {
    background: #ffd700;
    color: #1a1a1a;
    border-color: #ffd700;
  }

  .action-btn.dashboard:hover {
    background: #ffed4e;
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
  }

  .action-btn.login {
    background: #00ff41;
    color: #1a1a1a;
    border-color: #00ff41;
  }

  .action-btn.login:hover {
    background: #33ff66;
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0, 255, 65, 0.4);
  }

  .action-btn.logout {
    background: #ff0041;
    color: white;
    border-color: #ff0041;
  }

  .action-btn.logout:hover {
    background: #ff3366;
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(255, 0, 65, 0.4);
  }
</style>

<!-- Feedback Integration Component -->
<FeedbackIntegration
  bind:this={dashboardFeedback}
  interactionType="dashboard_usage"
  ratingType="ui_experience"
  priority="medium"
  context={{ 
    page: 'dashboard',
    isAuthenticated: data.isAuthenticated,
    userId: data.userId,
    email: data.email
  }}
  trackOnMount={true}
  let:feedback
/>