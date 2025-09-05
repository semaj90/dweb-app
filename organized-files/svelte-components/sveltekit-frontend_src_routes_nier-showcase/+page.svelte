<script lang="ts">
  import NierThemeShowcase from '$lib/components/NierThemeShowcase.svelte'
  import NierHeader from '$lib/components/NierHeader.svelte'
  import CaseCard from '$lib/components/cases/CaseCard.svelte'
  import NierAIAssistant from '$lib/components/ai/NierAIAssistant.svelte'
  import AIAssistant from '$lib/components/AIAssistant.svelte'
  import { Button } from 'bits-ui'
  import type { User } from '$lib/types'

  let isDarkMode = $state(false);
  let showAIAssistant = $state(false);

  // Sample user data
  const user: User = {
    id: 'user-white-001',
    name: 'Commander White',
    email: 'commander@yorha.mil',
    firstName: 'Commander',
    lastName: 'White',
    avatarUrl: '',
    role: 'admin',
    isActive: true,
    permissions: ['read', 'write', 'admin']
  }

  // Sample case data
  const sampleCases = [
    {
      id: 'CASE-2025-001',
      title: 'Machine Lifeform Network Breach',
      description: 'Investigation into unauthorized access of YoRHa combat data through compromised machine network protocols.',
      status: 'active' as const,
      priority: 'critical' as const,
      created: new Date('2025-07-14'),
      assignee: {
        name: '2B',
        avatar: null
      },
      stats: {
        evidence: 24,
        witnesses: 5,
        documents: 18
      },
      tags: ['cybersecurity', 'data-breach', 'priority'],
      progress: 75
    },
    {
      id: 'CASE-2025-002',
      title: 'Android Rights Violation',
      description: 'Analysis of potential violations regarding android autonomy protocols in civilian sectors.',
      status: 'pending' as const,
      priority: 'high' as const,
      created: new Date('2025-07-12'),
      assignee: {
        name: '9S',
        avatar: null
      },
      stats: {
        evidence: 12,
        witnesses: 8,
        documents: 15
      },
      tags: ['civil-rights', 'android-law'],
      progress: 45
    },
    {
      id: 'CASE-2025-003',
      title: 'Resource Allocation Dispute',
      description: 'Legal review of resource distribution protocols between resistance camps.',
      status: 'active' as const,
      priority: 'medium' as const,
      created: new Date('2025-07-10'),
      assignee: {
        name: 'A2',
        avatar: null
      },
      stats: {
        evidence: 8,
        witnesses: 3,
        documents: 10
      },
      tags: ['resources', 'dispute-resolution'],
      progress: 60
    }
  ]

  // Case actions
  const handleViewCase = (id: string) => {
    console.log('View case:', id)
  }

  const handleEditCase = (id: string) => {
    console.log('Edit case:', id)
  }

  const handleArchiveCase = (id: string) => {
    console.log('Archive case:', id)
  }

  const handleDeleteCase = (id: string) => {
    console.log('Delete case:', id)
  }

  // Apply dark mode class to body
  $effect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark')
      document.documentElement.setAttribute('data-theme', 'dark')
    } else {
      document.documentElement.classList.remove('dark')
      document.documentElement.setAttribute('data-theme', 'light')
    }
  })
</script>

<svelte:head>
  <title>NieR: Automata Legal System - SvelteKit</title>
  <meta name="description" content="A NieR: Automata themed legal case management system built with SvelteKit, Bits UI, and Melt UI" />
</svelte:head>

<div class="min-h-screen bg-nier-white dark:bg-nier-black transition-colors duration-500">
  <!-- Header -->
  <NierHeader {user} bind:isDarkMode />

  <!-- Main Content -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <!-- Hero Section -->
    <section class="text-center py-12 mb-12">
      <h1 class="text-5xl md:text-6xl font-display font-bold nier-heading mb-4 animate-fade-in">
        YoRHa Legal Division
      </h1>
      <p class="text-xl text-nier-gray dark:text-nier-silver mb-8 animate-fade-in" style="animation-delay: 0.2s">
        Advanced Case Management System • For the Glory of Mankind
      </p>

      <div class="flex justify-center gap-4 animate-fade-in" style="animation-delay: 0.4s">
        <button
          onclick={() => showAIAssistant = true}
          class="nier-button-digital px-6 py-3"
        >
          <span class="mr-2">🤖</span>
          Launch AI Assistant
        </button>
        <a href="#showcase" class="nier-button-outline px-6 py-3 text-nier-black dark:text-nier-white">
          View Components
        </a>
      </div>
    </section>

    <!-- Stats Overview -->
    <section class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
      <div class="nier-card p-6 text-center group hover:border-digital-green dark:hover:border-digital-green">
        <div class="text-3xl font-bold text-harvard-crimson dark:text-digital-green mb-2">
          127
        </div>
        <p class="text-sm text-nier-gray dark:text-nier-silver">Active Cases</p>
      </div>

      <div class="nier-card p-6 text-center group hover:border-digital-green dark:hover:border-digital-green">
        <div class="text-3xl font-bold text-harvard-crimson dark:text-digital-green mb-2">
          89%
        </div>
        <p class="text-sm text-nier-gray dark:text-nier-silver">Success Rate</p>
      </div>

      <div class="nier-card p-6 text-center group hover:border-digital-green dark:hover:border-digital-green">
        <div class="text-3xl font-bold text-harvard-crimson dark:text-digital-green mb-2">
          1,247
        </div>
        <p class="text-sm text-nier-gray dark:text-nier-silver">Evidence Items</p>
      </div>

      <div class="nier-card p-6 text-center group hover:border-digital-green dark:hover:border-digital-green">
        <div class="text-3xl font-bold text-harvard-crimson dark:text-digital-green mb-2">
          24/7
        </div>
        <p class="text-sm text-nier-gray dark:text-nier-silver">AI Support</p>
      </div>
    </section>

    <!-- Recent Cases -->
    <section class="mb-12">
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-2xl font-display font-bold nier-heading">Recent Cases</h2>
        <a href="/cases" class="nier-link text-sm">
          View all cases →
        </a>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {#each sampleCases as caseData}
          <CaseCard
            case={caseData}
            onView={handleViewCase}
            onEdit={handleEditCase}
            onArchive={handleArchiveCase}
            onDelete={handleDeleteCase}
          />
        {/each}
      </div>
    </section>

    <!-- Component Showcase -->
    <section id="showcase" class="py-12">
      <div class="text-center mb-8">
        <h2 class="text-3xl font-display font-bold nier-heading mb-4">
          Component Showcase
        </h2>
        <p class="text-lg text-nier-gray dark:text-nier-silver">
          Explore the complete NieR: Automata themed component library
        </p>
      </div>

      <NierThemeShowcase />
    </section>

    <!-- Features Section -->
    <section class="py-16 border-t border-nier-light-gray dark:border-nier-gray/30">
      <h2 class="text-3xl font-display font-bold nier-heading text-center mb-12">
        System Features
      </h2>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div class="text-center">
          <div class="w-20 h-20 mx-auto mb-4 bg-nier-gradient-digital rounded-2xl flex items-center justify-center text-3xl animate-pulse">
            🔐
          </div>
          <h3 class="text-xl font-semibold mb-2">Secure Evidence Chain</h3>
          <p class="text-sm text-nier-gray dark:text-nier-silver">
            Military-grade encryption and blockchain verification for all case evidence
          </p>
        </div>

        <div class="text-center">
          <div class="w-20 h-20 mx-auto mb-4 bg-nier-gradient-crimson rounded-2xl flex items-center justify-center text-3xl">
            🧠
          </div>
          <h3 class="text-xl font-semibold mb-2">AI-Powered Analysis</h3>
          <p class="text-sm text-nier-gray dark:text-nier-silver">
            Advanced pattern recognition and predictive analytics for case insights
          </p>
        </div>

        <div class="text-center">
          <div class="w-20 h-20 mx-auto mb-4 bg-nier-gradient-gold rounded-2xl flex items-center justify-center text-3xl">
            ⚡
          </div>
          <h3 class="text-xl font-semibold mb-2">Real-time Collaboration</h3>
          <p class="text-sm text-nier-gray dark:text-nier-silver">
            Seamless team coordination with live updates and shared workspaces
          </p>
        </div>
      </div>
    </section>
  </main>

  <!-- Footer -->
  <footer class="mt-24 py-8 border-t border-nier-light-gray dark:border-nier-gray/30">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
      <p class="text-sm text-nier-gray dark:text-nier-silver mb-2">
        YoRHa Legal System v2.0 • Built with SvelteKit + Bits UI + Melt UI
      </p>
      <p class="text-xs text-nier-light-gray dark:text-nier-gray">
        © 2025 YoRHa • For the Glory of Mankind
      </p>
    </div>
  </footer>

  <!-- AI Assistant -->
  <NierAIAssistant
    bind:isOpen={showAIAssistant}
    onClose={() => showAIAssistant = false}
    caseContext={sampleCases[0]}
  />

  <!-- Floating Action Button -->
  <button
    onclick={() => showAIAssistant = true}
    class="fixed bottom-6 right-6 w-14 h-14 bg-nier-gradient-digital rounded-full flex items-center justify-center shadow-lg hover:shadow-xl transform hover:scale-110 nier-transition animate-digital-glow"
    aria-label="Open AI Assistant"
  >
    <span class="text-2xl">🤖</span>
  </button>
</div>

<style>
  /* @unocss-include */
  /* Page-specific animations */
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

  .animate-fade-in {
    animation: fade-in 0.6s ease-out forwards;
  }
</style>
