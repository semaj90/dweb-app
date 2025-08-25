<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import Button from '$lib/components/ui/Button.svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import UniversalSearchBar from '$lib/components/search/UniversalSearchBar.svelte';
  import type { SearchResult } from '$lib/components/search/types.js';

  let { data }: {
    data: {
      userId: string | null;
      sessionId: string | null;
      email: string | null;
      isAuthenticated: boolean;
    }
  } = $props();

  // Modern website state
  let currentSlide = $state(0);
  let stats = $state({
    cases: '2,847',
    evidence: '18,592',
    users: '324',
    accuracy: '99.7%'
  });

  // Hero carousel slides
  let heroSlides = $state([
    {
      title: 'AI-Powered Legal Research',
      subtitle: 'Advanced case analysis with machine learning',
      image: '/api/placeholder/800/400',
      cta: 'Start Research'
    },
    {
      title: 'Evidence Management',
      subtitle: 'Secure chain of custody with blockchain technology',
      image: '/api/placeholder/800/400',
      cta: 'View Evidence'
    },
    {
      title: 'Person of Interest Tracking',
      subtitle: 'Comprehensive criminal database integration',
      image: '/api/placeholder/800/400',
      cta: 'Search POI'
    }
  ]);

  // Featured services
  let services = $state([
    {
      icon: '🔍',
      title: 'Case Investigation',
      description: 'AI-powered case analysis and evidence correlation',
      link: '/cases'
    },
    {
      icon: '👤',
      title: 'Person Tracking',
      description: 'Comprehensive person of interest database',
      link: '/poi'
    },
    {
      icon: '📄',
      title: 'Document Analysis',
      description: 'OCR and semantic analysis of legal documents',
      link: '/documents'
    },
    {
      icon: '⚖️',
      title: 'Legal Research',
      description: 'Access to precedents and legal databases',
      link: '/research'
    }
  ]);

  onMount(() => {
    // Auto-advance hero carousel
    setInterval(() => {
      currentSlide = (currentSlide + 1) % heroSlides.length;
    }, 5000);
  });

  function handleSearch(event: CustomEvent<{ query: string; results: SearchResult[] }>) {
    const { query, results } = event.detail;
    console.log('Search performed:', query, results.length, 'results');
    // Could navigate to search results page
  }

  function handleSearchSelect(event: CustomEvent<{ result: SearchResult }>) {
    const { result } = event.detail;
    console.log('Selected result:', result);

    // Navigate based on result type
    switch (result.type) {
      case 'case':
        goto(`/cases/${result.id}`);
        break;
      case 'criminal':
        goto(`/poi/${result.id}`);
        break;
      case 'evidence':
        goto(`/evidence/${result.id}`);
        break;
      default:
        goto(`/search?q=${encodeURIComponent(result.title)}`);
    }
  }

  // Missing variables used in auth handler
  let email = '';
  let password = '';
  let firstName = '';
  let lastName = '';
  let loading = false;
  let error = '';
  let message = '';
  let isLogin = true;

  // Missing systemInfo used in template
  let systemInfo = {
    uptime: '72h 14m',
    activeServices: '8/9',
    lastSync: '2m ago'
  };

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
  <title>Legal AI Platform - Advanced Case Management & Evidence Analysis</title>
  <meta name="description" content="Modern legal AI platform for case management, evidence analysis, and legal research with advanced AI-powered search capabilities." />
</svelte:head>

<!-- Modern Legal AI Platform Homepage -->
<div class="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
  <!-- Header -->
  <header class="bg-white shadow-lg">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between items-center py-6">
        <div class="flex items-center">
          <div class="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center mr-4">
            <span class="text-white font-bold text-lg">⚖️</span>
          </div>
          <div>
            <h1 class="text-2xl font-bold text-gray-900">Legal AI Platform</h1>
            <p class="text-sm text-gray-500">Advanced Legal Case Management</p>
          </div>
        </div>

        <nav class="hidden md:flex items-center space-x-8">
          <a href="/cases" class="text-gray-700 hover:text-blue-600 font-medium">Cases</a>
          <a href="/evidence" class="text-gray-700 hover:text-blue-600 font-medium">Evidence</a>
          <a href="/poi" class="text-gray-700 hover:text-blue-600 font-medium">Person Tracking</a>
          <a href="/research" class="text-gray-700 hover:text-blue-600 font-medium">Legal Research</a>
          {#if data.isAuthenticated}
            <a href="/dashboard" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">Dashboard</a>
          {:else}
            <a href="/auth/login" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">Sign In</a>
          {/if}
        </nav>
      </div>
    </div>
  </header>

  <!-- Hero Section with Search -->
  <section class="relative bg-white py-20">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="text-center mb-12">
        <h2 class="text-5xl font-bold text-gray-900 mb-6">
          AI-Powered Legal Intelligence
        </h2>
        <p class="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
          Streamline case management, evidence analysis, and legal research with advanced artificial intelligence. Search across cases, persons of interest, and legal documents instantly.
        </p>

        <!-- Featured Search Bar -->
        <div class="max-w-4xl mx-auto mb-8">
          <UniversalSearchBar
            placeholder="Search cases, persons of interest, evidence, or documents..."
            showRecentSearches={true}
            showTrendingSearches={true}
            enableAISuggestions={true}
            on:search={handleSearch}
            on:select={handleSearchSelect}
          />
        </div>

        <p class="text-sm text-gray-500">
          Try searching: "fraud investigation", "John Smith", "contract analysis"
        </p>
      </div>
    </div>
  </section>

  <!-- Services Grid -->
  <section class="py-16 bg-gray-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="text-center mb-12">
        <h3 class="text-3xl font-bold text-gray-900 mb-4">Comprehensive Legal Solutions</h3>
        <p class="text-lg text-gray-600">Everything you need for modern legal case management</p>
      </div>

      <div class="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
        {#each services as service}
          <div class="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300 border border-gray-100 hover:border-blue-200">
            <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
              <span class="text-2xl">{service.icon}</span>
            </div>
            <h4 class="text-xl font-semibold text-gray-900 mb-2">{service.title}</h4>
            <p class="text-gray-600 mb-4">{service.description}</p>
            <a
              href={service.link}
              class="text-blue-600 hover:text-blue-700 font-medium flex items-center"
            >
              Learn more →
            </a>
          </div>
        {/each}
      </div>
    </div>
  </section>

  <!-- Statistics Section -->
  <section class="py-16 bg-blue-600">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="text-center mb-12">
        <h3 class="text-3xl font-bold text-white mb-4">Trusted by Legal Professionals</h3>
        <p class="text-xl text-blue-100">Our platform delivers results that matter</p>
      </div>

      <div class="grid md:grid-cols-4 gap-8">
        <div class="text-center">
          <div class="text-4xl font-bold text-white mb-2">{stats.cases}</div>
          <div class="text-blue-100">Cases Managed</div>
        </div>
        <div class="text-center">
          <div class="text-4xl font-bold text-white mb-2">{stats.evidence}</div>
          <div class="text-blue-100">Evidence Files Processed</div>
        </div>
        <div class="text-center">
          <div class="text-4xl font-bold text-white mb-2">{stats.users}</div>
          <div class="text-blue-100">Legal Professionals</div>
        </div>
        <div class="text-center">
          <div class="text-4xl font-bold text-white mb-2">{stats.accuracy}</div>
          <div class="text-blue-100">AI Accuracy Rate</div>
        </div>
      </div>
    </div>
  </section>

  <!-- Hero Carousel -->
  <section class="py-16 bg-white">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl overflow-hidden shadow-2xl">
        <div class="relative h-96">
          {#each heroSlides as slide, index}
            <div
              class="absolute inset-0 transition-opacity duration-1000 {index === currentSlide ? 'opacity-100' : 'opacity-0'}"
            >
              <div class="flex h-full">
                <div class="flex-1 flex items-center p-12 text-white">
                  <div>
                    <h3 class="text-4xl font-bold mb-4">{slide.title}</h3>
                    <p class="text-xl mb-6 text-blue-100">{slide.subtitle}</p>
                    <button
                      class="bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
                    >
                      {slide.cta}
                    </button>
                  </div>
                </div>
                <div class="flex-1 flex items-center justify-center">
                  <div class="w-80 h-48 bg-white bg-opacity-20 rounded-lg flex items-center justify-center">
                    <span class="text-6xl text-white opacity-50">📊</span>
                  </div>
                </div>
              </div>
            </div>
          {/each}

          <!-- Carousel Indicators -->
          <div class="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2">
            {#each heroSlides as _, index}
              <button
                class="w-3 h-3 rounded-full transition-all duration-300 {index === currentSlide ? 'bg-white' : 'bg-white bg-opacity-40'}"
                on:click={() => currentSlide = index}
                aria-label="Go to slide {index + 1}"
              ></button>
            {/each}
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- CTA Section -->
  <section class="py-16 bg-gray-900">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
      <h3 class="text-3xl font-bold text-white mb-4">Ready to Transform Your Legal Practice?</h3>
      <p class="text-xl text-gray-300 mb-8">Join hundreds of legal professionals already using our platform</p>
      <div class="flex justify-center space-x-4">
        {#if data.isAuthenticated}
          <a href="/dashboard" class="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors">
            Go to Dashboard
          </a>
        {:else}
          <a href="/auth/register" class="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors">
            Start Free Trial
          </a>
          <a href="/auth/login" class="border-2 border-white text-white px-8 py-3 rounded-lg font-semibold hover:bg-white hover:text-gray-900 transition-colors">
            Sign In
          </a>
        {/if}
      </div>
    </div>
  </section>

  <!-- Footer -->
  <footer class="bg-gray-50 py-12 border-t border-gray-200">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="grid md:grid-cols-4 gap-8">
        <div>
          <div class="flex items-center mb-4">
            <div class="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center mr-3">
              <span class="text-white font-bold">⚖️</span>
            </div>
            <span class="text-xl font-bold text-gray-900">Legal AI</span>
          </div>
          <p class="text-gray-600">Advanced legal technology for modern law practices.</p>
        </div>

        <div>
          <h4 class="font-semibold text-gray-900 mb-4">Platform</h4>
          <ul class="space-y-2">
            <li><a href="/cases" class="text-gray-600 hover:text-blue-600">Case Management</a></li>
            <li><a href="/evidence" class="text-gray-600 hover:text-blue-600">Evidence Analysis</a></li>
            <li><a href="/poi" class="text-gray-600 hover:text-blue-600">Person Tracking</a></li>
            <li><a href="/research" class="text-gray-600 hover:text-blue-600">Legal Research</a></li>
          </ul>
        </div>

        <div>
          <h4 class="font-semibold text-gray-900 mb-4">System</h4>
          <ul class="space-y-2">
            <li><a href="/all-routes" class="text-gray-600 hover:text-blue-600">All Features</a></li>
            <li><a href="/yorha" class="text-gray-600 hover:text-blue-600">Advanced Interface</a></li>
            <li><a href="/api-docs" class="text-gray-600 hover:text-blue-600">API Documentation</a></li>
          </ul>
        </div>

        <div>
          <h4 class="font-semibold text-gray-900 mb-4">System Status</h4>
          <ul class="space-y-2 text-sm">
            <li class="text-gray-600">Uptime: <span class="text-green-600 font-medium">{systemInfo.uptime}</span></li>
            <li class="text-gray-600">Services: <span class="text-blue-600 font-medium">{systemInfo.activeServices}</span></li>
            <li class="text-gray-600">Last Sync: <span class="text-gray-500">{systemInfo.lastSync}</span></li>
          </ul>
        </div>
      </div>

      <div class="border-t border-gray-200 mt-8 pt-8 text-center">
        <p class="text-gray-500">&copy; 2024 Legal AI Platform. Advanced case management and evidence analysis.</p>
      </div>
    </div>
  </footer>
</div>

<style>
  /* Modern website styling using Tailwind utilities */
  /* Additional responsive behaviors handled by Tailwind classes in template */
</style>