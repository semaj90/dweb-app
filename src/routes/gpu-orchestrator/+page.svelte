<!--
  GPU Processing Orchestrator Page
  Main interface for the unified AI processing system
-->

<script lang="ts">
  import GPUProcessingOrchestrator from '$lib/components/GPUProcessingOrchestrator.svelte';
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import type { DocumentInput } from '$lib/state/gpu-processing-machine';

  // Sample documents for testing
  let sampleDocuments: DocumentInput[] = [
    {
      documentId: 'contract_001',
      title: 'Software License Agreement',
      content: `This Software License Agreement ("Agreement") is entered into as of the date of last signature below ("Effective Date") between Company A, a corporation organized under the laws of Delaware ("Licensor") and Company B, a limited liability company organized under the laws of California ("Licensee"). WHEREAS, Licensor has developed certain proprietary software and related documentation; WHEREAS, Licensee desires to obtain a license to use such software; NOW, THEREFORE, in consideration of the mutual covenants and agreements contained herein, the parties agree as follows: 1. GRANT OF LICENSE. Subject to the terms and conditions of this Agreement, Licensor hereby grants to Licensee a non-exclusive, non-transferable license to use the Software solely for Licensee's internal business purposes.`,
      options: {
        processType: 'full',
        priority: 8,
        timeout: 30000,
        retries: 3,
        batchSize: 1
      }
    },
    {
      documentId: 'lease_001',
      title: 'Commercial Lease Agreement',
      content: `This Commercial Lease Agreement ("Lease") is made this 15th day of January, 2024, between ABC Properties LLC, a Delaware limited liability company ("Landlord"), and XYZ Corporation, a Nevada corporation ("Tenant"). The Landlord hereby leases to Tenant and Tenant hereby leases from Landlord the premises described as Suite 200, located at 123 Business Park Drive, Anytown, State 12345 ("Premises"). The term of this Lease shall be for a period of five (5) years, commencing on February 1, 2024, and ending on January 31, 2029, unless sooner terminated in accordance with the provisions hereof. The monthly base rent shall be Twenty-Five Thousand Dollars ($25,000) per month, payable in advance on the first day of each calendar month.`,
      options: {
        processType: 'full',
        priority: 7,
        timeout: 30000,
        retries: 3,
        batchSize: 1
      }
    },
    {
      documentId: 'employment_001',
      title: 'Employment Contract',
      content: `This Employment Agreement ("Agreement") is entered into on March 1, 2024, between Tech Innovations Inc., a Delaware corporation ("Company"), and John Smith ("Employee"). Employee shall serve as Senior Software Engineer reporting to the Chief Technology Officer. The initial base salary shall be One Hundred Fifty Thousand Dollars ($150,000) per year, paid in accordance with Company's standard payroll practices. Employee shall be eligible to participate in Company's employee benefit plans and programs as they may exist from time to time, subject to the terms and conditions of such plans. This Agreement shall commence on March 15, 2024, and shall continue until terminated by either party in accordance with the terms hereof.`,
      options: {
        processType: 'full',
        priority: 6,
        timeout: 30000,
        retries: 3,
        batchSize: 1
      }
    }
  ];

  // Page metadata
  let pageTitle = 'GPU Processing Orchestrator';
  let pageDescription = 'Advanced AI document processing with GPU acceleration and XState orchestration';

  onMount(() => {
    // Set page title
    document.title = `${pageTitle} | Legal AI System`;
  });
</script>

<svelte:head>
  <title>{pageTitle} | Legal AI System</title>
  <meta name="description" content={pageDescription} />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  
  <!-- Open Graph / Facebook -->
  <meta property="og:type" content="website" />
  <meta property="og:url" content="{$page.url}" />
  <meta property="og:title" content={pageTitle} />
  <meta property="og:description" content={pageDescription} />
  
  <!-- Twitter -->
  <meta property="twitter:card" content="summary_large_image" />
  <meta property="twitter:url" content="{$page.url}" />
  <meta property="twitter:title" content={pageTitle} />
  <meta property="twitter:description" content={pageDescription} />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
  <!-- Navigation Breadcrumb -->
  <nav class="bg-white shadow-sm border-b">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <div class="flex items-center">
          <a href="/" class="text-xl font-bold text-gray-900 hover:text-blue-600 transition-colors">
            ⚖️ Legal AI
          </a>
          <span class="mx-2 text-gray-400">/</span>
          <span class="text-gray-600">GPU Orchestrator</span>
        </div>
        
        <div class="flex items-center space-x-4">
          <a 
            href="/ai-workspace" 
            class="text-gray-600 hover:text-blue-600 transition-colors text-sm font-medium"
          >
            🧠 AI Workspace
          </a>
          <a 
            href="/system-monitor" 
            class="text-gray-600 hover:text-blue-600 transition-colors text-sm font-medium"
          >
            📊 System Monitor
          </a>
          <a 
            href="/api-docs" 
            class="text-gray-600 hover:text-blue-600 transition-colors text-sm font-medium"
          >
            📚 API Docs
          </a>
        </div>
      </div>
    </div>
  </nav>

  <!-- Main Content -->
  <main class="py-8">
    <GPUProcessingOrchestrator 
      documents={sampleDocuments}
      autoStart={false}
      maxConcurrent={5}
    />
  </main>

  <!-- Footer -->
  <footer class="bg-white border-t mt-12">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div>
          <h3 class="text-lg font-semibold text-gray-900 mb-4">🎮 GPU Processing</h3>
          <p class="text-gray-600 text-sm">
            Advanced document processing powered by WebGPU acceleration, SIMD optimization, 
            and intelligent orchestration using XState state machines.
          </p>
        </div>
        
        <div>
          <h3 class="text-lg font-semibold text-gray-900 mb-4">🔗 Architecture</h3>
          <ul class="space-y-2 text-sm text-gray-600">
            <li>• Go SIMD Service (JSON parsing)</li>
            <li>• Node.js GPU Service (WebGPU)</li>
            <li>• XState Orchestration</li>
            <li>• gRPC Communication</li>
            <li>• WGSL Compute Shaders</li>
          </ul>
        </div>
        
        <div>
          <h3 class="text-lg font-semibold text-gray-900 mb-4">📊 Features</h3>
          <ul class="space-y-2 text-sm text-gray-600">
            <li>• Real-time processing monitoring</li>
            <li>• Automatic retry and error handling</li>
            <li>• Concurrent batch processing</li>
            <li>• Performance metrics & analytics</li>
            <li>• Service health monitoring</li>
          </ul>
        </div>
      </div>
      
      <div class="border-t mt-8 pt-8 flex items-center justify-between">
        <div class="text-sm text-gray-500">
          © 2024 Legal AI System. Powered by SvelteKit 2, XState, WebGPU & Go.
        </div>
        <div class="flex space-x-4">
          <a href="/docs" class="text-sm text-gray-500 hover:text-blue-600 transition-colors">
            Documentation
          </a>
          <a href="/support" class="text-sm text-gray-500 hover:text-blue-600 transition-colors">
            Support
          </a>
        </div>
      </div>
    </div>
  </footer>
</div>

<style>
  /* Custom scrollbar for webkit browsers */
  :global(html) {
    scroll-behavior: smooth;
  }
  
  :global(.gpu-processing-orchestrator) {
    animation: fadeIn 0.5s ease-out;
  }
  
  @keyframes fadeIn {
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