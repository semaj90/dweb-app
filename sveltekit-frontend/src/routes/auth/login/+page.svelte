<!--
  Login Page - Legal AI Platform
  Enhanced with GPU security analysis and XState integration
-->
<script lang="ts">
  import LoginForm from '$lib/components/auth/LoginForm.svelte';
  import * as Card from '$lib/components/ui/card';
  import { Shield } from 'lucide-svelte';
  
  interface Props {
    data: any;
  }
  
  let { data }: Props = $props();
</script>

<svelte:head>
  <title>Sign In - Legal AI Platform</title>
  <meta name="description" content="Secure login for legal professionals using GPU-enhanced authentication" />
</svelte:head>

<div class="min-h-screen flex items-center justify-center bg-background px-4">
  <div class="w-full max-w-md space-y-6">
    <!-- Header -->
    <div class="text-center space-y-2">
      <div class="flex items-center justify-center mb-4">
        <Shield class="h-10 w-10 text-primary mr-3" />
        <h1 class="text-3xl font-bold">Legal AI Platform</h1>
      </div>
      <p class="text-muted-foreground">
        Secure access for legal professionals
      </p>
    </div>

    <!-- Login Form -->
    <LoginForm 
      {data} 
      enableGPUAuth={true}
      showRegistration={true}
    />

    <!-- Footer -->
    <div class="text-center text-sm text-muted-foreground">
      <p>
        Protected by GPU-enhanced security analysis
      </p>
      <p class="mt-2">
        <a href="/legal/privacy" class="hover:underline">Privacy Policy</a> • 
        <a href="/legal/terms" class="hover:underline">Terms of Service</a> • 
        <a href="/support" class="hover:underline">Support</a>
      </p>
    </div>
  </div>
</div>