<!--
  Registration Page - Legal AI Platform
  Enhanced with GPU validation and legal professional verification
-->
<script lang="ts">
  import RegisterForm from '$lib/components/auth/RegisterForm.svelte';
  import * as Card from '$lib/components/ui/card';
  import { Shield, Scale } from 'lucide-svelte';
  
  interface Props {
    data: any;
  }
  
  let { data }: Props = $props();
</script>

<svelte:head>
  <title>Register - Legal AI Platform</title>
  <meta name="description" content="Register as a legal professional with GPU-enhanced validation" />
</svelte:head>

<div class="min-h-screen flex items-center justify-center bg-background px-4 py-8">
  <div class="w-full max-w-2xl space-y-6">
    <!-- Header -->
    <div class="text-center space-y-2">
      <div class="flex items-center justify-center mb-4">
        <Shield class="h-10 w-10 text-primary mr-3" />
        <h1 class="text-3xl font-bold">Legal AI Platform</h1>
      </div>
      <h2 class="text-xl font-semibold flex items-center justify-center gap-2">
        <Scale class="h-5 w-5" />
        Professional Registration
      </h2>
      <p class="text-muted-foreground max-w-lg mx-auto">
        Create your account to access AI-powered legal tools and case management features. 
        All registrations are verified for legal professional credentials.
      </p>
    </div>

    <!-- Registration Form -->
    <RegisterForm 
      {data} 
      enableGPUValidation={true}
      showLogin={true}
    />

    <!-- Footer -->
    <div class="text-center text-sm text-muted-foreground space-y-2">
      <p class="flex items-center justify-center gap-2">
        <Shield class="h-4 w-4" />
        Protected by AI-enhanced credential verification
      </p>
      <div class="space-y-1">
        <p>
          By registering, you agree to our 
          <a href="/legal/terms" class="text-primary hover:underline">Terms of Service</a> and 
          <a href="/legal/privacy" class="text-primary hover:underline">Privacy Policy</a>
        </p>
        <p>
          Need help? <a href="/support" class="text-primary hover:underline">Contact Support</a>
        </p>
      </div>
    </div>
  </div>
</div>