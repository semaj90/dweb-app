<script lang="ts">
  import DetectiveBoard from '$lib/components/detective/DetectiveBoard.svelte';
  import CaseForm from '$lib/components/forms/CaseForm.svelte';
  import { getAuthContext } from "$lib/stores/auth";
  import { onMount } from 'svelte';
  
  const auth = getAuthContext();
  
  let currentView = 'board'; // 'board' | 'create-case' | 'auth-demo'
  let mounted = false;
  
  // Sample evidence data for the detective board
  let sampleEvidence = [
    {
      id: 'evidence-1',
      title: 'Security Camera Footage',
      fileName: 'camera_feed_001.mp4',
      evidenceType: 'video',
      status: 'new',
      fileSize: 45678912,
      createdAt: new Date('2024-01-15T10:30:00'),
      uploadedAt: new Date('2024-01-15T10:30:00'),
      updatedAt: new Date('2024-01-15T10:30:00'),
      description: 'Video shows suspect entering building at 10:23 PM wearing dark clothing',
      tags: ['surveillance', 'timestamp', 'suspect-entry'],
      hash: 'abc123',
      thumbnailUrl: '/api/thumbnails/evidence-1.jpg',
      aiSummary: 'Video shows suspect entering building at 10:23 PM wearing dark clothing'
    },
    {
      id: 'evidence-2', 
      title: 'Witness Statement - John Doe',
      fileName: 'witness_statement_001.pdf',
      evidenceType: 'document',
      status: 'reviewing',
      fileSize: 1234567,
      createdAt: new Date('2024-01-16T14:20:00'),
      uploadedAt: new Date('2024-01-16T14:20:00'),
      updatedAt: new Date('2024-01-16T14:20:00'),
      description: 'Witness observed suspicious activity near the crime scene around 10:15 PM',
      tags: ['witness', 'testimony', 'timeline'],
      aiSummary: 'Witness observed suspicious activity near the crime scene around 10:15 PM'
    },
    {
      id: 'evidence-3',
      title: 'Crime Scene Photos',
      fileName: 'scene_photos_batch1.zip', 
      evidenceType: 'image',
      status: 'approved',
      fileSize: 23456789,
      createdAt: new Date('2024-01-17T09:15:00'),
      uploadedAt: new Date('2024-01-17T09:15:00'),
      updatedAt: new Date('2024-01-17T09:15:00'),
      tags: ['crime-scene', 'forensics', 'photography'],
      hash: 'xyz789',
      thumbnailUrl: '/api/thumbnails/evidence-3.jpg'
    },
    {
      id: 'evidence-4',
      title: 'Phone Call Recording',
      fileName: 'call_recording_suspect.mp3',
      evidenceType: 'audio',
      status: 'new',
      fileSize: 5678901,
      createdAt: new Date('2024-01-18T11:45:00'),
      uploadedAt: new Date('2024-01-18T11:45:00'),
      updatedAt: new Date('2024-01-18T11:45:00'),
      description: 'Audio recording contains discussion about meeting location',
      tags: ['wiretap', 'conversation', 'evidence'],
      aiSummary: 'Audio recording contains discussion about meeting location'
    },
    {
      id: 'evidence-5',
      title: 'Financial Records',
      fileName: 'bank_statements_2024.pdf',
      evidenceType: 'document',
      status: 'reviewing',
      fileSize: 3456789,
      createdAt: new Date('2024-01-19T16:30:00'),
      uploadedAt: new Date('2024-01-19T16:30:00'),
      updatedAt: new Date('2024-01-19T16:30:00'),
      description: 'Bank statements showing suspicious transactions',
      tags: ['financial', 'money-laundering', 'transactions'],
      aiSummary: 'Bank statements reveal suspicious large cash deposits totaling $50,000 over 3 months'
    },
    {
      id: 'evidence-6',
      title: 'DNA Analysis Report',
      fileName: 'dna_report_lab_001.pdf',
      evidenceType: 'document',
      status: 'approved',
      fileSize: 987654,
      createdAt: new Date('2024-01-20T09:00:00'),
      uploadedAt: new Date('2024-01-20T09:00:00'),
      updatedAt: new Date('2024-01-20T09:00:00'),
      description: 'Forensic DNA analysis results',
      tags: ['dna', 'forensics', 'lab-report'],
      hash: 'dna456',
      aiSummary: 'DNA evidence confirms 99.7% match with suspect sample collected at scene'
}
  ];
  
  onMount(() => {
    mounted = true;
  });
  
  function handleCaseFormSuccess(event: CustomEvent) {
    console.log('Case created/updated:', event.detail);
    currentView = 'board';
}
  function handleCaseFormCancel() {
    currentView = 'board';
}
  // Demo collaboration simulation
  function simulateCollaboration() {
    if (!mounted) return;
    
    // Simulate other users joining
    const mockUsers = [
      { id: 'user-1', name: 'Sarah Chen', email: 'sarah.chen@prosecutor.office' },
      { id: 'user-2', name: 'Mike Rodriguez', email: 'mike.rodriguez@prosecutor.office' },
      { id: 'user-3', name: 'Dr. Lisa Kim', email: 'lisa.kim@forensics.office' }
    ];
    
    // Add one random user as if they joined
    const randomUser = mockUsers[Math.floor(Math.random() * mockUsers.length)];
    
    // Show notification
    if (typeof window !== 'undefined') {
      // Create a temporary notification
      const notification = document.createElement('div');
      notification.className = 'fixed top-20 right-4 bg-blue-50 border border-blue-200 text-blue-800 px-4 py-3 rounded-lg shadow-lg z-50 transition-all duration-300';
      notification.innerHTML = `
        <div class="space-y-4">
          <div class="space-y-4">
            ${randomUser.name.charAt(0)}
          </div>
          <span class="space-y-4">${randomUser.name} joined the case</span>
        </div>
      `;
      
      document.body.appendChild(notification);
      
      // Remove after 3 seconds
      setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
          document.body.removeChild(notification);
        }, 300);
      }, 3000);
}
    console.log('🚀 Collaboration demo:', randomUser.name, 'joined the case');
}
</script>

<svelte:head>
  <title>Detective Mode - Google Slides for Prosecutors</title>
  <meta name="description" content="Advanced evidence management and case visualization tool for legal professionals" />
</svelte:head>

{#if mounted}
  <!-- Navigation Bar -->
  <nav class="space-y-4">
    <div class="space-y-4">
      <div class="space-y-4">
        <div class="space-y-4">
          <div class="space-y-4">
            <div class="space-y-4">
              <i class="space-y-4"></i>
            </div>
            <h1 class="space-y-4">Detective Mode</h1>
          </div>
          
          <div class="space-y-4">
            <button 
              onclick={() => currentView = 'board'}
              class="space-y-4"
            >
              <i class="space-y-4"></i>
              Evidence Board
            </button>
            
            <button 
              onclick={() => currentView = 'create-case'}
              class="space-y-4"
            >
              <i class="space-y-4"></i>
              Create Case
            </button>
            
            <button 
              onclick={() => currentView = 'auth-demo'}
              class="space-y-4"
            >
              <i class="space-y-4"></i>
              Auth Demo
            </button>
            
            <button 
              onclick={() => simulateCollaboration()}
              class="space-y-4"
            >
              <i class="space-y-4"></i>
              Demo Collab
            </button>
          </div>
        </div>
        
        <div class="space-y-4">
          <!-- Demo Status Badge -->
          <div class="space-y-4">
            <div class="space-y-4"></div>
            <span class="space-y-4">Live Demo</span>
          </div>
          
          {#if $auth.isAuthenticated}
            <div class="space-y-4">
              <span class="space-y-4">Welcome, {$auth.user?.name || $auth.user?.email}</span>
              <button 
                onclick={() => auth.logout()}
                class="space-y-4"
              >
                Logout
              </button>
            </div>
          {:else}
            <div class="space-y-4">
              <i class="space-y-4"></i>
              <span class="space-y-4">Not authenticated</span>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </nav>

  <!-- Main Content -->
  <main class="space-y-4">
    {#if currentView === 'board'}
      <DetectiveBoard 
        caseId="demo-case-001" 
        evidence={sampleEvidence} 
      />
    {:else if currentView === 'create-case'}
      <div class="space-y-4">
        <CaseForm 
          onsuccess={handleCaseFormSuccess}
          oncancel={handleCaseFormCancel}
        />
      </div>
    {:else if currentView === 'auth-demo'}
      <div class="space-y-4">
        <div class="space-y-4">
          <div class="space-y-4">
            <h2 class="space-y-4">Authentication Demo</h2>
            <p class="space-y-4">
              This demonstrates the Svelte Context API alternative to $lib for authentication.
            </p>
            
            <div class="space-y-4">
              <div class="space-y-4">
                <h3 class="space-y-4">Auth State</h3>
                <pre class="space-y-4">{JSON.stringify($auth, null, 2)}</pre>
              </div>
              
              <div class="space-y-4">
                <button 
                  onclick={() => auth.login('test@example.com', 'password123')}
                  class="space-y-4"
                  disabled={$auth.isLoading}
                >
                  Mock Login
                </button>
                
                <button 
                  onclick={() => auth.logout()}
                  class="space-y-4"
                  disabled={!$auth.isAuthenticated}
                >
                  Logout
                </button>
                
                <button 
                  onclick={() => auth.checkAuth()}
                  class="space-y-4"
                  disabled={$auth.isLoading}
                >
                  Check Auth
                </button>
              </div>
            </div>
            
            <div class="space-y-4">
              <h3 class="space-y-4">Features Demonstrated</h3>
              <div class="space-y-4">
                <div class="space-y-4">
                  <h4 class="space-y-4">✅ UnoCSS Only</h4>
                  <p class="space-y-4">PicoCSS removed, using UnoCSS utility classes throughout</p>
                </div>
                
                <div class="space-y-4">
                  <h4 class="space-y-4">✅ Drag & Drop</h4>
                  <p class="space-y-4">Evidence cards with svelte-dnd-action in 3-column layout</p>
                </div>
                
                <div class="space-y-4">
                  <h4 class="space-y-4">✅ Context Menus</h4>
                  <p class="space-y-4">Right-click menus using Melt UI primitives</p>
                </div>
                
                <div class="space-y-4">
                  <h4 class="space-y-4">✅ Superforms + Zod</h4>
                  <p class="space-y-4">Type-safe forms with validation and error handling</p>
                </div>
                
                <div class="space-y-4">
                  <h4 class="space-y-4">✅ Auth Context</h4>
                  <p class="space-y-4">Svelte Context API alternative to $lib for global state</p>
                </div>
                
                <div class="space-y-4">
                  <h4 class="space-y-4">✅ File Upload</h4>
                  <p class="space-y-4">Progress bar with drag/drop support</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    {/if}
  </main>

{:else}
  <!-- Loading Screen -->
  <div class="space-y-4">
    <div class="space-y-4">
      <div class="space-y-4"></div>
      <p class="space-y-4">Loading Detective Mode...</p>
      <p class="space-y-4">Initializing Google Slides-like interface</p>
    </div>
  </div>
{/if}
