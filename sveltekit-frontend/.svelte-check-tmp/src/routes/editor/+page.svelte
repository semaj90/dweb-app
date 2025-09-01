<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import ReportEditor from "$lib/components/editor/ReportEditor.svelte";
  import { report, reportActions } from "$lib/stores/report";
  
  // Sample evidence data for demonstration
  const sampleEvidence = [
    {
      id: '1',
      type: 'document' as const,
      title: 'Police Report #2024-001',
      description: 'Initial incident report filed by responding officer',
      url: '/docs/police-report-001.pdf',
      tags: ['official', 'police', 'incident'],
      metadata: { format: 'PDF', size: 1024000 },
      createdAt: new Date('2024-01-15'),
      updatedAt: new Date('2024-01-15')
    },
    {
      id: '2',
      type: 'image' as const,
      title: 'Crime Scene Photo 1',
      description: 'Overview of the crime scene from north entrance',
      url: 'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400',
      tags: ['evidence', 'scene', 'photography'],
      metadata: { format: 'JPG', size: 2048000 },
      createdAt: new Date('2024-01-16'),
      updatedAt: new Date('2024-01-16')
    },
    {
      id: '3',
      type: 'video' as const,
      title: 'Security Camera Footage',
      description: 'CCTV footage from building entrance showing suspect',
      url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
      tags: ['surveillance', 'suspect', 'timestamp'],
      metadata: { format: 'MP4', size: 15728640, duration: '00:02:14' },
      createdAt: new Date('2024-01-16'),
      updatedAt: new Date('2024-01-16')
    },
    {
      id: '4',
      type: 'link' as const,
      title: 'Forensic Lab Report',
      description: 'DNA analysis results from evidence collected at scene',
      url: 'https://example.com/forensic-report-2024-001',
      tags: ['forensics', 'dna', 'lab-results'],
      metadata: { format: 'HTML', size: 128000 },
      createdAt: new Date('2024-01-18'),
      updatedAt: new Date('2024-01-18')
    },
    {
      id: '5',
      type: 'audio' as const,
      title: 'Witness Interview Recording',
      description: 'Audio recording of witness statement taken on January 17th',
      url: '/audio/witness-interview-001.mp3',
      tags: ['witness', 'interview', 'statement'],
      metadata: { format: 'MP3', size: 5242880, duration: '00:12:34' },
      createdAt: new Date('2024-01-17'),
      updatedAt: new Date('2024-01-17')
    },
    {
      id: '6',
      type: 'image' as const,
      title: 'Evidence Item A-1',
      description: 'Photograph of recovered weapon with measurement scale',
      url: 'https://images.unsplash.com/photo-1568667256549-094345857637?w=400',
      tags: ['weapon', 'evidence', 'measurement'],
      metadata: { format: 'JPG', size: 1536000 },
      createdAt: new Date('2024-01-16'),
      updatedAt: new Date('2024-01-16')
}
  ];
  
  onMount(() => {
    // Load sample data for demonstration
    const sampleReport = {
      id: 'demo-report-001',
      title: 'Case #2024-001: Downtown Incident Investigation',
      content: `
        <h2>Executive Summary</h2>
        <p>This report details the investigation of the incident that occurred on January 15th, 2024, at approximately 10:30 PM in the downtown area. The investigation involved multiple agencies and utilized various forms of evidence to establish a comprehensive understanding of the events.</p>
        
        <h2>Initial Response</h2>
        <p>Officers responded to a 911 call reporting suspicious activity near the intersection of Main Street and 5th Avenue. Upon arrival, the scene was secured and preliminary evidence collection began.</p>
        
        <h2>Evidence Collection</h2>
        <p>The following evidence was collected and processed:</p>
        <ul>
          <li>Physical evidence from the scene</li>
          <li>Photographic documentation</li>
          <li>Video surveillance footage</li>
          <li>Witness statements</li>
        </ul>
        
        <h2>Analysis and Findings</h2>
        <p>Based on the evidence collected and witness testimonies, the following preliminary findings have been established...</p>
        
        <p><em>This is a demonstration report. Use the evidence panel to add supporting documentation and continue building your case file.</em></p>
      `,
      attachedEvidence: sampleEvidence,
      metadata: {
        createdAt: new Date('2024-01-15'),
        updatedAt: new Date(),
        version: 1,
        status: 'draft' as const,
        tags: ['investigation', 'downtown', '2024'],
        classification: 'Confidential'
      },
      settings: {
        autoSave: true,
        theme: 'light' as const,
        layout: 'dual' as const
}
    };
    
    reportActions.load(sampleReport);
  });
</script>

<svelte:head>
  <title>Legal Report Editor - Advanced Document Creation</title>
  <meta name="description" content="Professional legal report editor with rich text editing, evidence management, and advanced search capabilities." />
</svelte:head>

<div class="space-y-4">
  <ReportEditor />
</div>

<style>
  /* @unocss-include */
  .report-page {
    width: 100%;
    height: 100vh;
    overflow: hidden;
}
  /* Global styles for the report editor */
  :global(body) {
    margin: 0;
    padding: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
  /* Ensure TinyMCE works well with our layout */
  :global(.tox-tinymce) {
    border: 1px solid var(--pico-border-color, #e2e8f0) !important;
    border-radius: 0.5rem !important;
}
  :global(.tox-toolbar) {
    background: var(--pico-card-sectioning-background-color, #f8fafc) !important;
}
  :global(.tox-statusbar) {
    background: var(--pico-card-sectioning-background-color, #f8fafc) !important;
    border-top: 1px solid var(--pico-border-color, #e2e8f0) !important;
}
  /* Improve scrollbar styling */
  :global(::-webkit-scrollbar) {
    width: 8px;
    height: 8px;
}
  :global(::-webkit-scrollbar-track) {
    background: var(--pico-card-sectioning-background-color, #f1f5f9);
}
  :global(::-webkit-scrollbar-thumb) {
    background: var(--pico-border-color, #cbd5e1);
    border-radius: 4px;
}
  :global(::-webkit-scrollbar-thumb:hover) {
    background: var(--pico-muted-color, #94a3b8);
}
  /* Focus improvements */
  :global(*:focus-visible) {
    outline: 2px solid var(--pico-primary, #3b82f6);
    outline-offset: 2px;
}
  /* Animation improvements */
  :global(*) {
    transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}
</style>
