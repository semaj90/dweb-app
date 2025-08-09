

  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import { useMachine } from '@xstate/svelte';
  import { agentShellMachine } from '$lib/machines/agentShellMachine';
  import { io } from 'socket.io-client';
  import Loki from 'lokijs';
  import Fuse from 'fuse.js';

  // State machine for agent activity
  const { state, send } = useMachine(agentShellMachine);

  // Local cache for job state
  const db = new Loki('agentCache');
  const jobs = db.addCollection('jobs');
  const jobStore = writable([]);

  // Fuzzy search setup
  let fuse;
  let searchResults = [];

  // WebSocket for real-time updates
  let socket;
  onMount(() => {
    socket = io('/api/ws');
    socket.on('agent-update', (data) => {
      jobs.insert(data);
      jobStore.set(jobs.find());
      if (fuse) fuse.setCollection(jobs.find());
    });
    // Initialize Fuse.js
    fuse = new Fuse(jobs.find(), { keys: ['description', 'status'] });
  });

  function searchJobs(query: string) {
    searchResults = fuse.search(query).map(r => r.item);
  }

  function acceptPatch(jobId: string) {
    send({ type: 'ACCEPT_PATCH', jobId });
  }

  function rateSuggestion(jobId: string, rating: number) {
    send({ type: 'RATE_SUGGESTION', jobId, rating });
  }




  Agentic Legal AI Demo
   searchJobs(e.target.value)} />
  
    {#each $jobStore as job}
      
        {job.description} — {job.status}
         acceptPatch(job.id)}>Accept Patch
         rateSuggestion(job.id, 5)}>👍
         rateSuggestion(job.id, 1)}>👎
      
    {/each}
  
  
  
    Search Results
    
      {#each searchResults as result}
        {result.description} — {result.status}
      {/each}
    
  




.vector-intelligence-demo {
  max-width: 800px;
  margin: 2rem auto;
  padding: 1rem;
  border: 1px solid #ccc;
  border-radius: 8px;
}
input {
  width: 70%;
  padding: 0.5rem;
  margin-right: 0.5rem;
}
button {
  padding: 0.5rem 1rem;
}
ul {
  margin-top: 1rem;
}
canvas {
  margin-top: 2rem;
  border: 1px solid #aaa;
  border-radius: 4px;
}

