<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  export let sessionId: string;
  export let query: string;
  export let candidateIds: string[] = [];
  export let chosenId: string | null = null;

  const dispatch = createEventDispatcher();
  let sending = false;
  let lastResp: any = null;

  async function sendFeedback(reward: number) {
    sending = true;
    const payload = {
      sessionId,
      query,
      candidateIds,
      chosenId,
      reward,
      weightsProfile: 'default'
    };
    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const j = await res.json();
      lastResp = j;
      dispatch('feedbackSent', { ok: j.ok, payload, resp: j });
    } catch (e) {
      lastResp = { ok: false, error: String(e) };
      dispatch('feedbackError', lastResp);
    } finally {
      sending = false;
    }
  }
</script>

<style>
  .feedback-buttons {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px;
    background: rgba(26, 26, 26, 0.8);
    border: 1px solid #333;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
  }

  .btn { 
    padding: 8px 12px; 
    border-radius: 6px; 
    cursor: pointer; 
    display: inline-flex; 
    align-items: center; 
    gap: 8px; 
    border: 2px solid transparent;
    font-family: inherit;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
  }

  .btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transition: left 0.3s ease;
  }

  .btn:hover::before {
    left: 100%;
  }

  .up { 
    background: linear-gradient(135deg, #e6f6ea, #d1f5d3); 
    color: #047857; 
    border-color: #047857;
  }

  .up:hover {
    background: #047857;
    color: #e6f6ea;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(4, 120, 87, 0.3);
  }

  .down { 
    background: linear-gradient(135deg, #fff1f2, #ffe4e6); 
    color: #b91c1c; 
    border-color: #b91c1c;
  }

  .down:hover {
    background: #b91c1c;
    color: #fff1f2;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(185, 28, 28, 0.3);
  }

  .sending { 
    opacity: 0.6; 
    cursor: not-allowed; 
    transform: none !important;
    box-shadow: none !important;
  }

  .sending::before {
    display: none;
  }

  .status {
    margin-left: 10px;
    font-size: 0.7rem;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .status.success {
    background: rgba(4, 120, 87, 0.2);
    color: #047857;
    border: 1px solid #047857;
  }

  .status.error {
    background: rgba(185, 28, 28, 0.2);
    color: #b91c1c;
    border: 1px solid #b91c1c;
  }

  .status.sending {
    background: rgba(255, 191, 0, 0.2);
    color: #ffbf00;
    border: 1px solid #ffbf00;
  }

  .btn-icon {
    font-size: 1rem;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .sending .status {
    animation: pulse 1.5s infinite;
  }

  /* YoRHa theme integration */
  .feedback-buttons {
    background: linear-gradient(135deg, rgba(26, 26, 26, 0.9), rgba(15, 15, 15, 0.9));
    border: 1px solid #ffbf00;
    box-shadow: 0 2px 8px rgba(255, 191, 0, 0.1);
  }

  .btn {
    position: relative;
    z-index: 1;
  }

  .btn::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, transparent 40%, rgba(255, 191, 0, 0.1) 50%, transparent 60%);
    opacity: 0;
    transition: opacity 0.3s ease;
    z-index: -1;
  }

  .btn:hover::after {
    opacity: 1;
  }
</style>

<div class="feedback-buttons">
  <button 
    class="btn up" 
    class:sending 
    onclick={() => sendFeedback(1)} 
    disabled={sending}
    title="Mark as helpful - improves future results"
  >
    <span class="btn-icon">👍</span>
    <span>Helpful</span>
  </button>
  
  <button 
    class="btn down" 
    class:sending 
    onclick={() => sendFeedback(0)} 
    disabled={sending}
    title="Mark as not helpful - improves future results"
  >
    <span class="btn-icon">👎</span>
    <span>Not helpful</span>
  </button>
  
  {#if sending}
    <span class="status sending">sending…</span>
  {:else if lastResp}
    <span class="status" class:success={lastResp.ok} class:error={!lastResp.ok}>
      {lastResp.ok ? '✅ sent' : '❌ failed'}
    </span>
  {/if}
</div>
