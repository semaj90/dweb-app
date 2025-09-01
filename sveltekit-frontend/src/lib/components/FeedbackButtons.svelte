<script lang="ts">

  let { sessionId }: string;
  let { query }: string;
  let { candidateIds }: string[] = [];
  let { chosenId }: string | null = null;
let sending = $state(false);
let lastResp = $state<any >(null);

  async function sendFeedback(reward: number) {
    sending = true;
    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          query,
          candidateIds,
          chosenId,
          reward,
          weightsProfile: 'default',
        }),
      });
      lastResp = await res.json();
    } catch (e) {
      lastResp = { ok: false, error: String(e) };
    } finally {
      sending = false;
    }
  }
</script>

<div class="feedback-buttons">
  <button class="up" on:onclick={() => sendFeedback(1)} disabled={sending}>👍 Helpful</button>
  <button class="down" on:onclick={() => sendFeedback(0)} disabled={sending}>👎 Not helpful</button>
  {#if sending}
    <span>sending…</span>
  {:else if lastResp}
    <span>status: {String(lastResp.ok)}</span>
  {/if}
</div>

<style>
  .feedback-buttons {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  button {
    padding: 6px 10px;
    border-radius: 6px;
    cursor: pointer;
  }
  .up {
    background: #e6f6ea;
    color: #047857;
  }
  .down {
    background: #fff1f2;
    color: #b91c1c;
  }
</style>




