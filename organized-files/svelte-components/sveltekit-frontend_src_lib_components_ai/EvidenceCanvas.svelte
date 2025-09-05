<script lang="ts">
  import { apiFetch } from "$lib/api/clients/api-client";
  import { onMount } from "svelte";
  let canvasEl: HTMLCanvasElement
  let fabricCanvas: any

  let analyzing = false;
  let error: string | null = null;
  let result: {
    analysis?: string;
    summary?: string;
    confidence?: number;
    processing_time_ms?: number;
    status?: string;
    error?: string;
  } | null = null;
  let options = {
    analyze_layout: true,
    extract_entities: true,
    generate_summary: true,
    confidence_level: 0.8,
    context_window: 4096,
  };

  onMount(async () => {
    const { fabric } = await import("fabric");
    fabricCanvas = new fabric.Canvas(canvasEl);
    // Example: Add a rectangle
    fabricCanvas.add(
      new fabric.Rect({
        left: 100,
        top: 100,
        fill: "red",
        width: 60,
        height: 60,
      })
    );
  });

  function collectObjects() {
    const objs = (fabricCanvas?.getObjects?.() ?? []).map((o: any) => {
      const type = o.type || "object";
      const left = typeof o.left === "number" ? o.left : 0;
      const top = typeof o.top === "number" ? o.top : 0;
      const text = typeof o.text === "string" ? o.text: undefined
      return { type, position: { x: left, y: top }, ...(text ? { text } : {}) };
    });
    return objs;
  }

  async function analyzeCanvas() {
    analyzing = true;
    error = null;
    result = null;
    try {
      const payload = {
        task: "evidence_canvas_analysis",
        prompt:
          "Analyze the key evidence items and summarize relevant facts and entities.",
        context: [
          {
            canvas_json: {},
            objects: collectObjects(),
            canvas_size: { width: canvasEl.width, height: canvasEl.height },
          },
        ],
        instructions: "Provide concise legal-relevant insights.",
        options,
      };
      const resp = await apiFetch<{
        analysis: string
        summary: string
        confidence: number
        processing_time_ms: number
        status: string
        error?: string;
      }>("http://localhost:8081/api/evidence-canvas/analyze", "POST", {
        body: payload,
        retry: {
          attempts: 3,
          backoffMs: 400,
          maxBackoffMs: 2500,
          timeoutMs: 20000,
        },
      });
      result = resp;
      if (resp.error) error = resp.error;
    } catch (e: unknown) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      analyzing = false;
    }
  }
</script>

<div class="toolbar">
  <button on:click={analyzeCanvas} disabled={analyzing}
    >{analyzing ? "Analyzing…" : "Analyze Canvas"}</button
  >
  <label
    ><input type="checkbox" bind:checked={options.analyze_layout} /> Layout</label
  >
  <label
    ><input type="checkbox" bind:checked={options.extract_entities} /> Entities</label
  >
  <label
    ><input type="checkbox" bind:checked={options.generate_summary} /> Summary</label
  >
  <span class="spacer" />
  <small
    >Ctx: <input
      type="number"
      bind:value={options.context_window}
      min={512}
      max={16384}
      step={256}
      style="width:6rem"
    /></small
  >
  <small
    >Conf: <input
      type="number"
      bind:value={options.confidence_level}
      min={0}
      max={1}
      step={0.05}
      style="width:5rem"
    /></small
  >
  {#if error}<span class="error">{error}</span>{/if}
  {#if result && result.status === "success"}<span class="ok">✓</span>{/if}
  {#if analyzing}<span class="spinner">⏳</span>{/if}
</div>

<div class="evidence-canvas-wrapper">
  <canvas bind:this={canvasEl} width="800" height="600"></canvas>
</div>

{#if result}
  <div class="panel">
    <h3>Analysis</h3>
    <pre>{result.analysis}</pre>
    <h3>Summary</h3>
    <pre>{result.summary}</pre>
    <div class="meta">
      <span>Confidence: {result.confidence?.toFixed?.(2)}</span>
      <span>Time: {result.processing_time_ms} ms</span>
      <span>Status: {result.status}</span>
    </div>
  </div>
{/if}

<style>
  .toolbar {
    display: flex
    gap: 0.75rem;
    align-items: center
    padding: 0.5rem 0;
  }
  .toolbar .spacer {
    flex: 1;
  }
  .toolbar .error {
    color: #c00;
    margin-left: 0.5rem;
  }
  .toolbar .ok {
    color: #090;
    margin-left: 0.5rem;
  }
  .toolbar .spinner {
    color: #555;
    margin-left: 0.5rem;
  }

  .evidence-canvas-wrapper {
    display: flex
    justify-content: center
    align-items: center
    margin: 2rem auto;
    border: 1px solid #ccc;
    border-radius: 8px;
    width: 820px;
    height: 620px;
    background: #fafafa;
  }
  canvas {
    background: #fff;
    border-radius: 8px;
  }
  .panel {
    margin: 1rem auto;
    width: 820px;
    background: #fff;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    padding: 1rem;
  }
  .panel pre {
    white-space: pre-wrap;
    background: #f8f8f8;
    padding: 0.75rem;
    border-radius: 6px;
  }
  .panel .meta {
    display: flex
    gap: 1rem;
    color: #555;
  }
</style>
