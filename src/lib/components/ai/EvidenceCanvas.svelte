<script lang="ts">
  import { onMount } from 'svelte';
  let canvasEl: HTMLCanvasElement;
  let fabricCanvas: any;

  onMount(async () => {
    const { fabric } = await import('fabric');
    fabricCanvas = new fabric.Canvas(canvasEl);
    // Example: Add a rectangle
    fabricCanvas.add(new fabric.Rect({ left: 100, top: 100, fill: 'red', width: 60, height: 60 }));
  });
</script>

<div class="evidence-canvas-wrapper">
  <canvas bind:this={canvasEl} width="800" height="600"></canvas>
</div>

<style>
.evidence-canvas-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
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
</style>
