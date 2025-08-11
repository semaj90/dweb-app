<script lang="ts">
  // @ts-nocheck
  import { onMount } from 'svelte';
  let canvasEl: HTMLCanvasElement;
  let fabricCanvas: any;
  let pdfLoading = false;
  let userPrompt = '';
  let analyzing = false;

  onMount(async () => {
    const { fabric } = await import('fabric.js');
    fabricCanvas = new fabric.Canvas(canvasEl);
    // Example: Add a rectangle
    fabricCanvas.add(new fabric.Rect({ left: 100, top: 100, fill: 'red', width: 60, height: 60 }));
  });

  async function downloadPDF() {
    pdfLoading = true;
    try {
      // Dynamically import pdf-lib for SvelteKit best practice
      const { PDFDocument } = await import('pdf-lib');
      const dataUrl = canvasEl.toDataURL('image/png');
      const pdfDoc = await PDFDocument.create();
      const page = pdfDoc.addPage([canvasEl.width, canvasEl.height]);
      const pngImage = await pdfDoc.embedPng(dataUrl);
      page.drawImage(pngImage, {
        x: 0,
        y: 0,
        width: canvasEl.width,
        height: canvasEl.height,
      });
      // Accessibility metadata
      pdfDoc.setTitle('Evidence Canvas Export');
      pdfDoc.setAuthor('Legal AI System');
      pdfDoc.setSubject('Evidence Summary');
      pdfDoc.setKeywords(['evidence', 'canvas', 'legal', 'export']);
      const pdfBytes = await pdfDoc.save();
      const blob = new Blob([pdfBytes], { type: 'application/pdf' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'evidence-canvas.pdf';
      link.click();
      URL.revokeObjectURL(link.href);
    } catch (e) {
      alert('Failed to generate PDF: ' + (e?.message || e));
    } finally {
      pdfLoading = false;
    }
  }

  async function analyzeCanvas() {
    if (!userPrompt.trim()) return;
    analyzing = true;

    try {
      const canvasData = {
        task: "evidence_canvas_analysis",
        prompt: userPrompt,
        context: [{
          canvas_json: fabricCanvas?.toJSON() || {},
          objects: fabricCanvas?.getObjects().map(obj => ({
            type: obj.type,
            position: { x: obj.left, y: obj.top },
            text: obj.text || null
          })) || [],
          canvas_size: { width: canvasEl.width, height: canvasEl.height }
        }],
        instructions: "Analyze canvas content and respond with structured evidence summary"
      };

      console.log('go-llama payload:', canvasData);
      // TODO: Send to ollama, go-llama.cpp API endpoint for embeding, summarization ai features, .ts store?

    } catch (e) {
      console.error('Analysis failed:', e);
    } finally {
      analyzing = false;
    }
  }

</script>

    <input
      bind:value={userPrompt}
      placeholder="Ask Claude to analyze this canvas..."
      class="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
      on:keydown={(e) => e.key === 'Enter' && analyzeCanvas()}
    <style>
    .evidence-canvas-wrapper {
      display: flex;

      <div class="evidence-canvas">
        <div class="controls">
          <input
            bind:value={userPrompt}
            placeholder="Ask Claude to analyze this canvas..."
            class="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            on:keydown={(e) => e.key === 'Enter' && analyzeCanvas()}
          />
          <button
            on:click={analyzeCanvas}
            class="px-4 py-2 bg-indigo-600 text-white rounded"
            disabled={analyzing}
            aria-busy={analyzing}
          >
            {analyzing ? 'Analyzing...' : 'Analyze'}
          </button>
          <button
            on:click={downloadPDF}
            class="px-4 py-2 bg-blue-600 text-white rounded"
            disabled={pdfLoading}
            aria-busy={pdfLoading}
          >
            {pdfLoading ? 'Generating PDF...' : 'Download as PDF'}
          </button>
        </div>

        <div class="evidence-canvas-wrapper">
          <canvas bind:this={canvasEl} width="800" height="600"></canvas>
        </div>
      </div>

      <style>
      .evidence-canvas {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
      }
      .controls {
        display: flex;
        gap: 0.5rem;
        align-items: center;
      }
      .evidence-canvas-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0.5rem 0 2rem;
        border: 1px solid #ccc;
        border-radius: 8px;
        width: 820px;
        max-width: 100%;
        height: 620px;
        background: #fafafa;
        overflow: hidden;
      }
      canvas {
        background: #fff;
        border-radius: 8px;
      }
      </style>

