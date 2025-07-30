
<script lang="ts">
  import { onMount } from 'svelte';
  let canvasEl: HTMLCanvasElement;
  let fabricCanvas: any;
  let pdfLoading = false;

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
</script>


<div class="evidence-canvas-wrapper">
  <canvas bind:this={canvasEl} width="800" height="600"></canvas>
</div>
<button on:click={downloadPDF} class="mt-4 px-4 py-2 bg-blue-600 text-white rounded" disabled={pdfLoading} aria-busy={pdfLoading} aria-label="Download evidence canvas as PDF">
  {pdfLoading ? 'Generating PDF...' : 'Download as PDF'}
</button>

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
