<script lang="ts">
  interface Props {
    report: Report
  }
  let {
    report
  } = $props();



import { onMount } from "svelte";
import type { Report } from '$lib/data/types';
// UI Components
import * as ContextMenu from '$lib/components/ui/context-menu';
// Icons
import { Link, Sparkles } from "lucide-svelte";


let nodeElement: HTMLDivElement
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;

  // Add local position state for drag-and-drop
  let position = { x: 100, y: 100 };

  function handleMouseDown(event: MouseEvent) {
    if (
      event.target === nodeElement ||
      (event.target as Element)?.classList?.contains("node-header")
    ) {
      isDragging = true;
      dragStartX = event.clientX - position.x;
      dragStartY = event.clientY - position.y;
}
}
  function handleMouseMove(event: MouseEvent) {
    if (isDragging) {
      position.x = event.clientX - dragStartX;
      position.y = event.clientY - dragStartY;
}
}
  function handleMouseUp() {
    isDragging = false;
}
  async function saveCitation(text: string) {
    if (!text.trim()) return;

    // Implementation for saving citation
    console.log("Saving citation:", text);
}
  async function summarizeReport() {
    // Implementation for AI summary
    console.log("Summarizing report");
}
  onMount(() => {
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      nodeElement = null;
    };
  });
</script>

<ContextMenu.Root>
  <ContextMenu.Trigger>
    <div
      bind:this={nodeElement}
      class="space-y-4"
      style={`left: ${position.x}px; top: ${position.y}px; z-index: 10;`}
      onmousedown={handleMouseDown}
      onkeydown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          isDragging = true;
          dragStartX = 0;
          dragStartY = 0;
        }
      }}
      role="button"
      tabindex={0}
      aria-label="Drag report node"
    >
      <div>
        <div>
          {report.content}
        </div>
      </div>
    </div>
  </ContextMenu.Trigger>
  <ContextMenu.Content menu={true}>
    <ContextMenu.Item
    <ContextMenu.Item
      onselect={() => saveCitation(window.getSelection()?.toString() || "")}
    >
      <Link class="space-y-4" />
      Save as Citation
    </ContextMenu.Item>
    <ContextMenu.Item onselect={summarizeReport}>
      <Sparkles class="space-y-4" />
      AI Summary
    </ContextMenu.Item>
  </ContextMenu.Content>
</ContextMenu.Root>

<style>
  /* @unocss-include */
</style>
