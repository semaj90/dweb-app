<script lang="ts">
  interface Props {
    points?: number[];
    width?: number;
    height?: number;
    color?: string;
  }

  let { points = [], width = 160, height = 50, color = '#2563eb' }: Props = $props();
  
  let capped = $derived(points.slice(-60)); // last 60 samples
  let max = $derived(Math.max(1, ...capped));
  let d = $derived(capped.map((v,i) => {
    const x = (i/(capped.length-1||1))*width;
    const y = height - (v/max)*height;
    return `${x},${y}`;
  }).join(' '));
</script>
<svg {width} {height} class="overflow-visible">
  <polyline points={d} fill="none" stroke={color} stroke-width="2" stroke-linejoin="round" stroke-linecap="round" />
</svg>
<style>
  svg { display:block }
</style>
