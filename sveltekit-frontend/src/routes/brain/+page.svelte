<script lang="ts">
  import { onMount } from 'svelte';
  import { YoRHaAPIClient } from '$lib/components/three/yorha-ui/api/YoRHaAPIClient';
  import * as THREE from 'three';
let layout = $state<any >(null);
let graphData = $state<any >({ nodes: [], links: [] });
  const client = new YoRHaAPIClient({ onData: (id, data) => { if (id === 'brainGraph') { graphData = data; updateScene(); } }});
let canvasContainer = $state<HTMLDivElement | null >(null);
let renderer = $state<THREE.WebGLRenderer;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let animationId: number;
  let nodeMeshes: Record<string, THREE.Mesh> >({});
let linkLines = $state<THREE.Line[] >([]);

  const nodeGeometry = new THREE.SphereGeometry(0.25, 24, 24);
  const typeColor: Record<string, number> = {
    db: 0x3b82f6,
    cache: 0x0ea5e9,
    vector: 0x6366f1,
    llm: 0xf59e0b,
    service: 0x10b981,
    edge: 0xef4444,
    'ui-lib': 0x94a3b8,
    routing: 0x22c55e,
    automation: 0x14b8a6,
    table: 0x8b5cf6,
    default: 0xffffff
  };

  function initThree() {
    if (!canvasContainer) return;
    scene = new THREE.Scene();
    scene.background = new THREE.Color('#0b0d11');
    camera = new THREE.PerspectiveCamera(55, canvasContainer.clientWidth / 400, 0.1, 100);
    camera.position.set(0, 4, 9);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(canvasContainer.clientWidth, 400);
    canvasContainer.appendChild(renderer.domElement);
    const light = new THREE.DirectionalLight(0xffffff, 0.8);
    light.position.set(4, 6, 5);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0x404040));
  }

  function buildGraph() {
    // Clear existing
    Object.values(nodeMeshes).forEach(m => scene.remove(m));
    linkLines.forEach(l => scene.remove(l));
    nodeMeshes = {};
    linkLines = [];

    const radius = 4;
    graphData.nodes.forEach((n: any, idx: number) => {
      const mat = new THREE.MeshStandardMaterial({ color: typeColor[n.type] ?? typeColor.default, emissive: 0x111111 });
      const mesh = new THREE.Mesh(nodeGeometry, mat);
      // Initial circular placement; slight random Z for depth
      const a = (idx / graphData.nodes.length) * Math.PI * 2;
      mesh.position.set(Math.cos(a) * radius, (Math.sin(a * 2) * 0.5), Math.sin(a) * radius * 0.5 + (Math.random() - 0.5));
      scene.add(mesh);
      nodeMeshes[n.id] = mesh;
    });

    // Links
    graphData.links.forEach((l: any) => {
      const from = nodeMeshes[l.source];
      const to = nodeMeshes[l.target];
      if (!from || !to) return;
      const pts = [from.position, to.position];
      const geom = new THREE.BufferGeometry().setFromPoints(pts);
      const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0x334155 }));
      scene.add(line);
      linkLines.push(line);
    });
  }

  function updateScene() {
    buildGraph();
  }

  function animate() {
    animationId = requestAnimationFrame(animate);
    // Light weight drift animation
    Object.values(nodeMeshes).forEach(m => {
      m.rotation.y += 0.005;
    });
    renderer.render(scene, camera);
  }

  onMount(async () => {
    initThree();
    try {
      await client.loadLayout('/api/yorha/layout');
      layout = client.getLayout();
      client.startDataStreams();
      const res = await fetch('/api/brain/graph');
      if (res.ok) {
        graphData = await res.json();
        updateScene();
      }
      animate();
    } catch (e) {
      console.error('Brain page init failed', e);
    }
    return () => { cancelAnimationFrame(animationId); };
  });
</script>

<h1 class="text-2xl font-bold mb-4 font-mono">🧠 System Brain Graph</h1>
<p class="mb-4 opacity-80">Live topology of backend services, database entities and frontend modules.</p>

<div class="grid gap-4 md:grid-cols-3">
  <div class="col-span-2 flex flex-col gap-3">
    <div bind:this={canvasContainer} class="w-full border rounded bg-zinc-900/40 relative">
      {#if !graphData.nodes.length}
        <div class="absolute inset-0 flex items-center justify-center text-sm opacity-60">Loading 3D graph...</div>
      {/if}
    </div>
    <div class="p-3 border rounded bg-zinc-900/40">
      <h2 class="font-semibold mb-2">Nodes ({graphData.nodes.length})</h2>
      <div class="flex flex-wrap gap-2 text-[10px]">
        {#each graphData.nodes as n}
          <span class="px-2 py-1 rounded bg-zinc-800/70 border border-zinc-700/60"><code>{n.id}</code></span>
        {/each}
      </div>
    </div>
  </div>
  <div class="p-4 border rounded bg-zinc-900/40">
    <h2 class="font-semibold mb-2">Links ({graphData.links.length})</h2>
    <ul class="text-xs max-h-72 overflow-auto space-y-1">
      {#each graphData.links as l}
        <li><code>{l.source} → {l.target}</code> <span class="opacity-60">{l.kind}</span></li>
      {/each}
    </ul>
  </div>
</div>

<style>
  :global(body) { font-family: system-ui, sans-serif; }
</style>
