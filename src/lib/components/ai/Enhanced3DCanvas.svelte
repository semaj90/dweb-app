<script lang="ts">
  // Note: See mcp/ENHANCED-CONTEXT7-SERVER-GUIDE.md for related context and usage. #file:mcp

  import { onMount, onDestroy } from 'svelte';
  import { createActor } from 'xstate';
  import * as THREE from 'three';
  import { createCanvas3DMachine } from '$lib/state/canvas-3d-machine';
  import { webGPUProcessor } from '$lib/services/webgpu-processor';
  import { interactionTracker } from '$lib/services/interaction-tracker';
  import YoRHaForm from '$lib/components/yorha/YoRHaForm.svelte';
  import type { DocumentNode, UserInteraction } from '$lib/types/ai';

  // mount into an existing container (e.g. .evidence-canvas-wrapper) if provided
  export let mountSelector: string | HTMLElement = '.evidence-canvas-wrapper';

  // Props
  export let documents: DocumentNode[] = [];
  export let enableGPUAcceleration = true;
  export let enableInteractionTracking = true;
  export let somGridSize = { width: 50, height: 50 };

  // State (use definite assignment where appropriate)
  let canvasContainer!: HTMLDivElement;
  let scene!: THREE.Scene;
  let camera!: THREE.PerspectiveCamera;
  let renderer!: THREE.WebGLRenderer;
  let documentMeshes = new Map<string, THREE.Mesh>();
  let animationFrameId = 0;

  let yoRhaInstance: any = null;

  // XState actor for canvas state management
  const canvas3DActor = createActor(createCanvas3DMachine());
  let actorSubscription: { unsubscribe: () => void } | null = null;

  // GPU Processing and SOM state
  const emptyPixel = new Uint8Array([0, 0, 0, 255]);
  let somTexture: THREE.DataTexture = new THREE.DataTexture(emptyPixel, 1, 1);
  somTexture.needsUpdate = true;
  let gpuComputeBuffer: any = null;

  // Helper to build form fields reflecting current state
  function buildFormFields() {
    return [
      {
        id: 'enableGPUAcceleration',
        type: 'checkbox',
        label: 'GPU Acceleration',
        value: enableGPUAcceleration,
        placeholder: '',
        required: false
      },
      {
        id: 'enableInteractionTracking',
        type: 'checkbox',
        label: 'Interaction Tracking',
        value: enableInteractionTracking,
        placeholder: '',
        required: false
      },
      {
        id: 'somWidth',
        type: 'number',
        label: 'SOM Grid Width',
        value: somGridSize?.width ?? 50,
        placeholder: 'Width',
        required: true,
        validation: { min: 1 }
      },
      {
        id: 'somHeight',
        type: 'number',
        label: 'SOM Grid Height',
        value: somGridSize?.height ?? 50,
        placeholder: 'Height',
        required: true,
        validation: { min: 1 }
      },
      {
        id: 'documentsJson',
        type: 'textarea',
        label: 'Documents (JSON array)',
        value: JSON.stringify(documents || [], null, 2),
        placeholder: '[{ "id": "doc1", "title": "Doc 1" }]',
        required: false,
        validation: { minLength: 2 }
      }
    ];
  }

  function handleFormSubmit(payload: Record<string, any>) {
    try {
      // Update toggles
      enableGPUAcceleration = !!payload.enableGPUAcceleration;
      enableInteractionTracking = !!payload.enableInteractionTracking;

      // Update SOM grid size
      const w = Number(payload.somWidth) || somGridSize.width;
      const h = Number(payload.somHeight) || somGridSize.height;
      somGridSize = { width: Math.max(1, w), height: Math.max(1, h) };

      // Parse documents JSON if provided
      if (payload.documentsJson) {
        try {
          const parsed = JSON.parse(payload.documentsJson);
          if (Array.isArray(parsed)) {
            documents = parsed;
          } else {
            console.warn('documentsJson did not parse to an array, ignoring');
          }
        } catch (e) {
          console.warn('Failed to parse documentsJson:', e);
        }
      }

      // Rebuild SOM overlay: remove any existing shader-based SOM meshes and recreate
      if (scene) {
        const somCandidates: THREE.Mesh[] = [];
        scene.traverse((obj) => {
          if (obj instanceof THREE.Mesh && obj.material instanceof THREE.ShaderMaterial) {
            somCandidates.push(obj);
          }
        });
        somCandidates.forEach((m) => {
          try {
            scene.remove(m);
            m.geometry?.dispose?.();
            (m.material as any)?.dispose?.();
          } catch (err) {
            // ignore
          }
        });

        createSOMGrid();
      }

      // (Re)initialize or teardown GPU processing depending on toggle
      if (enableGPUAcceleration) {
        setupWebGPUProcessing().catch((e) => console.warn('GPU setup failed', e));
      } else if (gpuComputeBuffer) {
        try {
          gpuComputeBuffer.destroy?.();
        } catch (e) {
          // ignore
        } finally {
          gpuComputeBuffer = null;
        }
      }

      // Rebuild document visuals
      updateDocumentVisualization();

      // If enabling/disabling interaction tracking at runtime, (un)register listeners
      // Simple approach: teardown and re-setup tracking
      if (enableInteractionTracking) {
        try {
          setupInteractionTracking();
        } catch (e) {
          console.warn('Failed to setup interaction tracking:', e);
        }
      }
    } catch (err) {
      console.error('Error applying YoRHaForm values:', err);
    }
  }

  function handleFormCancel() {
    // close/destroy the form instance (UI-only)
    try {
      yoRhaInstance?.$destroy?.();
      const el = yoRhaInstance?._target || yoRhaInstance?._$$?.target;
      // best-effort remove wrapper node
      if (el && el.parentNode) el.parentNode.removeChild(el);
    } catch (e) {
      // ignore
    } finally {
      yoRhaInstance = null;
    }
  }

  // Programmatically mount the YoRHa form into the canvas container as an overlay
  onMount(() => {
    // Defer mounting until canvasContainer exists
    if (!canvasContainer) {
      // attempt again next tick if binding hasn't happened yet
      setTimeout(() => {
        if (canvasContainer) mountYoRHaForm();
      }, 10);
    } else {
      mountYoRHaForm();
    }

    return () => {
      if (yoRhaInstance) {
        try {
          yoRhaInstance.$destroy?.();
        } catch (e) {}
        yoRhaInstance = null;
      }
    };
  });

  onDestroy(() => {
    if (yoRhaInstance) {
      try {
        yoRhaInstance.$destroy?.();
      } catch (e) {}
      yoRhaInstance = null;
    }
  });

  function mountYoRHaForm() {
    if (!canvasContainer) return;

    // wrapper to ensure absolute overlay inside canvas container
    const wrapper = document.createElement('div');
    wrapper.style.position = 'absolute';
    wrapper.style.top = '12px';
    wrapper.style.right = '12px';
    wrapper.style.zIndex = '1000';
    wrapper.style.minWidth = '320px';
    // keep pointer events for the form but let canvas still be interactive if needed
    wrapper.style.pointerEvents = 'auto';

    canvasContainer.appendChild(wrapper);

    yoRhaInstance = new YoRHaForm({
      target: wrapper,
      props: {
        title: 'Canvas Controls',
        subtitle: 'Adjust rendering and data',
        fields: buildFormFields(),
        submitLabel: 'Apply',
        cancelLabel: 'Close',
        loading: false,
        showCancel: true,
        onsubmit: (data: Record<string, any>) => {
          handleFormSubmit(data);
          // keep form open; you can destroy on cancel instead
        },
        oncancel: () => {
          handleFormCancel();
        }
      }
    });
  }

  // Best-effort helper to push canvas context to MCP / local API (non-blocking)
  async function updateMCPContext(context: any) {
    try {
      await fetch('/api/mcp/context72/update-context', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context })
      });
    } catch (err) {
      console.warn('Failed to update MCP context (non-fatal):', err);
    }
  }

  onMount(async () => {
    // Ensure canvasContainer is available (bound) before initializing
    if (!canvasContainer) {
      // If not bound, attempt to find via mountSelector fallback
      if (typeof mountSelector === 'string') {
        const el = document.querySelector(mountSelector);
        if (el) canvasContainer = el as HTMLDivElement;
      } else if (mountSelector instanceof HTMLElement) {
        canvasContainer = mountSelector as HTMLDivElement;
      }
    }

    if (!canvasContainer) {
      console.warn('Enhanced3DCanvas: no canvas container available, aborting initialization');
      return;
    }

    await initializeCanvas();
    await setupWebGPUProcessing();
    startRenderLoop();
    canvas3DActor.start();

    // subscribe once
    actorSubscription = canvas3DActor.subscribe((state) => {
      if (state.context?.activeDocumentId || state.context?.hoveredDocumentId) {
        updateMCPContext({
          activeDocumentId: state.context.activeDocumentId,
          hoveredDocumentId: state.context.hoveredDocumentId,
          cameraPosition: camera?.position?.toArray?.(),
          visibleDocuments: getVisibleDocuments()
        });
      }
    });

    if (enableInteractionTracking) {
      setupInteractionTracking();
    }
  });

  onDestroy(() => {
    cleanup();
    try {
      canvas3DActor.stop();
    } catch (e) {}
    actorSubscription?.unsubscribe();
  });

  async function initializeCanvas() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);

    camera = new THREE.PerspectiveCamera(
      75,
      canvasContainer.clientWidth / canvasContainer.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 10);

    renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance'
    });
    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    canvasContainer.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    createSOMGrid();
  }

  async function setupWebGPUProcessing() {
    if (!enableGPUAcceleration || !('gpu' in navigator)) {
      console.log('WebGPU not available, falling back to CPU processing');
      return;
    }

    try {
      await webGPUProcessor.initialize();
      console.log('✅ WebGPU processor initialized');

      const bufferSize = documents.length * 768 * 4;
      gpuComputeBuffer = webGPUProcessor.createBuffer(bufferSize);
    } catch (error) {
      console.error('WebGPU initialization failed:', error);
    }
  }

  function createSOMGrid() {
    const gridGeometry = new THREE.PlaneGeometry(
      somGridSize.width,
      somGridSize.height,
      Math.max(1, somGridSize.width - 1),
      Math.max(1, somGridSize.height - 1)
    );

    const somMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        resolution: { value: new THREE.Vector2(somGridSize.width, somGridSize.height) },
        somTexture: { value: somTexture }
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vPosition;

        void main() {
          vUv = uv;
          vPosition = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform vec2 resolution;
        uniform sampler2D somTexture;
        varying vec2 vUv;
        varying vec3 vPosition;

        void main() {
          vec2 uv = vUv;
          vec4 somData = texture2D(somTexture, uv);
          float density = somData.r;
          vec3 color = mix(
            vec3(0.1, 0.1, 0.2),
            vec3(0.9, 0.4, 0.1),
            density
          );
          color += 0.1 * sin(time + vPosition.x * 0.1 + vPosition.y * 0.1);
          gl_FragColor = vec4(color, 0.8);
        }
      `,
      transparent: true
    });

    const somMesh = new THREE.Mesh(gridGeometry, somMaterial);
    somMesh.position.set(0, 0, -2);
    scene.add(somMesh);
  }

  function setupInteractionTracking() {
    const canvas = renderer.domElement;

    // Raw movement recording
    const handleMouseMoveRaw = (event: MouseEvent) => {
      const interaction: UserInteraction = {
        type: 'mouse_move',
        timestamp: Date.now(),
        position: { x: event.clientX, y: event.clientY },
        target: event.target as HTMLElement
      };
      interactionTracker.recordInteraction(interaction);
    };

    // Hover detection (raycasting)
    const handleMouseMoveHover = (event: MouseEvent) => {
      const mouse = new THREE.Vector2();
      mouse.x = (event.clientX / canvas.clientWidth) * 2 - 1;
      mouse.y = -(event.clientY / canvas.clientHeight) * 2 + 1;

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, camera);

      const intersects = raycaster.intersectObjects(Array.from(documentMeshes.values()));

      if (intersects.length > 0) {
        const hoveredObject = intersects[0].object;
        const documentId = findDocumentIdByMesh(hoveredObject as THREE.Mesh);

        const interaction: UserInteraction = {
          type: 'document_hover',
          timestamp: Date.now(),
          position: { x: event.clientX, y: event.clientY },
          documentId,
          target: event.target as HTMLElement
        };

        interactionTracker.recordInteraction(interaction);
      }
    };

    const handleClick = (event: MouseEvent) => {
      const mouse = new THREE.Vector2();
      mouse.x = (event.clientX / canvas.clientWidth) * 2 - 1;
      mouse.y = -(event.clientY / canvas.clientHeight) * 2 + 1;

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, camera);

      const intersects = raycaster.intersectObjects(Array.from(documentMeshes.values()));

      if (intersects.length > 0) {
        const clickedObject = intersects[0].object;
        const documentId = findDocumentIdByMesh(clickedObject as THREE.Mesh);

        const interaction: UserInteraction = {
          type: 'document_click',
          timestamp: Date.now(),
          position: { x: event.clientX, y: event.clientY },
          documentId,
          target: event.target as HTMLElement
        };

        interactionTracker.recordInteraction(interaction);
        canvas3DActor.send({ type: 'DOCUMENT_SELECTED', documentId });
      }
    };

    // Register handlers
    canvas.addEventListener('mousemove', handleMouseMoveRaw);
    canvas.addEventListener('mousemove', handleMouseMoveHover);
    canvas.addEventListener('click', handleClick);
  }

  function findDocumentIdByMesh(mesh: THREE.Mesh): string | undefined {
    for (const [documentId, docMesh] of documentMeshes.entries()) {
      if (docMesh === mesh) return documentId;
    }
    return undefined;
  }

  function startRenderLoop() {
    function animate() {
      animationFrameId = requestAnimationFrame(animate);

      scene.traverse((object) => {
        if (object instanceof THREE.Mesh && object.material instanceof THREE.ShaderMaterial) {
          object.material.uniforms.time.value = Date.now() * 0.001;
        }
      });

      const time = Date.now() * 0.0005;
      camera.position.x = Math.cos(time) * 15;
      camera.position.z = Math.sin(time) * 15;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    }
    animate();
  }

  function cleanup() {
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
    }

    try {
      renderer?.dispose?.();
    } catch (e) {}

    if (gpuComputeBuffer) {
      try {
        gpuComputeBuffer.destroy?.();
      } catch (e) {
        console.warn('Failed to destroy GPU buffer (non-fatal):', e);
      }
    }

    try {
      scene?.clear?.();
    } catch (e) {}
  }

  // Reactive updates for documents
  $: if (documents.length > 0 && scene) {
    updateDocumentVisualization();
  }

  // Ensure renderer and camera update on window/container resize
  function handleResize() {
    if (!renderer || !camera || !canvasContainer) return;
    const w = canvasContainer.clientWidth || window.innerWidth;
    const h = canvasContainer.clientHeight || window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  // Return a simple list of visible document IDs (used when pushing context)
  function getVisibleDocuments(): string[] {
    return Array.from(documentMeshes.keys());
  }

  // Minimal document visualization sync: create simple meshes for documents and layout them in a grid
  function updateDocumentVisualization() {
    if (!scene) return;

    const existing = new Set(documentMeshes.keys());
    const count = Math.max(1, documents.length);
    const cols = Math.ceil(Math.sqrt(count));
    const spacingX = 2.0;
    const spacingY = 1.6;

    documents.forEach((doc, idx) => {
      let mesh = documentMeshes.get(doc.id);
      if (!mesh) {
        const geo = new THREE.PlaneGeometry(1.6, 1.0);
        const color = new THREE.Color().setHSL((idx / Math.max(1, count)) * 0.6, 0.6, 0.6);
        const mat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide });
        mesh = new THREE.Mesh(geo, mat);
        // attach id so we can lookup later
        (mesh as any).userData = { documentId: doc.id };
        documentMeshes.set(doc.id, mesh);
        scene.add(mesh);
      }
      existing.delete(doc.id);

      const col = idx % cols;
      const row = Math.floor(idx / cols);
      const offsetX = (cols - 1) * spacingX * 0.5;
      const offsetY = (Math.ceil(count / cols) - 1) * spacingY * 0.5;
      mesh.position.set(col * spacingX - offsetX, -(row * spacingY - offsetY), 0);
    });

    // Remove meshes for documents that no longer exist
    for (const id of existing) {
      const m = documentMeshes.get(id);
      if (m) {
        scene.remove(m);
        try {
          m.geometry?.dispose?.();
          (m.material as any)?.dispose?.();
        } catch (e) {
          // ignore disposal errors
        }
        documentMeshes.delete(id);
      }
    }
  }
</script>
<svelte:window on:resize={handleResize} />

<div
  bind:this={canvasContainer}
  class="w-full h-full relative overflow-hidden bg-slate-900 rounded-lg"
  style="min-height: 600px;"
>
  <div class="absolute top-4 left-4 bg-black/50 text-green-400 p-2 rounded text-xs font-mono">
    Documents: {documents.length}<br>
    GPU: {enableGPUAcceleration ? 'Enabled' : 'Disabled'}<br>
    Tracking: {enableInteractionTracking ? 'Active' : 'Inactive'}<br>
    MCP: Connected
  </div>

  <div class="absolute top-4 right-4 flex gap-2">
    <button
      class="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm"
      on:click={() => canvas3DActor.send({ type: 'RESET_VIEW' })}
    >
      Reset View
    </button>

    <button
      class="bg-purple-600 hover:bg-purple-700 text-white px-3 py-1 rounded text-sm"
      on:click={() => canvas3DActor.send({ type: 'TOGGLE_SOM_OVERLAY' })}
    >
      Toggle SOM
    </button>
  </div>
</div>

<style>
  :global(canvas) {
    display: block;
    outline: none;
  }
</style>
```
  // Reactive updates for documents
  $: if (documents.length > 0 && scene) {
    updateDocumentVisualization();
  }

  // Ensure renderer and camera update on window/container resize
  function handleResize() {
    if (!renderer || !camera || !canvasContainer) return;
    const w = canvasContainer.clientWidth || window.innerWidth;
    const h = canvasContainer.clientHeight || window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  // Return a simple list of visible document IDs (used when pushing context)
  function getVisibleDocuments(): string[] {
    return Array.from(documentMeshes.keys());
  }

  // Minimal document visualization sync: create simple meshes for documents and layout them in a grid
  function updateDocumentVisualization() {
    if (!scene) return;

    const existing = new Set(documentMeshes.keys());
    const count = Math.max(1, documents.length);
    const cols = Math.ceil(Math.sqrt(count));
    const spacingX = 2.0;
    const spacingY = 1.6;

    documents.forEach((doc, idx) => {
      let mesh = documentMeshes.get(doc.id);
      if (!mesh) {
        const geo = new THREE.PlaneGeometry(1.6, 1.0);
        const color = new THREE.Color().setHSL((idx / Math.max(1, count)) * 0.6, 0.6, 0.6);
        const mat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide });
        mesh = new THREE.Mesh(geo, mat);
        // attach id so we can lookup later
        (mesh as any).userData = { documentId: doc.id };
        documentMeshes.set(doc.id, mesh);
        scene.add(mesh);
      }
      existing.delete(doc.id);

      const col = idx % cols;
      const row = Math.floor(idx / cols);
      const offsetX = (cols - 1) * spacingX * 0.5;
      const offsetY = (Math.ceil(count / cols) - 1) * spacingY * 0.5;
      mesh.position.set(col * spacingX - offsetX, -(row * spacingY - offsetY), 0);
    });

    // Remove meshes for documents that no longer exist
    for (const id of existing) {
      const m = documentMeshes.get(id);
      if (m) {
        scene.remove(m);
        try {
          m.geometry?.dispose?.();
          (m.material as any)?.dispose?.();
        } catch (e) {
          // ignore disposal errors
        }
        documentMeshes.delete(id);
      }
    }
  }
</script>
<svelte:window on:resize={handleResize} />

<div
  bind:this={canvasContainer}
  class="w-full h-full relative overflow-hidden bg-slate-900 rounded-lg"
  style="min-height: 600px;"
>
  <div class="absolute top-4 left-4 bg-black/50 text-green-400 p-2 rounded text-xs font-mono">
    Documents: {documents.length}<br>
    GPU: {enableGPUAcceleration ? 'Enabled' : 'Disabled'}<br>
    Tracking: {enableInteractionTracking ? 'Active' : 'Inactive'}<br>
    MCP: Connected
  </div>

  <div class="absolute top-4 right-4 flex gap-2">
    <button
      class="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm"
      on:click={() => canvas3DActor.send({ type: 'RESET_VIEW' })}
    >
      Reset View
    </button>

    <button
      class="bg-purple-600 hover:bg-purple-700 text-white px-3 py-1 rounded text-sm"
      on:click={() => canvas3DActor.send({ type: 'TOGGLE_SOM_OVERLAY' })}
    >
      Toggle SOM
    </button>
  </div>
</div>

<style>
  :global(canvas) {
    display: block;
    outline: none;
  }
</style>
