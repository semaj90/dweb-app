// Node test loader for the emscripten-generated module (works with MODULARIZE=1)
const path = require('path');
(async () => {
  const modPath = path.resolve(__dirname, '../dist/bvh_accelerator.js');
  if (!require('fs').existsSync(modPath)) {
    console.error('Module not found:', modPath);
    process.exit(1);
  }
  const modFactory = require(modPath);
  const factory = modFactory.default || modFactory.createBVHModule || modFactory.createModule || modFactory;
  const Module = await factory();
  console.log('Module ready');
  const build_index = Module.cwrap('build_index', 'void', ['number','number','number']);
  const knn_search = Module.cwrap('knn_search', 'number', ['number','number']);
  const free_index = Module.cwrap('free_index', 'void', []);

  const pts = new Float32Array([1,0,0, 0,1,0, 0,0,1, 1,1,0]);
  const dim = 3; const n = 4;
  const ptr = Module._malloc(pts.length * pts.BYTES_PER_ELEMENT);
  Module.HEAPF32.set(pts, ptr >> 2);
  build_index(ptr, dim, n);
  const q = new Float32Array([1,0,0]);
  const qptr = Module._malloc(q.length * q.BYTES_PER_ELEMENT);
  Module.HEAPF32.set(q, qptr >> 2);
  const k = 3;
  const resPtr = knn_search(qptr, k);
  if (!resPtr) { console.log('knn_search returned null'); }
  else {
    const indices = [];
    for (let i = 0; i < k; i++) indices.push(Module.HEAP32[(resPtr >> 2) + i]);
    console.log('knn indices:', indices);
  }
  free_index();
  Module._free(ptr); Module._free(qptr);
})();
