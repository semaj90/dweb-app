// Minimal gpu-worker stub for dev
self.onmessage = function(e){
  // echo back
  self.postMessage({ok: true, data: e.data});
};
