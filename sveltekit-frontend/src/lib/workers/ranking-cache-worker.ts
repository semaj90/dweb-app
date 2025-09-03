// ranking-cache-worker.ts
// Web Worker for WASM-accelerated ranking cache packing/unpacking & QUIC fetch

// Message protocol
// { type: 'init', wasmUrl?: string }
// { type: 'pack', payload: RankingSet }
// { type: 'unpack', blob: ArrayBuffer }
// { type: 'fetch', key: string, endpoint?: string, format?: 'raw'|'json' }

interface CanonicalResult { docId: string; score: number; flags: number; summaryHash: string; targetUrlId?: string }
interface RankingSet { results: CanonicalResult[]; query: string; totalResults: number; timestamp: number; version: number }

let wasm: any = null;
let wasmReady = false;
// Attempt to detect pre-injected WASM module (e.g., from wasm-pack bundle attaching to self.RankingWasm)
declare const self: any;
if (typeof self !== 'undefined' && self.RankingWasm) {
  wasm = self.RankingWasm;
  wasmReady = true;
}
let defaultEndpoint = '/quic/rankings';

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    switch(msg.type) {
      case 'init': {
        if (wasmReady) { (self as any).postMessage({ type:'init', ok:true }); return; }
        if (!wasmReady) {
          if (msg.wasmUrl) {
            try {
              const resp = await fetch(msg.wasmUrl);
              const bytes = await resp.arrayBuffer();
              const mod = await WebAssembly.instantiate(bytes, {});
              wasm = mod.instance.exports;
              wasmReady = true;
            } catch (e) {
              wasmReady = false; // fallback JS
            }
          } else {
            // Optional dynamic import stub (future Rust/wasm-pack bundle)
            try {
              // @ts-ignore - optional chunk, may fail silently
              const m = await import('../wasm/ranking_wasm_stub.js');
              if (m && m.default) { wasm = m.default; wasmReady = true; }
            } catch {}
          }
        }
        if (msg.endpoint) defaultEndpoint = msg.endpoint;
        (self as any).postMessage({ type:'init', ok:true, wasm: wasmReady });
        break;
      }
      case 'pack': {
        const { payload } = msg as { payload: RankingSet };
        // If future WASM export present use it, else JS fallback
        let packed: Uint8Array;
        if (wasmReady && typeof wasm.pack_rankings === 'function') {
          try {
            // Expect wasm.pack_rankings to accept JSON string and return pointer/len pair or Uint8Array.
            const json = JSON.stringify(payload);
            const res: any = wasm.pack_rankings(json);
            if (res instanceof Uint8Array) packed = res; else packed = packRankingSetJS(payload);
          } catch { packed = packRankingSetJS(payload); }
        } else {
          packed = packRankingSetJS(payload);
        }
        (self as any).postMessage({ type:'pack:done', blob: packed }, [packed.buffer]);
        break;
      }
      case 'unpack': {
        const { blob } = msg as { blob: ArrayBuffer };
        let rs: RankingSet;
        if (wasmReady && typeof wasm.unpack_rankings === 'function') {
          try {
            const u8 = new Uint8Array(blob);
            const res: any = wasm.unpack_rankings(u8);
            if (res && typeof res === 'string') { rs = JSON.parse(res); } else { rs = unpackRankingSetJS(u8); }
          } catch { rs = unpackRankingSetJS(new Uint8Array(blob)); }
        } else {
          rs = unpackRankingSetJS(new Uint8Array(blob));
        }
        (self as any).postMessage({ type:'unpack:done', rankingSet: rs });
        break;
      }
      case 'fetch': {
        const { key, endpoint, format } = msg as { key: string; endpoint?: string; format?: string };
        const url = `${endpoint || defaultEndpoint}/${encodeURIComponent(key)}${format==='json'? '?format=json':''}`;
        const res = await fetch(url);
        if (!res.ok) { (self as any).postMessage({ type:'fetch:error', error: res.status }); break; }
        if (format==='json') {
          const json = await res.json();
          (self as any).postMessage({ type:'fetch:json', data: json });
        } else {
          const buf = await res.arrayBuffer();
            (self as any).postMessage({ type:'fetch:raw', blob: buf }, [buf]);
        }
        break;
      }
    }
  } catch (err:any) {
    (self as any).postMessage({ type:'error', error: err?.message || String(err) });
  }
};

// --- Minimal JS pack/unpack (mirrors canonical-result-cache.ts logic) ---
function packRankingSetJS(rankingSet: RankingSet): Uint8Array {
  const MAX_RESULTS = 1024;
  if (!rankingSet.results.length || rankingSet.results.length>MAX_RESULTS) throw new Error('invalid result count');
  const buffer = new ArrayBuffer(8192);
  const view = new DataView(buffer);
  let offset = 0;
  const version = 1;
  const count = Math.min(rankingSet.results.length, 1023);
  view.setUint8(offset++, (version<<2)|((count>>8)&0x03));
  view.setUint8(offset++, count & 0xFF);
  view.setUint8(offset++, 0); // strategy
  view.setUint8(offset++, 0); // flags placeholder
  const crcOffset = offset; offset +=4; // reserve crc
  let prevId = '';
  for (let i=0;i<count;i++) {
    const r = rankingSet.results[i];
    const scoreQ = Math.min(1023, Math.max(0, Math.round(r.score*1023)));
    const packed16 = (scoreQ<<6) | ((r.flags & 0xF)<<2);
    view.setUint16(offset, packed16); offset+=2;
    const delta = computeDocDelta(r.docId, prevId); prevId = r.docId;
    offset = writeVarint(view, offset, delta);
    const sh = computeSummaryHash(r.summaryHash||'');
    offset = write22Bits(view, offset, sh);
    if (r.targetUrlId) { view.setUint8(offset++,1); offset = writeString(view, offset, r.targetUrlId); } else { view.setUint8(offset++,0); }
  }
  const payloadSize = offset - 8;
  const payload = new Uint8Array(buffer, 8, payloadSize);
  const crc = computeCRC32(payload);
  view.setUint32(crcOffset, crc);
  return new Uint8Array(buffer,0,offset);
}

function unpackRankingSetJS(packed: Uint8Array): RankingSet {
  const view = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  let offset=0;
  const b0=view.getUint8(offset++), b1=view.getUint8(offset++);
  const version = (b0>>2)&0x3F;
  const count = ((b0 & 0x03)<<8)|b1;
  const strategy=view.getUint8(offset++); const flags=view.getUint8(offset++);
  const crc=view.getUint32(offset); offset+=4;
  const payload = packed.slice(8);
  if (computeCRC32(payload)!==crc) throw new Error('crc mismatch');
  const results: CanonicalResult[] = [];
  let prev='';
  for (let i=0;i<count;i++) {
    const packed16 = view.getUint16(offset); offset+=2;
    const scoreQ = (packed16>>6)&0x3FF; const fl = (packed16>>2)&0xF;
    const deltaRes = readVarint(view, offset); offset = deltaRes.newOffset; const docId = applyDocDelta(prev, deltaRes.value); prev = docId;
    const shRes = read22Bits(view, offset); offset = shRes.newOffset; const sh = shRes.value.toString(16);
    const hasUrl = view.getUint8(offset++); let targetUrlId: string|undefined;
    if (hasUrl) { const urlRes = readString(view, offset); targetUrlId = urlRes.value; offset = urlRes.newOffset; }
    results.push({ docId, score: scoreQ/1023, flags: fl, summaryHash: sh, targetUrlId });
  }
  return { results, query:'', totalResults: results.length, timestamp: Date.now(), version };
}

function computeDocDelta(cur:string, prev:string){ if(!prev) return parseInt(cur)||0; return (parseInt(cur)||0)-(parseInt(prev)||0); }
function applyDocDelta(prev:string, delta:number){ if(!prev) return delta.toString(); return ((parseInt(prev)||0)+delta).toString(); }
function computeSummaryHash(s:string){ let h=0; for(let i=0;i<s.length;i++){ h=((h<<5)-h+s.charCodeAt(i)) & 0x3FFFFF;} return h; }
function writeVarint(view:DataView, offset:number, value:number){ while(value>=0x80){ view.setUint8(offset++, (value & 0xFF)|0x80); value>>>=7;} view.setUint8(offset++, value & 0xFF); return offset; }
function readVarint(view:DataView, offset:number){ let v=0, shift=0; while(true){ const b=view.getUint8(offset++); v|=(b&0x7F)<<shift; if((b&0x80)===0) break; shift+=7;} return { value:v, newOffset: offset}; }
function write22Bits(view:DataView, offset:number, value:number){ value &=0x3FFFFF; view.setUint8(offset++, (value>>16)&0xFF); view.setUint8(offset++, (value>>8)&0xFF); view.setUint8(offset++, value & 0xFF); return offset; }
function read22Bits(view:DataView, offset:number){ const b0=view.getUint8(offset++), b1=view.getUint8(offset++), b2=view.getUint8(offset++); return { value: (b0<<16)|(b1<<8)|b2, newOffset: offset}; }
function writeString(view:DataView, offset:number, str:string){ const bytes=new TextEncoder().encode(str); offset=writeVarint(view, offset, bytes.length); for(let i=0;i<bytes.length;i++){ view.setUint8(offset++, bytes[i]); } return offset; }
function readString(view:DataView, offset:number){ const lenRes=readVarint(view, offset); const len=lenRes.value; offset = lenRes.newOffset; const bytes=new Uint8Array(len); for(let i=0;i<len;i++){ bytes[i]=view.getUint8(offset++);} return { value: new TextDecoder().decode(bytes), newOffset: offset}; }
function computeCRC32(data:Uint8Array){ let crc=0xFFFFFFFF; for(let i=0;i<data.length;i++){ crc^=data[i]; for(let j=0;j<8;j++){ crc = (crc>>>1) ^ (crc & 1 ? 0xEDB88320:0); } } return (~crc)>>>0; }

export {};
