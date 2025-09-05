// wasm-ranking-cache-service.ts
// High-level client for WebWorker + (optional) WASM accelerated ranking cache.

import { browser } from '$app/environment';

export interface CanonicalResult { docId: string; score: number; flags: number; summaryHash: string; targetUrlId?: string }
export interface RankingSet { results: CanonicalResult[]; query: string; totalResults: number; timestamp: number; version: number }

interface WorkerRequestInit { type: 'init'; wasmUrl?: string; endpoint?: string }
interface WorkerRequestPack { type: 'pack'; payload: RankingSet }
interface WorkerRequestUnpack { type: 'unpack'; blob: ArrayBuffer }
interface WorkerRequestFetch { type: 'fetch'; key: string; endpoint?: string; format?: 'raw' | 'json' }

type WorkerRequest = WorkerRequestInit | WorkerRequestPack | WorkerRequestUnpack | WorkerRequestFetch

type Listener = (msg: any) => void

export class WASMRankingCacheService {
  private worker: Worker | null = null;
  private ready = false;
  private listeners = new Set<Listener>();
  private endpoint = '/quic/rankings';

  constructor(private wasmUrl?: string) {}

  async init(endpoint?: string): Promise<void> {
    if (!browser) return;
    if (endpoint) this.endpoint = endpoint;
    if (!this.worker) {
      this.worker = new Worker(new URL('../workers/ranking-cache-worker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = (ev) => {
        const data = ev.data;
        if (data?.type === 'init') { this.ready = true; }
        this.listeners.forEach(l => { try { l(data); } catch(e){} });
      };
    }
    await this.postAwait({ type: 'init', wasmUrl: this.wasmUrl, endpoint: this.endpoint }, 'init');
  }

  private post(msg: WorkerRequest) { this.worker?.postMessage(msg); }

  private postAwait(msg: WorkerRequest, waitType: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const handler = (m:any) => {
        if (m.type === 'error') { this.listeners.delete(handler); reject(new Error(m.error)); }
        if (m.type === waitType || m.type.startsWith(waitType+':')) { this.listeners.delete(handler); resolve(m); }
      };
      this.listeners.add(handler);
      this.post(msg);
      setTimeout(()=>{ this.listeners.delete(handler); reject(new Error('timeout waiting for ' + waitType)); }, 5000);
    });
  }

  async pack(r: RankingSet): Promise<Uint8Array> {
    const res = await this.postAwait({ type:'pack', payload: r }, 'pack:done');
    return new Uint8Array(res.blob);
  }

  async unpack(blob: ArrayBuffer): Promise<RankingSet> {
    const res = await this.postAwait({ type:'unpack', blob }, 'unpack:done');
    return res.rankingSet as RankingSet;
  }

  async fetchRaw(key: string): Promise<Uint8Array> {
    const res = await this.postAwait({ type:'fetch', key, endpoint: this.endpoint, format: 'raw' }, 'fetch:raw');
    return new Uint8Array(res.blob);
  }

  async fetchJsonMeta(key: string): Promise<any> {
    const res = await this.postAwait({ type:'fetch', key, endpoint: this.endpoint, format: 'json' }, 'fetch:json');
    return res.data;
  }

  isReady(): boolean { return this.ready; }
}

export const wasmRankingCacheService = new WASMRankingCacheService();
