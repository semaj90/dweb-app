// Lightweight realtime pipeline store subscribing to ws-fanout events
import { writable, derived } from 'svelte/store';

// ---- Types ----
export interface StageStatus {
	id: string;
	gpu?: boolean;
	wasm?: boolean;
	embedding?: boolean;
	retrieval?: boolean;
	llm?: boolean;
	final?: boolean;
	receivedAt?: number;
	completedAt?: number;
	// Allow additional dynamic stage flags without TS complaints
	[key: string]: any; // eslint-disable-line @typescript-eslint/no-explicit-any
}

export interface FinalResultEntry {
	id: string;
	llmResult?: any; // Domain-specific shape not enforced here
	context?: any;
	ts: number;
}

export const connectionStatus = writable<string>('disconnected');
export const stages = writable<Record<string, StageStatus>>({}); // traceId -> stage status object
export const finalResults = writable<FinalResultEntry[]>([]); // list of final LLM outputs
export const recentEvents = writable<any[]>([]); // rolling window (loosely typed)

let ws: WebSocket | null = null;

export function connectRealtime(url = 'ws://localhost:8080') {
	if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
	connectionStatus.set('connecting');
	try {
		ws = new WebSocket(url);
		ws.onopen = () => connectionStatus.set('connected');
		ws.onclose = () => { connectionStatus.set('disconnected'); setTimeout(()=>connectRealtime(url), 3000); };
		ws.onerror = () => connectionStatus.set('error');
		ws.onmessage = (ev) => {
			try {
				const data = JSON.parse(ev.data);
				handleEvent(data);
			} catch {}
		};
	} catch {
		connectionStatus.set('error');
	}
}

export function disconnectRealtime(){ if (ws){ ws.close(); ws = null; } }

function pushRecent(evt){
	recentEvents.update(list => { const next = [evt, ...list]; return next.slice(0,100); });
}

function handleEvent(wrapper){
	// wrapper shape { type, msg }
	const type = wrapper?.type;
	const msg = wrapper?.msg || {};
	pushRecent({ type, msg, at: Date.now() });
	if (type === 'ai.response') {
		const { id, stage, final } = msg;
		if (!id) return;
		stages.update(map => {
				const next: Record<string, StageStatus> = { ...map };
				const curr: StageStatus = next[id] || { id };
				if (stage) (curr as any)[stage] = true; // dynamic stage name
				if (final) { curr.final = true; curr.completedAt = Date.now(); }
				next[id] = curr; return next;
			});
		if (final){
			finalResults.update(arr => [{ id, llmResult: msg.llmResult, context: msg.context, ts: Date.now() }, ...arr].slice(0,50));
		}
	} else if (type === 'evidence.upload'){ // seed initial trace
		const id = msg?.traceId;
		if (id) { stages.update(map => { const next: Record<string, StageStatus> = { ...map }; if (!next[id]) next[id] = { id, receivedAt: Date.now() }; return next; }); }
	}
}

export const activePipelines = derived(stages, ($s) => Object.values($s as Record<string, StageStatus>).filter(v => !v.final));
export const completedPipelines = derived(stages, ($s) => Object.values($s as Record<string, StageStatus>).filter(v => v.final).sort((a, b) => ((b.completedAt || 0) - (a.completedAt || 0))).slice(0, 20));

// Convenience start on import (optional). Comment out if you prefer manual control.
if (typeof window !== 'undefined') {
	connectRealtime();
}

export default {
	connect: connectRealtime,
	disconnect: disconnectRealtime,
	connectionStatus,
	stages,
	finalResults,
	recentEvents,
	activePipelines,
	completedPipelines
};
